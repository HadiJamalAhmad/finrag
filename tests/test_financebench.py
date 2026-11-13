"""FinanceBench held-out suite: conversion rules, evidence matching, freezing. Offline."""

from pathlib import Path

import httpx
import pytest

from finrag import edgar, evaluate, metrics
from finrag import financebench as fb
from finrag.edgar import Filing

SUITE = Path(__file__).resolve().parents[1] / "eval" / "suites" / "financebench.jsonl"


def _filing(form: str, period: str, acc: str = "a") -> Filing:
    return Filing(ticker="MMM", company="3M CO", cik=66740, form=form, accession=acc,
                  filed=period, period=period, url=f"https://x/{acc}")


def test_doc_names_are_parsed():
    assert fb.parse_doc_name("3M_2018_10K") == fb.DocRef("3M_2018_10K", "10-K", 2018, None)
    assert fb.parse_doc_name("JPMORGAN_2021Q1_10Q").quarter == 1
    assert fb.parse_doc_name("Pfizer_2023Q2_10Q").form == "10-Q"
    assert fb.parse_doc_name("3M_2023_8K") is None


def test_only_10k_10q_with_a_known_company_are_kept():
    meta = {"A_2020_10K": {"doc_type": "10k"}, "B_2020_8K": {"doc_type": "8k"},
            "C_2020_10K": {"doc_type": "10k"}}
    qs = [{"financebench_id": "1", "doc_name": "A_2020_10K", "company": "3M"},
          {"financebench_id": "2", "doc_name": "B_2020_8K", "company": "3M"},
          {"financebench_id": "3", "doc_name": "C_2020_10K", "company": "Nowhere Corp"}]
    keep, out = fb.usable(qs, meta)
    assert [q["financebench_id"] for q in keep] == ["1"]
    assert {o["id"]: o["reason"][:13] for o in out} == {"2": "document type",
                                                         "3": "company 'Nowh"}


def test_pdf_text_matches_html_text_despite_split_words():
    html = "Consolidated Statement of Cash Flows | Purchases of property, plant | (1,577)"
    pdf = "Consolidated Statement of Cash Flow s\n Purchases of property,\nplant\n(1,577)"
    assert fb.coverage(pdf, fb.shingles(fb.normalize(html))) == 1.0
    assert fb.coverage("Net revenue of the Lunar segment", fb.shingles(fb.normalize(html))) < 0.2


def test_evidence_is_located_in_its_item():
    sections = {"7": fb.shingles(fb.normalize("Revenue grew because of strong demand.")),
                "8": fb.shingles(fb.normalize("Purchases of property, plant and equipment"))}
    assert fb.locate("purchases of property plant and equipment", sections)[0] == "8"


def test_short_evidence_falls_back_to_the_full_page():
    ev = {"evidence_text": "$1,577", "evidence_text_full_page": "The full page " * 10}
    assert fb.evidence_text(ev).startswith("The full page")


def test_10k_candidates_start_with_the_labelled_year():
    files = [_filing("10-K", "2019-12-31", "k19"), _filing("10-K", "2018-12-31", "k18"),
             _filing("10-K", "2017-12-31", "k17"), _filing("10-Q", "2018-06-30", "q")]
    got = [f.accession for f in fb.candidates(fb.parse_doc_name("3M_2018_10K"), files)]
    assert got == ["k18", "k19", "k17"]


def test_10q_candidates_start_with_the_nth_quarter():
    files = [_filing("10-Q", "2023-03-31", "q1"), _filing("10-Q", "2023-06-30", "q2"),
             _filing("10-Q", "2023-09-30", "q3")]
    got = [f.accession for f in fb.candidates(fb.parse_doc_name("3M_2023Q2_10Q"), files)]
    assert got[0] == "q2" and set(got) == {"q1", "q2", "q3"}


def test_the_filing_holding_the_evidence_wins():
    texts = {"k18": "nothing to see", "k19": "capital expenditure was 1,577 million"}
    files = [_filing("10-K", "2018-12-31", "k18"), _filing("10-K", "2019-12-31", "k19")]
    f, c = fb.resolve(["Capital expenditure was 1,577 million"], files,
                      lambda f: fb.shingles(fb.normalize(texts[f.accession])))
    assert f.accession == "k19" and c == 1.0
    f, c = fb.resolve(["Something no filing says at all"], files,
                      lambda f: fb.shingles(fb.normalize(texts[f.accession])))
    assert f is None and c < fb.MIN_COVERAGE


def test_a_partial_match_on_comparatives_loses_to_the_real_filing():
    """Build v1 bug: the prior-year 10-K (0.68, comparatives) beat the real one."""
    texts = {"k21": "revenue 100 cost 60", "k22": "revenue 100 cost 60 margin 40 tax 9"}
    files = [_filing("10-K", "2022-01-02", "k21"), _filing("10-K", "2023-01-01", "k22")]
    f, _ = fb.resolve(["revenue 100 cost 60 margin 40 tax 9"], files,
                      lambda f: fb.shingles(fb.normalize(texts[f.accession])))
    assert f.accession == "k22"


def test_a_tie_keeps_the_labelled_year():
    files = [_filing("10-K", "2018-12-31", "k18"), _filing("10-K", "2019-12-31", "k19")]
    f, _ = fb.resolve(["capital expenditure 1,577"], files,
                      lambda f: fb.shingles(fb.normalize("capital expenditure 1,577")))
    assert f.accession == "k18"


def test_year_variant_moves_every_year_out_of_the_corpus():
    q = fb.year_variant("What is 3M's FY2018 capex, and the FY22 quick ratio vs 2021?")
    assert q == "What is 3M's FY2003 capex, and the FY07 quick ratio vs 2006?"
    assert fb.year_variant("Does 3M pay a stable dividend?") is None


def test_company_variant_swaps_the_name_or_gives_up():
    assert fb.company_variant("Is Paypal's margin up?", "PayPal", "Tesla") == \
        "Is Tesla's margin up?"
    assert fb.company_variant("Is the margin up?", "PayPal", "Tesla") is None


def test_unanswerable_variants_alternate_by_rule():
    cases = [{"id": f"id{i}", "question": f"3M revenue in FY20{10 + i}?", "company": "3M"}
             for i in range(4)]
    out = fb.unanswerable(cases)
    assert [c["id"] for c in out] == ["id0-year", "id1-company", "id2-year", "id3-company"]
    assert all(not c["answerable"] and c["why"] for c in out)
    assert "Tesla" in out[1]["question"] and "2010" not in out[0]["question"]


def test_suite_and_lock_are_written_together(tmp_path):
    path = tmp_path / "s.jsonl"
    digest = fb.write_suite(path, [{"id": "x", "question": "q", "answerable": False}])
    assert fb.lock_path(path).read_text().strip() == digest == fb.sha256(path)


def test_build_refuses_to_overwrite_a_frozen_suite(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("{}\n")
    with pytest.raises(FileExistsError):
        fb.build(client=None, settings=None, source_dir=tmp_path, suite_path=path)


@pytest.mark.skipif(not SUITE.exists(), reason="suite not built yet")
def test_the_frozen_suite_has_not_changed():
    """Editing the held-out suite is a new version (new file), never a quiet fix."""
    assert fb.sha256(SUITE) == fb.lock_path(SUITE).read_text().strip()


def test_all_filings_reads_the_paged_history():
    recent = {"form": ["10-K", "424B2"], "accessionNumber": ["k25", "p"],
              "filingDate": ["2025-02-01", "2025-01-01"],
              "reportDate": ["2024-12-31", ""], "primaryDocument": ["k.htm", "p.htm"]}
    older = {"form": ["10-Q"], "accessionNumber": ["q21"], "filingDate": ["2021-05-01"],
             "reportDate": ["2021-03-31"], "primaryDocument": ["q.htm"]}
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path.endswith("CIK0000019617.json"):
            return httpx.Response(200, json={"name": "JPMORGAN CHASE & CO", "filings": {
                "recent": recent, "files": [{"name": "CIK0000019617-submissions-001.json"}]}})
        return httpx.Response(200, json=older)

    client = edgar.EdgarClient("Test User t@finrag.test", transport=httpx.MockTransport(handler))
    out = client.all_filings("jpm", 19617, ("10-K", "10-Q"))
    assert [(f.form, f.accession) for f in out] == [("10-K", "k25"), ("10-Q", "q21")]
    assert seen[1] == "https://data.sec.gov/submissions/CIK0000019617-submissions-001.json"


def test_gold_may_list_several_sections():
    gold = {"ticker": "MMM", "section": ["7", "8"]}
    assert metrics.is_relevant({"ticker": "MMM", "section": "8"}, gold)
    assert not metrics.is_relevant({"ticker": "MMM", "section": "1A"}, gold)


def test_wilson_interval_and_mcnemar():
    assert metrics.wilson(0, 0) is None
    lo, hi = metrics.wilson(55, 60)
    assert lo < 55 / 60 < hi and hi <= 1.0
    assert metrics.mcnemar(0, 0) == 1.0
    assert metrics.mcnemar(3, 2) > 0.5  # 3 vs 2 disagreements: no evidence either way
    assert metrics.mcnemar(12, 1) < 0.01


def test_compare_pairs_cases_by_id():
    a = {"cases": [{"id": "1", "answerable": True, "refused": False},
                   {"id": "2", "answerable": False, "refused": True},
                   {"id": "3", "answerable": True, "refused": True}]}
    b = {"cases": [{"id": "1", "answerable": True, "refused": True},
                   {"id": "2", "answerable": False, "refused": True},
                   {"id": "3", "answerable": True, "refused": True}]}
    r = evaluate.compare(a, b)
    assert (r["n_paired"], r["only_a"], r["only_b"]) == (3, ["1"], [])
