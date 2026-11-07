import httpx
import pytest

from finrag import answer as ans
from finrag import edgar, evaluate, metrics, scope
from finrag.config import Settings

RISK = {"ticker": "ACME", "form": "10-K", "period": "2025-09-27", "section": "1A",
        "section_title": "Risk Factors", "url": "u", "fiscal_year": 2025,
        "text": "The Company depends on a single supplier in Taiwan. Net sales | 12,345"}


class FakeStore:
    def __init__(self, score=0.8):
        self.score = score

    def search(self, query, k, tickers=frozenset(), years=frozenset()):
        return [(self.score, RISK)]


@pytest.fixture
def sc(filing_10k, sec_names):
    return scope.Scope.build([filing_10k], sec_names)


S = Settings(sec_user_agent="x y@z", min_score=0.55)


# ---------------------------------------------------------------- answer path
def test_answer_with_citation(sc):
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S,
                   llm=lambda p, s: "A single supplier in Taiwan [1].")
    assert not a.refused and a.stage == "answered" and a.sources[0]["section"] == "1A"


def test_each_stage_can_refuse(sc):
    assert ans.answer("Tesla revenue?", sc, FakeStore(), S).stage == "scope"
    assert ans.answer("Who supplies Acme?", sc, FakeStore(0.3), S).stage == "evidence"
    model = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: ans.REFUSAL)
    assert model.stage == "model"
    nums = ans.answer("Acme net sales?", sc, FakeStore(), S,
                      llm=lambda p, s: "Net sales were $99,999 million [1].")
    assert nums.refused and nums.stage == "numbers" and nums.ungrounded_numbers == ["$99,999"]


# ---------------------------------------------------------------- metrics + eval
def test_metrics():
    gold = {"ticker": "ACME", "section": "1A"}
    retrieved = [{"ticker": "ACME", "section": "7"}, {"ticker": "ACME", "section": "1A"}]
    assert metrics.recall_at_k(retrieved, gold, 1) == 0.0
    assert metrics.recall_at_k(retrieved, gold, 5) == 1.0
    assert metrics.reciprocal_rank(retrieved, gold) == 0.5
    r = metrics.refusal_scores([(False, False), (False, True), (True, False), (True, True)])
    assert r["false_accept_rate"] == 0.5 and r["over_refusal_rate"] == 0.5


def test_eval_reports_gates(sc):
    suite = [
        {"id": "a", "question": "Who supplies Acme?", "answerable": True,
         "gold": {"ticker": "ACME", "form": "10-K", "section": "1A"}},
        {"id": "b", "question": "Tesla revenue?", "answerable": False},
    ]
    rep = evaluate.run(suite, sc, FakeStore(), S, llm=lambda p, s: "Taiwan supplier [1].")
    assert rep["recall@5"] == 1.0 and rep["false_accept_rate"] == 0.0
    assert all(g["pass"] for g in rep["gates"].values())


# ---------------------------------------------------------------- EDGAR client
def test_user_agent_is_required():
    with pytest.raises(ValueError):
        edgar.EdgarClient("")


def test_recent_filings_keeps_the_newest_per_form():
    payload = {"name": "Acme Widgets Inc.", "filings": {"recent": {
        "form": ["10-Q", "8-K", "10-K", "10-Q", "10-K"],
        "accessionNumber": ["a-1", "a-2", "a-3", "a-4", "a-5"],
        "filingDate": ["2026-02-01", "2026-01-15", "2025-11-01", "2025-08-01", "2024-11-01"],
        "reportDate": ["2025-12-27", "", "2025-09-27", "2025-06-28", "2024-09-28"],
        "primaryDocument": ["q.htm", "k8.htm", "k.htm", "q2.htm", "k0.htm"],
    }}}
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload)

    client = edgar.EdgarClient("Test User test@example.com", transport=httpx.MockTransport(handler))
    out = client.recent_filings("acme", 1234, ("10-K", "10-Q"), limit=1)
    assert [(f.form, f.accession) for f in out] == [("10-Q", "a-1"), ("10-K", "a-3")]
    assert out[1].url == "https://www.sec.gov/Archives/edgar/data/1234/a3/k.htm"
    assert seen[0].url == "https://data.sec.gov/submissions/CIK0000001234.json"
    assert seen[0].headers["User-Agent"] == "Test User test@example.com"
