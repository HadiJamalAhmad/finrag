"""FinanceBench as a held-out suite: questions written by financial analysts, not by us.

FinanceBench (Patronus AI, arXiv 2311.11944) publishes 150 open questions on SEC
filings, each with the answer and the page holding the evidence. License: CC BY-NC
4.0 (Hugging Face card), so non-commercial use with attribution only.

Turning it into a finrag suite, with nothing chosen by hand:
1. Keep the 10-K / 10-Q questions (finrag does not ingest 8-Ks or earnings releases).
2. Resolve each document ("3M_2018_10K", "JPMORGAN_2021Q1_10Q") to its EDGAR filing.
   The fiscal-year label is the company's, so the first candidate is the filing whose
   period ends in that year, and the evidence text decides: the filing that contains
   it is the one. Neighbouring years are tried only when the first does not.
3. Find each evidence passage in the parsed filing to get its Item: gold stays
   metadata (ticker, form, period, section), never chunk ids (CLAUDE.md).
4. Questions whose filing or evidence cannot be found are EXCLUDED with their reason,
   and the count is published. Nothing is fixed by hand.
5. Unanswerable cases come from fixed rules, no LLM: the same question about a year
   no filing covers (every year moved 15 years back), or about an SEC-listed company
   not in the corpus.

Matching PDF text (FinanceBench) against HTML text (EDGAR) needs tolerance: PDF
extraction splits words ("Cash Flow s") and reorders table cells. Both sides are
reduced to lowercase letters and digits, then compared as sets of 12-character
shingles. Coverage = share of the evidence's shingles found in the filing text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Pinned: the suite is built from this exact commit of the FinanceBench repository.
SOURCE_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"  # 2024-12-03
RAW_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/{commit}/data/{name}"
QUESTIONS_FILE = "financebench_open_source.jsonl"
META_FILE = "financebench_document_information.jsonl"

SHINGLE = 12
MIN_COVERAGE = 0.5  # set before the first build: below it, "not found"
MIN_EVIDENCE_CHARS = 40  # shorter evidence snippets fall back to the full page
YEAR_SHIFT = 15  # unanswerable year variants: every year moves this far back
# SEC-listed companies outside the FinanceBench corpus, used in rotation for the
# "company not in corpus" variants. The scope guardrail knows them from SEC's list.
ABSENT_COMPANIES = ["Tesla", "Intel", "Starbucks", "Caterpillar", "Chevron", "Honeywell"]

# FinanceBench company name -> (ticker, CIK). CIKs, not tickers, find the filings:
# tickers move (Block: SQ -> XYZ) and disappear (Activision Blizzard, acquired 2023).
COMPANIES = {
    "3M": ("MMM", 66740), "AES Corporation": ("AES", 874761), "AMD": ("AMD", 2488),
    "Activision Blizzard": ("ATVI", 718877), "Adobe": ("ADBE", 796343),
    "Amazon": ("AMZN", 1018724), "Amcor": ("AMCR", 1748790),
    "American Express": ("AXP", 4962), "American Water Works": ("AWK", 1410636),
    "Best Buy": ("BBY", 764478), "Block": ("XYZ", 1512673), "Boeing": ("BA", 12927),
    "CVS Health": ("CVS", 64803), "Coca-Cola": ("KO", 21344), "Corning": ("GLW", 24741),
    "Costco": ("COST", 909832), "Foot Locker": ("FL", 850209),
    "General Mills": ("GIS", 40704), "JPMorgan": ("JPM", 19617),
    "Johnson & Johnson": ("JNJ", 200406), "Kraft Heinz": ("KHC", 1637459),
    "Lockheed Martin": ("LMT", 936468), "MGM Resorts": ("MGM", 789570),
    "Microsoft": ("MSFT", 789019), "Netflix": ("NFLX", 1065280), "Nike": ("NKE", 320187),
    "Paypal": ("PYPL", 1633917), "PepsiCo": ("PEP", 77476), "Pfizer": ("PFE", 78003),
    "Ulta Beauty": ("ULTA", 1403568), "Verizon": ("VZ", 732712), "Walmart": ("WMT", 104169),
}

DOC_RE = re.compile(r"^(?P<name>.+)_(?P<year>\d{4})(?:Q(?P<q>[1-4]))?_(?P<form>10K|10Q)$",
                    re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")
# "FY22", as analysts write it
FY2_RE = re.compile(r"\bFY\s?(\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class DocRef:
    name: str  # FinanceBench doc_name
    form: str  # "10-K" | "10-Q"
    year: int
    quarter: int | None


def parse_doc_name(doc_name: str) -> DocRef | None:
    m = DOC_RE.match(doc_name)
    if not m:
        return None
    form = "10-K" if m["form"].upper() == "10K" else "10-Q"
    return DocRef(doc_name, form, int(m["year"]), int(m["q"]) if m["q"] else None)


def usable(questions: list[dict], meta: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split into (10-K/10-Q questions, excluded with reason)."""
    keep, out = [], []
    for q in questions:
        doc_type = meta.get(q["doc_name"], {}).get("doc_type")
        if doc_type not in ("10k", "10q"):
            out.append(_excluded(q, f"document type {doc_type!r}: finrag ingests 10-K/10-Q"))
        elif q["company"] not in COMPANIES:
            out.append(_excluded(q, f"company {q['company']!r} has no CIK mapping"))
        elif parse_doc_name(q["doc_name"]) is None:
            out.append(_excluded(q, f"unreadable doc_name {q['doc_name']!r}"))
        else:
            keep.append(q)
    return keep, out


def _excluded(q: dict, reason: str) -> dict:
    return {"id": q["financebench_id"], "doc": q["doc_name"], "reason": reason}


# --- text matching -----------------------------------------------------------------

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def shingles(norm: str, n: int = SHINGLE) -> set[str]:
    return {norm[i:i + n] for i in range(max(0, len(norm) - n + 1))}


def evidence_text(ev: dict) -> str:
    """The evidence snippet, or its full page when the snippet is too short to match."""
    snippet = ev.get("evidence_text") or ""
    if len(normalize(snippet)) >= MIN_EVIDENCE_CHARS:
        return snippet
    return ev.get("evidence_text_full_page") or snippet


def coverage(evidence: str, haystack: set[str]) -> float:
    grams = shingles(normalize(evidence))
    return len(grams & haystack) / len(grams) if grams else 0.0


def locate(evidence: str, sections: dict[str, set[str]]) -> tuple[str | None, float]:
    """The section holding most of the evidence, and that coverage."""
    best, score = None, 0.0
    for key, grams in sections.items():
        c = coverage(evidence, grams)
        if c > score:
            best, score = key, c
    return best, score


# --- filing resolution -------------------------------------------------------------

def candidates(ref: DocRef, filings: list) -> list:
    """Filings to try for a document, most likely first.

    10-K: the one whose period ends in the labelled year, then the year after and the
    year before (fiscal years named after their start, retailers' January year-ends).
    10-Q: the labelled year's quarterly reports in period order with the Nth first,
    then the rest; then the neighbouring years (fiscal quarters that straddle).
    """
    same = sorted((f for f in filings if f.form == ref.form and int(f.period[:4]) == ref.year),
                  key=lambda f: f.period)
    near = sorted((f for f in filings if f.form == ref.form
                   and int(f.period[:4]) in (ref.year + 1, ref.year - 1)),
                  key=lambda f: (int(f.period[:4]) < ref.year, f.period))
    if ref.quarter and 0 < ref.quarter <= len(same):
        nth = same[ref.quarter - 1]
        same = [nth] + [f for f in same if f is not nth]
    return same + near


def resolve(evidences: list[str], filings: list, load_grams) -> tuple[object | None, float]:
    """First candidate whose text covers the evidence (mean coverage >= MIN_COVERAGE).

    load_grams(filing) -> shingle set of the whole filing (downloads, parses, caches).
    Returns (filing, coverage), or (None, best coverage seen).
    """
    best = 0.0
    for f in filings:
        grams = load_grams(f)
        if grams is None:
            continue
        c = sum(coverage(e, grams) for e in evidences) / len(evidences)
        if c >= MIN_COVERAGE:
            return f, c
        best = max(best, c)
    return None, best


# --- unanswerable variants ---------------------------------------------------------

def year_variant(question: str) -> str | None:
    """Every year moved YEAR_SHIFT years back: a period no filing in the corpus covers."""
    if not (YEAR_RE.search(question) or FY2_RE.search(question)):
        return None
    q = YEAR_RE.sub(lambda m: str(int(m.group(1)) - YEAR_SHIFT), question)
    return FY2_RE.sub(lambda m: f"FY{(int(m.group(1)) - YEAR_SHIFT) % 100:02d}", q)


def company_variant(question: str, company: str, absent: str) -> str | None:
    """The same question about a company outside the corpus; None if the name is absent."""
    pattern = re.compile(re.escape(company), re.IGNORECASE)
    if not pattern.search(question):
        return None
    return pattern.sub(absent, question)


def unanswerable(cases: list[dict]) -> list[dict]:
    """Fixed rule: cases sorted by id alternate between the two variants."""
    out = []
    for i, c in enumerate(sorted(cases, key=lambda c: c["id"])):
        if i % 2 == 0:
            q = year_variant(c["question"])
            why = "fiscal year not covered (year moved back)"
            vid = f"{c['id']}-year"
        else:
            absent = ABSENT_COMPANIES[(i // 2) % len(ABSENT_COMPANIES)]
            q = company_variant(c["question"], c["company"], absent)
            why = f"company not in corpus ({absent})"
            vid = f"{c['id']}-company"
        if q:
            out.append({"id": vid, "question": q, "answerable": False, "why": why,
                        "type": "unanswerable-rule", "source": "FinanceBench (variant)"})
    return out


# --- suite file --------------------------------------------------------------------

def to_case(q: dict, filing, sections: list[str]) -> dict:
    return {
        "id": q["financebench_id"],
        "question": q["question"],
        "answerable": True,
        "gold": {"ticker": filing.ticker, "form": filing.form, "period": filing.period,
                 "section": sections if len(sections) > 1 else sections[0]},
        "type": q["question_type"],
        "answer": q["answer"],
        "company": q["company"],
        "doc": q["doc_name"],
        "source": "FinanceBench",
    }


def write_suite(path: Path, cases: list[dict]) -> str:
    """Write the suite and its lock file; returns the SHA-256 recorded in the lock."""
    body = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    path.write_text(body, encoding="utf-8", newline="\n")
    digest = sha256(path)
    lock_path(path).write_text(digest + "\n", encoding="utf-8", newline="\n")
    return digest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


# --- build (network: GitHub for FinanceBench, EDGAR for the filings) ---------------

def download_source(dest: Path, commit: str = SOURCE_COMMIT) -> tuple[list[dict], dict]:
    """FinanceBench's two JSONL files at the pinned commit, cached under dest."""
    import httpx

    dest.mkdir(parents=True, exist_ok=True)
    for name in (QUESTIONS_FILE, META_FILE):
        path = dest / name
        if not path.exists():
            r = httpx.get(RAW_URL.format(commit=commit, name=name), timeout=60,
                          follow_redirects=True)
            r.raise_for_status()
            path.write_bytes(r.content)
    questions = _jsonl(dest / QUESTIONS_FILE)
    meta = {d["doc_name"]: d for d in _jsonl(dest / META_FILE)}
    return questions, meta


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build(client, settings, source_dir: Path, suite_path: Path, log=print) -> dict:
    """Resolve filings, locate evidence, write the suite + lock + build report.

    settings: a Settings whose data_dir is the BENCH corpus (never the demo's data/).
    Filings land in settings.raw_dir and settings.manifest_path, ready for `ingest`.
    """
    from finrag import edgar, parse

    if suite_path.exists():
        raise FileExistsError(
            f"{suite_path} exists and is frozen: a new build is a new version (new file).")
    questions, meta = download_source(source_dir)
    keep, excluded = usable(questions, meta)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.sec_names_path.write_text(json.dumps(client.company_tickers()), encoding="utf-8")

    history: dict[str, list] = {}  # company -> its 10-K/10-Q filings
    by_doc: dict[str, list[dict]] = {}
    for q in keep:
        by_doc.setdefault(q["doc_name"], []).append(q)

    def text_of(f) -> str:
        path = settings.raw_dir / f"{f.doc_id}.html"
        if not path.exists():
            path.write_text(client.download(f), encoding="utf-8")
        return parse.html_to_text(path.read_text(encoding="utf-8"))

    cases, resolved, details = [], {}, []
    for doc, qs in sorted(by_doc.items()):
        ref = parse_doc_name(doc)
        ticker, cik = COMPANIES[qs[0]["company"]]
        if ticker not in history:
            history[ticker] = client.all_filings(ticker, cik, ("10-K", "10-Q"))
        evidences = [evidence_text(e) for q in qs for e in q["evidence"]]
        filing, cov = resolve(evidences, candidates(ref, history[ticker]),
                              lambda f: shingles(normalize(text_of(f))))
        if filing is None:
            log(f"{doc}: filing NOT found (best coverage {cov:.2f})")
            excluded += [_excluded(q, f"filing not found on EDGAR (best coverage {cov:.2f})")
                         for q in qs]
            continue
        log(f"{doc}: {filing.form} period {filing.period} ({filing.accession}), "
            f"coverage {cov:.2f}")
        resolved[filing.accession] = filing
        sections = {s.key: shingles(normalize(s.text))
                    for s in parse.split_sections(text_of(filing), filing.form)}
        for q in qs:
            found = []
            for e in q["evidence"]:
                key, c = locate(evidence_text(e), sections)
                found.append((key if c >= MIN_COVERAGE else None, round(c, 3)))
            keys = sorted({k for k, _ in found if k})
            details.append({"id": q["financebench_id"], "doc": doc,
                            "accession": filing.accession, "evidence": found})
            if not keys:
                best = max(c for _, c in found)
                excluded.append(_excluded(q, f"evidence not located in any Item "
                                             f"(best coverage {best:.2f})"))
                continue
            cases.append(to_case(q, filing, keys))

    edgar.save_manifest(settings.manifest_path, list(resolved.values()))
    negatives = unanswerable(cases)
    digest = write_suite(suite_path, cases + negatives)
    report = {
        "source": {"dataset": "FinanceBench (Patronus AI), open-source sample",
                   "commit": SOURCE_COMMIT, "license": "CC BY-NC 4.0",
                   "paper": "arXiv:2311.11944"},
        "rules": {"shingle": SHINGLE, "min_coverage": MIN_COVERAGE,
                  "year_shift": YEAR_SHIFT, "absent_companies": ABSENT_COMPANIES},
        "counts": {"questions": len(questions), "answerable_kept": len(cases),
                   "unanswerable_variants": len(negatives), "excluded": len(excluded),
                   "filings": len(resolved)},
        "suite_sha256": digest,
        "excluded": excluded,
        "evidence": details,
    }
    report_path = suite_path.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    return report
