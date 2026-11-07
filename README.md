# finrag: answers from SEC filings, with citations, refusals and an evaluation bench

Ask a question about a company's annual (10-K) or quarterly (10-Q) report and get an
answer that cites the exact filing and section. When the filings do not contain the
answer, the system says so. Every figure in an answer is checked against the cited
passage before it is shown.

Built on the same principles as RAG Wikipedia: no RAG framework, every step explicit
and unit-tested, numbers measured on a held-out suite and never relaxed to look good.

## Why finance needs more than plain RAG

| Risk | Guardrail | Where |
|---|---|---|
| Question about a company that is not in the corpus: retrieval still returns *something* | **Scope check** before retrieval, using SEC's list of every listed company | `scope.py` |
| Question about a year no filing covers | **Year check**: a 10-K covers its year and the two before it | `scope.py` |
| Nothing relevant retrieved | **Evidence gate** on the best retrieval score | `answer.py` |
| The model answers anyway | Exact refusal phrase in the prompt, detected | `answer.py` |
| **A wrong figure stated with confidence** | **Numeric grounding**: every number in the answer must appear in the cited passages (scale-aware: "$391.0 billion" matches "391,035" in millions) | `grounding.py` |

Each refusal says which step decided, so a wrong refusal can be traced and fixed.

## Pipeline

```
SEC EDGAR (free, official) ─► edgar.py   download 10-K / 10-Q, manifest
                           ─► parse.py   HTML → text, split by Item (1A Risk Factors, 7 MD&A…)
                           ─► chunk.py   section-aware chunks, each carrying its citation
                           ─► store.py   bge-small-en-v1.5 (fastembed) → Qdrant (local mode)
question ─► scope ─► retrieve ─► evidence gate ─► llama3.2:3b (Ollama) ─► numeric check ─► answer + sources
```

## Quick start

```bash
cp .env.example .env            # then put YOUR name and email in SEC_USER_AGENT
uv sync --extra dev
uv run pytest -q                # 21 tests, offline
uv run finrag fetch AAPL MSFT NVDA JPM --forms 10-K --limit 1
uv run finrag ingest
uv run finrag ask "What risks does Apple describe about its suppliers?"
uv run finrag eval eval/suites/seed.jsonl --retrieval-only   # fast, no LLM
uv run finrag eval eval/suites/seed.jsonl                    # full, with the LLM
uv run uvicorn finrag.api:app --reload                       # POST /query
```

SEC requires every automated client to send a User-Agent with a name and a contact
address, and allows at most 10 requests per second. `edgar.py` enforces both.

## Evaluation

`eval/suites/seed.jsonl`: 20 cases to start (12 answerable, 8 unanswerable). An
answerable case names the filing section that holds the answer
(`{"ticker": "AAPL", "form": "10-K", "section": "1A"}`), so the suite stays valid
when chunking changes. Unanswerable cases cover four failure types: a company not in
the corpus, a year not covered, a forecast, and a detail no filing contains.

Gates, written before the first run: recall@5 ≥ 0.80, MRR ≥ 0.60, false accept rate ≤ 0.10.
`finrag eval` exits non-zero when a gate fails.

## Status (2026-11-07): starting point

Done and tested: EDGAR client, parser, chunker, both guardrails, evidence gate,
answer path, metrics, eval runner, CLI, API. 21 offline tests, ruff clean.
Verified end to end on a fixture filing: real embeddings and Qdrant retrieve the
right Item every time (supplier risk → Item 1A at 0.806, net sales → Item 7 at
0.743), and the scope check refuses Tesla and fiscal 2018.

Not yet run: the LLM step on this PC (Ollama could not get enough free memory
during the first test) and anything on real SEC data (needs your SEC_USER_AGENT).

First findings to act on:
- **The 0.55 evidence threshold is too low.** On the fixture, an off-topic question
  ("the CEO's dog") scored 0.561 and would reach the model. Calibrate on the real
  corpus with `--retrieval-only`, then consider a term-coverage gate like RAG Wikipedia's.
- **JPMorgan files most of its MD&A (Items 7 / 7A) as an exhibit** (the annual report,
  Exhibit 13), not in the 10-K body. Check `finrag ingest`'s section list for JPM
  after the first fetch; the `jpm-market-risk` case may need the exhibit ingested.

## Plan: 4 weeks

1. **Week 1: real data.** Fetch the 10-Ks for AAPL, MSFT, NVDA, JPM. Check every
   filing's section list, fix the parser on what breaks. Calibrate `min_score`.
2. **Week 2: the bench.** Grow the suite to 60 held-out questions, including
   numeric ones with gold figures ("Apple's total net sales in fiscal 2025").
   Add a numeric accuracy metric. First full report, red numbers included.
3. **Week 3: harder questions.** Comparisons across years and companies,
   10-Q quarters, earnings-call transcripts if a free licensed source fits.
   Try hybrid search (BM25 + vectors) and measure it against vectors alone.
4. **Week 4: show it.** React front end reusing RAG Wikipedia's, deployment
   (Modal GPU + Vercel), a silent captioned demo video, and a one-page write-up.
