# FinanceBench held-out suite: protocol

Written and committed **before** the suite was built and before any system ran on it.
The commit order in git is the proof.

## Why

The first bake-off ran on `seed.jsonl`: 20 questions written by us, and the same 20
used to tune `min_score`. It can rank models; it cannot settle a choice. This suite
uses questions nobody on this project wrote: the open sample of FinanceBench
(Patronus AI, [arXiv 2311.11944](https://arxiv.org/abs/2311.11944)), 150 questions
on SEC filings written and checked by financial analysts.

`seed.jsonl` stays the **development** set (tune on it freely). This suite is the
**test** set: never used to tune anything.

## Source and license

- Repository `patronus-ai/financebench`, pinned commit `cc39aeb4afdf33909ee1412188bf89035950c2eb`.
- License: CC BY-NC 4.0 (Hugging Face dataset card). Non-commercial use with
  attribution: fine for this portfolio project, not for client work.
- The filings themselves come from SEC EDGAR (public), fetched by `finrag bench-build`
  with the User-Agent from `.env`, under 10 requests per second.

## Conversion rules (code: `src/finrag/financebench.py`)

1. Keep questions on 10-K and 10-Q documents (127 of 150). 8-Ks and earnings
   releases are excluded: finrag does not ingest them.
2. Resolve each document to its EDGAR filing: the candidate whose text covers the
   evidence **best** (mean coverage of 12-character shingles, both texts reduced to
   lowercase letters and digits), and only if that coverage is >= **0.5**. Candidates
   in a fixed order: labelled fiscal year first, then the year after, then the year
   before; a tie keeps the earlier one; a coverage >= 0.99 ends the search.
3. Gold = (ticker, form, period, section), the Item that contains each evidence
   passage (same 0.5 coverage rule). Several evidence passages may give several Items;
   any of them counts as relevant.
4. A question whose filing or evidence cannot be found is **excluded**, with its
   reason, in `financebench.build.json`. Nothing is fixed by hand.
5. Unanswerable cases by fixed rule, no LLM: answerable cases sorted by id alternate
   between (a) every year moved 15 years back and (b) the company name replaced by an
   SEC-listed company outside the corpus (Tesla, Intel, Starbucks, Caterpillar,
   Chevron, Honeywell, in rotation). Cases whose question has no year (a) or does not
   contain the company name (b) give no variant.
   Limit, stated now: these test the **scope** guardrail. The harder unanswerable
   kinds (a forecast, a detail no filing contains) stay in `seed.jsonl`.
6. The suite is written once with its SHA-256 in `financebench.sha256`; a test fails
   if the file changes. A correction is a new version (new file), never an edit.

## Build history

- **v1 (2026-11-13), discarded before any system ran on it.** Rule 2 then took the
  *first* candidate above 0.5. A 10-K repeats the two previous years' statements, so
  a neighbouring year's filing could pass on its comparatives: J&J "2022" resolved
  to the FY2021 10-K (coverage 0.68), and CVS 2018 (0.51), Amcor 2023Q2 (0.58) and
  Best Buy 2024Q2 (0.62) were partial matches of the same kind. Rule 2 now takes the
  best candidate (test: `test_a_partial_match_on_comparatives_loses_to_the_real_filing`).
  v1 counts, for the record: 122 answerable kept, 113 unanswerable variants,
  28 excluded (23 document type, 5 evidence outside any parsed Item), 72 filings.
- **v2**: the frozen suite (`financebench.jsonl`, hash in `financebench.sha256`).

## Corpus

Its own directory, `data-bench/` (gitignored), with its own index. The demo corpus
(`data/`: AAPL, MSFT, NVDA, JPM) is untouched.

## Metrics and gates

- Gates unchanged (`evaluate.GATES`): recall@5 >= 0.80, MRR >= 0.60, false accept
  rate <= 0.10. **Never relaxed to pass.**
- Every rate is reported with its 95% Wilson interval, and broken down by
  FinanceBench question type (metrics-generated, domain-relevant, novel-generated).
- Recall is measured on unfiltered search over the whole bench corpus (72 filings,
  32 companies), the strictest reading.

## Expected findings, written before the run

- Over half of FinanceBench needs a calculation (a ratio, a margin, a change). The
  numeric grounding check refuses figures that do not appear in the sources, so the
  plain RAG path should **over-refuse** those questions. That is the cost of the
  design (a refusal rather than an invented figure), to be measured, not tuned away.
  The agent mode (`calculate` tool, operands only from tool results) should recover
  part of it.
- Questions written with two-digit years ("FY22") are not read by the scope
  guardrail's year rule; the year variants of those questions may be answered.

## Next phases (rules to be fixed before each runs)

- **Answer accuracy (full mode).** Numeric answers: correct within 1% of
  FinanceBench's figure. Text answers: a judge model from a different family than
  the one evaluated, with 20 judgements checked by hand to report agreement.
- **Model choice.** Proposed rule, to be confirmed before the run: change the default
  model only if the candidate keeps false accept <= 0.10 and answers more answerable
  questions correctly than `llama-3b`, with McNemar p < 0.05 on the paired cases
  (`finrag compare`). Otherwise the cheaper model stays.
