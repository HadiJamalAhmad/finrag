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

## Agent mode: tools, under a guard (`finrag agent`)

Some questions need several steps: "How much did Microsoft's revenue grow in fiscal
2026?" needs two reported figures and a calculation. In agent mode the model plans
those steps by calling tools, in a LangGraph loop (`agent.py`):

```
START ─► scope ─(out of scope)─► refused: scope
           │
           ▼
         model ─(tool calls)─► tools ─► model …      every call goes through ToolGuard
           │  └─(budget spent)─► refused: limit
           ▼
         verify ─► answer, or refused: model / evidence / numbers
```

| Tool (read-only) | What it returns |
|---|---|
| `search_filings(query, ticker?, fiscal_year?)` | Filing passages above the evidence gate, each with a source id |
| `get_financial_fact(ticker, metric, fiscal_year)` | A figure the company reported in its 10-K, from SEC XBRL data (full-year only, latest filing wins) |
| `calculate(expression)` | Arithmetic **only on figures a tool already returned**; any other operand is refused |

`ToolGuard` (`tools.py`) sits between the model and the tools: an allowlist of tool
names, pydantic validation of every argument, companies limited to the corpus, a
budget of tool calls per question, and an audit log of every proposal (accepted or
refused, with arguments and latency), returned with the answer. The model sits behind
a two-method `ChatModel` interface, so Mistral, OpenAI or Claude can replace Ollama
without touching the graph.

```bash
uv run finrag agent "How much did Microsoft's revenue grow in fiscal 2026?"
# POST /agent  {"question": "..."}   -> answer, stage, sources, steps (audit log)
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

## Deployment (2026-11-09)

Live: **https://finrag-ten.vercel.app** (static page + proxy on Vercel, backend on Modal).

```
browser ─► Vercel  web/index.html, web/api/{query,agent}.js   (adds x-api-key server-side)
        ─► Modal   deploy/modal_app.py: finrag.api + Ollama, 3 models, one 24 GB GPU (L4 / A10G)
```

- **Backend (Modal).** The image bakes in the index (`data/qdrant`), the manifest, the
  SEC company list, the cached company facts (`data/facts`), the bge-small embedder and
  the three models of `deploy/models.lock`. The build fails if a pulled digest differs.
  SEC is never called from Modal: the User-Agent stays in your `.env`.
- **Auth.** Every route but `/health` needs `x-api-key` (Modal secret `finrag-api`).
  Only the Vercel functions hold the key (env `FINRAG_API_KEY`, `FINRAG_BACKEND_URL`).
- **Cost.** At most one container; it scales to zero after 5 idle minutes. A cold start
  (GPU + model load) takes 1.5 to 2 minutes; a model not yet loaded adds its load time.
  Every answer reports its tokens, model seconds and estimated GPU cost.
- **CI/CD** (`.github/workflows/ci.yml`). Every push and PR: ruff, the offline tests,
  `pip-audit` on the locked dependencies and `gitleaks` on the whole history. A push to
  `main` with all of them green deploys the backend to Modal (secrets `MODAL_TOKEN_ID`,
  `MODAL_TOKEN_SECRET`); Vercel builds `web/` from the same push (Root Directory = web).
- **The corpus is a Modal Volume** (`finrag-data`), not part of the image: `data/` is
  gitignored, so CI deploys code and models while the index stays what was uploaded.
  After any re-ingest, from the machine that ran it:

```bash
uv sync --extra dev --extra deploy
uv run modal run deploy/modal_app.py::upload_data   # data/ -> Volume finrag-data
uv run modal deploy deploy/modal_app.py             # or push to main and let CI do it
```

First checks on the live site: NVIDIA export controls answered from Item 1A; JPMorgan
market risk answered from Item 7A (the Week 1 cross-reference fix); Tesla refused at
`scope`; the NVIDIA share-price forecast passes the evidence gate (0.799) and is
refused at `model`. The agent route works but llama3.2:3b sometimes calls `calculate`
before fetching any figure; the guard refuses it and the answer is a refusal.

## Three models (2026-11-09)

Like JobHack: a registry of open-weight models (`src/finrag/models.py`), each pinned by
digest (`deploy/models.lock`), chosen by `FINRAG_MODEL`, `--model`, or per request in the
API and on the web page. No question or filing passage leaves our infrastructure.

| Preset | Model | Context / output cap | Notes |
|---|---|---|---|
| `llama-3b` (default) | Meta Llama 3.2 3B | 4096 / 1024 | the baseline of every earlier number |
| `qwen3-4b` | Alibaba Qwen3 4B | 12288 / 8192 | always reasons; `think: true` keeps the reasoning out of the answer |
| `mistral-24b` | Mistral Small 3.2 24B | 4096 / 1024 | ~15 GB at 4-bit: needs a 24 GB GPU |

**Bake-off** (`uv run modal run deploy/modal_app.py::bakeoff_all`): the seed suite, full
pipeline with the LLM, one GPU container per model.

| Model | GPU | Gates | False accept | Over-refusal | p50 | p95 | Output tokens | Est. cost / LLM answer |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 3B | L4 | all PASS | 0.00 | **0.083** (1/12) | **3.4 s** | 47 s | 5,273 | **$0.0020** |
| Mistral Small 24B | A10G | all PASS | 0.00 | 0.167 (2/12) | 7.9 s | 43 s | 4,134 | $0.0044 |
| Qwen3 4B | A10G | all PASS | 0.00 | 0.250 (3/12) | 10.7 s | 80 s | 35,533 | $0.0085 |

Read with care:
- **20 cases, tuned on the same seed suite, not held-out.** One case moves over-refusal
  by 0.083. It ranks the models on this suite; it does not settle the choice.
- **Not the same GPU** (Modal had no L4 free for two runs): latency and cost compare
  only roughly. Cost = model seconds x the GPU's list price; idle and cold starts excluded.
  p95 includes the first call, which loads the model.
- **False accepts 0.00 for all three**: the NVIDIA share-price forecast passes the
  evidence gate (0.799) and every model refuses it at `model`. Retrieval-only mode
  (scope + evidence gate only) still reports 0.125 with `min_score` 0.72.
- Over-refusals by stage: Llama: `nvda-datacenter` (numbers). Mistral: `nvda-export`
  (numbers), `jpm-legal` (model). Qwen: `aapl-net-sales-drivers` and `nvda-export`
  (model), `msft-ai-risk` (numbers). Each is a case to read, not a number to tune away.
- **Qwen3 4B reasons on every answer**: 7x Llama's output tokens, 4x its cost, the
  slowest. First live run (before `think: true` and the larger context): 6,891 output
  tokens for one answer, then a refusal. Same lesson as JobHack L84.
- **Found on the live site, not in the suite:** "Who supplies Apple?" is refused at
  `evidence` (0.713 < 0.72). Terse answerable questions sit near the threshold: add such
  cases to the Week 2 suite before touching `min_score`.

The default stays `llama-3b`: best on this suite, cheapest, fastest. Mistral is the
candidate to re-measure on the held-out Week 2 suite (and in the agent, where a 3B model
struggles with multi-step tool use).

## GraphQL

`POST /graphql` (Strawberry; explorer at `GET /graphql`, live at
https://finrag-ten.vercel.app/api/graphql) exposes the same answer path as REST, typed:

```graphql
{ models { name provider available } }
{ answer(question: "How does JPMorgan Chase measure and manage market risk?",
         model: "mistral-24b", mode: PIPELINE) {
    answer stage reason
    sources { ticker section score }
    usage { model promptTokens completionTokens estCostUsd } } }
```

Not a second implementation: each resolver calls the REST handlers, so the scope check,
the evidence gate, numeric grounding, the ToolGuard (`mode: AGENT`), the cost estimate
and the Langfuse trace are identical. Input is validated by the same pydantic model.
Queries go by POST only: a cross-site GET can never start a paid model call. The model
call runs off the event loop (async resolver, thread pool).

## Tracing (Langfuse)

Every question becomes one Langfuse trace, sent by the official Python SDK
(`langfuse` 5.2, OpenTelemetry), following the Langfuse skill (github.com/langfuse/skills)
and Langfuse's observability best practices. The first version posted raw events to
`/api/public/ingestion`, which Langfuse Cloud stops accepting for traces on 2026-11-16.

```
answer-question      chain        input: question          output: the answer (or refusal)
  check-scope        guardrail    WARNING when it refuses
  retrieve-passages  retriever    the exact passages (text + scores) the model will see
  check-evidence     guardrail    best score vs min_score
  generate-answer    generation   messages, model, parameters, tokens, cost, reasoning
  check-figures      guardrail    grounded? ungrounded figures

answer-with-tools    agent        one generation per model turn; each tool call a `tool`
  generate-response  generation   sibling of the generation that asked for it, with its
  get-financial-fact tool         tool_call_id; refused calls are WARNING; verify-answer last
```

- **Names are stable and verb-first** (dashboards and evaluators key on them); dynamic
  values go to metadata.
- **Tags at creation** (immutable): `finrag`, the channel (`api`, `cli` or `eval`),
  `pipeline` or `agent`, the model preset. Stage and reason are root metadata.
  **Scores for what is known only at the end**: `outcome` (the deciding stage), `refused`,
  and `correct` on eval traces.
- **Environments**: `development` (local), `production` (Modal), `eval` (bake-offs and
  `finrag eval`), so test traffic never pollutes live dashboards.
- **Cost**: Langfuse cannot price a self-hosted model, so every generation carries our
  estimate (GPU seconds x price, or tokens x list price for GPT and Claude).
- **Flushed** before every short-lived process exits (CLI, eval, bake-off) and on server
  shutdown (FastAPI lifespan, Modal exit hook). Without keys, tracing is off: no client.
- **Tests stay offline**: the real SDK exports to an in-memory OpenTelemetry exporter and
  its HTTP calls go to a mock (`tests/test_tracing.py`).

Setup: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env`
(project in Langfuse Cloud EU), then `uv run python deploy/push_secrets.py` for Modal.

**Audit of real traces (2026-11-09).** Production traces (pipeline answer, evidence
refusal, agent, and the 20 eval traces of a bake-off) were fetched with `langfuse-cli`
and checked against Langfuse's best-practices page, fetched fresh. Found and fixed:
roots nested under a foreign OpenTelemetry span that the export filter dropped
(orphaned, in a borrowed trace id); root output as a JSON blob instead of the answer; no
channel tag; no ground truth on eval traces; `unknown_service:python`; a duplicate tag.
Re-checked on new traces: every observation present and typed, no orphan, tool calls
paired by id, `outcome` / `refused` on every trace, `correct` on all 20 eval traces
(18 true, 2 false: the two over-refusals). Not applicable: user IDs (no login),
sessions (single request/response), Langfuse prompt management (prompts live in code).

## Evaluation

`eval/suites/seed.jsonl`: 20 cases to start (12 answerable, 8 unanswerable). An
answerable case names the filing section that holds the answer
(`{"ticker": "AAPL", "form": "10-K", "section": "1A"}`), so the suite stays valid
when chunking changes. Unanswerable cases cover four failure types: a company not in
the corpus, a year not covered, a forecast, and a detail no filing contains.

Gates, written before the first run: recall@5 ≥ 0.80, MRR ≥ 0.60, false accept rate ≤ 0.10.
`finrag eval` exits non-zero when a gate fails.

## Status (2026-11-09): agent mode

Added: `agent.py` (LangGraph), `tools.py` (three tools + ToolGuard), SEC XBRL facts in
`edgar.py`, `finrag agent`, `POST /agent`. 46 offline tests (one per refusal stage,
every guard rule), ruff clean.

Live, on the real corpus (AAPL, MSFT, NVDA, JPM 10-Ks):
- **Tools: correct.** Reported revenue matches the 10-Ks (Apple FY2025 $416,161M,
  Microsoft FY2026 $331,839M, NVIDIA FY2026 $215,938M, JPMorgan 2025 $182,447M); a
  TSLA call is refused by the guard.
- **Single-figure question: answered and grounded** ("Apple's revenue in fiscal 2025"),
  but without the [S1] citation the prompt asks for.
- **Multi-step question: refused, not answered.** llama3.2:3b wrote tool calls *inside*
  the calculator's argument; the guard rejected the call, then the model answered
  from memory and the evidence stage refused it. Safe, but useless.
- **Search question: over-refused.** The right Apple passage was retrieved; the 3B
  model still replied "I don't know".

Reading: the guardrails hold (no ungrounded answer got through), and the bottleneck
is the 3B model's tool planning. Next: an agent eval suite (expected tools + gold
figures, answer rate and false-accept rate, gates set before the first run), then the
same suite on a larger open-weight model and on a hosted one through `ChatModel`, plus
traces in Langfuse.

Week 1, retrieval-only on the real corpus (2026-11-07, `min_score` 0.72): recall@5
0.917, MRR 0.861, false accept rate **0.125 (gate ≤ 0.10: FAIL)**.

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
