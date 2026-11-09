# CLAUDE.md: finrag

RAG over SEC filings (10-K / 10-Q). No RAG framework on purpose: every step is a small
module in `src/finrag/` with its own tests. Read `README.md` for the pipeline and plan.

## Commands

```bash
uv sync --extra dev
uv run pytest -q          # offline; never hits SEC, Ollama or the embedder
uv run ruff check .       # line-length 100, E,F,I,UP
uv run finrag fetch|ingest|ask|agent|eval
```

Always `uv run`. Python 3.12.

## Invariants

- **Never relax a gate to make the eval pass** (`evaluate.GATES`). A red number is a
  finding: fix the cause or record why it cannot be fixed.
- **The SEC User-Agent comes from the user's .env only.** Never invent one, never
  hard-code an address. Respect the 10 requests/second limit (`edgar.MIN_INTERVAL`).
- **Every refusal names its stage** (scope / evidence / model / numbers). Do not add
  a refusal path without a stage and a test.
- **Numeric grounding runs on every answer.** Do not bypass it for "obvious" figures.
- **Eval gold is metadata** (ticker, form, section), never chunk ids, so the suite
  survives re-chunking. Re-ingest after any change to parse.py or chunk.py.
- **The agent's tools are read-only and go through `ToolGuard`.** A new tool needs an
  argument schema, a guard test and a test for each way it can refuse. Never let the
  model call a function directly, and never let `calculate` accept operands that no
  tool returned.
- **Tests stay offline.** Use fixtures (`tests/fixtures/`), MockTransport, fake stores.
- Ollama: always pass `num_ctx` (recent versions default to a huge context and the
  machine runs out of memory). This PC is often short of RAM: stop other heavy
  services before a full eval.
