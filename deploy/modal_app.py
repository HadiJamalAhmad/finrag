"""finrag backend on Modal: the FastAPI app (finrag.api) + Ollama with three models.

    uv run modal deploy deploy/modal_app.py                 # the API (one L4 GPU)
    uv run modal run deploy/modal_app.py::bakeoff_all       # same eval, every model

Everything the API reads is baked into the image at deploy time, from this PC:
- data/qdrant, data/manifest.json, data/company_tickers.json (run `finrag ingest` first)
- data/facts/*.json (company facts for the agent; cached, so SEC is never called
  from Modal and the SEC User-Agent never leaves .env)
- the bge-small embedder and the three models of deploy/models.lock, digests checked
  at build time (deploy/pull_models.py): the build fails if Ollama serves other weights.
Re-deploy after any re-ingest.

Models (src/finrag/models.py): llama-3b (default), qwen3-4b, mistral-24b. Mistral
Small 24B is ~15 GB at 4-bit, so the GPU has 24 GB: an L4, or an A10G when no L4 is
free. Each answer's cost estimate uses the GPU the container actually got.

Every route except /health requires the header `x-api-key`, checked against the
Modal secret `finrag-api` (FINRAG_API_KEY). The Vercel proxy holds the key; the
browser never sees it. One API container at most: a public page cannot fan out GPUs.
"""

from __future__ import annotations

import json
import subprocess
import time
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# 24 GB for Mistral Small 24B. L4 first; A10G when Modal has no L4 free (seen on
# 2026-11-09: the first bake-off waited 30 minutes for an L4 and never started).
GPUS = ["L4", "A10G"]
DEFAULT_MODEL = "llama-3b"
EMBED = "BAAI/bge-small-en-v1.5"
OLLAMA = "http://127.0.0.1:11434"
# Modal re-imports this file inside the container, where pyproject.toml does not exist:
# read the dependency list on the deploying machine only (the image is built from it there).
DEPS = (
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    if modal.is_local()
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "zstd", "pciutils")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    # Pull the pinned weights at build time (a cold start must not download 20 GB),
    # and stop the build if a digest differs from models.lock.
    .add_local_file(ROOT / "deploy" / "models.lock", "/deploy/models.lock", copy=True)
    .add_local_file(ROOT / "deploy" / "pull_models.py", "/deploy/pull_models.py", copy=True)
    .run_commands("python3 /deploy/pull_models.py /deploy/models.lock")
    .uv_pip_install(*DEPS)
    .env({
        "FASTEMBED_CACHE_PATH": "/models/fastembed",  # default is /tmp: lost per container
        "FINRAG_DATA": "/data",
        "OLLAMA_URL": OLLAMA,
        "FINRAG_MODEL": DEFAULT_MODEL,
        "OLLAMA_KEEP_ALIVE": "-1",
    })
    .run_commands(f"python -c \"from fastembed import TextEmbedding; TextEmbedding('{EMBED}')\"")
    .add_local_dir(DATA / "qdrant", "/data/qdrant", copy=True)
    .add_local_dir(DATA / "facts", "/data/facts", copy=True)
    .add_local_file(DATA / "manifest.json", "/data/manifest.json", copy=True)
    .add_local_file(DATA / "company_tickers.json", "/data/company_tickers.json", copy=True)
    .add_local_python_source("finrag")
)

app = modal.App("finrag")


def start_ollama() -> subprocess.Popen:
    """Start `ollama serve` and wait until it answers.

    Own session: when Modal cancels a request (client gone), its signal must not reach
    the Ollama server. Seen on the first deploy: one cancelled request killed Ollama
    and every later question failed with "Connection refused".
    """
    import httpx

    proc = subprocess.Popen(["ollama", "serve"], start_new_session=True)
    for _ in range(120):
        try:
            httpx.get(f"{OLLAMA}/api/tags", timeout=1).raise_for_status()
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("ollama serve did not come up")


def which_gpu() -> str:
    """The GPU this container got (L4 or A10G), for the cost estimate."""
    from finrag.models import detect_gpu

    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return detect_gpu(name)


def load_model(name: str) -> None:
    """Load a preset on the GPU, with the num_ctx answer.py uses (always explicit)."""
    import httpx

    from finrag.models import get, ollama_options

    httpx.post(f"{OLLAMA}/api/generate", json={
        "model": get(name).ollama_tag, "prompt": "", "keep_alive": -1,
        **ollama_options(get(name))}, timeout=600).raise_for_status()


@app.cls(
    image=image,
    gpu=GPUS,
    memory=32768,  # Ollama maps the 15 GB Mistral weights through host memory
    timeout=900,
    scaledown_window=300,  # idle 5 min -> scale to zero (no GPU cost while idle)
    max_containers=1,
    secrets=[modal.Secret.from_name("finrag-api")],
)
class Backend:
    def _ensure_ollama(self) -> None:
        if self.ollama.poll() is not None:  # exited: restart instead of failing forever
            print(f"ollama exited with code {self.ollama.returncode}; restarting")
            self.ollama = start_ollama()

    @modal.enter()
    def start(self) -> None:
        import os

        # Before finrag.config is imported: Settings reads FINRAG_GPU at import time.
        os.environ["FINRAG_GPU"] = which_gpu()
        self.ollama = start_ollama()
        # The default model is warm; another preset loads on its first request.
        load_model(DEFAULT_MODEL)
        from finrag.api import _deps
        from finrag.store import embed

        _deps()
        embed(["warm up"], EMBED)

    @modal.exit()
    def stop(self) -> None:
        self.ollama.terminate()

    @modal.asgi_app()
    def web(self):
        import hmac
        import os

        from fastapi import Request
        from fastapi.responses import JSONResponse

        from finrag.api import app as api

        key = os.environ["FINRAG_API_KEY"]

        @api.middleware("http")
        async def require_key(request: Request, call_next):
            given = request.headers.get("x-api-key", "")
            if request.url.path != "/health" and not hmac.compare_digest(given, key):
                return JSONResponse({"detail": "missing or wrong x-api-key"}, status_code=401)
            if request.url.path != "/health":
                self._ensure_ollama()
            return await call_next(request)

        return api


@app.function(image=image, gpu=GPUS, memory=32768, timeout=3600)
def bakeoff(model: str, suite_jsonl: str) -> dict:
    """Run the full eval (with the LLM) for one model preset, on the same GPU as the API."""
    from dataclasses import replace

    from finrag import evaluate
    from finrag.cli import _scope
    from finrag.config import get_settings
    from finrag.store import Store

    start_ollama()
    load_model(model)
    s = replace(get_settings(), model=model, gpu=which_gpu())
    suite = [json.loads(ln) for ln in suite_jsonl.splitlines() if ln.strip()]
    return evaluate.run(suite, _scope(s), Store(s), s, with_llm=True)


@app.local_entrypoint()
def bakeoff_all(models: str = "llama-3b,qwen3-4b,mistral-24b",
                suite: str = "eval/suites/seed.jsonl") -> None:
    """One container per model, in parallel; reports land in eval/reports/bakeoff-*.json."""
    names = [m.strip() for m in models.split(",") if m.strip()]
    text = (ROOT / suite).read_text(encoding="utf-8")
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, rep in zip(names, bakeoff.starmap([(n, text) for n in names]), strict=True):
        (out_dir / f"bakeoff-{name}.json").write_text(json.dumps(rep, indent=2),
                                                     encoding="utf-8")
        g, lat, u = rep["gates"], rep["latency_s"], rep["usage"]
        rows.append(f"{name:12} [{rep['model']['gpu']}] recall@5 {rep['recall@5']}  "
                    f"mrr {rep['mrr']}  "
                    f"false_accept {rep['false_accept_rate']}"
                    f"{'' if g['false_accept_rate']['pass'] else ' FAIL'}  "
                    f"over_refusal {rep.get('over_refusal_rate')}  "
                    f"p50 {lat['p50']} s  p95 {lat['p95']} s  "
                    f"cost/answer {u['est_cost_usd_per_llm_answer']} USD")
    print("\n".join(rows))
