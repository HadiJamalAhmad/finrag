"""finrag backend on Modal: the FastAPI app (finrag.api) + Ollama on one T4 GPU.

    uv run --extra deploy modal deploy deploy/modal_app.py

Everything the API reads is baked into the image at deploy time, from this PC:
- data/qdrant, data/manifest.json, data/company_tickers.json (run `finrag ingest` first)
- data/facts/*.json (company facts for the agent; cached, so SEC is never called
  from Modal and the SEC User-Agent never leaves .env)
- the bge-small embedder and llama3.2:3b weights.
Re-deploy after any re-ingest.

Every route except /health requires the header `x-api-key`, checked against the
Modal secret `finrag-api` (FINRAG_API_KEY). The Vercel proxy holds the key; the
browser never sees it. One container at most: a public page cannot fan out GPUs.
"""

from __future__ import annotations

import subprocess
import time
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LLM = "llama3.2:3b"  # same model as local runs: the eval numbers stay comparable
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
    # Pull the weights at build time so a cold start does not download 2 GB.
    .run_commands(
        f"bash -c 'ollama serve > /tmp/ollama-build.log 2>&1 & "
        f"for i in $(seq 60); do curl -sf {OLLAMA}/api/tags >/dev/null && break; sleep 1; done; "
        f"ollama pull {LLM}'"
    )
    .uv_pip_install(*DEPS)
    .env({
        "FASTEMBED_CACHE_PATH": "/models/fastembed",  # default is /tmp: lost per container
        "FINRAG_DATA": "/data",
        "OLLAMA_URL": OLLAMA,
        "FINRAG_LLM": LLM,
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


@app.cls(
    image=image,
    gpu="T4",
    memory=8192,
    timeout=600,
    scaledown_window=300,  # idle 5 min -> scale to zero (no GPU cost while idle)
    max_containers=1,
    secrets=[modal.Secret.from_name("finrag-api")],
)
class Backend:
    def _start_ollama(self) -> None:
        import httpx

        # Own session: when Modal cancels a request (client gone), its signal must not
        # reach the Ollama server. Seen on the first deploy: one cancelled request
        # killed Ollama and every later question failed with "Connection refused".
        self.ollama = subprocess.Popen(["ollama", "serve"], start_new_session=True)
        for _ in range(120):
            try:
                httpx.get(f"{OLLAMA}/api/tags", timeout=1).raise_for_status()
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("ollama serve did not come up")

    def _ensure_ollama(self) -> None:
        if self.ollama.poll() is not None:  # exited: restart instead of failing forever
            print(f"ollama exited with code {self.ollama.returncode}; restarting")
            self._start_ollama()

    @modal.enter()
    def start(self) -> None:
        import httpx

        self._start_ollama()
        # Load the model on the GPU now, with the same num_ctx as answer.py, so the
        # first question does not pay for it. num_ctx always explicit (CLAUDE.md).
        httpx.post(f"{OLLAMA}/api/generate", json={
            "model": LLM, "prompt": "", "keep_alive": -1, "options": {"num_ctx": 4096},
        }, timeout=300).raise_for_status()
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
