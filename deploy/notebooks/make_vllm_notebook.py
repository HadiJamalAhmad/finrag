"""Writes deploy/notebooks/vllm_eval.ipynb (kept as code so the notebook stays reviewable).

The notebook, on a free Kaggle GPU (T4 x2): serve Mistral 7B Instruct v0.3 with vLLM at a
pinned Hugging Face revision, run finrag's held-out evaluation through the `vllm` provider,
then measure throughput (one request at a time vs many at once: vLLM's continuous batching).
    python deploy/notebooks/make_vllm_notebook.py
"""
import json
from pathlib import Path

from finrag.models import get

P = get("mistral-7b-vllm")
REPO = "unsloth/mistral-7b-instruct-v0.3"


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md(f"""
# finrag with vLLM: Mistral 7B on a free GPU

Serves **{P.label}** with vLLM (OpenAI-compatible API) from `{REPO}` at revision
`{P.hf_revision}`, then runs finrag's held-out French evaluation through the `vllm` provider
(`FINRAG_MODEL={P.name}`), and measures throughput with 1 vs 32 concurrent requests.

Kaggle: Accelerator **GPU T4 x2**, Internet on. Corpus: the 45 sheets frozen in the repository.
"""),
    code("""
import subprocess

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True,
                          capture_output=True, text=True).stdout)
print("GPUs:", NGPU)
"""),
    code("""
# vLLM in its own environment (Python 3.12), pinned
!pip -q install uv
!uv venv -q -p 3.12 /tmp/vllm-env
!uv pip install -q --python /tmp/vllm-env vllm==0.33.0
!/tmp/vllm-env/bin/python -c "import vllm; print('vllm', vllm.__version__)"
"""),
    code(f"""
# Start the server at the pinned revision; T4 has no bfloat16, so float16
import time
import urllib.request

log = open("/kaggle/working/vllm.log", "w")
server = subprocess.Popen(
    ["/tmp/vllm-env/bin/vllm", "serve", "{REPO}",
     "--revision", "{P.hf_revision}",
     "--served-model-name", "{P.model_id}", "--dtype", "half",
     "--tensor-parallel-size", str(NGPU), "--max-model-len", "{P.num_ctx}",
     "--gpu-memory-utilization", "0.90", "--port", "8000"],
    stdout=log, stderr=subprocess.STDOUT)
for i in range(240):  # up to 20 minutes for download + load
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        print(f"vLLM ready after {{i * 5}} s")
        break
    except OSError:
        if server.poll() is not None:
            break
        time.sleep(5)
if server.poll() is not None:
    print(open("/kaggle/working/vllm.log").read()[-4000:])
    raise RuntimeError("vLLM did not start")
print(urllib.request.urlopen("http://localhost:8000/v1/models").read().decode()[:300])
"""),
    code("""
# finrag at the vllm-provider branch, with its locked dependencies
%cd /tmp
!rm -rf finrag && git clone -q --branch vllm-provider https://github.com/HadiJamalAhmad/finrag.git
%cd /tmp/finrag
!uv sync --frozen -q && git log --oneline -1
"""),
    code(f"""
import os

os.environ.update({{
    "VLLM_URL": "http://localhost:8000/v1", "FINRAG_MODEL": "{P.name}", "FINRAG_GPU": "T4",
    "FINRAG_DATA": "eval/fixtures/fr-interieur", "PYTHONUNBUFFERED": "1",
    "FINRAG_DOMAIN": "documents",
    "FINRAG_EMBED_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "FINRAG_COLLECTION": "docs_fr", "FINRAG_MIN_SCORE": "0.61", "FINRAG_TOP_K": "3",
}})
!uv run finrag ingest > /dev/null
!uv run finrag eval eval/suites/fr-interieur-heldout.jsonl \\
    --out eval/reports/fr-heldout-{P.name}.json
"""),
    code(f"""
# Throughput: 1 request at a time vs 32 at once (continuous batching)
import asyncio
import json

import httpx

PROMPT = ("Réponds en français en citant la source. Question : quelles pièces faut-il "
          "fournir pour une première demande de passeport ? " + "Contexte. " * 300)

async def one(c):
    r = await c.post("http://localhost:8000/v1/chat/completions", timeout=600, json={{
        "model": "{P.model_id}", "temperature": 0, "max_tokens": 256,
        "messages": [{{"role": "user", "content": PROMPT}}]}})
    return r.json()["usage"]["completion_tokens"]

async def bench(n, conc):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient() as c:
        async def lim():
            async with sem:
                return await one(c)
        t0 = time.perf_counter()
        toks = await asyncio.gather(*[lim() for _ in range(n)])
        dt = time.perf_counter() - t0
    return {{"requests": n, "concurrency": conc, "seconds": round(dt, 2),
            "output_tokens": sum(toks), "tokens_per_s": round(sum(toks) / dt, 1)}}

res = {{"sequential": await bench(8, 1), "concurrent": await bench(32, 32), "gpus": NGPU}}
res["speedup"] = round(res["concurrent"]["tokens_per_s"] / res["sequential"]["tokens_per_s"], 1)
print(json.dumps(res, indent=1))
json.dump(res, open("eval/reports/vllm-throughput.json", "w"), indent=1)
"""),
    code("""
# Only the reports go to the notebook output (not the environments)
!mkdir -p /kaggle/working/reports
!cp eval/reports/*.json /kaggle/working/reports/
!ls -la /kaggle/working/reports
server.terminate()
"""),
]
kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb = {"cells": cells, "metadata": {"kernelspec": kernel, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
out = Path(__file__).with_name("vllm_eval.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)
