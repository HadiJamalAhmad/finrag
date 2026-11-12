"""One-shot init container: pull the served models into Ollama, check their digests, load.

    python pull_models.py          (env: OLLAMA_URL, FINRAG_OLLAMA_MODELS)

The Oracle twin of deploy/pull_models.py, over Ollama's HTTP API because Ollama runs in
its own container. Same rule: a pulled digest that differs from the pin in
finrag.models (and deploy/models.lock) stops everything; the API container only starts
when this one exits 0 (compose: service_completed_successfully).
Then the default model is loaded with an explicit num_ctx (CLAUDE.md) and kept loaded.
"""

from __future__ import annotations

import os
import sys

import httpx

from finrag.models import MODEL_REGISTRY, ollama_options


def served(spec: str) -> list[str]:
    names = [n.strip() for n in spec.split(",") if n.strip()]
    bad = [n for n in names
           if n not in MODEL_REGISTRY or MODEL_REGISTRY[n].provider != "ollama"]
    if not names or bad:
        raise SystemExit(f"FINRAG_OLLAMA_MODELS must list Ollama presets; got {spec!r}")
    return names


def pull_and_check(client: httpx.Client, names: list[str]) -> None:
    for n in names:
        tag = MODEL_REGISTRY[n].model_id
        print(f"pulling {tag} ...", flush=True)
        client.post("/api/pull", json={"model": tag, "stream": False},
                    timeout=3600).raise_for_status()
    pulled = {m["name"]: m["digest"]
              for m in client.get("/api/tags", timeout=30).json()["models"]}
    bad = [f"{MODEL_REGISTRY[n].model_id}: pinned {MODEL_REGISTRY[n].digest[:12]}, "
           f"pulled {pulled.get(MODEL_REGISTRY[n].model_id, 'nothing')[:12]}"
           for n in names if pulled.get(MODEL_REGISTRY[n].model_id) != MODEL_REGISTRY[n].digest]
    if bad:
        raise SystemExit("digest mismatch (update models.lock AND src/finrag/models.py "
                         "together, then re-run the eval):\n  " + "\n  ".join(bad))
    print("digests match:", ", ".join(names), flush=True)


def load(client: httpx.Client, name: str) -> None:
    p = MODEL_REGISTRY[name]
    client.post("/api/generate", json={"model": p.model_id, "prompt": "", "keep_alive": -1,
                                       **ollama_options(p)}, timeout=900).raise_for_status()
    print(f"{name} loaded", flush=True)


def main() -> None:
    names = served(os.environ.get("FINRAG_OLLAMA_MODELS", ""))
    default = os.environ.get("FINRAG_MODEL", names[0])
    if default not in names:
        sys.exit(f"FINRAG_MODEL={default!r} is not in FINRAG_OLLAMA_MODELS")
    with httpx.Client(base_url=os.environ.get("OLLAMA_URL", "http://ollama:11434")) as c:
        pull_and_check(c, names)
        load(c, default)


if __name__ == "__main__":
    main()
