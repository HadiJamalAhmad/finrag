"""Build the data of the static Hugging Face Space: the 45 frozen sheets' passages with their
vectors, computed by finrag's own embed() (the function Store.rebuild calls), and the
reference scores of the 69 evaluation questions, for check_browser.mjs to compare against.

    FINRAG_DATA=eval/fixtures/fr-interieur FINRAG_DOMAIN=documents \
    FINRAG_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    uv run python deploy/static-space/build_index.py deploy/static-space/site
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from finrag import documents
from finrag.config import Settings
from finrag.store import embed, query_prefix

SUITES = ["eval/suites/fr-interieur.jsonl", "eval/suites/fr-interieur-heldout.jsonl"]


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def main(out: Path) -> None:
    s = Settings()
    chunks = documents.to_chunks(documents.load(s.docs_dir))
    vectors = unit(np.array(embed([c.text for c in chunks], s.embed_model), dtype=np.float32))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(json.dumps({
        "model": s.embed_model, "min_score": s.min_score, "top_k": s.top_k,
        "finrag_commit": commit,
        "chunks": [{"t": c.section_title, "u": c.url, "x": c.text,
                    "v": [round(float(x), 6) for x in v]} for c, v in zip(chunks, vectors)],
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    questions = [json.loads(ln)["question"] for f in SUITES
                 for ln in Path(f).read_text(encoding="utf-8").splitlines() if ln.strip()]
    qv = unit(np.array(embed([query_prefix(s.embed_model) + q for q in questions],
                             s.embed_model), dtype=np.float32))
    best = (qv @ vectors.T).max(axis=1)
    (out / "reference.json").write_text(json.dumps([
        {"q": q, "best": round(float(b), 6), "passes": bool(b >= s.min_score)}
        for q, b in zip(questions, best)], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(chunks)} passages, {len(questions)} reference questions -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
