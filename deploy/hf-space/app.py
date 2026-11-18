"""finrag on a free Hugging Face Space (CPU): the answer path WITHOUT the language model.

A question goes through finrag's real answer() with the French documents settings. The model
call is replaced by a stop: when the evidence gate passes, the page shows the passages the
model would receive, with their links; when a step refuses, it shows the refusal and the
step that decided it. Nothing here is a simulation: it is the production code path, cut
before generation (a free CPU Space has no GPU for the model).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from finrag.answer import answer
from finrag.config import Settings
from finrag.store import Store

S = Settings()
STORE = Store(S)
SCOPE = None  # documents mode has no company scope: answer() never reads it


class ModelStop(Exception):
    """Raised where the model would be called: the evidence gate has passed."""


def no_model(prompt, settings):
    raise ModelStop


def run(question: str) -> dict:
    try:
        a = answer(question, SCOPE, STORE, S, llm=no_model)
    except ModelStop:
        hits = STORE.search(question, S.top_k, frozenset(), frozenset())
        return {"stage": "passages", "min_score": S.min_score,
                "passages": [{"score": round(s, 3), "title": p.get("section_title", ""),
                              "url": p.get("url", ""), "text": p["text"]} for s, p in hits]}
    return {"stage": a.stage, "reason": a.reason, "answer": a.answer, "min_score": S.min_score,
            "best": a.sources[0]["score"] if a.sources else None}


app = FastAPI(title="finrag, démonstration")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=300)


@app.post("/ask")
def ask(q: Ask) -> dict:
    return run(q.question)


PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE
