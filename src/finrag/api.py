"""HTTP API: POST /query, POST /agent, GET /health.  Run: uv run uvicorn finrag.api:app --reload"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from finrag.answer import answer
from finrag.cli import _scope
from finrag.config import get_settings

app = FastAPI(title="finrag", version="0.1.0")


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@lru_cache(maxsize=1)
def _deps():
    from finrag.store import Store

    s = get_settings()
    return s, _scope(s), Store(s)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(q: Query) -> dict:
    s, sc, store = _deps()
    return asdict(answer(q.question, sc, store, s))


@app.post("/agent")
def agent_query(q: Query) -> dict:
    """Answer with tools. The response includes the audit log of every tool call."""
    from finrag import agent

    s, sc, store = _deps()
    guard, model = agent.from_settings(s, store, sc)  # a fresh guard per question
    return asdict(agent.run(q.question, sc, guard, model))
