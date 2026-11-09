"""HTTP API: POST /query, POST /agent, GET /models, GET /health.

Run: uv run uvicorn finrag.api:app --reload
"""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from finrag.answer import answer
from finrag.cli import _scope
from finrag.config import get_settings
from finrag.models import MODEL_REGISTRY

app = FastAPI(title="finrag", version="0.1.0")


@app.exception_handler(RuntimeError)
async def not_configured(request: Request, exc: RuntimeError):
    from finrag.providers import ProviderNotConfigured

    if isinstance(exc, ProviderNotConfigured):  # a hosted model without its key
        return JSONResponse({"detail": str(exc)}, status_code=503)
    raise exc


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    # A preset from finrag.models (GET /models). None: the server default (FINRAG_MODEL).
    model: str | None = None

    @field_validator("model")
    @classmethod
    def known_model(cls, v: str | None) -> str | None:
        if v is not None and v not in MODEL_REGISTRY:
            raise ValueError(f"unknown model; options: {', '.join(MODEL_REGISTRY)}")
        return v


@lru_cache(maxsize=1)
def _deps():
    from finrag.store import Store

    s = get_settings()
    return s, _scope(s), Store(s)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    s = get_settings()
    return {"default": s.model, "models": [
        {"name": p.name, "label": p.label, "model_id": p.model_id, "digest": p.digest,
         "provider": p.provider, "available": _available(p, s)}
        for p in MODEL_REGISTRY.values()]}


def _available(p, s) -> bool:
    return {"ollama": True, "openai": bool(s.openai_api_key),
            "anthropic": bool(s.anthropic_api_key)}[p.provider]


def _settings_for(q: Query):
    s, sc, store = _deps()
    return (replace(s, model=q.model) if q.model else s), sc, store


@app.post("/query")
def query(q: Query) -> dict:
    s, sc, store = _settings_for(q)
    return asdict(answer(q.question, sc, store, s))


@app.post("/agent")
def agent_query(q: Query) -> dict:
    """Answer with tools. The response includes the audit log of every tool call."""
    from finrag import agent

    s, sc, store = _settings_for(q)
    guard, model = agent.from_settings(s, store, sc)  # a fresh guard per question
    return asdict(agent.run(q.question, sc, guard, model, gpu=s.gpu))
