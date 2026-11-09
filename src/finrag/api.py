"""HTTP API: POST /query, POST /agent, POST /documents, GET /models, GET /health.

Run: uv run uvicorn finrag.api:app --reload

Two credentials, two rights: asking (x-api-key, checked by the deployment, Modal) and
pushing documents (x-ingest-key, checked here). A connector that feeds the index cannot
read answers with its key, and a client that asks cannot write into the index.
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import date
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from finrag.answer import answer
from finrag.cli import _scope
from finrag.config import get_settings
from finrag.models import MODEL_REGISTRY


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # A long-running server: flush buffered traces and stop the exporter on shutdown.
    if _tracer.cache_info().currsize:
        _tracer().shutdown()


app = FastAPI(title="finrag", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RuntimeError)
async def not_configured(request: Request, exc: RuntimeError):
    from finrag.providers import ProviderNotConfigured

    if isinstance(exc, ProviderNotConfigured):  # a hosted model without its key
        return JSONResponse({"detail": str(exc)}, status_code=503)
    raise exc


GROUP = r"^[a-z0-9_-]{1,32}$"


def _check_groups(v: list[str] | None) -> list[str] | None:
    import re

    if v is not None and not all(re.fullmatch(GROUP, g) for g in v):
        raise ValueError("group names: lowercase letters, digits, - and _ (1-32)")
    return v


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    # A preset from finrag.models (GET /models). None: the server default (FINRAG_MODEL).
    model: str | None = None
    # The user's groups, set by the SSO gateway in production (never by the end user).
    # None: no identity, public documents only.
    groups: list[str] | None = Field(default=None, max_length=10)

    @field_validator("groups")
    @classmethod
    def group_names(cls, v: list[str] | None) -> list[str] | None:
        return _check_groups(v)

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


@lru_cache(maxsize=1)
def _tracer():
    from finrag.tracing import Tracer

    return Tracer.from_settings(get_settings(), channel="api")  # no-op without keys


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
    return asdict(answer(q.question, sc, store, s, tracer=_tracer(), groups=q.groups))


@app.post("/agent")
def agent_query(q: Query) -> dict:
    """Answer with tools. The response includes the audit log of every tool call."""
    from finrag import agent

    s, sc, store = _settings_for(q)
    # A fresh guard per question; its search reads only what this user may read.
    guard, model = agent.from_settings(s, store, sc, groups=q.groups)
    return asdict(agent.traced_run(q.question, sc, guard, model, s, _tracer()))


class DocumentIn(BaseModel):
    """A document pushed by an information-system connector (GED, CRM export, RPA)."""

    id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    # Who may read it. Required, at least one: nothing becomes public by omission.
    groups: list[str] = Field(min_length=1, max_length=10)
    source: str = Field(default="connector", pattern=r"^[a-z0-9_-]{1,32}$")
    url: str = Field(default="", max_length=500)

    @field_validator("groups")
    @classmethod
    def group_names(cls, v: list[str]) -> list[str]:
        return _check_groups(v)


def require_ingest_key(x_ingest_key: str = Header(default="")) -> None:
    expected = os.environ.get("FINRAG_INGEST_KEY", "")
    if not expected:  # not configured: ingestion is off, never open
        raise HTTPException(503, "document ingestion is not enabled on this server")
    if not hmac.compare_digest(x_ingest_key, expected):
        raise HTTPException(401, "missing or wrong x-ingest-key")


@app.post("/documents")
def ingest(doc: DocumentIn, x_ingest_key: str = Header(default="")) -> dict:
    """Index a document now, readable only by its groups. Same id again: replaced."""
    from finrag.chunk import chunk_document

    require_ingest_key(x_ingest_key)
    _, _, store = _deps()
    chunks = chunk_document(doc.id, doc.title, doc.text, doc.groups, doc.source,
                            doc.url or f"{doc.source}://{doc.id}", date.today().isoformat())
    return {"id": doc.id, "chunks": store.add(chunks), "groups": doc.groups}


# GraphQL: the same handlers, typed (finrag.graphql_api). Imported last: it calls back
# into this module's handlers.
from finrag.graphql_api import router as graphql_router  # noqa: E402

app.include_router(graphql_router, prefix="/graphql")
