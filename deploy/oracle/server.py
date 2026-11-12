"""finrag.api behind the same x-api-key rule as on Modal, for the Oracle CPU host.

    uvicorn server:build --factory --host 0.0.0.0 --port 8000   (the container's command)

Same rule as deploy/modal_app.py: every route except /health and /documents needs the
header x-api-key = FINRAG_API_KEY (/documents checks its own x-ingest-key). The Vercel
proxy holds the key; the browser never sees it. Without FINRAG_API_KEY the server does
not start: a public host must never serve the API open.

No Ollama watchdog here (Modal's _ensure_ollama): Ollama is its own container and
Docker restarts it (restart: unless-stopped).
"""

from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

OPEN_PATHS = frozenset({"/health", "/documents"})


def protect(api: FastAPI, key: str) -> FastAPI:
    """Add the x-api-key check to *api* and return it."""
    if not key:
        raise RuntimeError("FINRAG_API_KEY is not set: refusing to serve the API open")

    @api.middleware("http")
    async def require_key(request: Request, call_next):
        given = request.headers.get("x-api-key", "")
        if request.url.path not in OPEN_PATHS and not hmac.compare_digest(given, key):
            return JSONResponse({"detail": "missing or wrong x-api-key"}, status_code=401)
        return await call_next(request)

    return api


def build() -> FastAPI:
    from finrag.api import app as api

    return protect(api, os.environ.get("FINRAG_API_KEY", ""))
