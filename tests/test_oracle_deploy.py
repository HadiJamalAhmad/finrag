"""The Oracle CPU host (deploy/oracle/): API key rule, served models, digest check.

All offline: a FastAPI stub for the key rule, MockTransport for Ollama.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from finrag import api
from finrag.config import Settings
from finrag.models import GPU_USD_PER_HOUR, MODEL_REGISTRY, Usage

ORACLE = Path(__file__).resolve().parents[1] / "deploy" / "oracle"


def load(name):
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", ORACLE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


server, puller = load("server"), load("pull_models")


# ---------------------------------------------------------------- the x-api-key rule
@pytest.fixture
def guarded():
    app = FastAPI()
    for path in ("/health", "/documents", "/query", "/models"):
        app.add_api_route(path, lambda: {"ok": True}, methods=["GET"])
    return TestClient(server.protect(app, "s3cret"))


def test_key_required_except_health_and_documents(guarded):
    assert guarded.get("/health").status_code == 200
    assert guarded.get("/documents").status_code == 200  # its own x-ingest-key, in the route
    assert guarded.get("/query").status_code == 401
    assert guarded.get("/models", headers={"x-api-key": "wrong"}).status_code == 401
    assert guarded.get("/query", headers={"x-api-key": "s3cret"}).status_code == 200


def test_refuses_to_serve_open_without_a_key():
    with pytest.raises(RuntimeError, match="FINRAG_API_KEY"):
        server.protect(FastAPI(), "")


# ---------------------------------------------------------------- served models
S = Settings(sec_user_agent="x y@z", gpu="A1-CPU", ollama_models="llama-3b")


def test_models_lists_unserved_ollama_presets_as_unavailable(monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: S)
    d = TestClient(api.app).get("/models").json()
    avail = {m["name"]: m["available"] for m in d["models"]}
    assert d["hardware"] == "A1-CPU"
    assert avail["llama-3b"] is True
    assert avail["qwen3-4b"] is False and avail["mistral-24b"] is False


def test_empty_setting_serves_every_ollama_preset():
    s = replace(S, ollama_models="")
    assert all(api._available(p, s) for p in MODEL_REGISTRY.values() if p.provider == "ollama")


def test_request_for_an_unserved_model_is_a_503(monkeypatch):
    monkeypatch.setattr(api, "_deps", lambda: (S, None, None))
    r = TestClient(api.app).post("/query", json={"question": "Apple revenue 2025?",
                                                "model": "mistral-24b"})
    assert r.status_code == 503
    assert "not served" in r.json()["detail"]


def test_free_cpu_costs_nothing():
    assert GPU_USD_PER_HOUR["A1-CPU"] == 0.0
    assert Usage("llama-3b", 100, 50, 30.0, 1).cost_usd("A1-CPU") == 0.0


# ---------------------------------------------------------------- pull + digest check
def ollama(digest_for_llama):
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.url.path)
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "llama3.2:3b", "digest": digest_for_llama}]})
        return httpx.Response(200, json={})

    return httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler)), calls


def test_pull_passes_when_digests_match():
    client, calls = ollama(MODEL_REGISTRY["llama-3b"].digest)
    puller.pull_and_check(client, ["llama-3b"])
    assert calls == ["/api/pull", "/api/tags"]


def test_pull_stops_on_a_digest_mismatch():
    client, _ = ollama("0" * 64)
    with pytest.raises(SystemExit, match="digest mismatch"):
        puller.pull_and_check(client, ["llama-3b"])


@pytest.mark.parametrize("spec", ["", "gpt-luna", "llama-3b,nope"])
def test_served_list_must_name_ollama_presets(spec):
    with pytest.raises(SystemExit):
        puller.served(spec)
