"""What a container deployment (deploy/k8s) relies on: the API key check inside the app,
the readiness probe, and the Qdrant server switch. Offline: in-memory Qdrant, fake stores.
"""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from finrag import api, store
from finrag.config import Settings

client = TestClient(api.app)


# ---------------------------------------------------------------- x-api-key
@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("FINRAG_API_KEY", "k-123")


def test_without_a_configured_key_the_api_is_open(monkeypatch):
    monkeypatch.delenv("FINRAG_API_KEY", raising=False)
    assert client.get("/models").status_code == 200


def test_a_configured_key_is_required_on_every_route(api_key):
    assert client.get("/models").status_code == 401
    assert client.get("/me").status_code == 401
    assert client.post("/query", json={"question": "anything"}).status_code == 401
    assert client.post("/graphql", json={"query": "{__typename}"}).status_code == 401


def test_a_wrong_key_is_refused_and_the_right_one_accepted(api_key):
    assert client.get("/models", headers={"x-api-key": "k-12"}).status_code == 401
    assert client.get("/models", headers={"x-api-key": "k-123"}).status_code == 200


def test_probes_need_no_key(api_key):
    """Kubernetes calls them without credentials; behind the key they would always fail."""
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code in (200, 503)  # never 401


# ---------------------------------------------------------------- readiness
class FakeStore:
    chunks = 0
    error: Exception | None = None

    def __init__(self, settings):
        pass

    def count(self):
        if FakeStore.error:
            raise FakeStore.error
        return FakeStore.chunks


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("FINRAG_DATA", str(tmp_path))
    monkeypatch.setattr(store, "Store", FakeStore)
    FakeStore.chunks, FakeStore.error = 0, None
    return tmp_path


def test_not_ready_while_the_index_is_unreachable(corpus):
    FakeStore.error = ConnectionError("refused")
    r = client.get("/ready")
    assert r.status_code == 503 and "unreachable" in r.json()["reason"]


def test_not_ready_before_ingestion(corpus):
    (corpus / "manifest.json").write_text("[]")
    r = client.get("/ready")
    assert r.status_code == 503 and "empty" in r.json()["reason"]


def test_not_ready_without_the_corpus_manifest(corpus):
    FakeStore.chunks = 12
    r = client.get("/ready")
    assert r.status_code == 503 and "manifest" in r.json()["reason"]


def test_ready_with_chunks_and_a_manifest(corpus):
    FakeStore.chunks = 12
    (corpus / "manifest.json").write_text("[]")
    r = client.get("/ready")
    assert r.status_code == 200 and r.json() == {"ready": True, "chunks": 12}


def test_liveness_never_touches_the_index(corpus):
    """A slow index must not get healthy pods restarted: /health stays 200."""
    FakeStore.error = ConnectionError("refused")
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------- Qdrant server switch
def test_local_mode_by_default_and_a_server_when_a_url_is_set(tmp_path):
    local = Settings(sec_user_agent="x y@z", data_dir=tmp_path, qdrant_url="")
    assert store._location(local) == str(tmp_path / "qdrant")
    server = Settings(sec_user_agent="x y@z", data_dir=tmp_path,
                      qdrant_url="http://qdrant:6333")
    assert store._location(server) == "http://qdrant:6333"


def test_count_is_zero_before_the_collection_exists_then_the_chunks():
    s = object.__new__(store.Store)
    s.s = Settings(sec_user_agent="x y@z", collection="c")
    s.client = QdrantClient(":memory:")
    assert s.count() == 0
    s.client.create_collection("c", vectors_config=VectorParams(size=2,
                                                                distance=Distance.COSINE))
    s.client.upsert("c", points=[PointStruct(id=i, vector=[1, 0]) for i in range(3)])
    assert s.count() == 3


# ---------------------------------------------------------------- model pull (Ollama service)
def _ollama(served_digest):
    import httpx

    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        if req.url.path == "/api/pull":
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(200, json={"models": [
            {"name": "llama3.2:3b", "digest": served_digest}]})

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_pull_returns_the_digest_when_it_matches_the_pin():
    from finrag.models import get, pull_verified

    p = get("llama-3b")
    http, calls = _ollama(p.digest)
    assert pull_verified(p, "http://ollama:11434", http) == p.digest
    assert calls == [("POST", "/api/pull"), ("GET", "/api/tags")]


def test_pull_fails_when_the_server_has_other_weights():
    from finrag.models import get, pull_verified

    http, _ = _ollama("0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        pull_verified(get("llama-3b"), "http://ollama:11434", http)


def test_a_hosted_model_cannot_be_pulled():
    from finrag.models import get, pull_verified

    with pytest.raises(ValueError, match="not a pinned Ollama model"):
        pull_verified(get("claude-haiku"), "http://ollama:11434")


# ---------------------------------------------------------------- rebuild without a gap
def _chunk(i):
    from finrag.chunk import Chunk

    return Chunk(id=f"{i:032x}", text=f"passage {i}", ticker="ACME", company="Acme",
                 form="10-K", fiscal_year=2025, period="2025-09-27", filed="2025-11-01",
                 section="1A", section_title="Risk Factors", url="u")


@pytest.fixture
def live(monkeypatch):
    """A Store on in-memory Qdrant with a fake embedder; `seen` records what a reader
    sees through the alias while each embedding batch runs."""
    s = object.__new__(store.Store)
    s.s = Settings(sec_user_agent="x y@z", collection="docs", embed_batch=0)
    s.client = QdrantClient(":memory:")
    seen = []

    def fake_embed(texts, model, batch_size=0):
        seen.append(s.count())
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(store, "embed", fake_embed)
    return s, seen


def test_first_rebuild_creates_the_index_behind_its_name(live):
    s, _ = live
    assert s.rebuild([_chunk(i) for i in range(5)], dim=2) == 5
    assert s.count() == 5 and s._alias_target().startswith("docs--")


def test_readers_see_the_old_index_until_the_new_one_is_complete(live):
    s, seen = live
    s.rebuild([_chunk(i) for i in range(3)], dim=2)
    seen.clear()
    s.rebuild([_chunk(i) for i in range(10)], dim=2, batch=2)
    assert seen == [3] * 5  # every batch of the new index: readers still see the old 3
    assert s.count() == 10


def test_the_previous_collection_is_dropped_after_the_switch(live):
    s, _ = live
    s.rebuild([_chunk(0)], dim=2)
    s.rebuild([_chunk(1)], dim=2)
    names = [c.name for c in s.client.get_collections().collections]
    assert names == [s._alias_target()]


def test_a_plain_collection_from_before_aliases_is_replaced(live):
    s, _ = live
    s.client.create_collection("docs", vectors_config=VectorParams(size=2,
                                                                   distance=Distance.COSINE))
    s.client.upsert("docs", points=[PointStruct(id=i, vector=[1, 0]) for i in range(7)])
    s.rebuild([_chunk(i) for i in range(2)], dim=2)
    assert s.count() == 2 and s._alias_target() is not None


def test_leftovers_of_an_interrupted_rebuild_are_dropped(live):
    s, _ = live
    s.rebuild([_chunk(0)], dim=2)
    s.client.create_collection("docs--crashed", vectors_config=VectorParams(
        size=2, distance=Distance.COSINE))
    s.rebuild([_chunk(1)], dim=2)
    assert "docs--crashed" not in [c.name for c in s.client.get_collections().collections]


def test_connector_documents_go_into_the_live_index(live):
    s, _ = live
    s.rebuild([_chunk(0)], dim=2)
    s.add([_chunk(1), _chunk(2)])
    assert s.count() == 3


def test_the_embedding_batch_size_reaches_fastembed(monkeypatch):
    import numpy as np

    calls = []

    class FakeModel:
        def embed(self, texts, **kw):
            calls.append(kw)
            return [np.zeros(2) for _ in texts]

    monkeypatch.setattr(store, "_embedder", lambda model: FakeModel())
    store.embed(["a", "b"], "m")
    store.embed(["a", "b"], "m", 16)
    assert calls == [{}, {"batch_size": 16}]


# ---------------------------------------------------------------- slow model
def test_a_model_timeout_is_a_504_not_an_internal_error(monkeypatch):
    """Seen on a CPU-only cluster: the model took over 300 s and the API answered 500."""
    import httpx

    def slow(*a, **kw):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.delenv("FINRAG_API_KEY", raising=False)
    monkeypatch.setattr(api, "_settings_for", lambda q: (None, None, None))
    monkeypatch.setattr(api, "answer", slow)
    r = TestClient(api.app, raise_server_exceptions=False).post(
        "/query", json={"question": "anything at all"})
    assert r.status_code == 504 and "did not answer in time" in r.json()["detail"]
