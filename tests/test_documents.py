"""Information-system connector: document ingestion and document-level permissions.

The permission filter is checked against a real (in-memory) Qdrant: no mock of the
semantics that decide who reads what. Everything offline.
"""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from test_agent_tools import FACTS

from finrag import answer as ans
from finrag import api, scope
from finrag.chunk import chunk_document
from finrag.config import Settings
from finrag.store import permission_filter
from finrag.tools import SearchArgs, build_guard

S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4")
RISK_DOC = ("Exposure to a single counterparty must not exceed 8% of CET1 capital. "
            "Above 5%, escalate to the Risk Committee within two business days.")


# ---------------------------------------------------------------- who reads what
@pytest.fixture
def qdrant():
    c = QdrantClient(":memory:")
    c.create_collection("t", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    c.upsert("t", points=[
        PointStruct(id=1, vector=[1, 0], payload={"name": "10-K", "groups": []}),
        PointStruct(id=2, vector=[1, 0], payload={"name": "public, no field"}),
        PointStruct(id=3, vector=[1, 0], payload={"name": "risk policy", "groups": ["risques"]}),
        PointStruct(id=4, vector=[1, 0], payload={"name": "branch memo", "groups": ["agence"]}),
    ])
    return c


def visible(c, groups):
    pts = c.query_points("t", query=[1, 0], limit=10,
                         query_filter=permission_filter(frozenset(groups))).points
    return sorted(p.payload["name"] for p in pts)


def test_no_identity_sees_public_documents_only(qdrant):
    assert visible(qdrant, []) == ["10-K", "public, no field"]


def test_a_group_sees_public_plus_its_own_documents(qdrant):
    assert visible(qdrant, ["risques"]) == ["10-K", "public, no field", "risk policy"]
    assert "risk policy" not in visible(qdrant, ["agence"])


def test_several_groups_see_the_union(qdrant):
    assert len(visible(qdrant, ["risques", "agence"])) == 4


# ---------------------------------------------------------------- chunking
def test_pushing_the_same_document_again_replaces_it():
    a = chunk_document("pol-7", "Risk policy", RISK_DOC, ["risques"], "ged", "ged://pol-7",
                       "2026-11-09")
    b = chunk_document("pol-7", "Risk policy", RISK_DOC + " v2", ["risques"], "ged",
                       "ged://pol-7", "2026-11-09")
    assert [c.id for c in a] == [c.id for c in b]  # same ids: an upsert overwrites
    assert a[0].groups == ("risques",) and a[0].text.startswith("Risk policy\n")


# ---------------------------------------------------------------- the API
class RecordingStore:
    def __init__(self, hits=None):
        self.added, self.calls, self.hits = [], [], hits or []

    def add(self, chunks):
        self.added += chunks
        return len(chunks)

    def search(self, query, k, tickers=frozenset(), years=frozenset(), **kw):
        self.calls.append(kw)
        return self.hits


@pytest.fixture
def client(monkeypatch, filing_10k, sec_names):
    store = RecordingStore()
    sc = scope.Scope.build([filing_10k], sec_names)
    monkeypatch.setattr(api, "_deps", lambda: (S, sc, store))
    return TestClient(api.app), store


DOC = {"id": "pol-7", "title": "Counterparty risk policy", "text": RISK_DOC,
       "groups": ["risques"], "source": "ged"}


def test_ingestion_is_off_until_its_key_is_configured(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("FINRAG_INGEST_KEY", raising=False)
    assert c.post("/documents", json=DOC).status_code == 503


def test_ingestion_needs_its_own_key(client, monkeypatch):
    c, store = client
    monkeypatch.setenv("FINRAG_INGEST_KEY", "ingest-secret")
    assert c.post("/documents", json=DOC, headers={"x-api-key": "a-query-key"}).status_code == 401
    r = c.post("/documents", json=DOC, headers={"x-ingest-key": "ingest-secret"})
    assert r.status_code == 200 and r.json()["chunks"] == len(store.added) >= 1
    assert store.added[0].groups == ("risques",) and store.added[0].url == "ged://pol-7"


@pytest.mark.parametrize("bad", [{"groups": []}, {"groups": ["Risques!"]}, {"id": "a b"}])
def test_a_document_must_say_who_may_read_it(client, monkeypatch, bad):
    c, _ = client
    monkeypatch.setenv("FINRAG_INGEST_KEY", "k")
    r = c.post("/documents", json={**DOC, **bad}, headers={"x-ingest-key": "k"})
    assert r.status_code == 422  # nothing becomes public by omission


def test_the_users_groups_reach_retrieval(client, monkeypatch):
    from finrag.auth import Identity

    c, store = client

    class StubAuth:  # the verified identity (finrag.auth, tested in test_auth.py)
        def identify(self, authorization):
            return Identity("user_1", ("risques",))

    monkeypatch.setattr(api, "_auth", lambda: StubAuth())
    c.post("/query", json={"question": "Who supplies Acme?"},
           headers={"authorization": "Bearer t"})
    assert store.calls[-1] == {"groups": frozenset({"risques"})}
    c.post("/query", json={"question": "Who supplies Acme?"})
    assert store.calls[-1] == {}  # no token: the store's default, public only
    # Groups in the body are not a thing any more: identity comes from the token only.
    c.post("/query", json={"question": "Who supplies Acme?", "groups": ["risques"]})
    assert store.calls[-1] == {}


def test_the_agent_reads_only_what_its_user_may_read():
    store = RecordingStore()
    guard = build_guard(store, lambda t: FACTS, {"ACME"}, top_k=5, min_score=0.55,
                        groups=["agence"])
    guard.tools["search_filings"][1](SearchArgs(query="risk policy"), guard)
    assert store.calls[-1] == {"groups": frozenset({"agence"})}


def test_answer_without_groups_leaves_the_store_default(filing_10k, sec_names):
    store = RecordingStore()
    sc = scope.Scope.build([filing_10k], sec_names)
    ans.answer("Who supplies Acme?", sc, store, S, llm=lambda p, s: "x")
    assert store.calls[-1] == {}
