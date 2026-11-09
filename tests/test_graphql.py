"""GraphQL: the same answer path as REST, typed. Offline (fake store, mocked model)."""

import pytest
from fastapi.testclient import TestClient
from test_agent_tools import FACTS, FakeStore, Scripted, call, say

from finrag import agent, api, models, scope
from finrag import answer as ans
from finrag.config import Settings
from finrag.tools import build_guard
from finrag.tracing import Tracer

S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4", openai_api_key="",
             anthropic_api_key="", langfuse_public_key="", langfuse_secret_key="")
GEN = ans.Generation("A single supplier in Taiwan [1].",
                     models.Usage("llama-3b", 900, 40, 4.5, 1), "stop")


@pytest.fixture
def client(monkeypatch, filing_10k, sec_names):
    sc = scope.Scope.build([filing_10k], sec_names)
    monkeypatch.setattr(api, "_deps", lambda: (S, sc, FakeStore()))
    monkeypatch.setattr(api, "_tracer", lambda: Tracer(None))
    monkeypatch.setattr(api, "get_settings", lambda: S)
    monkeypatch.setattr(ans, "generate", lambda p, s: GEN)
    # answer() binds llm=generate at definition: route the default through the patch.
    real = ans.answer
    monkeypatch.setattr(api, "answer", lambda *a, **k: real(*a, llm=lambda p, s: GEN, **k))
    return TestClient(api.app), sc


def gql(c, query: str) -> dict:
    r = c.post("/graphql", json={"query": query})
    assert r.status_code == 200
    return r.json()


def test_models_lists_every_preset_and_availability(client):
    c, _ = client
    data = gql(c, "{ models { name provider available digest } }")["data"]["models"]
    assert [m["name"] for m in data] == list(models.MODEL_REGISTRY)
    hosted = {m["name"]: m["available"] for m in data if m["provider"] != "ollama"}
    assert hosted == {"gpt-luna": False, "claude-haiku": False}  # no keys in this test


def test_answer_runs_the_same_pipeline_as_rest(client):
    c, _ = client
    q = '''{ answer(question: "Who supplies Acme?") {
              answer refused stage sources { n ticker section score }
              usage { model promptTokens estCostUsd } } }'''
    a = gql(c, q)["data"]["answer"]
    assert a["refused"] is False and a["stage"] == "answered"
    assert a["answer"] == GEN.text
    assert a["sources"][0] == {"n": "1", "ticker": "ACME", "section": "1A", "score": 0.8}
    assert a["usage"]["model"] == "llama-3b" and a["usage"]["promptTokens"] == 900
    # Identical to REST, field for field.
    rest = c.post("/query", json={"question": "Who supplies Acme?"}).json()
    assert rest["answer"] == a["answer"] and rest["stage"] == a["stage"]


def test_refusals_keep_their_stage(client):
    c, _ = client
    a = gql(c, '{ answer(question: "What was Tesla revenue?") { refused stage reason } }')
    assert a["data"]["answer"] == {"refused": True, "stage": "scope",
                                   "reason": "company not in corpus: TSLA"}


@pytest.mark.parametrize("args,needle", [
    ('question: "hi"', "at least 3 characters"),
    ('question: "Who supplies Acme?", model: "gpt-5"', "unknown model"),
])
def test_input_is_validated_like_rest(client, args, needle):
    c, _ = client
    out = gql(c, f"{{ answer({args}) {{ stage }} }}")
    assert out["data"] is None and needle in out["errors"][0]["message"]


def test_a_hosted_model_without_its_key_names_the_variable(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(api, "answer", ans.answer)  # the real model path, through providers
    out = gql(c, '{ answer(question: "Who supplies Acme?", model: "gpt-luna") { stage } }')
    assert out["data"] is None
    assert "needs OPENAI_API_KEY" in out["errors"][0]["message"]


def test_agent_mode_returns_the_tool_audit_log(client, monkeypatch):
    c, sc = client
    guard = build_guard(FakeStore(), lambda t: FACTS, {"ACME"}, top_k=5, min_score=0.55,
                        max_calls=4)
    model = Scripted(call("get_financial_fact", ticker="ACME", metric="revenue",
                          fiscal_year=2025),
                     say("Revenue was $12,345 million [S1]."))
    monkeypatch.setattr(agent, "from_settings", lambda s, store, scope_: (guard, model))
    q = '''{ answer(question: "Acme revenue in fiscal 2025?", mode: AGENT) {
              stage sources { n tool } steps } }'''
    a = gql(c, q)["data"]["answer"]
    assert a["stage"] == "answered"
    assert a["sources"][0]["n"] == "S1" and a["sources"][0]["tool"] == "get_financial_fact"
    assert a["steps"][0]["tool"] == "get_financial_fact" and a["steps"][0]["ok"] is True


def test_queries_over_get_are_refused(client):
    # A cross-site GET must never start a paid model call.
    c, _ = client
    r = c.get("/graphql", params={"query": "{ models { name } }"})
    assert r.status_code in (400, 405)
