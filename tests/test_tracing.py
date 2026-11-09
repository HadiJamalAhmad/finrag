"""Langfuse tracing: what a trace holds per stage, and that it never breaks an answer."""

import base64
import json
from dataclasses import replace

import httpx
import pytest
from test_agent_tools import FACTS, FakeStore, Scripted, call, say

from finrag import agent, evaluate, models, scope
from finrag import answer as ans
from finrag.config import Settings
from finrag.tools import build_guard
from finrag.tracing import NULL, Tracer

S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4")
GEN = ans.Generation("A single supplier in Taiwan [1].",
                     models.Usage("llama-3b", 900, 40, 4.5, 1))


def recording_tracer(status=207, body=None):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(status, json=body or {"successes": [], "errors": []})

    return Tracer("pk-lf-test", "sk-lf-test", "https://cloud.langfuse.com",
                  transport=httpx.MockTransport(handler)), sent


def events(request):
    return json.loads(request.content)["batch"]


@pytest.fixture
def sc(filing_10k, sec_names):
    return scope.Scope.build([filing_10k], sec_names)


def test_no_keys_no_tracing():
    t = Tracer("", "", "https://cloud.langfuse.com")
    assert not t.enabled and t.trace("query", "q") is NULL


def test_an_answer_is_one_trace_with_a_span_per_stage(sc):
    tracer, sent = recording_tracer()
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN,
                   tracer=tracer)
    assert not a.refused and len(sent) == 1  # one batch per question
    req = sent[0]
    assert str(req.url) == "https://cloud.langfuse.com/api/public/ingestion"
    assert req.headers["authorization"] == "Basic " + base64.b64encode(
        b"pk-lf-test:sk-lf-test").decode()
    batch = events(req)
    trace = batch[0]
    assert trace["type"] == "trace-create"
    assert trace["body"]["tags"] == ["llama-3b", "stage:answered"]
    assert trace["body"]["output"]["stage"] == "answered"
    names = [e["body"]["name"] for e in batch[1:]]
    assert names == ["scope", "retrieve", "evidence_gate", "llm", "numbers"]
    assert all(e["body"]["traceId"] == trace["body"]["id"] for e in batch[1:])
    gen = next(e for e in batch if e["type"] == "generation-create")["body"]
    assert gen["usageDetails"] == {"input": 900, "output": 40}
    assert gen["costDetails"]["total"] == pytest.approx(4.5 * 0.80 / 3600, abs=1e-6)


def test_a_refusal_is_tagged_with_its_stage_and_the_stage_is_a_warning(sc):
    tracer, sent = recording_tracer()
    a = ans.answer("Who supplies Acme?", sc, FakeStore(score=0.3), S,
                   llm=lambda p, s: GEN, tracer=tracer)
    assert a.stage == "evidence"
    batch = events(sent[0])
    assert "stage:evidence" in batch[0]["body"]["tags"]
    gate = next(e["body"] for e in batch if e["body"].get("name") == "evidence_gate")
    assert gate["level"] == "WARNING" and gate["output"]["pass"] is False
    assert not any(e["type"] == "generation-create" for e in batch)  # model never called


def test_tracing_failure_never_breaks_the_answer(sc):
    def boom(request):
        raise httpx.ConnectError("langfuse down")

    tracer = Tracer("pk", "sk", "https://cloud.langfuse.com", transport=httpx.MockTransport(boom))
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=tracer)
    assert not a.refused and a.answer == GEN.text
    assert tracer.errors and "langfuse send failed" in tracer.errors[0]


def test_ingestion_errors_are_reported_not_raised(sc):
    tracer, _ = recording_tracer(207, {"successes": [], "errors": [{"id": "x", "status": 400}]})
    ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=tracer)
    assert tracer.errors and "207" in tracer.errors[0]


def test_eval_traces_are_tagged_eval_and_case(sc):
    tracer, sent = recording_tracer()
    suite = [{"id": "acme-supplier", "question": "Who supplies Acme?", "answerable": True,
              "gold": {"ticker": "ACME", "form": "10-K", "section": "1A"}}]
    evaluate.run(suite, sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=tracer)
    body = events(sent[0])[0]["body"]
    assert body["name"] == "eval:query"
    assert {"eval", "acme-supplier", "llama-3b"} <= set(body["tags"])


def test_agent_trace_has_a_span_per_tool_call(sc):
    tracer, sent = recording_tracer()
    guard = build_guard(FakeStore(), lambda t: FACTS, {"ACME"}, top_k=5, min_score=0.55,
                        max_calls=4)
    model = Scripted(call("get_financial_fact", ticker="ACME", metric="revenue",
                          fiscal_year=2025),
                     call("delete_everything"),  # refused by the guard: a WARNING span
                     say("Revenue was $12,345 million [S1]."))
    agent.traced_run("Acme revenue in fiscal 2025?", sc, guard, model,
                     replace(S, model="llama-3b"), tracer)
    batch = events(sent[0])
    spans = [e["body"] for e in batch if e["type"] == "span-create"]
    assert [s["name"] for s in spans] == ["tool:get_financial_fact", "tool:delete_everything"]
    assert spans[1]["level"] == "WARNING"
    assert "agent" in batch[0]["body"]["tags"]
