"""Langfuse tracing through the real SDK, offline.

The SDK runs as in production, but its spans go to an in-memory OpenTelemetry exporter
(isolated tracer provider) and its HTTP calls (scores) to a MockTransport: nothing
leaves the machine (CLAUDE.md: tests stay offline).
"""

import json
import uuid
from dataclasses import replace

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_agent_tools import FACTS, FakeStore, Scripted, call, say

from finrag import agent, evaluate, models, scope
from finrag import answer as ans
from finrag.config import Settings
from finrag.tools import build_guard
from finrag.tracing import NULL, Tracer

# Keys pinned empty: the developer's .env must never turn a test into a live call.
S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4", langfuse_public_key="",
             langfuse_secret_key="")
GEN = ans.Generation("A single supplier in Taiwan [1].",
                     models.Usage("llama-3b", 900, 40, 4.5, 1), "stop", "Item 1A says...")


class Capture:
    """A Tracer on the real SDK whose output stays in memory."""

    def __init__(self, fail_http: bool = False):
        self.exporter, self.requests = InMemorySpanExporter(), []

        def handler(request):
            if fail_http:
                raise httpx.ConnectError("langfuse down")
            self.requests.append(request)
            return httpx.Response(200, json={})

        s = replace(S, langfuse_public_key=f"pk-lf-test-{uuid.uuid4()}",
                    langfuse_secret_key="sk-lf-test", langfuse_base_url="http://langfuse.test")
        self.tracer = Tracer.from_settings(
            s, environment="test", tracer_provider=TracerProvider(),
            span_exporter=self.exporter,
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def spans(self) -> list:
        self.tracer.flush()
        return list(self.exporter.get_finished_spans())

    def scores(self) -> dict:
        self.tracer.flush()
        out = {}
        for r in self.requests:
            if r.url.path.endswith("/api/public/scores"):
                for sc in json.loads(r.content):
                    out[sc["name"]] = sc
        return out


def attr(span, key):
    v = span.attributes.get(f"langfuse.{key}")
    try:
        return json.loads(v) if isinstance(v, str) else v
    except json.JSONDecodeError:
        return v


def by_name(spans, name):
    return [s for s in spans if s.name == name]


@pytest.fixture
def sc(filing_10k, sec_names):
    return scope.Scope.build([filing_10k], sec_names)


# ---------------------------------------------------------------- off without keys
def test_no_keys_no_client_no_tracing(sc):
    t = Tracer.from_settings(S)
    assert not t.enabled and t.client is None
    with t.root("answer-question", "chain", input="q", tags=[]) as obs:
        assert obs is NULL
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=t)
    assert not a.refused and a.trace_id is None


# ---------------------------------------------------------------- pipeline
def test_an_answer_is_one_trace_typed_stage_by_stage(sc):
    cap = Capture()
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN,
                   tracer=cap.tracer)
    spans = cap.spans()
    root = by_name(spans, "answer-question")[0]
    children = [s for s in spans if s.parent and s.parent.span_id == root.context.span_id]
    assert [(s.name, attr(s, "observation.type")) for s in children] == [
        ("check-scope", "guardrail"), ("retrieve-passages", "retriever"),
        ("check-evidence", "guardrail"), ("generate-answer", "generation"),
        ("check-figures", "guardrail")]
    assert attr(root, "observation.type") == "chain"
    assert attr(root, "observation.input") == "Who supplies Acme?"
    # The root shows the answer at a glance; the stage is metadata (and a score).
    assert attr(root, "observation.output") == GEN.text
    assert attr(root, "observation.metadata.stage") == "answered"
    # Tags set at creation, on every observation; environment separates test traffic.
    assert set(root.attributes["langfuse.trace.tags"]) == {"finrag", "pipeline", "llama-3b"}
    assert all(s.attributes["langfuse.environment"] == "test" for s in spans)
    assert a.trace_id == format(root.context.trace_id, "032x")


def test_the_generation_carries_messages_model_usage_cost_and_reasoning(sc):
    cap = Capture()
    ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=cap.tracer)
    gen = by_name(cap.spans(), "generate-answer")[0]
    assert attr(gen, "observation.model.name") == "llama3.2:3b"
    assert attr(gen, "observation.model.parameters")["num_ctx"] == 4096
    messages = attr(gen, "observation.input")
    assert messages[0]["role"] == "user" and "Excerpts:" in messages[0]["content"]
    assert attr(gen, "observation.output") == {"role": "assistant", "content": GEN.text}
    # Exclusive input/output buckets; our cost (Langfuse cannot price a self-hosted model).
    assert attr(gen, "observation.usage_details") == {"input": 900, "output": 40}
    assert attr(gen, "observation.cost_details")["total"] == pytest.approx(
        4.5 * 0.80 / 3600, abs=1e-6)
    assert attr(gen, "observation.metadata.reasoning") == "Item 1A says..."


def test_the_retriever_shows_the_passages_the_model_saw(sc):
    cap = Capture()
    ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=cap.tracer)
    out = attr(by_name(cap.spans(), "retrieve-passages")[0], "observation.output")
    assert out[0]["section"] == "1A" and "single supplier in Taiwan" in out[0]["text"]


def test_a_refusal_warns_at_its_stage_and_is_scored_not_tagged(sc):
    cap = Capture()
    a = ans.answer("Who supplies Acme?", sc, FakeStore(score=0.3), S,
                   llm=lambda p, s: GEN, tracer=cap.tracer)
    spans = cap.spans()
    gate = by_name(spans, "check-evidence")[0]
    assert attr(gate, "observation.level") == "WARNING"
    assert not by_name(spans, "generate-answer")  # the model was never called
    scores = cap.scores()
    assert scores["outcome"]["value"] == "evidence" == a.stage
    assert scores["refused"]["value"] == 1
    assert "stage:evidence" not in by_name(spans, "answer-question")[0].attributes[
        "langfuse.trace.tags"]


def test_the_channel_is_a_tag_set_at_creation(sc):
    cap = Capture()
    cap.tracer.channel = "api"
    ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=cap.tracer)
    root = by_name(cap.spans(), "answer-question")[0]
    assert "api" in root.attributes["langfuse.trace.tags"]


def test_tracing_failures_never_break_the_answer(sc):
    cap = Capture(fail_http=True)  # scores cannot be sent
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN,
                   tracer=cap.tracer)
    assert not a.refused and a.answer == GEN.text
    cap.tracer.flush()  # must not raise either


# ---------------------------------------------------------------- eval
def test_eval_traces_go_to_the_eval_environment_with_a_correct_score(sc):
    cap = Capture()
    suite = [{"id": "acme-supplier", "question": "Who supplies Acme?", "answerable": True,
              "gold": {"ticker": "ACME", "form": "10-K", "section": "1A"}}]
    evaluate.run(suite, sc, FakeStore(), S, llm=lambda p, s: GEN, tracer=cap.tracer)
    root = by_name(cap.spans(), "answer-question")[0]
    assert root.attributes["langfuse.environment"] == "eval"
    assert attr(root, "observation.metadata.expected")["section"] == "1A"  # ground truth
    assert "eval" in root.attributes["langfuse.trace.tags"]
    assert attr(root, "observation.metadata.case") == "acme-supplier"
    assert cap.scores()["correct"]["value"] == 1  # answerable and answered


# ---------------------------------------------------------------- agent
def test_agent_trace_one_generation_per_turn_tools_as_siblings(sc):
    cap = Capture()
    guard = build_guard(FakeStore(), lambda t: FACTS, {"ACME"}, top_k=5, min_score=0.55,
                        max_calls=4)
    first = call("get_financial_fact", ticker="ACME", metric="revenue", fiscal_year=2025)
    first["tool_calls"][0]["id"] = "call_1"
    model = Scripted(first, call("delete_everything"), say("Revenue was $12,345 million [S1]."))
    a = agent.traced_run("Acme revenue in fiscal 2025?", sc, guard, model, S, cap.tracer)
    spans = cap.spans()
    root = by_name(spans, "answer-with-tools")[0]
    assert attr(root, "observation.type") == "agent"
    children = [s for s in spans if s.parent and s.parent.span_id == root.context.span_id]
    assert [s.name for s in children] == [
        "check-scope", "generate-response", "get-financial-fact", "generate-response",
        "unknown-tool", "generate-response", "verify-answer"]
    tool = by_name(spans, "get-financial-fact")[0]
    assert attr(tool, "observation.type") == "tool"
    assert attr(tool, "observation.metadata.tool_call_id") == "call_1"
    unknown = by_name(spans, "unknown-tool")[0]
    assert attr(unknown, "observation.level") == "WARNING"
    assert attr(unknown, "observation.metadata.tool") == "delete_everything"
    # Tool-call arguments as JSON strings in the role-labelled output of the generation.
    out = attr(by_name(spans, "generate-response")[0], "observation.output")
    assert json.loads(out["tool_calls"][0]["function"]["arguments"])["ticker"] == "ACME"
    assert cap.scores()["outcome"]["value"] == a.stage


def test_the_root_never_attaches_to_an_ambient_opentelemetry_span(sc):
    # Production 2026-11-09: a foreign span (runtime / library) was active; the root nested
    # under it, the export filter dropped it, and the trace was orphaned and borrowed.
    cap = Capture()
    ambient = TracerProvider().get_tracer("some-library")
    with ambient.start_as_current_span("ambient-request") as outer:
        ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: GEN,
                   tracer=cap.tracer)
    root = by_name(cap.spans(), "answer-question")[0]
    assert root.parent is None
    assert root.context.trace_id != outer.get_span_context().trace_id
