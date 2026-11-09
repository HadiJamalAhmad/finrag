"""Hosted models (OpenAI, Anthropic): request shape, usage, cost, agent conversion. Offline."""

import asyncio
import json
from dataclasses import replace

import pytest
from test_agent_tools import FACTS, FakeStore, Scripted, say

from finrag import agent, models, providers
from finrag import answer as ans
from finrag.config import Settings
from finrag.tools import build_guard

S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4",
             openai_api_key="sk-test", anthropic_api_key="sk-ant-test")


class Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def capture(monkeypatch, *bodies):
    """Replace httpx.post in providers; return the list of (url, json, headers) sent."""
    sent, queue = [], list(bodies)

    def fake_post(url, json, timeout, headers):
        sent.append((url, json, headers))
        return Resp(queue.pop(0))

    monkeypatch.setattr(providers.httpx, "post", fake_post)
    return sent


OPENAI_OK = {"choices": [{"message": {"role": "assistant", "content": "A supplier [1]."},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1200, "completion_tokens": 30}}
ANTHROPIC_OK = {"content": [{"type": "thinking", "thinking": "let me see"},
                            {"type": "text", "text": "A supplier [1]."}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1300, "output_tokens": 40}}


# ---------------------------------------------------------------- registry and cost
def test_five_presets_two_hosted_with_token_prices():
    assert list(models.MODEL_REGISTRY) == ["llama-3b", "qwen3-4b", "mistral-24b",
                                           "gpt-luna", "claude-haiku"]
    assert models.OLLAMA_MODELS == ["llama-3b", "qwen3-4b", "mistral-24b"]
    for name in ("gpt-luna", "claude-haiku"):
        p = models.get(name)
        assert p.digest is None and p.usd_per_mtok == (0.10, 0.50)


def test_hosted_cost_is_tokens_times_list_price_not_gpu_time():
    u = models.Usage("claude-haiku", 1_000_000, 1_000_000, seconds=3600, calls=1)
    assert u.cost_usd("L4") == pytest.approx(0.60)  # 0.10 + 0.50, the hour is ignored
    d = u.as_dict("L4")
    assert d["pricing"] == "tokens" and d["gpu"] is None


# ---------------------------------------------------------------- one-shot answer
def test_openai_request_shape_and_usage(monkeypatch):
    sent = capture(monkeypatch, OPENAI_OK)
    g = ans.generate("prompt", replace(S, model="gpt-luna"))
    url, body, headers = sent[0]
    assert url == providers.OPENAI_URL and headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-6-luna"
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "none"
    assert "temperature" not in body  # not documented for GPT-6: never sent
    assert g.text == "A supplier [1]." and g.done_reason == "stop"
    assert (g.usage.prompt_tokens, g.usage.completion_tokens) == (1200, 30)


def test_anthropic_request_shape_and_thinking_is_not_the_answer(monkeypatch):
    sent = capture(monkeypatch, ANTHROPIC_OK)
    g = ans.generate("prompt", replace(S, model="claude-haiku"))
    url, body, headers = sent[0]
    assert url == providers.ANTHROPIC_URL
    assert headers["x-api-key"] == "sk-ant-test" and headers["anthropic-version"]
    assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] == 4096
    assert "temperature" not in body  # Claude 5.5 rejects it
    assert g.text == "A supplier [1]."  # the thinking block is dropped
    assert (g.usage.prompt_tokens, g.usage.completion_tokens) == (1300, 40)


@pytest.mark.parametrize("model,body", [
    ("gpt-luna", {"choices": [{"message": {"content": "A suppl"}, "finish_reason": "length"}],
                  "usage": {}}),
    ("claude-haiku", {"content": [{"type": "text", "text": "A suppl"}],
                      "stop_reason": "max_tokens", "usage": {}}),
])
def test_a_cut_hosted_answer_is_refused_at_the_model_stage(monkeypatch, filing_10k,
                                                           sec_names, model, body):
    from finrag import scope

    capture(monkeypatch, body)
    sc = scope.Scope.build([filing_10k], sec_names)

    class Store:
        def search(self, *a, **k):
            return [(0.8, {**FakeStore().search("q", 5)[0][1]})]

    a = ans.answer("Who supplies Acme?", sc, Store(), replace(S, model=model))
    assert a.refused and a.stage == "model"
    assert a.reason == "the answer was cut at the output limit"


def test_a_missing_key_names_the_variable_and_maps_to_503():
    from finrag.api import not_configured

    with pytest.raises(providers.ProviderNotConfigured, match="needs ANTHROPIC_API_KEY"):
        providers.api_key(models.get("claude-haiku"), replace(S, anthropic_api_key=""))
    resp = asyncio.run(not_configured(None, providers.ProviderNotConfigured(
        "model 'gpt-luna' needs OPENAI_API_KEY")))
    assert resp.status_code == 503 and b"OPENAI_API_KEY" in resp.body


def test_keys_never_appear_in_a_printed_settings():
    text = repr(S)
    assert "sk-test" not in text and "sk-ant-test" not in text


# ---------------------------------------------------------------- agent conversion
CONVERSATION = [
    {"role": "system", "content": "rules"},
    {"role": "user", "content": "Revenue growth?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "get_financial_fact", "arguments": {"ticker": "ACME"}}},
        {"id": "c2", "function": {"name": "get_financial_fact", "arguments": {"ticker": "ACME"}}},
    ]},
    {"role": "tool", "tool_name": "get_financial_fact", "tool_call_id": "c1", "content": "[S1] 1"},
    {"role": "tool", "tool_name": "get_financial_fact", "tool_call_id": "c2", "content": "[S2] 2"},
]
TOOLS = [{"type": "function", "function": {"name": "calculate", "description": "math",
                                           "parameters": {"type": "object"}}}]


def test_anthropic_chat_converts_both_ways(monkeypatch):
    sent = capture(monkeypatch, {
        "content": [{"type": "tool_use", "id": "t9", "name": "calculate",
                     "input": {"expression": "2 - 1"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5}})
    reply = providers.AnthropicChat(models.get("claude-haiku"), "k").chat(CONVERSATION, TOOLS)
    body = sent[0][1]
    assert body["system"] == "rules"
    assert body["tools"][0] == {"name": "calculate", "description": "math",
                                "input_schema": {"type": "object"}}
    assert [b["id"] for b in body["messages"][1]["content"]] == ["c1", "c2"]  # tool_use
    # Both tool results travel in ONE user turn, each paired with its call.
    results = body["messages"][2]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["c1", "c2"]
    assert reply["tool_calls"] == [{"id": "t9", "function": {
        "name": "calculate", "arguments": {"expression": "2 - 1"}}}]


def test_openai_chat_converts_both_ways(monkeypatch):
    sent = capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": "x1", "type": "function", "function": {
            "name": "calculate", "arguments": "{\"expression\": \"2 - 1\"}"}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
    chat = providers.OpenAIChat(models.get("gpt-luna"), "k")
    reply = chat.chat(CONVERSATION, TOOLS)
    msgs = sent[0][1]["messages"]
    call = msgs[2]["tool_calls"][0]
    assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"ticker": "ACME"}
    assert [(m["role"], m["tool_call_id"]) for m in msgs[3:]] == [("tool", "c1"), ("tool", "c2")]
    assert reply["tool_calls"][0]["id"] == "x1" and chat.usage.prompt_tokens == 9


def test_the_graph_pairs_each_tool_result_with_its_call_id(filing_10k, sec_names):
    from finrag import scope

    sc = scope.Scope.build([filing_10k], sec_names)
    guard = build_guard(FakeStore(), lambda t: FACTS, {"ACME"}, top_k=5, min_score=0.55,
                        max_calls=4)
    model = Scripted(
        {"role": "assistant", "content": "", "tool_calls": [{"id": "abc", "function": {
            "name": "get_financial_fact",
            "arguments": {"ticker": "ACME", "metric": "revenue", "fiscal_year": 2025}}}]},
        say("Revenue was $12,345 million [S1]."))
    agent.run("Acme revenue in fiscal 2025?", sc, guard, model)
    tool_msgs = [m for m in model.seen[-1] if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "abc"
