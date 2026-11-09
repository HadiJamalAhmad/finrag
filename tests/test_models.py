"""The model registry, per-model generation, usage and cost: all offline."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from finrag import agent, evaluate, models, scope
from finrag import answer as ans
from finrag.config import Settings

S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4")
RISK = {"ticker": "ACME", "form": "10-K", "period": "2025-09-27", "section": "1A",
        "section_title": "Risk Factors", "url": "u", "fiscal_year": 2025,
        "text": "The Company depends on a single supplier in Taiwan."}


class FakeStore:
    def search(self, query, k, tickers=frozenset(), years=frozenset()):
        return [(0.8, RISK)]


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


OLLAMA_BODY = {"response": "<think>hmm</think>A single supplier in Taiwan [1].",
               "prompt_eval_count": 900, "eval_count": 40, "total_duration": 4_500_000_000}


# ---------------------------------------------------------------- registry
def test_registry_matches_the_lock_file():
    # The image build pulls deploy/models.lock and fails on a digest mismatch:
    # the registry and the lock must name the same weights.
    lock = models.read_lock()
    assert lock == {p.ollama_tag: p.digest for p in models.MODEL_REGISTRY.values()}
    assert all(len(d) == 64 for d in lock.values())


def test_three_models_and_a_loud_failure_on_unknown(monkeypatch):
    assert list(models.MODEL_REGISTRY) == ["llama-3b", "qwen3-4b", "mistral-24b"]
    monkeypatch.delenv("FINRAG_MODEL", raising=False)
    assert models.get().name == "llama-3b"
    monkeypatch.setenv("FINRAG_MODEL", "mistral-24b")
    assert models.get().ollama_tag == "mistral-small3.2:24b"
    with pytest.raises(ValueError, match="options: llama-3b, qwen3-4b, mistral-24b"):
        models.get("gpt-5")


def test_cost_is_model_seconds_times_the_gpu_price():
    u = models.Usage("llama-3b", 900, 40, 4.5, 1) + models.Usage("llama-3b", 100, 10, 1.5, 1)
    assert (u.prompt_tokens, u.completion_tokens, u.calls) == (1000, 50, 2)
    assert u.cost_usd("L4") == pytest.approx(6.0 * 0.80 / 3600)


# ---------------------------------------------------------------- generation
@pytest.mark.parametrize("name,think,ctx,cap", [("llama-3b", None, 4096, 1024),
                                                ("mistral-24b", None, 4096, 1024),
                                                ("qwen3-4b", True, 12288, 8192)])
def test_generate_uses_the_preset(monkeypatch, name, think, ctx, cap):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return FakeResponse(OLLAMA_BODY)

    monkeypatch.setattr(ans.httpx, "post", fake_post)
    g = ans.generate("prompt", replace(S, model=name))
    assert sent["model"] == models.get(name).ollama_tag
    assert sent["options"]["num_ctx"] == ctx  # always explicit (CLAUDE.md)
    assert sent["options"]["num_predict"] == cap  # output cap: latency and cost bound
    # qwen3:4b always reasons: think=True puts the reasoning in a separate field.
    assert sent.get("think") is think
    assert g.text == "A single supplier in Taiwan [1]."  # a stray <think> block is dropped
    assert (g.usage.model, g.usage.prompt_tokens, g.usage.seconds) == (name, 900, 4.5)


def test_answer_carries_usage_and_cost(filing_10k, sec_names):
    sc = scope.Scope.build([filing_10k], sec_names)
    gen = ans.Generation("A single supplier in Taiwan [1].",
                         models.Usage("qwen3-4b", 900, 40, 4.5, 1))
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: gen)
    assert not a.refused and a.usage["model"] == "qwen3-4b"
    assert a.usage["est_cost_usd"] == pytest.approx(4.5 * 0.80 / 3600, abs=1e-6)
    # Refused before the model: nothing was spent, so no usage.
    assert ans.answer("Tesla revenue?", sc, FakeStore(), S, llm=lambda p, s: gen).usage is None


def test_agent_chat_sums_usage_over_turns(monkeypatch):
    bodies = iter([
        {"message": {"role": "assistant", "content": "", "tool_calls": [{}]},
         "prompt_eval_count": 500, "eval_count": 20, "total_duration": 2_000_000_000},
        {"message": {"role": "assistant", "content": "<think>x</think>Done [S1]."},
         "prompt_eval_count": 800, "eval_count": 30, "total_duration": 3_000_000_000},
    ])
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json)
        return FakeResponse(next(bodies))

    monkeypatch.setattr(agent.httpx, "post", fake_post)
    chat = agent.OllamaChat("http://ollama", models.get("qwen3-4b"))
    chat.chat([], [])
    reply = chat.chat([], [])
    assert reply["content"] == "Done [S1]."
    # The reasoning preset keeps its larger context (12288 > the agent's 8192).
    assert all(b["think"] is True and b["options"]["num_ctx"] == 12288 for b in sent)
    assert (chat.usage.calls, chat.usage.prompt_tokens, chat.usage.seconds) == (2, 1300, 5.0)


@pytest.mark.parametrize("gen,reason", [
    (ans.Generation("A single supplier in Tai", models.Usage("llama-3b"), "length"),
     "the answer was cut at the output limit"),
    (ans.Generation("", models.Usage("qwen3-4b"), "stop"), "the model returned an empty answer"),
    (ans.Generation("<think>still going", models.Usage("qwen3-4b"), "stop"),
     "the model's reasoning never ended"),
])
def test_an_unfinished_answer_is_refused_at_the_model_stage(filing_10k, sec_names, gen, reason):
    sc = scope.Scope.build([filing_10k], sec_names)
    a = ans.answer("Who supplies Acme?", sc, FakeStore(), S, llm=lambda p, s: gen)
    assert a.refused and a.stage == "model" and a.reason == reason
    assert a.answer == ans.REFUSAL and a.usage is not None  # what it cost is still reported


def test_a_cut_agent_message_becomes_no_answer(monkeypatch):
    body = {"message": {"role": "assistant", "content": "Revenue grew by"},
            "done_reason": "length", "prompt_eval_count": 1, "eval_count": 1024}
    monkeypatch.setattr(agent.httpx, "post", lambda url, json, timeout: FakeResponse(body))
    reply = agent.OllamaChat("http://ollama", models.get("llama-3b")).chat([], [])
    assert reply["content"] == ""  # verify_node then refuses at the `model` stage


def test_cost_uses_the_gpu_that_actually_ran():
    assert models.detect_gpu("NVIDIA A10G") == "A10G"
    assert models.detect_gpu("NVIDIA L4") == "L4"
    with pytest.raises(ValueError, match="no price"):
        models.detect_gpu("NVIDIA H100 80GB HBM3")


# ---------------------------------------------------------------- eval report
def test_full_eval_reports_model_latency_and_cost(filing_10k, sec_names):
    sc = scope.Scope.build([filing_10k], sec_names)
    suite = [
        {"id": "a", "question": "Who supplies Acme?", "answerable": True,
         "gold": {"ticker": "ACME", "form": "10-K", "section": "1A"}},
        {"id": "b", "question": "Tesla revenue?", "answerable": False},
    ]
    gen = ans.Generation("Taiwan supplier [1].", models.Usage("mistral-24b", 1000, 50, 9.0, 1))
    rep = evaluate.run(suite, sc, FakeStore(), replace(S, model="mistral-24b"),
                       llm=lambda p, s: gen)
    assert rep["model"]["preset"] == "mistral-24b"
    assert rep["model"]["digest"] == models.get("mistral-24b").digest  # provenance
    assert rep["usage"]["llm_answers"] == 1  # the scope refusal never reached the model
    assert rep["usage"]["est_cost_usd_total"] == pytest.approx(9.0 * 0.80 / 3600, abs=1e-4)
    assert rep["latency_s"]["p50"] is not None
    assert "seconds" in rep["cases"][0] and "est_cost_usd" in rep["cases"][0]


# ---------------------------------------------------------------- API
def test_api_lists_the_models_and_rejects_an_unknown_one():
    from finrag.api import app

    client = TestClient(app)
    names = [m["name"] for m in client.get("/models").json()["models"]]
    assert names == ["llama-3b", "qwen3-4b", "mistral-24b"]
    r = client.post("/query", json={"question": "Who supplies Apple?", "model": "gpt-5"})
    assert r.status_code == 422  # refused before any retrieval or model call
