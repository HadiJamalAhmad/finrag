"""Hosted models (OpenAI, Anthropic) behind the same two calls as Ollama.

- generate_hosted(prompt, preset, settings) -> (text, Usage, done_reason): the one-shot
  answer of answer.py, same contract as the Ollama path.
- OpenAIChat / AnthropicChat: the agent's ChatModel. The agent keeps its own message
  format (Ollama's: tool_calls with {"function": {"name", "arguments"}}, tool results
  as {"role": "tool", "tool_call_id", "content"}); each adapter converts both ways.

Plain HTTP with httpx, like the rest of finrag: no vendor SDK. A missing key raises
ProviderNotConfigured, which the API turns into a 503 naming the model.
"""

from __future__ import annotations

import json
import time

import httpx

from finrag.models import ModelPreset, Usage

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TIMEOUT = 120.0


class ProviderNotConfigured(RuntimeError):
    """The preset is hosted and its vendor key is not set."""


def api_key(preset: ModelPreset, settings) -> str:
    key = {"openai": settings.openai_api_key,
           "anthropic": settings.anthropic_api_key}.get(preset.provider, "")
    if not key:
        env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[preset.provider]
        raise ProviderNotConfigured(f"model {preset.name!r} needs {env}")
    return key


# ---------------------------------------------------------------- OpenAI
def _openai_call(preset: ModelPreset, key: str, messages: list[dict],
                 tools: list[dict] | None = None) -> tuple[dict, Usage, str | None]:
    body = {"model": preset.model_id, "messages": messages,
            # GPT-6 models take max_completion_tokens; no temperature (see models.py).
            "max_completion_tokens": preset.num_predict, **dict(preset.api_params)}
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    r = httpx.post(OPENAI_URL, json=body, timeout=TIMEOUT,
                   headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    u = data.get("usage") or {}
    usage = Usage(preset.name, int(u.get("prompt_tokens") or 0),
                  int(u.get("completion_tokens") or 0), time.perf_counter() - t0, 1)
    done = "length" if choice.get("finish_reason") == "length" else "stop"
    return choice["message"], usage, done


def _to_openai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{
                            "id": c.get("id") or f"call_{i}", "type": "function",
                            "function": {"name": c["function"]["name"],
                                         "arguments": _as_json(c["function"].get("arguments"))}}
                            for i, c in enumerate(m["tool_calls"])]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id") or "call_0",
                        "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


class OpenAIChat:
    def __init__(self, preset: ModelPreset, key: str):
        self.preset, self.key, self.usage = preset, key, None
        self.last_usage, self.last_reasoning = None, None  # the latest turn, for its trace

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        msg, turn, done = _openai_call(self.preset, self.key, _to_openai(messages), tools)
        self.last_usage, self.last_reasoning = turn, None
        self.usage = turn if self.usage is None else self.usage + turn
        calls = [{"id": c["id"], "function": {"name": c["function"]["name"],
                                              "arguments": c["function"].get("arguments")}}
                 for c in msg.get("tool_calls") or []]
        content = msg.get("content") or ""
        return {"role": "assistant", "content": "" if done == "length" else content,
                **({"tool_calls": calls} if calls else {})}


# ---------------------------------------------------------------- Anthropic
def _anthropic_call(preset: ModelPreset, key: str, messages: list[dict], system: str = "",
                    tools: list[dict] | None = None) -> tuple[list[dict], Usage, str | None]:
    body = {"model": preset.model_id, "max_tokens": preset.num_predict, "messages": messages,
            **dict(preset.api_params)}  # no temperature: Claude 5.5 rejects it
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [{"name": t["function"]["name"],
                          "description": t["function"].get("description", ""),
                          "input_schema": t["function"]["parameters"]} for t in tools]
    t0 = time.perf_counter()
    r = httpx.post(ANTHROPIC_URL, json=body, timeout=TIMEOUT, headers={
        "x-api-key": key, "anthropic-version": ANTHROPIC_VERSION})
    r.raise_for_status()
    data = r.json()
    u = data.get("usage") or {}
    usage = Usage(preset.name, int(u.get("input_tokens") or 0),
                  int(u.get("output_tokens") or 0), time.perf_counter() - t0, 1)
    done = "length" if data.get("stop_reason") == "max_tokens" else "stop"
    return data.get("content") or [], usage, done


def _reasoning(blocks: list[dict]) -> str | None:
    return "\n".join(b.get("thinking", "") for b in blocks
                     if b.get("type") == "thinking").strip() or None


def _text(blocks: list[dict]) -> str:
    # Thinking blocks are the model's reasoning: never part of the answer.
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system, out = "", []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "call_0",
                     "content": m["content"]}
            # Consecutive tool results travel in ONE user turn.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            blocks += [{"type": "tool_use", "id": c.get("id") or f"call_{i}",
                        "name": c["function"]["name"],
                        "input": _as_dict(c["function"].get("arguments"))}
                       for i, c in enumerate(m.get("tool_calls") or [])]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m.get("content") or ""})
    return system, out


class AnthropicChat:
    def __init__(self, preset: ModelPreset, key: str):
        self.preset, self.key, self.usage = preset, key, None
        self.last_usage, self.last_reasoning = None, None  # the latest turn, for its trace

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        system, msgs = _to_anthropic(messages)
        blocks, turn, done = _anthropic_call(self.preset, self.key, msgs, system, tools)
        self.last_usage, self.last_reasoning = turn, _reasoning(blocks)
        self.usage = turn if self.usage is None else self.usage + turn
        calls = [{"id": b["id"], "function": {"name": b["name"], "arguments": b.get("input")}}
                 for b in blocks if b.get("type") == "tool_use"]
        content = _text(blocks)
        return {"role": "assistant", "content": "" if done == "length" else content,
                **({"tool_calls": calls} if calls else {})}


# ---------------------------------------------------------------- one-shot answer
def generate_hosted(prompt: str, preset: ModelPreset,
                    settings) -> tuple[str, Usage, str | None, str | None]:
    """(answer, usage, done_reason, reasoning)."""
    key = api_key(preset, settings)
    user = [{"role": "user", "content": prompt}]
    if preset.provider == "openai":
        msg, usage, done = _openai_call(preset, key, user)
        return (msg.get("content") or "").strip(), usage, done, None
    if preset.provider == "anthropic":
        blocks, usage, done = _anthropic_call(preset, key, user)
        return _text(blocks), usage, done, _reasoning(blocks)
    raise ValueError(f"not a hosted provider: {preset.provider!r}")


def chat_model(preset: ModelPreset, settings):
    key = api_key(preset, settings)
    return (OpenAIChat if preset.provider == "openai" else AnthropicChat)(preset, key)


def _as_json(args) -> str:
    return args if isinstance(args, str) else json.dumps(args or {})


def _as_dict(args) -> dict:
    if isinstance(args, str):
        try:
            return json.loads(args or "{}")
        except json.JSONDecodeError:
            return {}
    return args or {}
