"""An agent that answers with tools: scope -> (model <-> guarded tools)* -> verify.

The plain answer path (answer.py) retrieves once and answers. Some questions need
more: "How much did Microsoft's revenue grow in fiscal 2025?" needs two reported
figures and a calculation. Here the model plans those steps itself by calling
tools, and LangGraph runs the loop:

    START -> scope --out of scope--> END (refused: scope)
               |
               v
             model --tool calls--> tools --> model ...   (ToolGuard: allowlist,
               |  +--budget spent--> limit -> END          validation, budget, log)
               v
             verify -> END   (refusal phrase / no evidence / numeric grounding)

The guardrails do not trust the model: the scope check runs before it, every tool
call goes through ToolGuard, and every figure in the final answer must appear in
what the tools returned. Each refusal names its stage, like answer.py.

The model sits behind a two-method interface (`ChatModel`), so a hosted model
(Mistral, OpenAI, Claude) can replace Ollama without touching the graph.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from finrag import grounding, scope
from finrag.answer import REFUSAL, Answer
from finrag.config import Settings
from finrag.models import ModelPreset, Usage, ollama_options, strip_thinking, unfinished
from finrag.tools import ToolGuard

SYSTEM = """You are a financial analyst assistant. You answer questions about the SEC
filings of these companies only: {companies}.

Rules:
- Never answer from memory. Use the tools to find every fact and figure.
- Use get_financial_fact for reported figures (revenue, net income...), search_filings
  for anything written in the filings (risks, strategy, legal matters...), and
  calculate for any growth rate, difference or ratio. Never do arithmetic yourself.
- Work one step at a time: first get each figure, then call calculate with the plain
  numbers the tools returned, e.g. (331839000000 - 281724000000) / 281724000000 * 100.
  A tool cannot be called inside another tool's arguments.
- Cite the source id of every fact, like [S1] or [S2][S3].
- Copy figures exactly as the tools return them.
- If the tools do not give you the answer, reply exactly: {refusal}"""


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return one assistant message: {"role", "content", optional "tool_calls"}."""


@dataclass
class OllamaChat:
    url: str
    preset: ModelPreset
    num_ctx: int = 8192  # tool results pile up across turns; still far below the default
    usage: Usage | None = None  # summed over every turn of one question

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.preset.model_id, "messages": messages, "tools": tools,
                "stream": False,
                # num_ctx explicitly, always: see CLAUDE.md (the default exhausts RAM here).
                # A reasoning preset needs its larger context here too.
                **ollama_options(self.preset, max(self.num_ctx, self.preset.num_ctx))}
        r = httpx.post(f"{self.url}/api/chat", json=body, timeout=300.0)
        r.raise_for_status()
        data = r.json()
        turn = Usage.from_ollama(self.preset.name, data)
        self.usage = turn if self.usage is None else self.usage + turn
        # Only content and tool calls go back into the conversation: the reasoning
        # (msg["thinking"] with think=True) would fill the context turn after turn.
        msg = {k: v for k, v in data["message"].items() if k != "thinking"}
        content = strip_thinking(msg.get("content") or "")
        # A cut final message is no answer: verify refuses it at the `model` stage.
        msg["content"] = "" if unfinished(content or "x", data.get("done_reason")) else content
        return msg


@dataclass
class AgentAnswer(Answer):
    steps: list[dict] = field(default_factory=list)  # ToolGuard audit log


class State(TypedDict, total=False):
    question: str
    messages: list[dict]
    refused: bool
    stage: str
    reason: str
    answer: str


def cited_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[(S\d+)\]", text)), key=lambda s: int(s[1:]))


def build_graph(sc: scope.Scope, guard: ToolGuard, model: ChatModel):
    def scope_node(state: State) -> State:
        decision = scope.check(state["question"], sc)
        if not decision.in_scope:
            return {"refused": True, "stage": "scope", "reason": decision.reason}
        system = SYSTEM.format(companies=", ".join(sorted(sc.corpus_tickers)),
                               refusal=REFUSAL)
        return {"refused": False, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": state["question"]}]}

    def model_node(state: State) -> State:
        reply = model.chat(state["messages"], guard.schemas())
        msg = {"role": "assistant", "content": reply.get("content") or ""}
        if reply.get("tool_calls"):
            msg["tool_calls"] = reply["tool_calls"]
        return {"messages": [*state["messages"], msg]}

    def tools_node(state: State) -> State:
        out = []
        for call in state["messages"][-1]["tool_calls"]:
            fn = call.get("function", {})
            # tool_call_id: OpenAI and Anthropic pair each result with its call.
            out.append({"role": "tool", "tool_name": fn.get("name", ""),
                        "tool_call_id": call.get("id"),
                        "content": guard.run(fn.get("name", ""), fn.get("arguments"))})
        return {"messages": [*state["messages"], *out]}

    def limit_node(state: State) -> State:
        return {"refused": True, "stage": "limit",
                "reason": f"no answer within the budget of {guard.max_calls} tool calls"}

    def verify_node(state: State) -> State:
        text = state["messages"][-1]["content"].strip()
        if not text or REFUSAL.lower().rstrip(".") in text.lower():
            return {"refused": True, "stage": "model", "reason": "the model found no answer"}
        if not guard.evidence:
            return {"refused": True, "stage": "evidence",
                    "reason": "the model answered without any tool result to rely on"}
        ids = set(cited_ids(text)) or {ev.id for ev in guard.evidence}
        check = grounding.check(text, [ev.text for ev in guard.evidence if ev.id in ids])
        if not check.grounded:
            return {"refused": True, "stage": "numbers",
                    "reason": "figures not found in the cited tool results: "
                              + ", ".join(check.ungrounded)}
        return {"refused": False, "stage": "answered", "reason": "ok", "answer": text}

    def after_scope(state: State) -> str:
        return END if state["refused"] else "model"

    def after_model(state: State) -> str:
        if not state["messages"][-1].get("tool_calls"):
            return "verify"
        return "limit" if guard.exhausted else "tools"

    g = StateGraph(State)
    for name, fn in [("scope", scope_node), ("model", model_node), ("tools", tools_node),
                     ("limit", limit_node), ("verify", verify_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "scope")
    g.add_conditional_edges("scope", after_scope, ["model", END])
    g.add_conditional_edges("model", after_model, ["tools", "limit", "verify"])
    g.add_edge("tools", "model")
    g.add_edge("limit", END)
    g.add_edge("verify", END)
    return g.compile()


def run(question: str, sc: scope.Scope, guard: ToolGuard, model: ChatModel,
        gpu: str = "L4") -> AgentAnswer:
    graph = build_graph(sc, guard, model)
    # Each model turn is 2 graph steps (model + tools): the budget, not this limit,
    # is what normally stops a loop. This is the backstop.
    state = graph.invoke({"question": question},
                         {"recursion_limit": 2 * guard.max_calls + 10})
    use = getattr(model, "usage", None)
    usage = use.as_dict(gpu) if use else None
    if state["refused"]:
        return AgentAnswer(question, REFUSAL, True, state["stage"], state["reason"],
                           _sources(guard, None), usage=usage, steps=guard.log)
    ids = cited_ids(state["answer"])
    return AgentAnswer(question, state["answer"], False, "answered", "ok",
                       _sources(guard, set(ids) or None), usage=usage, steps=guard.log)


def traced_run(question: str, sc: scope.Scope, guard: ToolGuard, model: ChatModel,
               settings: Settings, tracer=None) -> AgentAnswer:
    """run(), plus one Langfuse trace: a span per tool call (from the guard's audit log)
    and one generation with the model's summed usage."""
    trace = (tracer.trace("agent", question, tags=[settings.model, "agent"],
                          metadata={"model_id": settings.preset.model_id,
                                    "max_calls": guard.max_calls}) if tracer else None)
    t0 = time.time()
    a = run(question, sc, guard, model, gpu=settings.gpu)
    if trace is None:
        return a
    for st in a.steps:
        with trace.span(f"tool:{st['tool']}", input=st["args"]) as sp:
            sp.output = {"ok": st["ok"], "error": st.get("error")}
            sp.metadata = {"ms": st.get("ms")}
            if not st["ok"]:
                sp.warning = st.get("error") or "refused by ToolGuard"
    trace.generation("llm (all turns)", t0, settings.preset.model_id, question, a.answer,
                     a.usage, {"preset": settings.model})
    trace.end({"answer": a.answer, "refused": a.refused, "stage": a.stage, "reason": a.reason},
              tags=[f"stage:{a.stage}"], metadata={"usage": a.usage,
                                                   "tool_calls": len(a.steps)})
    return a


def _sources(guard: ToolGuard, ids: set[str] | None) -> list[dict]:
    return [{"n": ev.id, "tool": ev.tool, **ev.source}
            for ev in guard.evidence if ids is None or ev.id in ids]


def from_settings(settings: Settings, store, sc: scope.Scope, max_calls: int = 6):
    """Wire the real store, SEC facts and Ollama. Returns (guard, model)."""
    from finrag import edgar
    from finrag.tools import build_guard, facts_loader

    filings = edgar.load_manifest(settings.manifest_path)
    ciks = {f.ticker: f.cik for f in filings}
    load = facts_loader(lambda: edgar.EdgarClient(settings.sec_user_agent), ciks,
                        settings.data_dir / "facts")
    guard = build_guard(store, load, ciks, settings.top_k, settings.min_score, max_calls)
    if settings.preset.provider != "ollama":
        from finrag.providers import chat_model

        return guard, chat_model(settings.preset, settings)
    return guard, OllamaChat(settings.ollama_url, settings.preset)
