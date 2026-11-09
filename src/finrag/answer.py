"""The answer path: scope check -> retrieval -> evidence gate -> LLM -> number check.

Each step can refuse, and the response says which one did. A refusal with a
reason is something an analyst can act on; a silent wrong figure is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from finrag import grounding, scope
from finrag.config import Settings
from finrag.models import Usage, model_parameters, ollama_options, strip_thinking, unfinished
from finrag.tracing import NULL

REFUSAL = "I don't know based on the filings provided."

PROMPT = """You are a financial analyst assistant. Answer ONLY from the numbered SEC filing
excerpts below. Cite every sentence with its excerpt number, like [1] or [2][3].
Copy figures exactly as written in the excerpts, with their units.
If the excerpts do not contain the answer, reply exactly: {refusal}

Excerpts:
{context}

Question: {question}
Answer:"""


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    stage: str  # which step decided: scope | evidence | model | numbers | answered
    reason: str
    sources: list[dict] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    # Tokens, model seconds and estimated GPU cost, when the model was called.
    usage: dict | None = None
    # The Langfuse trace of this answer (None when tracing is off).
    trace_id: str | None = None


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage
    # Ollama's done_reason: "stop", or "length" when the answer hit num_predict.
    done_reason: str | None = "stop"
    # The model's reasoning, when it returns one apart from the answer (Qwen3's
    # `thinking`, Claude's thinking blocks): traced, never shown as the answer.
    reasoning: str | None = None


def build_prompt(question: str, hits: list[tuple[float, dict]]) -> str:
    context = "\n\n".join(f"[{i}] ({p['ticker']} {p['form']} {p['period']}, {p['section_title']})\n"
                          f"{p['text']}" for i, (_, p) in enumerate(hits, 1))
    return PROMPT.format(refusal=REFUSAL, context=context, question=question)


def cited_indices(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


def generate(prompt: str, settings: Settings) -> Generation:
    preset = settings.preset
    if preset.provider != "ollama":
        from finrag.providers import generate_hosted

        text, usage, done, reasoning = generate_hosted(prompt, preset, settings)
        return Generation(text, usage, done, reasoning)
    # num_ctx and num_predict always explicit, per preset (finrag.models): recent Ollama
    # defaults to a huge context and tried to allocate a 12 GB KV cache on this machine.
    body = {"model": preset.model_id, "prompt": prompt, "stream": False,
            **ollama_options(preset)}
    r = httpx.post(f"{settings.ollama_url}/api/generate", json=body, timeout=300.0)
    r.raise_for_status()
    data = r.json()
    # With think=True the reasoning comes back in data["thinking"], never in the answer.
    return Generation(strip_thinking(data["response"]), Usage.from_ollama(preset.name, data),
                      data.get("done_reason"), data.get("thinking") or None)


def answer(question: str, sc: scope.Scope, store, settings: Settings, llm=generate,
           tracer=None, trace_context: dict | None = None) -> Answer:
    """The answer path. With a tracer (finrag.tracing): one Langfuse trace per question.

    trace_context: extra "tags", "metadata" and "environment" for the trace (eval runs).
    """
    ctx = trace_context or {}
    if tracer is None or not tracer.enabled:
        return _answer(question, sc, store, settings, llm, NULL)
    with tracer.root("answer-question", "chain", input=question,
                     tags=["pipeline", settings.model, *ctx.get("tags", [])],
                     metadata={"model_id": settings.preset.model_id,
                               "min_score": settings.min_score, "top_k": settings.top_k,
                               **ctx.get("metadata", {})},
                     environment=ctx.get("environment")) as root:
        a = _answer(question, sc, store, settings, llm, root)
        # What a reviewer needs at a glance; sources and usage go to metadata.
        # The answer itself, at a glance; the deciding stage is in metadata and scores.
        root.update(output=a.answer,
                    metadata={"stage": a.stage, "reason": a.reason, "sources": a.sources,
                              "usage": a.usage, "ungrounded_numbers": a.ungrounded_numbers})
        # Known only at the end: scores, not tags (tags are fixed at creation).
        root.score("outcome", a.stage, "CATEGORICAL", comment=a.reason)
        root.score("refused", 1 if a.refused else 0, "BOOLEAN")
        a.trace_id = root.trace_id
    return a


def _answer(question: str, sc: scope.Scope, store, settings: Settings, llm, obs) -> Answer:
    with obs.child("check-scope", "guardrail", input=question) as sp:
        decision = scope.check(question, sc)
        sp.update(output={"in_scope": decision.in_scope, "tickers": sorted(decision.tickers),
                          "years": sorted(decision.years), "reason": decision.reason})
        if not decision.in_scope:
            sp.warn(decision.reason)
    if not decision.in_scope:
        return Answer(question, REFUSAL, True, "scope", decision.reason)

    with obs.child("retrieve-passages", "retriever", input=question) as sp:
        hits = store.search(question, settings.top_k, decision.tickers, decision.years)
        sources = [{"n": i, "score": round(s, 3), **{k: p[k] for k in
                   ("ticker", "form", "period", "section", "section_title", "url")}}
                   for i, (s, p) in enumerate(hits, 1)]
        # The exact passages the model will see: what a reviewer needs to judge the answer.
        sp.update(output=[{"n": i, "ticker": p["ticker"], "section": p["section"],
                           "score": round(s, 3), "text": p["text"]}
                          for i, (s, p) in enumerate(hits, 1)],
                  metadata={"k": settings.top_k, "tickers": sorted(decision.tickers),
                            "years": sorted(decision.years)})
    best = hits[0][0] if hits else None
    passed = best is not None and best >= settings.min_score
    with obs.child("check-evidence", "guardrail",
                   input={"best_score": round(best, 3) if best is not None else None,
                          "min_score": settings.min_score}) as sp:
        sp.update(output={"pass": passed})
        if not passed:
            sp.warn("best retrieval score below min_score")
    if not passed:
        shown = f"{best:.3f}" if best is not None else "none"
        return Answer(question, REFUSAL, True, "evidence",
                      f"best score {shown} < {settings.min_score}", sources)

    prompt = build_prompt(question, hits)
    preset = settings.preset
    with obs.child("generate-answer", "generation", model=preset.model_id,
                   model_parameters=model_parameters(preset),
                   input=[{"role": "user", "content": prompt}]) as gen:
        out = llm(prompt, settings)
        # A test double may return a plain string; the real generate() returns a Generation.
        text, use, done, reasoning = ((out, None, "stop", None) if isinstance(out, str)
                                      else (out.text, out.usage, out.done_reason, out.reasoning))
        usage = use.as_dict(settings.gpu) if use else None
        gen.update(output={"role": "assistant", "content": text},
                   metadata={"preset": settings.model, "done_reason": done,
                             "reasoning": reasoning,
                             "pricing": usage and usage["pricing"],
                             "llm_seconds": usage and usage["llm_seconds"]},
                   **_usage_fields(usage))
    problem = unfinished(text, done)
    if problem:  # cut at the cap, empty, or reasoning that never ended: not an answer
        return Answer(question, REFUSAL, True, "model", problem, sources, usage=usage)
    if REFUSAL.lower().rstrip(".") in text.lower():
        return Answer(question, REFUSAL, True, "model", "the model found no answer", sources,
                      usage=usage)

    with obs.child("check-figures", "guardrail", input=text) as sp:
        cited = cited_indices(text, len(hits)) or list(range(1, len(hits) + 1))
        check = grounding.check(text, [hits[i - 1][1]["text"] for i in cited])
        sp.update(output={"grounded": check.grounded, "ungrounded": check.ungrounded},
                  metadata={"cited": cited})
        if not check.grounded:
            sp.warn("figures not found in the cited filings")
    if not check.grounded:
        return Answer(question, REFUSAL, True, "numbers",
                      "figures not found in the cited filings", sources, check.ungrounded,
                      usage=usage)
    return Answer(question, text, False, "answered", "ok",
                  [s for s in sources if s["n"] in cited], usage=usage)


def _usage_fields(usage: dict | None) -> dict:
    """Langfuse generation fields: exclusive input/output buckets, and our cost estimate
    (Langfuse cannot price a self-hosted model; ingested cost takes priority)."""
    if not usage:
        return {}
    return {"usage_details": {"input": usage["prompt_tokens"],
                              "output": usage["completion_tokens"]},
            "cost_details": {"total": usage["est_cost_usd"]}}
