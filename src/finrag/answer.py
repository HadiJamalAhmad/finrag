"""The answer path: scope check -> retrieval -> evidence gate -> LLM -> number check.

Each step can refuse, and the response says which one did. A refusal with a
reason is something an analyst can act on; a silent wrong figure is not.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx

from finrag import grounding, scope
from finrag.config import Settings
from finrag.models import Usage, ollama_options, strip_thinking, unfinished
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


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage
    # Ollama's done_reason: "stop", or "length" when the answer hit num_predict.
    done_reason: str | None = "stop"


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

        text, usage, done = generate_hosted(prompt, preset, settings)
        return Generation(text, usage, done)
    # num_ctx and num_predict always explicit, per preset (finrag.models): recent Ollama
    # defaults to a huge context and tried to allocate a 12 GB KV cache on this machine.
    body = {"model": preset.model_id, "prompt": prompt, "stream": False,
            **ollama_options(preset)}
    r = httpx.post(f"{settings.ollama_url}/api/generate", json=body, timeout=300.0)
    r.raise_for_status()
    data = r.json()
    # With think=True the reasoning comes back in data["thinking"], never in the answer.
    return Generation(strip_thinking(data["response"]), Usage.from_ollama(preset.name, data),
                      data.get("done_reason"))


def answer(question: str, sc: scope.Scope, store, settings: Settings, llm=generate,
           tracer=None) -> Answer:
    """The answer path. With a tracer (finrag.tracing), one Langfuse trace per question."""
    trace = (tracer.trace("query", question, tags=[settings.model],
                          metadata={"model_id": settings.preset.model_id,
                                    "min_score": settings.min_score, "top_k": settings.top_k})
             if tracer else NULL)
    try:
        a = _answer(question, sc, store, settings, llm, trace)
    except Exception as exc:
        trace.end({"error": f"{type(exc).__name__}: {exc}"}, tags=["error"])
        raise
    trace.end({"answer": a.answer, "refused": a.refused, "stage": a.stage, "reason": a.reason},
              tags=[f"stage:{a.stage}"], metadata={"usage": a.usage})
    return a


def _answer(question: str, sc: scope.Scope, store, settings: Settings, llm, trace) -> Answer:
    with trace.span("scope", input=question) as sp:
        decision = scope.check(question, sc)
        sp.output = {"in_scope": decision.in_scope, "tickers": sorted(decision.tickers),
                     "years": sorted(decision.years), "reason": decision.reason}
        if not decision.in_scope:
            sp.warning = decision.reason
    if not decision.in_scope:
        return Answer(question, REFUSAL, True, "scope", decision.reason)

    with trace.span("retrieve", input={"k": settings.top_k}) as sp:
        hits = store.search(question, settings.top_k, decision.tickers, decision.years)
        sources = [{"n": i, "score": round(s, 3), **{k: p[k] for k in
                   ("ticker", "form", "period", "section", "section_title", "url")}}
                   for i, (s, p) in enumerate(hits, 1)]
        sp.output = [{k: src[k] for k in ("n", "ticker", "section", "score")} for src in sources]
    with trace.span("evidence_gate") as sp:
        best = hits[0][0] if hits else None
        passed = best is not None and best >= settings.min_score
        sp.output = {"best": round(best, 3) if best is not None else None,
                     "min_score": settings.min_score, "pass": passed}
        if not passed:
            sp.warning = "best score below min_score"
    if not passed:
        shown = f"{best:.3f}" if best is not None else "none"
        return Answer(question, REFUSAL, True, "evidence",
                      f"best score {shown} < {settings.min_score}", sources)

    prompt, t0 = build_prompt(question, hits), time.time()
    out = llm(prompt, settings)
    # A test double may return a plain string; the real generate() returns a Generation.
    text, use, done = ((out, None, "stop") if isinstance(out, str)
                       else (out.text, out.usage, out.done_reason))
    usage = use.as_dict(settings.gpu) if use else None
    trace.generation("llm", t0, settings.preset.model_id, prompt, text, usage,
                     {"preset": settings.model, "done_reason": done})
    problem = unfinished(text, done)
    if problem:  # cut at the cap, empty, or reasoning that never ended: not an answer
        return Answer(question, REFUSAL, True, "model", problem, sources, usage=usage)
    if REFUSAL.lower().rstrip(".") in text.lower():
        return Answer(question, REFUSAL, True, "model", "the model found no answer", sources,
                      usage=usage)

    with trace.span("numbers", input=text) as sp:
        cited = cited_indices(text, len(hits)) or list(range(1, len(hits) + 1))
        check = grounding.check(text, [hits[i - 1][1]["text"] for i in cited])
        sp.output = {"grounded": check.grounded, "ungrounded": check.ungrounded,
                     "cited": cited}
        if not check.grounded:
            sp.warning = "figures not found in the cited filings"
    if not check.grounded:
        return Answer(question, REFUSAL, True, "numbers",
                      "figures not found in the cited filings", sources, check.ungrounded,
                      usage=usage)
    return Answer(question, text, False, "answered", "ok",
                  [s for s in sources if s["n"] in cited], usage=usage)
