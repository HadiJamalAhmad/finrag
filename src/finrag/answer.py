"""The answer path: scope check -> retrieval -> evidence gate -> LLM -> number check.

Each step can refuse, and the response says which one did. A refusal with a
reason is something an analyst can act on; a silent wrong figure is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from finrag import grounding, pii, scope
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
    # Personal data masked in the question, by kind ({"NIR": 1}): counts, never values.
    pii: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage
    # Ollama's done_reason: "stop", or "length" when the answer hit num_predict.
    done_reason: str | None = "stop"
    # The model's reasoning, when it returns one apart from the answer (Qwen3's
    # `thinking`, Claude's thinking blocks): traced, never shown as the answer.
    reasoning: str | None = None


# Documents mode (FINRAG_DOMAIN=documents): any document corpus, here French public
# administration pages. The refusal is in French because the corpus and its users are;
# the model is told to answer in the question's language.
REFUSAL_DOCS = "Je ne sais pas d'après les documents fournis."

PROMPT_DOCS = """You are an assistant for public administration documents. Answer ONLY from
the numbered excerpts below, in the language of the question. Cite every sentence with
its excerpt number, like [1] or [2][3]. Copy amounts, delays and dates exactly as written.
If the excerpts do not contain the answer, reply exactly: {refusal}

Excerpts:
{context}

Question: {question}
Answer:"""


def refusal_for(settings: Settings) -> str:
    return REFUSAL_DOCS if settings.domain == "documents" else REFUSAL


def build_prompt(question: str, hits: list[tuple[float, dict]], domain: str = "sec") -> str:
    if domain == "documents":
        context = "\n\n".join(f"[{i}] ({p['section_title']})\n{p['text']}"
                              for i, (_, p) in enumerate(hits, 1))
        return PROMPT_DOCS.format(refusal=REFUSAL_DOCS, context=context, question=question)
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
           tracer=None, trace_context: dict | None = None,
           groups: list[str] | None = None) -> Answer:
    """The answer path. With a tracer (finrag.tracing): one Langfuse trace per question.

    trace_context: extra "tags", "metadata" and "environment" for the trace (eval runs).
    groups: the user's groups (from SSO in production). None = no identity: public
    documents only (the store's default). Restricted documents never reach the model.
    """
    # Personal data is masked first: the model, the trace and the passage search only
    # ever see placeholders. The user gets the values back in the answer.
    masked = pii.protector(settings.pii)(question)
    ctx = trace_context or {}
    if tracer is None or not tracer.enabled:
        a = _answer(masked.text, sc, store, settings, llm, NULL, groups)
        return unmask(a, question, masked)
    with tracer.root("answer-question", "chain", input=masked.text,
                     tags=["pipeline", settings.model, *ctx.get("tags", [])],
                     metadata={"model_id": settings.preset.model_id,
                               "min_score": settings.min_score, "top_k": settings.top_k,
                               "pii_masked": masked.kinds(), **ctx.get("metadata", {})},
                     environment=ctx.get("environment")) as root:
        a = _answer(masked.text, sc, store, settings, llm, root, groups)
        # What a reviewer needs at a glance; sources and usage go to metadata.
        # The answer itself, at a glance; the deciding stage is in metadata and scores.
        root.update(output=a.answer,
                    metadata={"stage": a.stage, "reason": a.reason, "sources": a.sources,
                              "usage": a.usage, "ungrounded_numbers": a.ungrounded_numbers})
        # Known only at the end: scores, not tags (tags are fixed at creation).
        root.score("outcome", a.stage, "CATEGORICAL", comment=a.reason)
        root.score("refused", 1 if a.refused else 0, "BOOLEAN")
        a.trace_id = root.trace_id
    return unmask(a, question, masked)  # after the trace: it keeps the placeholders


def unmask(a: Answer, question: str, masked: pii.Masked) -> Answer:
    """The user's own values back in the answer, once every check has run on the masked text
    (placeholders hold no digits, so the figure check never mistakes one for a number)."""
    a.question, a.answer, a.pii = question, masked.restore(a.answer), masked.kinds()
    return a


def _answer(question: str, sc: scope.Scope, store, settings: Settings, llm, obs,
            groups: list[str] | None = None) -> Answer:
    refusal = refusal_for(settings)
    with obs.child("check-scope", "guardrail", input=question) as sp:
        # Companies and fiscal years are SEC notions. A documents corpus has neither, so
        # nothing is refused here: an off-topic question meets the evidence gate instead.
        decision = (scope.ScopeDecision(True, "documents mode: no company scope",
                                        frozenset(), frozenset())
                    if settings.domain == "documents" else scope.check(question, sc))
        sp.update(output={"in_scope": decision.in_scope, "tickers": sorted(decision.tickers),
                          "years": sorted(decision.years), "reason": decision.reason})
        if not decision.in_scope:
            sp.warn(decision.reason)
    if not decision.in_scope:
        return Answer(question, refusal, True, "scope", decision.reason)

    with obs.child("retrieve-passages", "retriever", input=question) as sp:
        extra = {"groups": frozenset(groups)} if groups is not None else {}
        hits = store.search(question, settings.top_k, decision.tickers, decision.years, **extra)
        sources = [{"n": i, "score": round(s, 3), **{k: p[k] for k in
                   ("ticker", "form", "period", "section", "section_title", "url")}}
                   for i, (s, p) in enumerate(hits, 1)]
        # The exact passages the model will see: what a reviewer needs to judge the answer.
        sp.update(output=[{"n": i, "ticker": p["ticker"], "section": p["section"],
                           "score": round(s, 3), "text": p["text"]}
                          for i, (s, p) in enumerate(hits, 1)],
                  metadata={"k": settings.top_k, "tickers": sorted(decision.tickers),
                            "years": sorted(decision.years), "groups": sorted(groups or [])})
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
        return Answer(question, refusal, True, "evidence",
                      f"best score {shown} < {settings.min_score}", sources)

    prompt = build_prompt(question, hits, settings.domain)
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
        return Answer(question, refusal, True, "model", problem, sources, usage=usage)
    if refusal.lower().rstrip(".") in text.lower():
        return Answer(question, refusal, True, "model", "the model found no answer", sources,
                      usage=usage)

    with obs.child("check-figures", "guardrail", input=text) as sp:
        cited = cited_indices(text, len(hits)) or list(range(1, len(hits) + 1))
        check = grounding.check(text, [hits[i - 1][1]["text"] for i in cited])
        sp.update(output={"grounded": check.grounded, "ungrounded": check.ungrounded},
                  metadata={"cited": cited})
        if not check.grounded:
            sp.warn("figures not found in the cited filings")
    if not check.grounded:
        return Answer(question, refusal, True, "numbers",
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
