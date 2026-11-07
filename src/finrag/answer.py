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


def build_prompt(question: str, hits: list[tuple[float, dict]]) -> str:
    context = "\n\n".join(f"[{i}] ({p['ticker']} {p['form']} {p['period']}, {p['section_title']})\n"
                          f"{p['text']}" for i, (_, p) in enumerate(hits, 1))
    return PROMPT.format(refusal=REFUSAL, context=context, question=question)


def cited_indices(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})


def generate(prompt: str, settings: Settings) -> str:
    r = httpx.post(
        f"{settings.ollama_url}/api/generate",
        json={"model": settings.llm_model, "prompt": prompt, "stream": False,
              # num_ctx explicitly: recent Ollama defaults to a huge context and tried to
              # allocate a 12 GB KV cache on this machine. 5 passages of ~300 words fit in 4k.
              "options": {"temperature": 0, "num_ctx": 4096}},
        timeout=180.0,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def answer(question: str, sc: scope.Scope, store, settings: Settings, llm=generate) -> Answer:
    decision = scope.check(question, sc)
    if not decision.in_scope:
        return Answer(question, REFUSAL, True, "scope", decision.reason)

    hits = store.search(question, settings.top_k, decision.tickers, decision.years)
    sources = [{"n": i, "score": round(s, 3), **{k: p[k] for k in
               ("ticker", "form", "period", "section", "section_title", "url")}}
               for i, (s, p) in enumerate(hits, 1)]
    if not hits or hits[0][0] < settings.min_score:
        best = f"{hits[0][0]:.3f}" if hits else "none"
        return Answer(question, REFUSAL, True, "evidence",
                      f"best score {best} < {settings.min_score}", sources)

    text = llm(build_prompt(question, hits), settings)
    if REFUSAL.lower().rstrip(".") in text.lower():
        return Answer(question, REFUSAL, True, "model", "the model found no answer", sources)

    cited = cited_indices(text, len(hits)) or list(range(1, len(hits) + 1))
    check = grounding.check(text, [hits[i - 1][1]["text"] for i in cited])
    if not check.grounded:
        return Answer(question, REFUSAL, True, "numbers",
                      "figures not found in the cited filings", sources, check.ungrounded)
    return Answer(question, text, False, "answered", "ok",
                  [s for s in sources if s["n"] in cited])
