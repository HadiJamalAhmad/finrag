"""Retrieval and refusal metrics, the same family as RAG Wikipedia's bench.

A retrieved passage is RELEVANT when it comes from the expected filing section
(ticker + form + section, and fiscal year when the case names one). Matching
on metadata instead of chunk ids keeps the suite valid when chunking changes.
"""

from __future__ import annotations


def is_relevant(payload: dict, gold: dict) -> bool:
    return all(payload.get(k) == v for k, v in gold.items())


def recall_at_k(retrieved: list[dict], gold: dict, k: int) -> float:
    return 1.0 if any(is_relevant(p, gold) for p in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[dict], gold: dict) -> float:
    for rank, p in enumerate(retrieved, 1):
        if is_relevant(p, gold):
            return 1.0 / rank
    return 0.0


def refusal_scores(rows: list[tuple[bool, bool]]) -> dict:
    """rows: (answerable, refused). Returns the four rates that matter.

    false_accept_rate: unanswerable questions the system answered anyway. The
    one to watch: it is the confident-wrong-answer rate.
    """
    unans = [r for a, r in rows if not a]
    ans = [r for a, r in rows if a]
    return {
        "false_accept_rate": round(sum(not r for r in unans) / len(unans), 3) if unans else None,
        "correct_refusal_rate": round(sum(unans) / len(unans), 3) if unans else None,
        "over_refusal_rate": round(sum(ans) / len(ans), 3) if ans else None,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
    }
