"""Retrieval and refusal metrics, the same family as RAG Wikipedia's bench.

A retrieved passage is RELEVANT when it comes from the expected filing section
(ticker + form + section, and fiscal year when the case names one). Matching
on metadata instead of chunk ids keeps the suite valid when chunking changes.
A gold value may be a list: the answer's evidence sits in several sections
(FinanceBench cites up to three pages), and any one of them is relevant.
"""

from __future__ import annotations

import math


def is_relevant(payload: dict, gold: dict) -> bool:
    return all(payload.get(k) in v if isinstance(v, list) else payload.get(k) == v
               for k, v in gold.items())


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


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for k successes out of n. With 60 cases one case
    moves a rate by 1.7 points: the interval says how much a rate can be trusted."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value. b = cases only system A got right, c = only B.
    Paired on the same questions, so cases both get right (or wrong) say nothing."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)
