"""Run an eval suite: retrieval quality on answerable cases, refusals on all of them.

Suite format (one JSON object per line):
  {"id": "...", "question": "...", "answerable": true,
   "gold": {"ticker": "AAPL", "form": "10-K", "section": "1A"}}
  {"id": "...", "question": "...", "answerable": false, "why": "company not in corpus"}

Thresholds are written down BEFORE the first run and never relaxed to pass:
a red number is a finding, and it is reported as one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from finrag import metrics
from finrag.answer import answer

GATES = {"recall@5": (">=", 0.80), "mrr": (">=", 0.60), "false_accept_rate": ("<=", 0.10)}


def load_suite(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run(suite: list[dict], sc, store, settings, llm=None, with_llm: bool = True) -> dict:
    rec, rr, refusal_rows, cases = [], [], [], []
    for case in suite:
        if case["answerable"]:
            hits = store.search(case["question"], settings.top_k)
            retrieved = [p for _, p in hits]
            rec.append(metrics.recall_at_k(retrieved, case["gold"], 5))
            rr.append(metrics.reciprocal_rank(retrieved, case["gold"]))
        kwargs = {"llm": llm} if llm else {}
        if with_llm:
            a = answer(case["question"], sc, store, settings, **kwargs)
            refused, stage = a.refused, a.stage
        else:
            # Retrieval-only mode: refusals decided by scope + evidence gate only.
            from finrag import scope

            d = scope.check(case["question"], sc)
            if not d.in_scope:
                refused, stage = True, "scope"
            else:
                hits = store.search(case["question"], settings.top_k, d.tickers, d.years)
                refused = not hits or hits[0][0] < settings.min_score
                stage = "evidence" if refused else "answered"
        refusal_rows.append((case["answerable"], refused))
        cases.append({"id": case["id"], "answerable": case["answerable"], "refused": refused,
                      "stage": stage})

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_cases": len(suite),
        "mode": "full" if with_llm else "retrieval-only",
        "recall@5": round(mean(rec), 3) if rec else None,
        "mrr": round(mean(rr), 3) if rr else None,
        **metrics.refusal_scores(refusal_rows),
        "cases": cases,
    }
    report["gates"] = {name: _gate(report.get(name), op, t) for name, (op, t) in GATES.items()}
    return report


def _gate(value, op: str, threshold: float) -> dict:
    if value is None:
        return {"value": None, "threshold": f"{op} {threshold}", "pass": False}
    ok = value >= threshold if op == ">=" else value <= threshold
    return {"value": value, "threshold": f"{op} {threshold}", "pass": ok}
