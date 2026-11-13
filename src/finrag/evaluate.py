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
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from finrag import metrics
from finrag.answer import answer

GATES = {"recall@5": (">=", 0.80), "mrr": (">=", 0.60), "false_accept_rate": ("<=", 0.10)}


def load_suite(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run(suite: list[dict], sc, store, settings, llm=None, with_llm: bool = True,
        tracer=None) -> dict:
    rec, rr, refusal_rows, cases, seconds, usages = [], [], [], [], [], []
    for case in suite:
        if case["answerable"]:
            hits = store.search(case["question"], settings.top_k)
            retrieved = [p for _, p in hits]
            rec.append(metrics.recall_at_k(retrieved, case["gold"], 5))
            rr.append(metrics.reciprocal_rank(retrieved, case["gold"]))
        kwargs = {"llm": llm} if llm else {}
        if tracer:  # each case: a trace in the "eval" environment, never mixed with live use
            kwargs["tracer"] = tracer
            kwargs["trace_context"] = {"tags": ["eval"], "environment": "eval",
                                       "metadata": {"case": case["id"],
                                                    "answerable": case["answerable"],
                                                    # ground truth, for evaluators
                                                    "expected": case.get("gold")
                                                    or {"refuse": case.get("why")}}}
        extra = {}
        if with_llm:
            t0 = time.perf_counter()
            a = answer(case["question"], sc, store, settings, **kwargs)
            dt = time.perf_counter() - t0
            refused, stage = a.refused, a.stage
            if tracer and a.trace_id:
                # Correct = answered when answerable, refused when not: the eval's own label.
                tracer.score_trace(a.trace_id, "correct",
                                   1 if refused != case["answerable"] else 0, "BOOLEAN",
                                   comment=f"case {case['id']}: {stage}")
            seconds.append(dt)
            extra = {"seconds": round(dt, 2)}
            if a.usage:
                usages.append(a.usage)
                extra |= {k: a.usage[k] for k in
                          ("prompt_tokens", "completion_tokens", "est_cost_usd")}
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
        row = {"id": case["id"], "answerable": case["answerable"], "refused": refused,
               "stage": stage, **extra}
        if case["answerable"]:
            row["hit@5"] = rec[-1] == 1.0
        if "type" in case:
            row["type"] = case["type"]
        cases.append(row)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_cases": len(suite),
        "mode": "full" if with_llm else "retrieval-only",
        "recall@5": round(mean(rec), 3) if rec else None,
        "mrr": round(mean(rr), 3) if rr else None,
        **metrics.refusal_scores(refusal_rows),
    }
    report["ci95"] = _intervals(rec, refusal_rows)
    if any("type" in c for c in cases):
        report["by_type"] = _by_type(cases)
    if with_llm:
        report |= _cost_and_latency(settings, seconds, usages)
    report["cases"] = cases
    report["gates"] = {name: _gate(report.get(name), op, t) for name, (op, t) in GATES.items()}
    return report


def _intervals(rec: list[float], rows: list[tuple[bool, bool]]) -> dict:
    """95% Wilson intervals: a rate on a few dozen cases is a range, not a point."""
    unans = [r for a, r in rows if not a]
    ans = [r for a, r in rows if a]
    return {"recall@5": metrics.wilson(int(sum(rec)), len(rec)),
            "false_accept_rate": metrics.wilson(sum(not r for r in unans), len(unans)),
            "over_refusal_rate": metrics.wilson(sum(ans), len(ans))}


def _by_type(cases: list[dict]) -> dict:
    """Per question type: where the system is weak, not just how weak on average."""
    out: dict[str, dict] = {}
    for t in sorted({c["type"] for c in cases if "type" in c}):
        group = [c for c in cases if c.get("type") == t]
        ans = [c for c in group if c["answerable"]]
        out[t] = {"n": len(group),
                  "recall@5": round(mean(c["hit@5"] for c in ans), 3) if ans else None,
                  "refused": round(mean(c["refused"] for c in group), 3)}
    return out


def compare(a: dict, b: dict) -> dict:
    """Two reports on the same suite, paired case by case (McNemar).

    Correct = answered when answerable, refused when not. Only the cases where
    the two systems disagree carry information about which one is better.
    """
    ra = {c["id"]: c["refused"] != c["answerable"] for c in a["cases"]}
    rb = {c["id"]: c["refused"] != c["answerable"] for c in b["cases"]}
    shared = sorted(ra.keys() & rb.keys())
    only_a = [i for i in shared if ra[i] and not rb[i]]
    only_b = [i for i in shared if rb[i] and not ra[i]]
    return {"n_paired": len(shared), "a_correct": sum(ra[i] for i in shared),
            "b_correct": sum(rb[i] for i in shared), "only_a": only_a, "only_b": only_b,
            "mcnemar_p": metrics.mcnemar(len(only_a), len(only_b))}


def _cost_and_latency(settings, seconds: list[float], usages: list[dict]) -> dict:
    """Which model answered (digest = provenance), how fast, and at what GPU cost."""
    p = settings.preset
    srt = sorted(seconds)

    def pct(q: float) -> float | None:
        return round(srt[min(len(srt) - 1, int(q * len(srt)))], 2) if srt else None

    total = sum(u["est_cost_usd"] for u in usages)
    return {
        "model": {"preset": p.name, "label": p.label, "model_id": p.model_id,
                  "provider": p.provider, "digest": p.digest,
                  "gpu": settings.gpu if p.provider == "ollama" else None},
        # End-to-end per question (scope + retrieval + model + checks), refusals included.
        "latency_s": {"mean": round(mean(seconds), 2) if seconds else None,
                      "p50": pct(0.5), "p95": pct(0.95)},
        "usage": {"llm_answers": len(usages),
                  "prompt_tokens": sum(u["prompt_tokens"] for u in usages),
                  "completion_tokens": sum(u["completion_tokens"] for u in usages),
                  "llm_seconds": round(sum(u["llm_seconds"] for u in usages), 1),
                  "est_cost_usd_total": round(total, 4),
                  "est_cost_usd_per_llm_answer": round(total / len(usages), 5) if usages
                  else None},
    }


def _gate(value, op: str, threshold: float) -> dict:
    if value is None:
        return {"value": None, "threshold": f"{op} {threshold}", "pass": False}
    ok = value >= threshold if op == ">=" else value <= threshold
    return {"value": value, "threshold": f"{op} {threshold}", "pass": ok}
