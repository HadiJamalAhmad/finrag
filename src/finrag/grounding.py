"""Guardrail 2: every number in an answer must appear in the passages it cites.

In finance the dangerous hallucination is not a wrong sentence, it is a wrong
figure stated with confidence. A deterministic check, after generation: extract
the numbers from the answer, and look for each one in the cited evidence.

Filings state amounts in thousands or millions ("$ 391,035" in a table headed
"in millions") while an answer says "$391.0 billion". So a number counts as
grounded if it matches an evidence number at some power-of-1000 scale, within a
rounding tolerance. Years and citation markers like [2] are not checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUM_RE = re.compile(r"(?<![\w\[])\$?\(?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\)?\s*(%|percent)?")
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)
# Relative: "$391.0 billion" vs 391,035 million is 0.009% off; 0.6% allows rounding.
TOLERANCE = 0.006


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    checked: list[str]
    ungrounded: list[str]


def numbers(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        whole, frac = m.group(1), m.group(2)
        value = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
        if not frac and "," not in whole and 1900 <= value <= 2100:
            continue  # a year
        out.append((raw, value))
    return out


def _matches(value: float, evidence: list[float]) -> bool:
    for e in evidence:
        for s in SCALES:
            target = value * s
            if abs(target - e) <= TOLERANCE * max(abs(e), 1e-9):
                return True
    return False


def value_in(value: float, evidence_texts: list[str]) -> bool:
    """True if `value` appears in the evidence at some power-of-1000 scale."""
    return _matches(value, [v for t in evidence_texts for _, v in numbers(t)])


def check(answer: str, evidence_texts: list[str]) -> GroundingResult:
    # Citation markers are not figures.
    cleaned = re.sub(r"\[\d+\]", " ", answer)
    found = numbers(cleaned)
    pool = [v for t in evidence_texts for _, v in numbers(t)]
    ungrounded = [raw for raw, v in found if v > 10 and not _matches(v, pool)]
    # Small integers (1-10) are mostly counts in prose ("three segments"); skipped.
    return GroundingResult(not ungrounded, [raw for raw, _ in found], ungrounded)
