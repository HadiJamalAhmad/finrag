"""Personal data never reaches the model or the traces: find it, pseudonymise it, restore it.

A question such as "Je suis Camille Martin, NIR 2 85 05 78 006 084 ..., comment renouveler
mon passeport ?" is rewritten "Je suis [NAME_A], NIR [NIR_A], comment ..." before anything
leaves this process: the model (hosted or ours), Langfuse, the eval reports. The mapping
stays in memory for one question, and the answer shown to the user gets the values back.

Detectors, most specific first; a span already taken is never taken again:
- NIR (French social security number): 13 digits + a key, checked (97 - n mod 97; Corsica
  2A/2B handled). A 15-digit reference number with a wrong key is left alone.
- IBAN: groups of four, checked by ISO 13616 mod 97.
- CARD: 15 digits starting 34/37 (Amex) or 16 digits starting 4, 51-55 or 2221-2720, and
  Luhn-valid. Not "any Luhn number": a SIRET (14 digits, also Luhn) is a company's, not a
  person's, and must stay readable.
- EMAIL, and PHONE (French numbers, 0X or +33/0033, ten digits). Short service numbers
  (3939) are not personal and are not matched.
- NAME: people's names from spaCy's French model (entity PER), only with FINRAG_PII=full
  (optional extra `pii`). Regex cannot find names; a statistical model can, imperfectly:
  eval/suites/pii-fr.jsonl measures how well.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

Ner = Callable[[str], list[tuple[int, int]]]  # text -> (start, end) of each person's name

KINDS = ("NIR", "IBAN", "CARD", "EMAIL", "PHONE", "NAME")
MODES = ("off", "regex", "full")

_NIR = re.compile(r"(?<![\w])([12])[ .]?(\d{2})[ .]?(0[1-9]|1[0-2]|[2-9]\d)[ .]?"
                  r"(\d{2}|2A|2B)[ .]?(\d{3})[ .]?(\d{3})[ .]?(\d{2})(?![\w])")
_IBAN = re.compile(r"(?<![\w])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?(?![\w])")
_CARD = re.compile(r"(?<![\w])\d(?:[ -]?\d){14,15}(?![\w])")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}(?![\w])")
_PHONE = re.compile(r"(?<![\w+])(?:(?:\+|00)33[ .-]?(?:\(0\)[ .-]?)?|0)[1-9]"
                    r"(?:[ .-]?\d{2}){4}(?![\w])")


def nir_ok(m: re.Match) -> bool:
    sex, yy, mm, dept, commune, order, key = m.groups()
    body = sex + yy + mm + dept + commune + order
    if dept in ("2A", "2B"):  # Corsica: 2A -> 19 and minus 1,000,000; 2B -> 18, minus 2,000,000
        n = int(body.replace(dept, "19" if dept == "2A" else "18")) - (
            1_000_000 if dept == "2A" else 2_000_000)
    else:
        n = int(body)
    return 97 - n % 97 == int(key)


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def card_ok(raw: str) -> bool:
    d = re.sub(r"\D", "", raw)
    if len(d) == 15:
        brand = d[:2] in ("34", "37")
    else:
        brand = d[0] == "4" or 51 <= int(d[:2]) <= 55 or 2221 <= int(d[:4]) <= 2720
    return brand and luhn_ok(d)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    value: str


def find(text: str, ner: Ner | None = None) -> list[Span]:
    """Every personal datum in text, non-overlapping, in reading order."""
    found: list[Span] = []

    def take(start: int, end: int, kind: str) -> None:
        if not any(s.start < end and start < s.end for s in found):
            found.append(Span(start, end, kind, text[start:end]))

    for m in _NIR.finditer(text):
        if nir_ok(m):
            take(m.start(), m.end(), "NIR")
    for m in _IBAN.finditer(text):
        if iban_ok(m.group()):
            take(m.start(), m.end(), "IBAN")
    for m in _CARD.finditer(text):
        if card_ok(m.group()):
            take(m.start(), m.end(), "CARD")
    for m in _EMAIL.finditer(text):
        take(m.start(), m.end(), "EMAIL")
    for m in _PHONE.finditer(text):
        take(m.start(), m.end(), "PHONE")
    if ner is not None:
        for start, end in ner(text):
            take(start, end, "NAME")
    return sorted(found, key=lambda s: s.start)


def _label(i: int) -> str:
    """A, B, ... Z, AA, AB: no digits, so citations [2] and the figure check never see one."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


@dataclass
class Masked:
    text: str
    # placeholder -> original value. Never printed, never traced.
    mapping: dict[str, str] = field(default_factory=dict, repr=False)

    def kinds(self) -> dict[str, int]:
        """How many of each kind were masked: safe to log (no values)."""
        return dict(Counter(p[1:].split("_")[0] for p in self.mapping))

    def restore(self, s: str) -> str:
        for placeholder, value in self.mapping.items():
            s = s.replace(placeholder, value)
        return s


def mask(text: str, ner: Ner | None = None) -> Masked:
    spans = find(text, ner)
    by_value: dict[tuple[str, str], str] = {}
    counts: Counter = Counter()
    out, last = [], 0
    for s in spans:
        key = (s.kind, s.value)
        if key not in by_value:  # the same value twice -> the same placeholder
            by_value[key] = f"[{s.kind}_{_label(counts[s.kind])}]"
            counts[s.kind] += 1
        out += [text[last:s.start], by_value[key]]
        last = s.end
    out.append(text[last:])
    return Masked("".join(out), {p: v for (_, v), p in by_value.items()})


# Written BEFORE the first run on eval/suites/pii-fr.jsonl, never relaxed to pass.
# Checked numbers, emails and phones: the suite is synthetic and these are deterministic,
# so anything below 1.0 is a bug. Names come from a statistical model: 0.80 each way.
PII_GATES = {"leaks": ("<=", 0), "false_masks": ("<=", 0),
             "name_recall": (">=", 0.80), "name_precision": (">=", 0.80)}


def score(suite: list[dict], protect: Callable[[str], Masked]) -> dict:
    """Per kind: found / missed / wrong, by exact value. A leak: a labelled value still in
    the masked text. A false mask: a non-name value masked in a case labelled clean."""
    tp, fp, fn = Counter(), Counter(), Counter()
    leaks, false_masks, rows = 0, 0, []
    for case in suite:
        m = protect(case["text"])
        want = {(p["kind"], p["value"]) for p in case["pii"]}
        got = {(p[1:].split("_")[0], v) for p, v in m.mapping.items()}
        for k, _ in want & got:
            tp[k] += 1
        for k, _ in got - want:
            fp[k] += 1
        for k, _ in want - got:
            fn[k] += 1
        leaked = [p["value"] for p in case["pii"] if p["value"] in m.text]
        leaks += len(leaked)
        if not case["pii"] and any(k != "NAME" for k, _ in got):
            false_masks += 1
        rows.append({"id": case["id"], "masked": m.text, "missed": sorted(want - got),
                     "wrong": sorted(got - want)})

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    per_kind = {k: {"found": tp[k], "missed": fn[k], "wrong": fp[k],
                    "recall": ratio(tp[k], tp[k] + fn[k]),
                    "precision": ratio(tp[k], tp[k] + fp[k])} for k in KINDS
                if tp[k] + fn[k] + fp[k]}
    report = {"n_cases": len(suite), "leaks": leaks, "false_masks": false_masks,
              "name_recall": per_kind.get("NAME", {}).get("recall"),
              "name_precision": per_kind.get("NAME", {}).get("precision"),
              "per_kind": per_kind, "cases": rows}
    report["gates"] = {
        k: {"value": report[k], "threshold": f"{op} {t}",
            # None: not measured (names without the NER), neither a pass nor a fail.
            "pass": None if report[k] is None else
            (report[k] >= t if op == ">=" else report[k] <= t)}
        for k, (op, t) in PII_GATES.items()}
    return report


@lru_cache(maxsize=2)
def spacy_ner(model: str = "fr_core_news_sm") -> Ner:
    """Person names from spaCy (optional extra `pii`). Loaded once, lazily."""
    import spacy

    nlp = spacy.load(model, exclude=["parser", "lemmatizer", "morphologizer",
                                     "attribute_ruler", "senter"])

    def ner(text: str) -> list[tuple[int, int]]:
        return [(e.start_char, e.end_char) for e in nlp(text).ents if e.label_ == "PER"]

    return ner


def protector(mode: str) -> Callable[[str], Masked]:
    """FINRAG_PII: off (nothing masked), regex (checked numbers, emails, phones), full (+ names)."""
    if mode not in MODES:
        raise ValueError(f"FINRAG_PII must be one of {MODES}, not {mode!r}")
    if mode == "off":
        return lambda text: Masked(text)
    ner = spacy_ner() if mode == "full" else None
    return lambda text: mask(text, ner)
