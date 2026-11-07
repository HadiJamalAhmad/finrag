"""Guardrail 1: refuse BEFORE retrieval when the question is outside the corpus.

Retrieval always returns something. Ask about Tesla in a corpus without Tesla
and the nearest passages will be about some other carmaker or supplier, and a
small model will happily answer from them. So two deterministic checks run first:

- COMPANY: the question names an SEC-listed company that is not in the corpus.
  Names come from SEC's own company_tickers.json, so "Tesla" is recognised as
  a company even though no Tesla filing was downloaded.
- YEAR: the question names a fiscal year no filing in the corpus covers.
  A 10-K reports its year and the two before it (comparative statements).

Known limitation, on purpose for now: company detection only looks at
capitalised words and all-caps tickers, so "the target price" is not read as
Target Corp. A question starting with a company name ("Target reported...")
is still caught, because the first word is checked against the alias list too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finrag.edgar import Filing

GENERIC = {
    "the", "first", "american", "united", "general", "international", "global", "national",
    "bank", "capital", "financial", "group", "holdings", "trust", "new", "north", "south",
    "east", "west", "western", "eastern", "pacific", "atlantic", "federal", "public", "royal",
    "standard", "energy", "health", "data", "digital", "life", "home", "us", "usa", "what",
    "how", "which", "who", "when", "why", "does", "did", "is", "was", "are", "item", "form",
    "risk", "net", "total", "free", "cash", "fiscal", "annual", "report", "sec", "inc", "corp",
    "company", "co", "ltd", "plc", "management", "according", "describe", "compare", "list",
    "summarize", "explain", "give", "tell", "show", "can", "could", "should", "will", "would",
}
TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
WORD_RE = re.compile(r"\b[A-Z][a-zA-Z&.\-]{2,}\b")
YEAR_RE = re.compile(r"\b(?:fiscal\s+(?:year\s+)?|FY\s?)?((?:19|20)\d{2})\b", re.IGNORECASE)


def _norm(word: str) -> str:
    """'Apple's' / 'Apple’s' / 'Apple.' -> 'apple'."""
    word = word.lower().rstrip(".")
    for suffix in ("'s", "’s"):
        word = word.removesuffix(suffix)
    return word


def alias(name: str) -> str | None:
    """'Apple Inc.' -> 'apple', 'JPMORGAN CHASE & CO' -> 'jpmorgan'. None if too generic."""
    first = re.split(r"[\s,]+", name.strip())[0].lower().strip(".&-")
    if len(first) < 4 or first in GENERIC:
        return None
    return first


@dataclass
class Scope:
    corpus_tickers: set[str]
    corpus_aliases: dict[str, str]  # alias -> corpus ticker
    years: dict[str, set[int]]  # ticker -> fiscal years covered
    sec_aliases: dict[str, str] = field(default_factory=dict)  # alias -> ticker
    sec_tickers: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, filings: list[Filing], sec_tickers_json: dict | None = None) -> Scope:
        years: dict[str, set[int]] = {}
        aliases: dict[str, str] = {}
        for f in filings:
            span = {f.fiscal_year - 2, f.fiscal_year - 1, f.fiscal_year} if f.form == "10-K" \
                else {f.fiscal_year}
            years.setdefault(f.ticker, set()).update(span)
            if a := alias(f.company):
                aliases[a] = f.ticker
        sec_aliases: dict[str, str] = {}
        sec_tickers: set[str] = set()
        for row in (sec_tickers_json or {}).values():
            sec_tickers.add(row["ticker"].upper())
            if a := alias(row["title"]):
                sec_aliases.setdefault(a, row["ticker"].upper())
        return cls(set(years), aliases, years, sec_aliases, sec_tickers)


@dataclass(frozen=True)
class ScopeDecision:
    in_scope: bool
    reason: str
    tickers: frozenset[str]  # corpus companies the question names (empty = not specific)
    years: frozenset[int]


def check(question: str, scope: Scope) -> ScopeDecision:
    named: set[str] = set()
    outside: set[str] = set()

    for tok in TICKER_RE.findall(question):
        if tok in scope.corpus_tickers:
            named.add(tok)
        elif tok in scope.sec_tickers and tok.lower() not in GENERIC:
            outside.add(tok)

    for w in {_norm(w) for w in WORD_RE.findall(question)}:
        if w in scope.corpus_aliases:
            named.add(scope.corpus_aliases[w])
        elif w in scope.sec_aliases and w not in GENERIC:
            outside.add(scope.sec_aliases[w])

    outside -= scope.corpus_tickers
    if outside and not named:
        return ScopeDecision(False, f"company not in corpus: {', '.join(sorted(outside))}",
                             frozenset(), frozenset())

    years = {int(y) for y in YEAR_RE.findall(question)}
    if years:
        pool = set().union(*(scope.years.get(t, set()) for t in (named or scope.corpus_tickers)))
        missing = years - pool
        if missing:
            return ScopeDecision(False, f"fiscal year not covered: {sorted(missing)}",
                                 frozenset(named), frozenset(years))
    return ScopeDecision(True, "ok", frozenset(named), frozenset(years))
