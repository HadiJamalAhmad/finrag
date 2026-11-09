"""The agent's tools, and the guard every tool call goes through.

An agent that calls tools is only as safe as the layer between the model and the
tools. So the model never calls a function directly: it proposes a call, and
`ToolGuard.run` decides.

- **Allowlist**: only the tools registered here exist. An unknown name is refused.
- **Validation**: arguments are checked against a pydantic schema before anything
  runs; a ticker outside the corpus is refused, not searched.
- **Budget**: at most `max_calls` calls per question, so a looping model stops.
- **Audit log**: every proposal, accepted or refused, is recorded with its
  arguments, outcome and latency.
- **Evidence**: every successful result is kept with a source id ([S1], [S2]...),
  and is the only material the final numeric check accepts.

The tools are read-only: they search filings, read reported figures and compute.
None of them writes, sends or deletes anything.
"""

from __future__ import annotations

import ast
import json
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from finrag import grounding

# Reported figures come from SEC XBRL data. Each metric maps to the us-gaap concepts
# companies use for it, in order of preference (revenue has several names).
METRICS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet"),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "research_and_development": ("ResearchAndDevelopmentExpense",),
    "total_assets": ("Assets",),
    "diluted_eps": ("EarningsPerShareDiluted",),
}
Metric = Literal["revenue", "net_income", "operating_income", "research_and_development",
                 "total_assets", "diluted_eps"]


class ToolError(Exception):
    """A refusal or failure the model is told about, so it can try something else."""


# ---- argument schemas (also the JSON schemas sent to the model) -----------------

class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300,
                       description="What to look for in the filings' text.")
    ticker: str | None = Field(None, description="Restrict to one company, e.g. AAPL.")
    fiscal_year: int | None = Field(None, ge=1994, le=2100,
                                    description="Restrict to filings covering this year.")


class FactArgs(BaseModel):
    ticker: str = Field(description="Company ticker, e.g. MSFT.")
    metric: Metric = Field(description="Which reported figure.")
    fiscal_year: int = Field(ge=2009, le=2100, description="Fiscal year, e.g. 2025.")


class CalcArgs(BaseModel):
    expression: str = Field(max_length=200, description=(
        "Arithmetic on figures already returned by other tools, e.g. "
        "'(416161 - 391035) / 391035 * 100'. Only + - * / ( ) and numbers."))


@dataclass
class Evidence:
    id: str
    tool: str
    text: str
    source: dict


@dataclass
class ToolGuard:
    """Runs proposed tool calls under the rules above. One instance per question."""

    tools: dict[str, tuple[type[BaseModel], Callable[[BaseModel, ToolGuard], str], str]]
    allowed_tickers: frozenset[str]
    max_calls: int = 6
    calls: int = 0
    log: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    def schemas(self) -> list[dict]:
        """Tool definitions in the function-calling format Ollama and OpenAI accept."""
        return [{"type": "function", "function": {
                    "name": name, "description": desc,
                    "parameters": schema.model_json_schema()}}
                for name, (schema, _, desc) in self.tools.items()]

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls

    def add_evidence(self, tool: str, text: str, source: dict) -> str:
        ev = Evidence(f"S{len(self.evidence) + 1}", tool, text, source)
        self.evidence.append(ev)
        return ev.id

    def run(self, name: str, raw_args: dict | str | None) -> str:
        """Execute one proposed call. Always returns text for the model, never raises."""
        start = time.perf_counter()
        entry = {"tool": name, "args": raw_args, "ok": False}
        try:
            if self.exhausted:
                raise ToolError(f"tool budget of {self.max_calls} calls used up")
            self.calls += 1
            if name not in self.tools:
                raise ToolError(f"unknown tool {name!r}; allowed: {sorted(self.tools)}")
            schema, fn, _ = self.tools[name]
            if isinstance(raw_args, str):  # some models send arguments as a JSON string
                raw_args = json.loads(raw_args or "{}")
            args = schema.model_validate(raw_args or {})
            ticker = getattr(args, "ticker", None)
            if ticker is not None:
                ticker = ticker.upper()
                if ticker not in self.allowed_tickers:
                    raise ToolError(f"{ticker} is not in the corpus "
                                    f"({', '.join(sorted(self.allowed_tickers))})")
                args = args.model_copy(update={"ticker": ticker})
            result = fn(args, self)
            entry["ok"] = True
            return result
        except ValidationError as e:
            entry["error"] = f"invalid arguments: {e.errors()[0]['msg']}"
        except (ToolError, json.JSONDecodeError) as e:
            entry["error"] = str(e)
        finally:
            entry["ms"] = round((time.perf_counter() - start) * 1000, 1)
            self.log.append(entry)
        return f"ERROR: {entry['error']}"


# ---- tool implementations ------------------------------------------------------

def make_search(store, top_k: int, min_score: float, groups=None):
    """groups: the user's groups. The agent reads only what its user may read."""
    def search_filings(args: SearchArgs, guard: ToolGuard) -> str:
        tickers = frozenset({args.ticker}) if args.ticker else frozenset()
        years = frozenset({args.fiscal_year}) if args.fiscal_year else frozenset()
        extra = {"groups": frozenset(groups)} if groups is not None else {}
        hits = [(s, p) for s, p in store.search(args.query, top_k, tickers, years, **extra)
                if s >= min_score]  # same evidence gate as the plain answer path
        if not hits:
            return "No passage in the filings is relevant enough to this query."
        out = []
        for score, p in hits:
            sid = guard.add_evidence("search_filings", p["text"], {
                **{k: p[k] for k in ("ticker", "form", "period", "section",
                                     "section_title", "url")}, "score": round(score, 3)})
            out.append(f"[{sid}] {p['ticker']} {p['form']} {p['period']}, "
                       f"Item {p['section']} {p['section_title']}:\n{p['text']}")
        return "\n\n".join(out)
    return search_filings


def pick_fact(facts: dict, metric: str, fiscal_year: int) -> dict | None:
    """The annual (10-K, full-year) value of `metric` for the fiscal year ending in
    `fiscal_year`, from SEC companyfacts JSON. Latest filing wins (restatements)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in METRICS[metric]:
        units = gaap.get(concept, {}).get("units", {})
        rows = units.get("USD") or units.get("USD/shares") or []
        best = None
        for r in rows:
            if r.get("form") != "10-K" or int(r["end"][:4]) != fiscal_year:
                continue
            if "start" in r:  # a flow (revenue, income): keep full-year periods only
                days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
                if not 350 <= days <= 380:
                    continue
            if best is None or r["filed"] > best["filed"]:
                best = r
        if best:
            return {**best, "concept": concept,
                    "unit": "USD/shares" if "USD/shares" in units else "USD"}
    return None


def make_facts(load_facts: Callable[[str], dict]):
    def get_financial_fact(args: FactArgs, guard: ToolGuard) -> str:
        row = pick_fact(load_facts(args.ticker), args.metric, args.fiscal_year)
        if row is None:
            return (f"No reported {args.metric} for {args.ticker} fiscal "
                    f"{args.fiscal_year} in SEC XBRL data.")
        period = f"{row.get('start', '')}..{row['end']}".lstrip(".")
        text = (f"{args.ticker} {args.metric} for fiscal {args.fiscal_year} "
                f"(period {period}): {row['val']:,} {row['unit']} "
                f"(us-gaap:{row['concept']}, 10-K accession {row['accn']}).")
        sid = guard.add_evidence("get_financial_fact", text, {
            "ticker": args.ticker, "form": "10-K (XBRL)", "period": row["end"],
            "section": row["concept"], "section_title": args.metric,
            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                   f"&CIK={args.ticker}&type=10-K", "accession": row["accn"]})
        return f"[{sid}] {text}"
    return get_financial_fact


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node: ast.AST, operands: list[float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, operands)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        operands.append(float(node.value))
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, operands), _eval(node.right, operands))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand, operands))
    raise ToolError("only numbers, + - * / and parentheses are allowed")


# Constants a calculation may use without them coming from a filing (percent, scales).
FREE_CONSTANTS = {0.0, 1.0, 2.0, 100.0, 1000.0, 1e6, 1e9}


def calculate(args: CalcArgs, guard: ToolGuard) -> str:
    """Arithmetic only on figures the tools already returned: a computed growth rate
    is grounded only if its inputs are. No eval(), only a whitelisted AST walk."""
    try:
        tree = ast.parse(args.expression.replace(",", ""), mode="eval")
    except SyntaxError as e:
        raise ToolError("not a valid arithmetic expression") from e
    operands: list[float] = []
    try:
        value = _eval(tree, operands)
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e
    pool = [ev.text for ev in guard.evidence]
    unknown = [o for o in operands
               if o not in FREE_CONSTANTS and not grounding.value_in(o, pool)]
    if unknown:
        raise ToolError("these operands do not come from any tool result: "
                        + ", ".join(f"{o:g}" for o in unknown))
    text = f"{args.expression} = {value:.4f}".rstrip("0").rstrip(".")
    sid = guard.add_evidence("calculate", text, {"ticker": "", "form": "calculation",
                             "period": "", "section": "", "section_title": args.expression,
                             "url": ""})
    return f"[{sid}] {text}"


def facts_loader(client_factory: Callable[[], object], ciks: dict[str, int], cache_dir: Path):
    """Load companyfacts JSON for a ticker, from cache or SEC (one request per company)."""
    def load(ticker: str) -> dict:
        path = cache_dir / f"{ticker}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        data = client_factory().company_facts(ciks[ticker])
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    return load


def build_guard(store, load_facts, allowed_tickers, top_k: int, min_score: float,
                max_calls: int = 6, groups=None) -> ToolGuard:
    return ToolGuard(
        tools={
            "search_filings": (SearchArgs, make_search(store, top_k, min_score, groups),
                               "Search the text of the companies' 10-K/10-Q filings "
                               "(risks, strategy, MD&A, legal proceedings...)."),
            "get_financial_fact": (FactArgs, make_facts(load_facts),
                                   "Get a figure the company reported in its 10-K "
                                   "(revenue, net income...) for one fiscal year."),
            "calculate": (CalcArgs, calculate,
                          "Compute growth, differences or ratios from figures already "
                          "returned by the other tools."),
        },
        allowed_tickers=frozenset(allowed_tickers),
        max_calls=max_calls,
    )
