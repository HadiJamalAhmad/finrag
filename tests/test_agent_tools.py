"""The agent and its tool guard, offline: a scripted model stands in for the LLM."""

import httpx
import pytest

from finrag import agent, edgar, scope
from finrag.answer import REFUSAL
from finrag.tools import build_guard, pick_fact

# companyfacts-shaped data for ACME (fiscal year ends late September).
FACTS = {"facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"start": "2024-09-29", "end": "2025-09-27", "val": 12345000000, "form": "10-K",
         "filed": "2025-11-01", "accn": "0001234-25-000001"},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 11400000000, "form": "10-K",
         "filed": "2024-11-01", "accn": "0001234-24-000001"},
        # The FY2025 10-K restates FY2024: the latest filing must win.
        {"start": "2023-10-01", "end": "2024-09-28", "val": 11431000000, "form": "10-K",
         "filed": "2025-11-01", "accn": "0001234-25-000001"},
        # A quarter reported inside a 10-K: not a full year, must be ignored.
        {"start": "2025-06-29", "end": "2025-09-27", "val": 3300000000, "form": "10-K",
         "filed": "2025-11-01", "accn": "0001234-25-000001"},
    ]}},
    "Assets": {"units": {"USD": [
        {"end": "2025-09-27", "val": 50000000000, "form": "10-K", "filed": "2025-11-01",
         "accn": "0001234-25-000001"}]}},
}}}

RISK = {"ticker": "ACME", "form": "10-K", "period": "2025-09-27", "section": "1A",
        "section_title": "Risk Factors", "url": "https://sec.example/acme",
        "text": "Acme depends on a single supplier in Taiwan for its widget sensors."}


class FakeStore:
    def __init__(self, score=0.8):
        self.score = score

    def search(self, query, k, tickers=frozenset(), years=frozenset()):
        return [(self.score, RISK)]


class Scripted:
    """Replays assistant messages in order and records what it was shown."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(messages)
        return self.replies.pop(0) if self.replies else {"role": "assistant", "content": ""}


def call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def say(text):
    return {"role": "assistant", "content": text}


@pytest.fixture
def sc(filing_10k, sec_names):
    return scope.Scope.build([filing_10k], sec_names)


@pytest.fixture
def guard():
    return build_guard(FakeStore(), lambda t: FACTS, {"ACME"}, top_k=5, min_score=0.55,
                       max_calls=4)


# ---- ToolGuard -----------------------------------------------------------------

def test_guard_refuses_unknown_tool_and_logs_it(guard):
    out = guard.run("delete_filings", {})
    assert out.startswith("ERROR: unknown tool")
    assert guard.log[-1]["ok"] is False and guard.log[-1]["tool"] == "delete_filings"


def test_guard_validates_arguments(guard):
    assert "invalid arguments" in guard.run("get_financial_fact",
                                            {"ticker": "ACME", "metric": "ebitda",
                                             "fiscal_year": 2025})
    assert "invalid arguments" in guard.run("search_filings", {})


def test_guard_refuses_ticker_outside_corpus(guard):
    out = guard.run("search_filings", {"query": "battery supply", "ticker": "tsla"})
    assert "TSLA is not in the corpus" in out
    assert not guard.evidence


def test_guard_accepts_json_string_arguments(guard):
    out = guard.run("get_financial_fact",
                    '{"ticker": "acme", "metric": "total_assets", "fiscal_year": 2025}')
    assert out.startswith("[S1]") and "50,000,000,000" in out


def test_guard_budget_stops_further_calls(guard):
    for _ in range(4):
        guard.run("search_filings", {"query": "supplier risk"})
    assert guard.exhausted
    assert "budget" in guard.run("search_filings", {"query": "supplier risk"})


def test_search_applies_the_evidence_gate():
    g = build_guard(FakeStore(score=0.40), lambda t: FACTS, {"ACME"}, 5, 0.55)
    assert g.run("search_filings", {"query": "the CEO's dog"}).startswith("No passage")
    assert not g.evidence


# ---- reported figures and calculator -------------------------------------------

def test_pick_fact_full_year_latest_filing_and_concept_fallback():
    fy25 = pick_fact(FACTS, "revenue", 2025)
    assert fy25["val"] == 12345000000  # not the 3.3B quarter
    assert fy25["concept"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert pick_fact(FACTS, "revenue", 2024)["val"] == 11431000000  # restated value
    assert pick_fact(FACTS, "net_income", 2025) is None


def test_calculate_only_on_figures_from_tool_results(guard):
    guard.run("get_financial_fact", {"ticker": "ACME", "metric": "revenue", "fiscal_year": 2025})
    guard.run("get_financial_fact", {"ticker": "ACME", "metric": "revenue", "fiscal_year": 2024})
    ok = guard.run("calculate", {"expression": "(12345 - 11431) / 11431 * 100"})
    assert ok.startswith("[S3]") and "= 7.9958" in ok
    invented = guard.run("calculate", {"expression": "(13000 - 11431) / 11431 * 100"})
    assert "do not come from any tool result: 13000" in invented


def test_calculate_rejects_code(guard):
    assert "only numbers" in guard.run("calculate", {"expression": "__import__('os').getcwd()"})


def test_edgar_company_facts_url():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=FACTS)

    client = edgar.EdgarClient("Jane Doe jane@doe.org", transport=httpx.MockTransport(handler))
    assert client.company_facts(1234) == FACTS
    assert seen == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000001234.json"]


# ---- the agent loop: one test per stage -----------------------------------------

def test_agent_answers_with_figures_calculation_and_citations(sc, guard):
    model = Scripted(
        call("get_financial_fact", ticker="ACME", metric="revenue", fiscal_year=2025),
        call("get_financial_fact", ticker="ACME", metric="revenue", fiscal_year=2024),
        call("calculate", expression="(12345000000 - 11431000000) / 11431000000 * 100"),
        say("Acme's revenue grew from $11,431 million [S2] to $12,345 million [S1], "
            "an increase of 8.0% [S3]."),
    )
    a = agent.run("How much did Acme's revenue grow in fiscal 2025?", sc, guard, model)
    assert (a.refused, a.stage) == (False, "answered")
    assert [s["n"] for s in a.sources] == ["S1", "S2", "S3"]
    assert [s["tool"] for s in a.steps] == ["get_financial_fact"] * 2 + ["calculate"]
    # Tool results were fed back to the model before its final answer.
    assert any(m["role"] == "tool" for m in model.seen[-1])


def test_agent_refuses_an_invented_figure(sc, guard):
    model = Scripted(
        call("get_financial_fact", ticker="ACME", metric="revenue", fiscal_year=2025),
        say("Acme's revenue was $12,345 million [S1], with a margin of 41.7% [S1]."),
    )
    a = agent.run("What was Acme's revenue and margin in fiscal 2025?", sc, guard, model)
    assert (a.refused, a.stage) == (True, "numbers")
    assert "41.7%" in a.reason and a.answer == REFUSAL


def test_agent_refuses_an_answer_without_tool_evidence(sc, guard):
    a = agent.run("What risks does Acme describe?", sc, guard,
                  Scripted(say("Acme faces supply chain risks.")))
    assert (a.refused, a.stage) == (True, "evidence")


def test_agent_stops_at_the_tool_budget(sc, guard):
    model = Scripted(*[call("search_filings", query="supplier risk")] * 10)
    a = agent.run("What risks does Acme describe?", sc, guard, model)
    assert (a.refused, a.stage) == (True, "limit")
    assert len(a.steps) == guard.max_calls


def test_agent_scope_refusal_never_calls_the_model(sc, guard):
    model = Scripted()
    a = agent.run("What was Tesla's revenue in 2025?", sc, guard, model)
    assert (a.refused, a.stage) == (True, "scope")
    assert model.seen == []


def test_agent_passes_on_the_model_refusal(sc, guard):
    model = Scripted(call("search_filings", query="CEO favourite sport"), say(REFUSAL))
    a = agent.run("What is Acme's CEO's favourite sport?", sc, guard, model)
    assert (a.refused, a.stage) == (True, "model")
