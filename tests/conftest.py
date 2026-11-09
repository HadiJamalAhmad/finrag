import json
from pathlib import Path

import pytest

from finrag.edgar import Filing

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def filing_10k() -> Filing:
    return Filing(ticker="ACME", company="Acme Widgets Inc.", cik=1234, form="10-K",
                  accession="0001234-25-000001", filed="2025-11-01", period="2025-09-27",
                  url="https://www.sec.gov/Archives/edgar/data/1234/000123425000001/acme-10k.htm")


@pytest.fixture
def html_10k() -> str:
    return (FIX / "acme_10k.html").read_text(encoding="utf-8")


@pytest.fixture
def sec_names() -> dict:
    return json.loads((FIX / "company_tickers.json").read_text(encoding="utf-8"))


# config.py loads the developer's .env at import. Tests stay offline (CLAUDE.md): no
# Settings built inside a test may carry a real provider or tracing key.
LIVE_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY",
             "LANGFUSE_SECRET_KEY", "CLERK_SECRET_KEY", "CLERK_PUBLISHABLE_KEY",
             "FINRAG_INGEST_KEY"]


@pytest.fixture(autouse=True)
def no_live_keys(monkeypatch):
    for name in LIVE_KEYS:
        monkeypatch.delenv(name, raising=False)
