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
