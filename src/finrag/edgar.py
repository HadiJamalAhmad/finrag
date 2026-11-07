"""Download 10-K / 10-Q filings from SEC EDGAR, the free official source.

Endpoints (documented at https://www.sec.gov/edgar/sec-api-documentation):
- https://www.sec.gov/files/company_tickers.json      ticker -> CIK, for every listed company
- https://data.sec.gov/submissions/CIK##########.json  a company's recent filings
- https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<primary document>

SEC rules for automated access: a User-Agent naming you and a contact address,
and at most 10 requests per second. Both are enforced here, not left to the caller.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
MIN_INTERVAL = 0.12  # seconds between requests: stays under the 10 req/s limit


@dataclass(frozen=True)
class Filing:
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filed: str  # filing date, YYYY-MM-DD
    period: str  # period of report, YYYY-MM-DD
    url: str

    @property
    def fiscal_year(self) -> int:
        # The year the reporting period ends. True for calendar years and for
        # fiscal years named after their end (Apple: Sept, Microsoft: June,
        # NVIDIA: January). Check it for any company that names years otherwise.
        return int(self.period[:4])

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}_{self.form}_{self.period}".replace("/", "-")


# The .env.example template and the reserved example domains (RFC 2606) are not a
# real identity: sending them would tell SEC nothing about who is calling.
PLACEHOLDER_USER_AGENT = "Your Name your.email@example.com"
EXAMPLE_DOMAIN_RE = re.compile(r"@(?:[\w-]+\.)*example\.(?:com|org|net)\b", re.IGNORECASE)


def check_user_agent(user_agent: str) -> None:
    """Refuse a User-Agent that SEC would not accept as an identification."""
    if "@" not in user_agent or len(user_agent.split()) < 2:
        raise ValueError(
            "SEC_USER_AGENT must be your name and your email "
            "(SEC requires a name and a contact address). Set it in .env."
        )
    if user_agent.strip() == PLACEHOLDER_USER_AGENT or EXAMPLE_DOMAIN_RE.search(user_agent):
        raise ValueError(
            "SEC_USER_AGENT is still the .env.example placeholder (or an example.* address). "
            "Put your own name and email in .env."
        )


class EdgarClient:
    def __init__(self, user_agent: str, transport: httpx.BaseTransport | None = None):
        check_user_agent(user_agent)
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._last = 0.0

    def _get(self, url: str) -> httpx.Response:
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def company_tickers(self) -> dict:
        return self._get(TICKERS_URL).json()

    def recent_filings(
        self, ticker: str, cik: int, forms: tuple[str, ...], limit: int
    ) -> list[Filing]:
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        out: list[Filing] = []
        count = dict.fromkeys(forms, 0)
        for i, form in enumerate(recent["form"]):  # newest first
            if form not in count or count[form] >= limit:
                continue
            count[form] += 1
            acc = recent["accessionNumber"][i]
            out.append(
                Filing(
                    ticker=ticker.upper(),
                    company=data["name"],
                    cik=cik,
                    form=form,
                    accession=acc,
                    filed=recent["filingDate"][i],
                    period=recent["reportDate"][i],
                    url=ARCHIVE_URL.format(
                        cik=cik, acc=acc.replace("-", ""), doc=recent["primaryDocument"][i]
                    ),
                )
            )
            if all(c >= limit for c in count.values()):
                break
        return out

    def download(self, filing: Filing) -> str:
        return self._get(filing.url).text


def cik_for(tickers_json: dict, ticker: str) -> tuple[int, str]:
    for row in tickers_json.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"]), row["title"]
    raise KeyError(f"ticker {ticker!r} not found in SEC company_tickers.json")


def fetch(
    client: EdgarClient,
    tickers: list[str],
    forms: tuple[str, ...],
    limit: int,
    raw_dir: Path,
    manifest_path: Path,
    names_path: Path,
) -> list[Filing]:
    """Download filings into raw_dir and record them in the manifest. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tickers_json = client.company_tickers()
    # Kept for the scope guardrail: it knows every SEC-listed company name.
    names_path.write_text(json.dumps(tickers_json), encoding="utf-8")

    manifest = load_manifest(manifest_path)
    known = {f.accession for f in manifest}
    for ticker in tickers:
        cik, _ = cik_for(tickers_json, ticker)
        for filing in client.recent_filings(ticker, cik, forms, limit):
            path = raw_dir / f"{filing.doc_id}.html"
            if not path.exists():
                path.write_text(client.download(filing), encoding="utf-8")
            if filing.accession not in known:
                manifest.append(filing)
                known.add(filing.accession)
    save_manifest(manifest_path, manifest)
    return manifest


def load_manifest(path: Path) -> list[Filing]:
    if not path.exists():
        return []
    return [Filing(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def save_manifest(path: Path, filings: list[Filing]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(f) for f in filings], indent=2), encoding="utf-8")
