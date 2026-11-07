"""Turn a filing's HTML into clean text split by its Items (sections).

Sections matter for finance: "risk factors" (Item 1A) and "MD&A" (Item 7) answer
different questions, and a citation should say which one it comes from.

The table of contents lists every Item before the body does, so a header can
appear twice. For each Item the LONGEST section wins: the TOC entry is a line,
the real section is pages long.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

TEN_K_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TEN_Q_TITLES = {
    "I-1": "Financial Statements",
    "I-2": "Management's Discussion and Analysis (MD&A)",
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk",
    "I-4": "Controls and Procedures",
    "II-1": "Legal Proceedings",
    "II-1A": "Risk Factors",
    "II-2": "Unregistered Sales of Equity Securities",
    "II-5": "Other Information",
    "II-6": "Exhibits",
}

ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.{0,150})$", re.IGNORECASE)
PART_RE = re.compile(r"^\s*part\s+(i{1,2}|iv|iii)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    key: str  # "1A" for a 10-K, "II-1A" for a 10-Q
    title: str
    text: str


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Inline XBRL keeps machine-readable facts in a hidden header: not prose.
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    # Table cells become " | "-separated rows so numbers keep their labels.
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", ")", "%"}]
        row.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def split_sections(text: str, form: str) -> list[Section]:
    is_q = form.upper().startswith("10-Q")
    titles = TEN_Q_TITLES if is_q else TEN_K_TITLES
    lines = text.splitlines()
    part = "I"
    heads: list[tuple[int, str]] = []  # (line index, key)
    for i, line in enumerate(lines):
        pm = PART_RE.match(line)
        if pm:
            part = pm.group(1).upper()
            continue
        m = ITEM_RE.match(line)
        if not m:
            continue
        item = m.group(1).upper()
        key = f"{part}-{item}" if is_q else item
        if key in titles:
            heads.append((i, key))

    best: dict[str, Section] = {}
    for n, (start, key) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if key not in best or len(body) > len(best[key].text):
            best[key] = Section(key=key, title=titles[key], text=body)
    order = list(titles)
    return sorted(best.values(), key=lambda s: order.index(s.key))
