"""Turn a filing's HTML into clean text split by its Items (sections).

Sections matter for finance: "risk factors" (Item 1A) and "MD&A" (Item 7) answer
different questions, and a citation should say which one it comes from.

The table of contents lists every Item before the body does, so a header can
appear twice. For each Item the LONGEST section wins: the TOC entry is a line,
the real section is pages long.

Two layouts found on real filings (Week 1):
- Running page headers ("PART II / Item 7" on top of every page, Microsoft): a run
  of consecutive headings for the same Item is ONE section, not one per page.
- Cross-reference 10-Ks (JPMorgan): Item 7 says only "MD&A appears on pages 46-160"
  and Item 3 "Refer to Note 30"; the pages sit later in the same document, after
  Item 15. A short Item that points to pages or to a Note receives those lines.
  When references overlap (7A's pages 133-142 lie inside Item 7's 46-160) the
  narrowest one wins, so every line belongs to exactly one section.
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
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
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

# Cross-references inside a short Item body: "on pages 133-142", "page 120", "Note 30".
# A pointer is a sentence or two (JPM: 13-46 words). NVDA Item 2 (137 words) is a real
# Properties section that mentions a lease Note in passing: it must not absorb the Note.
XREF_MAX_WORDS = 60
PAGES_RE = re.compile(
    r"\bpages?\s+(\d{1,3})(?:\s*(?:[-–—]|through|to)\s*(\d{1,3}))?", re.IGNORECASE
)
NOTE_REF_RE = re.compile(r"\bNote\s+(\d{1,2})\b", re.IGNORECASE)
# A Note's own heading ("Note 30 – Litigation", "NOTE 14 — CONTINGENCIES"). Lines with
# a "|" are table-of-contents rows ("Note 30 - Litigation | 302"), not headings.
NOTE_HEAD_RE = re.compile(r"^\s*note\s+(\d{1,2})\s*[-–—:.](?!.*\|)", re.IGNORECASE)
# A page footer naming the report: "45 | JPMorgan Chase & Co./2025 Form 10-K" or
# "JPMorgan Chase & Co./2025 Form 10-K | 46". The number closes that page.
FOOTER_RE = re.compile(
    r"^\s*(?:(\d{1,3})\s*\|\s*[^|]*(?:10-K|annual report)[^|]*"
    r"|[^|]*(?:10-K|annual report)[^|]*\|\s*(\d{1,3}))\s*$",
    re.IGNORECASE,
)


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

    # Running page headers repeat the same Item on every page: keep the first of a run.
    heads = [h for n, h in enumerate(heads) if n == 0 or heads[n - 1][1] != h[1]]

    spans: dict[str, tuple[int, int]] = {}  # key -> (first body line, end)
    for n, (start, key) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        old = spans.get(key)
        if old is None or _chars(lines, start + 1, end) > _chars(lines, *old):
            spans[key] = (start + 1, end)

    owner: list[str | None] = [None] * len(lines)
    for key, (a, b) in spans.items():
        owner[a:b] = [key] * (b - a)
    _apply_cross_references(lines, spans, owner)

    order = list(titles)
    out: list[Section] = []
    for key in sorted(spans, key=order.index):
        body = "\n".join(ln for ln, k in zip(lines, owner, strict=True) if k == key).strip()
        out.append(Section(key=key, title=titles[key], text=body))
    return out


def _chars(lines: list[str], a: int, b: int) -> int:
    return sum(len(ln) for ln in lines[a:b])


def _page_of_lines(lines: list[str]) -> list[int | None]:
    """Page number of every line, from footers that close each page (None if unknown)."""
    pages: list[int | None] = [None] * len(lines)
    prev = 0
    for i, line in enumerate(lines):
        m = FOOTER_RE.match(line)
        if not m:
            continue
        page = int(m.group(1) or m.group(2))
        if page <= 0:
            continue
        pages[prev : i + 1] = [page] * (i + 1 - prev)
        prev = i + 1
    return pages


def _note_lines(lines: list[str], note: int) -> range | None:
    """Lines of Note `note`: from its heading to the next different Note's heading."""
    start = None
    for i, line in enumerate(lines):
        m = NOTE_HEAD_RE.match(line)
        if not m:
            continue
        if start is None and int(m.group(1)) == note:
            start = i
        elif start is not None and int(m.group(1)) != note:
            return range(start, i)
    return range(start, len(lines)) if start is not None else None


def _apply_cross_references(
    lines: list[str], spans: dict[str, tuple[int, int]], owner: list[str | None]
) -> None:
    """Give a short Item the pages or the Note its body points to (in place)."""
    pages: list[int | None] | None = None
    claims: list[tuple[str, list[int]]] = []
    for key, (a, b) in spans.items():
        body = "\n".join(lines[a:b])
        if len(body.split()) > XREF_MAX_WORDS:
            continue
        pm, nm = PAGES_RE.search(body), NOTE_REF_RE.search(body)
        if pm and (not nm or pm.start() < nm.start()):
            if pages is None:
                pages = _page_of_lines(lines)
            lo = int(pm.group(1))
            hi = int(pm.group(2) or lo)
            idx = [i for i, p in enumerate(pages) if p is not None and lo <= p <= hi]
        elif nm:
            found = _note_lines(lines, int(nm.group(1)))
            idx = list(found) if found else []
        else:
            continue
        idx = [i for i in idx if not a <= i < b]  # never claim the pointer itself
        if idx:
            claims.append((key, idx))
    # Widest first, so a narrower reference nested inside it overwrites its lines.
    for key, idx in sorted(claims, key=lambda c: -len(c[1])):
        for i in idx:
            owner[i] = key
