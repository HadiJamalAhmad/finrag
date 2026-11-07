"""Section-aware chunking: a chunk never spans two Items, and carries its citation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from finrag.edgar import Filing
from finrag.parse import Section

MAX_WORDS = 300
OVERLAP_WORDS = 50


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    ticker: str
    company: str
    form: str
    fiscal_year: int
    period: str
    filed: str
    section: str
    section_title: str
    url: str

    def cite(self) -> str:
        return (
            f"{self.company} ({self.ticker}) {self.form}, period ending {self.period}, "
            f"Item {self.section.split('-')[-1]} {self.section_title}"
        )


def split_words(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Pack paragraphs up to max_words; a paragraph longer than that is cut with overlap."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    pieces: list[list[str]] = []
    for para in (p for p in text.split("\n") if p.strip()):
        words = para.split()
        while len(words) > max_words:
            pieces.append(words[:max_words])
            words = words[max_words - overlap :]
        pieces.append(words)

    chunks: list[str] = []
    current: list[str] = []
    for words in pieces:
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        current = current + words
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_filing(filing: Filing, sections: list[Section]) -> list[Chunk]:
    out: list[Chunk] = []
    for section in sections:
        for n, text in enumerate(split_words(section.text)):
            # Prefix with the context a reader of the passage would need.
            header = f"{filing.company} {filing.form} ({filing.period}) - {section.title}\n"
            digest = hashlib.sha1(f"{filing.accession}|{section.key}|{n}".encode()).hexdigest()
            out.append(
                Chunk(
                    id=digest,
                    text=header + text,
                    ticker=filing.ticker,
                    company=filing.company,
                    form=filing.form,
                    fiscal_year=filing.fiscal_year,
                    period=filing.period,
                    filed=filing.filed,
                    section=section.key,
                    section_title=section.title,
                    url=filing.url,
                )
            )
    return out
