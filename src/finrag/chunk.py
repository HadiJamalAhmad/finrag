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
    # Who may read it: () = public (the SEC filings). Otherwise the user must belong to
    # one of these groups (documents pushed by an information-system connector).
    groups: tuple[str, ...] = ()

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


def chunk_document(doc_id: str, title: str, text: str, groups: list[str], source: str,
                   url: str, date: str) -> list[Chunk]:
    """Chunks of a document pushed by a connector (GED, CRM export, RPA bot).

    Ids derive from the document id: pushing the same document again overwrites its
    chunks instead of duplicating them (idempotent upserts, what a connector needs).
    """
    out: list[Chunk] = []
    for n, piece in enumerate(split_words(text)):
        digest = hashlib.sha1(f"doc|{source}|{doc_id}|{n}".encode()).hexdigest()
        out.append(Chunk(
            id=digest, text=f"{title}\n{piece}", ticker="INTERNAL", company=title,
            form="DOC", fiscal_year=int(date[:4]), period=date, filed=date,
            section=source, section_title=title, url=url, groups=tuple(groups)))
    return out
