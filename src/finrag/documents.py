"""Documents mode (FINRAG_DOMAIN=documents): a corpus of documents instead of SEC filings.

The corpus used here: service-public.fr practical sheets ("fiches pratiques"), the
official open dataset published by the DILA under the Licence Ouverte / Etalab 2.0
(https://lecomarquage.service-public.fr). We keep the sheets to which the ministry
of the Interior contributes (identity card, passport, electoral rolls, proxy vote,
driving licence, vehicle registration...): the selection reads each sheet's own
`dc:contributor`, nothing is picked by hand.

Each sheet becomes a Document of titled sections ("Où faire la demande ?", "Coût"),
and each section is chunked with chunk.chunk_document, so a citation points to the
sheet and the section it came from. The documents are public: no groups.

    uv run finrag fetch-docs                 # dataset -> data/docs/*.json (+ SOURCE.md)
    FINRAG_DOMAIN=documents uv run finrag ingest
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from finrag.chunk import Chunk, chunk_document

DATASET_URL = "https://lecomarquage.service-public.fr/vdd/3.4/part/zip/vosdroits-latest.zip"
SOURCE = "service-public"
LICENCE = ("Source : service-public.fr, Direction de l'information légale et administrative "
           "(DILA). Licence Ouverte / Open Licence 2.0 (Etalab).")
DC = "{http://purl.org/dc/elements/1.1/}"

# Navigation, addresses, related links, legal references: not the answer to a question.
SKIP = {"OuSAdresser", "Reference", "ServiceEnLigne", "PourEnSavoirPlus", "QuiPeutMAider",
        "FilDAriane", "Theme", "DossierPere", "SousDossierPere", "SousThemePere", "Audience",
        "Canal", "Abreviation", "Source", "PivotLocal", "Illustration", "SurTitre"}


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    url: str
    date: str  # YYYY-MM-DD, last important change
    source: str = SOURCE
    # (heading, text) in reading order
    sections: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------- service-public.fr XML
def _flat(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _lines(el) -> list[str]:
    out: list[str] = []
    for child in el:
        tag = child.tag
        if tag in SKIP:
            continue
        if tag in ("Paragraphe", "Titre"):
            if text := _flat(child):
                out.append(text)
        elif tag == "Item":
            inner = _lines(child) or ([_flat(child)] if _flat(child) else [])
            out += [f"- {inner[0]}", *inner[1:]] if inner else []
        else:
            out += _lines(child)
    return out


def _section(el) -> tuple[str, str]:
    """A Chapitre or Situation: its first Titre is the heading, the rest the text."""
    titre = el.find("Titre")
    heading = _flat(titre) if titre is not None else ""
    body = [ln for ln in _lines(el) if ln != heading]
    return heading, "\n".join(body)


def parse_sheet(xml: bytes) -> Document:
    root = ET.fromstring(xml)
    title = _flat(root.find(f"{DC}title"))
    # 15 of the 45 Interior sheets have no dateDerniereModificationImportante; all carry
    # dc:date ("modified 2025-04-09").
    date = (root.get("dateDerniereModificationImportante") or "")[:10]
    if not date and (dc_date := root.find(f"{DC}date")) is not None:
        found = re.search(r"\d{4}-\d{2}-\d{2}", _flat(dc_date))
        date = found.group(0) if found else ""
    body: list[tuple[str, str]] = []
    for container in root.findall("Texte") + root.findall("ListeSituations"):
        parts = container.findall("Chapitre") + container.findall("Situation")
        if parts:
            body += [s for s in map(_section, parts) if s[1]]
        elif text := "\n".join(_lines(container)):
            body.append(("Détails", text))
    intro = [_flat(root.find(f"{DC}description"))] if root.find(f"{DC}description") is not None \
        else []
    if (el := root.find("Introduction")) is not None:
        intro += _lines(el)
    # The summary (dc:description) often repeats a sentence of the text: index it once.
    seen = {ln for _, text in body for ln in text.splitlines()}
    intro = [ln for ln in intro if ln and ln not in seen]
    sections = ([("Présentation", "\n".join(intro))] if intro else []) + body
    return Document(id=root.get("ID", ""), title=title, url=root.get("spUrl", ""),
                    date=date, sections=sections)


def contributors(xml: bytes) -> str:
    el = ET.fromstring(xml).find(f"{DC}contributor")
    return _flat(el) if el is not None else ""


def interior_sheets(zip_path: Path) -> list[Document]:
    """The practical sheets (F…) whose contributors include the ministry of the Interior."""
    docs = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not re.fullmatch(r"F\d+\.xml", name):
                continue
            xml = z.read(name)
            if "intérieur" in contributors(xml).lower():
                doc = parse_sheet(xml)
                if doc.sections:
                    docs.append(doc)
    return docs


# ---------------------------------------------------------------- corpus on disk
def save(docs: list[Document], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    for d in docs:
        (directory / f"{d.id}.json").write_text(
            json.dumps(asdict(d), ensure_ascii=False, indent=1), encoding="utf-8")
    (directory / "SOURCE.md").write_text(f"{LICENCE}\n\n{DATASET_URL}\n", encoding="utf-8")


def load(directory: Path) -> list[Document]:
    docs = []
    for p in sorted(directory.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        row["sections"] = [tuple(s) for s in row["sections"]]
        docs.append(Document(**row))
    return docs


def to_chunks(docs: list[Document]) -> list[Chunk]:
    """One chunk_document call per section: a citation names the sheet AND the section."""
    out: list[Chunk] = []
    for d in docs:
        for n, (heading, text) in enumerate(d.sections):
            out += chunk_document(f"{d.id}#{n}", f"{d.title} — {heading}", text, [],
                                  d.source, d.url, d.date or "1970-01-01")
    return out
