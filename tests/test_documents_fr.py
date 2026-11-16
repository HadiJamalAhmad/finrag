"""Documents mode: service-public.fr sheets, French prompt and refusal, no company scope.

Fixtures sp_F18704.xml and sp_other_F10041.xml are real sheets from the service-public.fr
open dataset (DILA, Licence Ouverte / Etalab 2.0), unmodified. Offline: fake stores.
"""

import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from finrag import answer as ans
from finrag import documents, scope
from finrag.config import Settings
from finrag.store import QUERY_PREFIX, query_prefix

FIX = Path(__file__).parent / "fixtures"
SHEET = (FIX / "sp_F18704.xml").read_bytes()
OTHER = (FIX / "sp_other_F10041.xml").read_bytes()
DOCS = Settings(sec_user_agent="x y@z", domain="documents", min_score=0.5)


# ---------------------------------------------------------------- reading a sheet
def test_a_sheet_keeps_its_title_url_date_and_text():
    d = documents.parse_sheet(SHEET)
    assert d.id == "F18704"
    assert d.title.startswith("Que faire si on retrouve sa carte d'identité")
    assert d.url == "https://www.service-public.gouv.fr/particuliers/vosdroits/F18704"
    assert d.date == "2025-04-09"  # from dc:date: this sheet has no dateDerniere... attribute
    text = "\n".join(t for _, t in d.sections)
    assert "invalidé de manière informatique" in text
    assert "irréversible" in text


def test_the_summary_is_not_indexed_twice():
    d = documents.parse_sheet(SHEET)
    lines = [ln for _, t in d.sections for ln in t.splitlines()]
    assert len(lines) == len(set(lines))


def test_only_sheets_the_interior_ministry_contributes_to_are_kept(tmp_path):
    assert "intérieur" in documents.contributors(SHEET).lower()
    assert "intérieur" not in documents.contributors(OTHER).lower()
    z = tmp_path / "vosdroits.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("F18704.xml", SHEET)
        f.writestr("F10041.xml", OTHER)
        f.writestr("N358.xml", SHEET)  # a theme page, not a practical sheet
    assert [d.id for d in documents.interior_sheets(z)] == ["F18704"]


def test_save_and_load_round_trip_with_the_licence(tmp_path):
    d = documents.parse_sheet(SHEET)
    documents.save([d], tmp_path)
    assert documents.load(tmp_path) == [d]
    assert "Licence Ouverte" in (tmp_path / "SOURCE.md").read_text(encoding="utf-8")


def test_chunks_are_public_and_cite_the_sheet_and_its_section():
    d = documents.parse_sheet(SHEET)
    chunks = documents.to_chunks([d])
    assert chunks and all(c.groups == () for c in chunks)  # public: no group needed
    assert all(c.url == d.url and c.section == "service-public" for c in chunks)
    assert all(c.section_title.startswith(d.title + " — ") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    # Same sheet again: same ids, so a re-import replaces instead of duplicating.
    assert [c.id for c in documents.to_chunks([d])] == [c.id for c in chunks]


# ---------------------------------------------------------------- answer path
class FakeStore:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, question, k, tickers=frozenset(), years=frozenset(), **kw):
        self.calls.append((tickers, years))
        return self.hits


PASSAGE = {"ticker": "INTERNAL", "form": "DOC", "period": "2026-04-08",
           "section": "service-public", "url": "https://www.service-public.gouv.fr/x",
           "section_title": "Carte d'identité d'un majeur — Coût",
           "text": "La première demande de carte d'identité est gratuite."}
EMPTY_SCOPE = scope.Scope.build([], {})


def test_a_year_in_the_question_is_not_refused_in_documents_mode():
    """The SEC scope refuses years it does not cover; a documents corpus has no years."""
    store = FakeStore([(0.9, PASSAGE)])
    a = ans.answer("Combien coûte une carte d'identité en 2026 ?", EMPTY_SCOPE, store, DOCS,
                   llm=lambda p, s: "Elle est gratuite [1].")
    assert a.stage == "answered" and store.calls == [(frozenset(), frozenset())]


def test_the_same_question_is_refused_at_scope_in_sec_mode():
    a = ans.answer("Combien coûte une carte d'identité en 2026 ?", EMPTY_SCOPE,
                   FakeStore([(0.9, PASSAGE)]), replace(DOCS, domain="sec"),
                   llm=lambda p, s: "x")
    assert a.stage == "scope"


def test_off_topic_questions_are_refused_in_french_at_the_evidence_gate():
    a = ans.answer("Quelle est la capitale du Pérou ?", EMPTY_SCOPE,
                   FakeStore([(0.31, PASSAGE)]), DOCS, llm=lambda p, s: "x")
    assert (a.stage, a.answer) == ("evidence", ans.REFUSAL_DOCS)


def test_the_model_refusing_in_french_is_recognised():
    a = ans.answer("Combien coûte une carte d'identité ?", EMPTY_SCOPE,
                   FakeStore([(0.9, PASSAGE)]), DOCS,
                   llm=lambda p, s: "Je ne sais pas d'après les documents fournis.")
    assert (a.stage, a.refused) == ("model", True)


def test_the_documents_prompt_shows_section_titles_and_asks_for_the_question_language():
    p = ans.build_prompt("Combien ?", [(0.9, PASSAGE)], "documents")
    assert "[1] (Carte d'identité d'un majeur — Coût)" in p
    assert "language of the question" in p and ans.REFUSAL_DOCS in p
    assert "SEC" not in p


def test_the_sec_prompt_is_unchanged():
    filing = {**PASSAGE, "ticker": "ACME", "form": "10-K", "section_title": "Risk Factors"}
    p = ans.build_prompt("Q?", [(0.9, filing)])
    assert "numbered SEC filing" in p and "[1] (ACME 10-K 2026-04-08, Risk Factors)" in p


# ---------------------------------------------------------------- embedder prefix
@pytest.mark.parametrize("model,prefix", [
    ("BAAI/bge-small-en-v1.5", QUERY_PREFIX),  # the SEC index: unchanged
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
])
def test_each_embedder_gets_the_query_prefix_it_was_trained_with(model, prefix):
    assert query_prefix(model) == prefix


# ---------------------------------------------------------------- API
def test_ready_needs_no_sec_manifest_in_documents_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from finrag import api, store

    class Filled:
        def __init__(self, s):
            pass

        def count(self):
            return 600

    monkeypatch.setenv("FINRAG_DATA", str(tmp_path))
    monkeypatch.setattr(store, "Store", Filled)
    monkeypatch.setattr(api, "get_settings", lambda: replace(DOCS, data_dir=tmp_path))
    c = TestClient(api.app)
    assert c.get("/ready").status_code == 200
    assert c.post("/agent", json={"question": "Combien ?"}).status_code == 404


def test_the_cluster_sample_is_what_the_reader_makes_of_the_fixture():
    """deploy/k8s/sample-fr (CI's French corpus) must not drift from the parser."""
    sample = Path(__file__).parent.parent / "deploy" / "k8s" / "sample-fr"
    assert documents.load(sample) == [documents.parse_sheet(SHEET)]
