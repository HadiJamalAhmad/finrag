"""Structured extraction: every kept field carries a sentence of the sheet that proves it.

Offline: the model is replaced by fixed outputs (dicts) or an httpx MockTransport.
The sheet text is from the real fixture sp_F1709.xml (service-public.fr, Licence Ouverte).
"""

import json
from pathlib import Path

import httpx

from finrag import extract
from finrag.config import Settings
from finrag.documents import parse_sheet

FIX = Path(__file__).parent / "fixtures"
SHEET = parse_sheet((FIX / "sp_F1709.xml").read_bytes())
TEXT = "\n".join(t for _, t in SHEET.sections)
PASSPORT = ("Vous devez acheter un timbre fiscal d’un montant de 86 € (43 € en Guyane).\n"
            "Le passeport doit être retiré dans les 3 mois. Passé ce délai, le passeport est "
            "détruit.\nVous pouvez vous rendre dans n'importe quelle mairie, à condition "
            "qu'elle soit équipée d'une station d'enregistrement.")
S = Settings(sec_user_agent="x y@z")


def raw(**kw):
    base = {n: None for n in extract.SCALARS} | {p: "" for p in extract.PROOF_OF.values()}
    return base | {"couts": []} | kw


# ---------------------------------------------------------------- what is kept
def test_a_value_with_a_verbatim_proof_containing_it_is_kept():
    e = extract.verify(raw(delai_retrait="3 mois",
                           delai_retrait_preuve="Le passeport doit être retiré dans les 3 "
                                                "mois."), PASSPORT)
    assert e.fields["delai_retrait"]["value"] == "3 mois" and not e.rejected


def test_amounts_are_kept_with_their_condition_when_the_proof_holds_them():
    proof = "Vous devez acheter un timbre fiscal d'un montant de 86 € (43 € en Guyane)."
    e = extract.verify(raw(couts=[{"montant": "86 €", "condition": "en France", "preuve": proof},
                                  {"montant": "43 €", "condition": "en Guyane", "preuve": proof}]),
                       PASSPORT)
    assert [c["montant"] for c in e.fields["couts"]] == ["86 €", "43 €"]


def test_typographic_apostrophes_and_spaces_do_not_break_a_true_proof():
    # The sheet writes d’un (U+2019); models usually return d'un.
    proof = "Vous devez acheter un timbre fiscal d'un montant de 86 €"
    e = extract.verify(raw(couts=[{"montant": "86 €", "condition": "", "preuve": proof}]),
                       PASSPORT)
    assert e.fields["couts"] and not e.rejected


def test_null_needs_no_proof():
    e = extract.verify(raw(), PASSPORT)
    assert all(e.fields[n]["value"] is None for n in extract.SCALARS) and not e.rejected


# ---------------------------------------------------------------- what is rejected
def test_a_proof_that_is_not_in_the_sheet_is_rejected():
    e = extract.verify(raw(delai_retrait="3 mois",
                           delai_retrait_preuve="Le passeport doit être retiré dans les 3 "
                                                "mois sinon il est envoyé par la poste."),
                       PASSPORT)
    assert e.fields["delai_retrait"]["value"] is None
    assert e.rejected[0]["reason"] == "proof is not a sentence of the sheet"


def test_an_invented_amount_is_rejected_even_with_a_real_proof():
    proof = "Vous devez acheter un timbre fiscal d'un montant de 86 € (43 € en Guyane)."
    e = extract.verify(raw(couts=[{"montant": "96 €", "condition": "à l'étranger",
                                   "preuve": proof}]), PASSPORT)
    assert e.fields["couts"] == []
    assert e.rejected == [{"field": "couts", "value": "96 €", "proof": proof,
                           "reason": "a number of the value is not in its proof"}]


def test_a_text_value_unrelated_to_its_proof_is_rejected():
    e = extract.verify(raw(lieu="préfecture", lieu_preuve="Vous pouvez vous rendre dans "
                           "n'importe quelle mairie, à condition qu'elle soit équipée"),
                       PASSPORT)
    assert e.fields["lieu"]["value"] is None
    assert e.rejected[0]["reason"] == "the value shares no word with its proof"


def test_a_proof_of_a_few_words_proves_nothing():
    e = extract.verify(raw(lieu="mairie", lieu_preuve="quelle mairie"), PASSPORT)
    assert e.fields["lieu"]["value"] is None and "shorter" in e.rejected[0]["reason"]


def test_a_cost_whose_proof_has_no_amount_is_rejected():
    proof = "Le passeport doit être retiré dans les 3 mois."
    e = extract.verify(raw(couts=[{"montant": "3", "condition": "", "preuve": proof}]),
                       PASSPORT)
    assert e.fields["couts"] == [] and e.rejected[0]["reason"] == "no amount in proof"


def test_on_the_real_sheet_a_true_free_of_charge_is_kept():
    proof = next(ln for ln in TEXT.splitlines() if "gratuitement" in ln)
    e = extract.verify(raw(gratuit=True, gratuit_preuve=proof), TEXT)
    assert e.fields["gratuit"] == {"value": True, "proof": proof}


# ---------------------------------------------------------------- the model call
def test_the_model_gets_the_schema_a_large_context_and_temperature_0():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"response": json.dumps(raw(gratuit=True))})

    out = extract.ask_model("T", "texte", S, httpx.Client(transport=httpx.MockTransport(handler)))
    assert out["gratuit"] is True
    assert seen["format"] == extract.SCHEMA and seen["stream"] is False
    assert seen["options"]["num_ctx"] == extract.NUM_CTX
    assert seen["options"]["temperature"] == 0


def test_unparsable_model_output_yields_nothing_invented():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"response": "not json"})))
    e = extract.verify(extract.ask_model("T", TEXT, S, client), TEXT)
    assert all(e.fields[n]["value"] is None for n in extract.SCALARS)
    assert e.fields["couts"] == []


# ---------------------------------------------------------------- scoring
def test_score_counts_right_fields_amounts_and_unsupported_values():
    e = extract.verify(raw(
        delai_retrait="3 mois",
        delai_retrait_preuve="Le passeport doit être retiré dans les 3 mois.",
        couts=[{"montant": "86 €", "condition": "en France",
                "preuve": "Vous devez acheter un timbre fiscal d'un montant de 86 €"}]),
        PASSPORT, "P")
    gold = [{"id": "P", "delai_retrait": "3 mois", "gratuit": None,
             "montants": ["86 €", "43 €"]}]
    s = extract.score({"P": e}, gold)
    assert s["field_accuracy"]["delai_retrait"] == 1.0 and s["field_accuracy"]["gratuit"] == 1.0
    assert s["amount_recall"] == 0.5 and s["amount_precision"] == 1.0
    assert s["unsupported_values_kept"] == 0


# ---------------------------------------------------------------- conditions
SECTIONS = [("En France", "Vous devez acheter un timbre fiscal d'un montant de 86 €."),
            ("À l'étranger", "Vous devez acheter un timbre fiscal d'un montant de 96 €.")]


def test_a_condition_stated_by_the_section_heading_is_kept():
    e = extract.verify(raw(couts=[{"montant": "96 €", "condition": "à l'étranger",
                                   "preuve": "Vous devez acheter un timbre fiscal d'un "
                                             "montant de 96 €."}]), SECTIONS)
    assert e.fields["couts"] == [{"montant": "96 €", "condition": "à l'étranger",
                                  "section": "À l'étranger",
                                  "proof": "Vous devez acheter un timbre fiscal d'un "
                                           "montant de 96 €."}]


def test_a_condition_neither_in_the_proof_nor_its_heading_is_left_empty_not_guessed():
    """Seen on F14929: 96 € (abroad) came back as "en général"."""
    e = extract.verify(raw(couts=[{"montant": "96 €", "condition": "en général",
                                   "preuve": "Vous devez acheter un timbre fiscal d'un "
                                             "montant de 96 €."}]), SECTIONS)
    assert e.fields["couts"][0]["montant"] == "96 €"
    assert e.fields["couts"][0]["condition"] == ""
    assert e.rejected[0]["field"] == "couts.condition"


def test_rejections_keep_the_offending_proof_for_review():
    e = extract.verify(raw(lieu="mairie équipée", lieu_preuve="Il faut aller dans une mairie "
                           "équipée pour la demande."), SECTIONS)
    assert e.rejected[0]["proof"].startswith("Il faut aller")


def test_the_model_reads_each_section_under_its_heading():
    assert extract.as_text(SECTIONS).splitlines()[0] == "## En France"


# ---------------------------------------------------------------- windows and merge
def test_long_sections_are_read_in_windows_that_all_fit():
    para = "Une phrase de la fiche qui compte dix mots au total ici."
    long_text = "\n".join([para] * 300)  # ~3 300 words, like the long F1372 sections
    ws = extract.windows([("En France", long_text), ("Vide", "  ")])
    assert len(ws) > 1 and all(len(w.split()) <= extract.WINDOW_WORDS for _, w, _ in ws)
    assert {h for h, _, _ in ws} == {"En France"} and all(t == long_text for _, _, t in ws)


def test_each_window_is_checked_against_its_whole_section():
    """A proof found in the section but cut across two windows still counts."""
    section = "Premier paragraphe.\nVous devez acheter un timbre fiscal d'un montant de 25 €."
    e = extract.verify(raw(couts=[{"montant": "25 €", "condition": "",
                                   "preuve": "Vous devez acheter un timbre fiscal d'un "
                                             "montant de 25 €."}]), [("En France", section)])
    assert e.fields["couts"][0]["section"] == "En France"


def test_merge_keeps_the_first_verified_value_and_every_cost_once():
    a = extract.Extracted("F", "T", "u", {"lieu": {"value": None, "proof": ""},
                                          "pre_demande_en_ligne": {"value": True, "proof": "p1"},
                                          "gratuit": {"value": None, "proof": ""},
                                          "delai_retrait": {"value": None, "proof": ""},
                                          "couts": [{"montant": "86 €", "condition": "en France",
                                                     "section": "", "proof": "x"}]})
    b = extract.Extracted("F", "T", "u", {"lieu": {"value": "mairie", "proof": "p2"},
                                          "pre_demande_en_ligne": {"value": False, "proof": "p3"},
                                          "gratuit": {"value": None, "proof": ""},
                                          "delai_retrait": {"value": "3 mois", "proof": "p4"},
                                          "couts": [{"montant": "86 €", "condition": "En France",
                                                     "section": "", "proof": "y"},
                                                    {"montant": "96 €", "condition": "",
                                                     "section": "", "proof": "z"}]},
                          [{"field": "lieu", "value": "x", "proof": "", "reason": "r"}])
    m = extract.merge([a, b], "F", "T", "u")
    assert m.fields["pre_demande_en_ligne"]["value"] is True  # first in reading order
    assert m.fields["lieu"]["value"] == "mairie" and m.fields["delai_retrait"]["value"] == "3 mois"
    assert [c["montant"] for c in m.fields["couts"]] == ["86 €", "96 €"]
    assert len(m.rejected) == 1


# ---------------------------------------------------------------- resumable, gentle runs
def test_a_cached_result_reads_back_exactly():
    e = extract.verify(raw(couts=[{"montant": "96 €", "condition": "à l'étranger",
                                   "preuve": "Vous devez acheter un timbre fiscal d'un "
                                             "montant de 96 €."}]), SECTIONS, "F1", "T")
    assert extract.from_json(extract.to_json(e)) == e


def test_the_pause_falls_between_model_calls_never_before_the_first(monkeypatch):
    from types import SimpleNamespace

    slept = []
    monkeypatch.setattr(extract.time, "sleep", slept.append)
    monkeypatch.setattr(extract, "ask_model", lambda *a, **k: raw())
    para = "Une phrase de la fiche qui compte dix mots au total ici."
    doc = SimpleNamespace(id="F", title="T", url="u",
                          sections=[("A", "\n".join([para] * 300)), ("B", para)])
    n = len(extract.windows(doc.sections))
    extract.extract(doc, S, pause=2.5)
    assert n > 2 and slept == [2.5] * (n - 1)
    slept.clear()
    extract.extract(doc, S)
    assert slept == []
