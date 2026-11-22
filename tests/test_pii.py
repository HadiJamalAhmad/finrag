"""Personal data: found by checked patterns, masked before the model and the traces, restored
for the user. Offline: names use a stub NER (the real spaCy model runs in pii-eval)."""

import pytest
from test_agent_tools import FakeStore
from test_tracing import GEN, Capture

from finrag import answer as ans
from finrag import pii, scope
from finrag.config import Settings

NIR = "1 85 05 78 006 084 91"        # key = 97 - 1850578006084 mod 97
NIR_CORSICA = "1 85 05 2A 004 012 97"  # 2A -> 19, minus 1,000,000
IBAN = "FR76 3000 6000 0112 3456 7890 189"
VISA, AMEX = "4111 1111 1111 1111", "378282246310005"


def kinds(text, ner=None):
    return [(s.kind, s.value) for s in pii.find(text, ner)]


# ---------------------------------------------------------------- detectors
@pytest.mark.parametrize("value,kind", [
    (NIR, "NIR"), (NIR.replace(" ", ""), "NIR"), (NIR_CORSICA, "NIR"),
    (IBAN, "IBAN"), (IBAN.replace(" ", ""), "IBAN"),
    (VISA, "CARD"), (AMEX, "CARD"),
    ("camille.martin@example.fr", "EMAIL"),
    ("06 12 34 56 78", "PHONE"), ("06.12.34.56.78", "PHONE"), ("+33 6 12 34 56 78", "PHONE"),
    ("0033612345678", "PHONE"), ("01 40 07 60 60", "PHONE"),
])
def test_each_kind_is_found_whole(value, kind):
    assert kinds(f"Mon identifiant : {value}, merci.") == [(kind, value)]


@pytest.mark.parametrize("text", [
    "NIR 1 85 05 78 006 084 92 (wrong key)",
    "IBAN FR76 3000 6000 0112 3456 7890 188 (wrong check digits)",
    "Carte 4111 1111 1111 1112 (fails Luhn)",
    "SIRET 732 829 320 00074",          # a company's number, Luhn-valid: not personal
    "Appelez le 3939, service public",  # short service number
    "Le timbre coûte 86 euros, délai 15 jours, code postal 78300.",
    "Formulaire cerfa n° 12100*03 du 1er janvier 2026",
    "Montant : 1 250 000 euros",
])
def test_lookalikes_are_left_alone(text):
    assert kinds(text) == []


def test_specific_beats_generic_and_nothing_overlaps():
    # A NIR is 15 digits: the card pattern must not take it, nor the phone one a part of it.
    spans = pii.find(f"{NIR} ou {VISA}")
    assert [s.kind for s in spans] == ["NIR", "CARD"]


def test_names_come_from_the_ner_and_never_override_a_number():
    def ner(text):
        i = text.index("Camille Martin")
        return [(i, i + len("Camille Martin")), (0, 3)]  # (0, 3) overlaps the NIR

    assert kinds(f"{NIR} Camille Martin", ner) == [("NIR", NIR), ("NAME", "Camille Martin")]


# ---------------------------------------------------------------- masking
def test_masking_is_consistent_reversible_and_digit_free():
    text = f"Je suis Camille, NIR {NIR}. Rappel : {NIR}, ou camille@example.fr."
    m = pii.mask(text)
    assert m.text == "Je suis Camille, NIR [NIR_A]. Rappel : [NIR_A], ou [EMAIL_A]."
    assert m.kinds() == {"NIR": 1, "EMAIL": 1}
    assert m.restore(m.text) == text
    assert NIR not in repr(m)  # the mapping is never printed


def test_labels_run_past_z_without_digits():
    assert [pii._label(i) for i in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]


def test_mode_off_masks_nothing_and_unknown_modes_fail_loudly():
    assert pii.protector("off")(NIR).text == NIR
    with pytest.raises(ValueError, match="FINRAG_PII"):
        pii.protector("names")


# ---------------------------------------------------------------- the answer path
S = Settings(sec_user_agent="x y@z", min_score=0.55, gpu="L4", langfuse_public_key="",
             langfuse_secret_key="")


@pytest.fixture
def sc(filing_10k, sec_names):
    return scope.Scope.build([filing_10k], sec_names)


def test_the_model_never_sees_the_data_and_the_user_gets_it_back(sc):
    prompts = []

    def llm(prompt, s):
        prompts.append(prompt)
        return ans.Generation("A single supplier in Taiwan [1]. Copy sent to [EMAIL_A].",
                              GEN.usage, "stop")

    q = f"Who supplies Acme? My NIR is {NIR}, write to camille@example.fr"
    a = ans.answer(q, sc, FakeStore(), S, llm=llm)
    assert NIR not in prompts[0] and "camille@example.fr" not in prompts[0]
    assert "[NIR_A]" in prompts[0]
    # Placeholders hold no digits: the figure check passes; the user sees their address.
    assert a.stage == "answered" and a.question == q
    assert a.answer.endswith("Copy sent to camille@example.fr.")
    assert a.pii == {"NIR": 1, "EMAIL": 1}


def test_no_trace_holds_the_data(sc):
    cap = Capture()
    a = ans.answer(f"Who supplies Acme? IBAN {IBAN}", sc, FakeStore(), S,
                   llm=lambda p, s: GEN, tracer=cap.tracer)
    assert a.pii == {"IBAN": 1}
    dump = repr([dict(s.attributes) for s in cap.spans()])
    assert IBAN not in dump and IBAN.replace(" ", "") not in dump
    assert "[IBAN_A]" in dump and '{"IBAN":1}' in dump  # the count is traced, not the value


# ---------------------------------------------------------------- scoring
def test_score_counts_found_missed_wrong_leaks_and_false_masks():
    suite = [
        {"id": "a", "text": f"NIR {NIR}, Camille Martin", "pii": [
            {"kind": "NIR", "value": NIR}, {"kind": "NAME", "value": "Camille Martin"}]},
        {"id": "b", "text": "Le timbre coûte 86 euros.", "pii": []},
    ]
    r = pii.score(suite, pii.protector("regex"))
    assert r["per_kind"]["NIR"] == {"found": 1, "missed": 0, "wrong": 0,
                                    "recall": 1.0, "precision": 1.0}
    assert r["per_kind"]["NAME"]["missed"] == 1 and r["name_recall"] == 0.0
    assert r["leaks"] == 1 and r["false_masks"] == 0  # the name is still in the text
    assert r["gates"]["leaks"]["pass"] is False


def test_the_real_french_model_finds_a_name_when_installed():
    pytest.importorskip("fr_core_news_sm")  # extra `pii`; the pii-eval workflow has it
    m = pii.protector("full")("Bonjour, je m'appelle Camille Martin et j'habite à Lyon.")
    assert "Camille Martin" not in m.text and m.kinds().get("NAME") == 1
