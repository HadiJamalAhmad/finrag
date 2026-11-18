"""Structured extraction: a procedure sheet -> fields, each with the sentence that proves it.

The model (Ollama, JSON-schema output, temperature 0) returns for every field a value AND
the sentence of the sheet it took it from. `verify` then keeps a field only when

- the proof is copied from the sheet, word for word (at least MIN_PROOF_WORDS words), and
- the value is in its proof: every number of the value appears in the proof, and a text
  value shares a content word with it.

A field that fails is dropped and listed in `rejected` with the reason: no extracted
value ever reaches the output without a sentence of the source that a reader can check.
The same rule as the answers' number check (finrag.grounding), applied to data.

    uv run finrag extract F14929 F1341        # data/docs -> data/structured/*.json
    uv run finrag eval-extract eval/suites/fr-extraction.jsonl
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

import httpx

from finrag.config import Settings
from finrag.models import ollama_options

MIN_PROOF_WORDS = 5
# The model reads one window of a section at a time: at most WINDOW_WORDS words (~1 300
# tokens) under the section heading. Whole sheets reach 20 000 words (F1372: three
# sections of 6 000 to 7 000), far past any small model's context, which silently cut
# them; a window fits 3 072 tokens with the prompt and the answer, and needs ~3x less
# memory than an 8 192-token context.
WINDOW_WORDS = 800
NUM_CTX = 3072

SCHEMA = {
    "type": "object",
    "properties": {
        "lieu": {"type": ["string", "null"]},
        "lieu_preuve": {"type": "string"},
        "pre_demande_en_ligne": {"type": ["boolean", "null"]},
        "pre_demande_preuve": {"type": "string"},
        "gratuit": {"type": ["boolean", "null"]},
        "gratuit_preuve": {"type": "string"},
        "delai_retrait": {"type": ["string", "null"]},
        "delai_retrait_preuve": {"type": "string"},
        "couts": {"type": "array", "items": {"type": "object", "properties": {
            "montant": {"type": "string"}, "condition": {"type": "string"},
            "preuve": {"type": "string"}}, "required": ["montant", "condition", "preuve"]}},
    },
    "required": ["lieu", "lieu_preuve", "pre_demande_en_ligne", "pre_demande_preuve",
                 "gratuit", "gratuit_preuve", "delai_retrait", "delai_retrait_preuve",
                 "couts"],
}

PROMPT = """Tu extrais des données structurées d'une fiche officielle de service-public.fr.
Remplis chaque champ UNIQUEMENT avec ce que dit la fiche. Pour chaque valeur, recopie MOT
POUR MOT, dans le champ de preuve, la phrase de la fiche qui la justifie. Si la fiche ne
donne pas l'information, mets null et une preuve vide "". N'invente rien.

Champs :
- lieu : où déposer la demande en France (par exemple « mairie équipée »)
- pre_demande_en_ligne : true si une pré-demande en ligne est possible, false si la fiche
  dit qu'elle ne l'est pas, null si elle n'en parle pas
- gratuit : true si la fiche dit que la démarche est gratuite, sinon null
- delai_retrait : le délai maximal pour aller chercher le titre une fois qu'il est prêt,
  après lequel il est détruit (par exemple « 3 mois »). PAS le temps de fabrication.
- couts : chaque montant à payer, tel qu'écrit (par exemple « 86 € »), avec sa condition
  (par exemple « en France », « en Guyane », « à l'étranger ») et sa phrase de preuve

Fiche : {title}
{text}
"""

SCALARS = ("lieu", "pre_demande_en_ligne", "gratuit", "delai_retrait")
PROOF_OF = {"lieu": "lieu_preuve", "pre_demande_en_ligne": "pre_demande_preuve",
            "gratuit": "gratuit_preuve", "delai_retrait": "delai_retrait_preuve"}


@dataclass
class Extracted:
    id: str
    title: str
    url: str
    fields: dict = field(default_factory=dict)  # name -> {"value", "proof"}; couts: list
    rejected: list[dict] = field(default_factory=list)  # {"field", "value", "reason"}


# ---------------------------------------------------------------- verification
def _norm(s: str) -> str:
    s = s.replace("’", "'").replace(" ", " ").replace("\xa0", " ")
    s = re.sub(r"^[-•]\s*", "", s.strip())
    return re.sub(r"\s+", " ", s).strip().lower()


def _numbers(s: str) -> set[str]:
    return {n.replace(".", ",") for n in re.findall(r"\d+(?:[.,]\d+)?", s)}


def _content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-zà-ÿ]{4,}", _norm(s))}


def proof_problem(proof: str, source: str) -> str | None:
    p = _norm(proof)
    if len(p.split()) < MIN_PROOF_WORDS:
        return f"proof shorter than {MIN_PROOF_WORDS} words"
    if p not in _norm(source):
        return "proof is not a sentence of the sheet"
    return None


def value_problem(value, proof: str) -> str | None:
    if isinstance(value, bool):
        return None  # a yes/no has nothing to match; the proof and the eval carry it
    text = str(value)
    if not _numbers(text) <= _numbers(proof):
        return "a number of the value is not in its proof"
    if not _numbers(text) and not (_content_words(text) & _content_words(proof)):
        return "the value shares no word with its proof"
    return None


def _sections(source) -> list[tuple[str, str]]:
    return [("", source)] if isinstance(source, str) else list(source)


def as_text(sections: list[tuple[str, str]]) -> str:
    """What the model reads: each section under its heading, so it can tell the 86 € in
    France from the 96 € abroad (the amount's own sentence often says neither)."""
    return "\n".join(f"## {h}\n{t}" if h else t for h, t in sections)


def _heading_of(proof: str, sections: list[tuple[str, str]]) -> str:
    p = _norm(proof)
    return next((h for h, t in sections if p in _norm(t)), "")


def verify(raw: dict, source, doc_id: str = "", title: str = "",
           url: str = "") -> Extracted:
    """*source*: the sheet's text, or its (heading, text) sections (headings let a cost's
    condition be checked against the section it comes from)."""
    sections = _sections(source)
    text = as_text(sections)
    out = Extracted(doc_id, title, url)
    for name in SCALARS:
        value, proof = raw.get(name), raw.get(PROOF_OF[name]) or ""
        if value is None:
            out.fields[name] = {"value": None, "proof": ""}
            continue
        why = proof_problem(proof, text) or value_problem(value, proof)
        if why:
            out.rejected.append({"field": name, "value": value, "proof": proof, "reason": why})
            out.fields[name] = {"value": None, "proof": ""}
        else:
            out.fields[name] = {"value": value, "proof": proof}
    kept = []
    for item in raw.get("couts") or []:
        amount, proof = item.get("montant", ""), item.get("preuve", "")
        why = (proof_problem(proof, text) or value_problem(amount, proof)
               or (None if "€" in proof or "euro" in _norm(proof) else "no amount in proof"))
        if why:
            out.rejected.append({"field": "couts", "value": amount, "proof": proof,
                                 "reason": why})
            continue
        # The condition ("à l'étranger") must be said by the proof or by the heading of
        # the section the proof sits in; otherwise it is left empty, never guessed.
        condition, heading = item.get("condition", ""), _heading_of(proof, sections)
        if condition and not (_content_words(condition)
                              <= _content_words(proof) | _content_words(heading)):
            out.rejected.append({"field": "couts.condition", "value": condition,
                                 "proof": proof, "reason": "condition not in its proof "
                                 "or in its section heading"})
            condition = ""
        kept.append({"montant": amount, "condition": condition, "section": heading,
                     "proof": proof})
    out.fields["couts"] = kept
    return out


# ---------------------------------------------------------------- the model call
def ask_model(title: str, text: str, settings: Settings, client=None) -> dict:
    preset = settings.preset
    body = {"model": preset.model_id, "stream": False, "format": SCHEMA,
            "prompt": PROMPT.format(title=title, text=text),
            **ollama_options(preset, num_ctx=NUM_CTX)}
    body["options"]["num_predict"] = 768
    http = client or httpx.Client(timeout=300.0)
    r = http.post(f"{settings.ollama_url}/api/generate", json=body)
    r.raise_for_status()
    try:
        return json.loads(r.json()["response"])
    except (KeyError, ValueError):
        return {}  # unparsable output: every field null, nothing invented


def windows(sections: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """(heading, window text, whole section text): paragraph-packed windows of at most
    WINDOW_WORDS words. A proof is checked against its whole section, so a sentence
    split across two windows is not lost to the check."""
    from finrag.chunk import split_words

    return [(h, w, t) for h, t in sections if t.strip()
            for w in split_words(t, max_words=WINDOW_WORDS, overlap=40)]


def merge(parts: list[Extracted], doc_id: str, title: str, url: str) -> Extracted:
    """One record per sheet: the first verified value of each field in reading order
    (the sheet's general case comes first), and every verified cost once."""
    out = Extracted(doc_id, title, url)
    for name in SCALARS:
        out.fields[name] = next((p.fields[name] for p in parts
                                 if p.fields[name]["value"] is not None),
                                {"value": None, "proof": ""})
    seen, costs = set(), []
    for p in parts:
        for c in p.fields["couts"]:
            key = (min(_numbers(c["montant"]) or {""}), _norm(c["condition"]))
            if key not in seen:
                seen.add(key)
                costs.append(c)
    out.fields["couts"] = costs
    out.rejected = [r for p in parts for r in p.rejected]
    return out


def extract(doc, settings: Settings, client=None, pause: float = 0.0) -> Extracted:
    """pause: seconds to wait between two model calls, so a laptop GPU can cool down."""
    parts = []
    for k, (h, w, t) in enumerate(windows(doc.sections)):
        if k and pause:
            time.sleep(pause)
        parts.append(verify(ask_model(doc.title, as_text([(h, w)]), settings, client), [(h, t)]))
    return merge(parts, doc.id, doc.title, doc.url)


def to_json(e: Extracted) -> str:
    return json.dumps(asdict(e), ensure_ascii=False, indent=1)


def from_json(text: str) -> Extracted:
    """The inverse of to_json: a cached result reads back exactly."""
    return Extracted(**json.loads(text))


# ---------------------------------------------------------------- evaluation
def score(results: dict[str, Extracted], gold: list[dict]) -> dict:
    """Field accuracy against gold read off the sheets; amounts as recall and precision.

    A field counts as right when the kept value equals the gold (text: gold contained in
    the value, case-insensitive). `unsupported` counts kept values that the gold says
    are wrong: the number that must stay at 0 for the data to be trusted.
    """
    right = {n: 0 for n in SCALARS}
    total = {n: 0 for n in SCALARS}
    wrong_kept = 0
    found = expected = kept_amounts = good_amounts = 0
    for g in gold:
        e = results[g["id"]]
        for n in SCALARS:
            if n not in g:
                continue
            total[n] += 1
            got, want = e.fields[n]["value"], g[n]
            if isinstance(want, str) and isinstance(got, str):
                ok = _norm(want) in _norm(got)
            else:
                ok = got == want
            right[n] += ok
            wrong_kept += (not ok) and got is not None
        want_amounts = {_numbers(a).pop() for a in g.get("montants", [])}
        got_amounts = {next(iter(_numbers(c["montant"])), "") for c in e.fields["couts"]}
        expected += len(want_amounts)
        found += len(want_amounts & got_amounts)
        kept_amounts += len(got_amounts)
        good_amounts += len(got_amounts & want_amounts)
    acc = {n: round(right[n] / total[n], 3) if total[n] else None for n in SCALARS}
    return {
        "n_sheets": len(gold),
        "field_accuracy": acc,
        "amount_recall": round(found / expected, 3) if expected else None,
        "amount_precision": round(good_amounts / kept_amounts, 3) if kept_amounts else None,
        "unsupported_values_kept": wrong_kept + (kept_amounts - good_amounts),
        "rejected_by_verification": sum(len(e.rejected) for e in results.values()),
    }
