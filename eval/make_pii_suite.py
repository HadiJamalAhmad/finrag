"""Writes eval/suites/pii-fr.jsonl: French questions to an administration assistant, with the
personal data each one holds (kind + exact value), and look-alikes that hold none.

Synthetic on purpose: no real person's data may sit in a repository. Every number is
invented but VALID (NIR key, IBAN mod 97, Luhn), so a detector cannot pass by checking
nothing; look-alikes carry wrong keys. Fixed seed: the file is reproducible.
    python eval/make_pii_suite.py
"""
import json
import random
from pathlib import Path

R = random.Random(20261123)
FIRST = ["Camille", "Julien", "Nadia", "Thomas", "Fatima", "Hugo", "Léa", "Karim", "Inès",
         "Mathieu", "Sophie", "Youssef", "Chloé", "Antoine", "Aïcha", "Nicolas", "Manon",
         "Mehdi", "Claire", "Pierre-Louis"]
LAST = ["Martin", "Bernard", "Haddad", "Lefèvre", "Nguyen", "Moreau", "Benali", "Garnier",
        "Rousseau", "Dubois", "Traoré", "Lambert", "Fontaine", "Mercier", "Diallo", "Girard"]
TOPICS = ["renouveler mon passeport", "refaire ma carte d'identité", "changer l'adresse de ma "
          "carte grise", "obtenir un extrait de casier judiciaire", "déclarer la perte de mon "
          "permis de conduire", "demander un titre de séjour", "faire une procuration de vote"]


def nir() -> str:
    body = (f"{R.choice('12')}{R.randint(40, 99):02d}{R.randint(1, 12):02d}"
            f"{R.choice(['75', '78', '92', '93', '13', '69'])}{R.randint(1, 999):03d}"
            f"{R.randint(1, 999):03d}")
    key = 97 - int(body) % 97
    return f"{body[0]} {body[1:3]} {body[3:5]} {body[5:7]} {body[7:10]} {body[10:13]} {key:02d}"


def iban() -> str:
    bban = "".join(str(R.randint(0, 9)) for _ in range(23))
    check = 98 - int("".join(str(int(c, 36)) for c in bban + "FR00")) % 97
    s = f"FR{check:02d}{bban}"
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


def visa() -> str:
    d = [4] + [R.randint(0, 9) for _ in range(14)]
    total = sum((x * 2 - 9 if x * 2 > 9 else x * 2) if i % 2 == 0 else x
                for i, x in enumerate(reversed(d)))
    d.append((10 - total % 10) % 10)
    s = "".join(map(str, d))
    return " ".join(s[i:i + 4] for i in range(0, 16, 4))


def phone() -> str:
    return f"0{R.choice('67')} " + " ".join(f"{R.randint(0, 99):02d}" for _ in range(4))


def person() -> tuple[str, str, str]:
    f, last = R.choice(FIRST), R.choice(LAST)
    mail = (f"{f}.{last}".lower().replace("é", "e").replace("è", "e").replace("ï", "i")
            .replace("ë", "e") + "@example.fr")
    return f, last, mail


cases = []


def add(text, *pii):
    cases.append({"id": f"pii-{len(cases) + 1:03d}", "text": text,
                  "pii": [{"kind": k, "value": v} for k, v in pii]})


for i in range(12):
    f, last, mail = person()
    full, n, t = f"{f} {last}", nir(), R.choice(TOPICS)
    form = i % 4
    if form == 0:
        add(f"Bonjour, je m'appelle {full}, comment {t} ?", ("NAME", full))
    elif form == 1:
        add(f"Je suis {full}, mon numéro de sécurité sociale est {n}. Comment {t} ?",
            ("NAME", full), ("NIR", n))
    elif form == 2:
        add(f"Pour {t}, vous pouvez me joindre au {(phone_ := phone())} ou à {mail}. {full}",
            ("PHONE", phone_), ("EMAIL", mail), ("NAME", full))
    else:
        add(f"Mon fils {f} {last} doit {t}. Mon IBAN pour le remboursement : {(ib := iban())}.",
            ("NAME", full), ("IBAN", ib))

for _ in range(4):
    add(f"Le paiement par carte {(v := visa())} a été refusé sur le site de l'ANTS, que faire ?",
        ("CARD", v))
for _ in range(3):
    add(f"Mon NIR {(n := nir())} est-il obligatoire pour la demande ?", ("NIR", n))
    n2 = nir().replace(" ", "")
    add(f"NIR sans espaces : {n2}, est-ce correct ?", ("NIR", n2))

# Hard cases for names: a lowercase name, a name after a title, a single family name.
add("bonjour je suis julien moreau et j'ai perdu mon passeport", ("NAME", "julien moreau"))
add("Madame Haddad souhaite savoir si sa demande est complète.", ("NAME", "Haddad"))
add("Mme Nadia Benali, née le 3 mars 1990, demande un titre de séjour.",
    ("NAME", "Nadia Benali"))
add("Mon voisin, M. Thomas Girard, peut-il signer la procuration ?", ("NAME", "Thomas Girard"))

# Look-alikes: nothing personal here.
bad_nir = nir()
bad_nir = bad_nir[:-2] + f"{(int(bad_nir[-2:]) % 97) + 1:02d}"
for text in [
    f"Le numéro {bad_nir} est-il un numéro de sécurité sociale valide ?",
    "Le timbre fiscal pour un passeport adulte coûte 86 euros.",
    "Appelez le 3939 pour joindre Allô Service Public.",
    "Le formulaire cerfa n° 12100*03 est-il encore valable ?",
    "La préfecture des Yvelines est-elle ouverte le samedi ?",
    "Quel est le délai pour une carte d'identité à Paris en mars 2026 ?",
    "Mon code postal est 78300, quelle mairie dois-je contacter ?",
    "L'amende de 135 euros peut-elle être minorée à 90 euros ?",
    "Le SIRET 732 829 320 00074 correspond-il à une entreprise ?",
    "Que dit l'article L. 121-1 du code de la route ?",
    "Le ministère de l'Intérieur publie-t-il les délais par département ?",
    "Victor Hugo est-il le nom de la rue de la mairie ?",
]:
    add(text)

out = Path(__file__).parent / "suites" / "pii-fr.jsonl"
out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
print(out, len(cases))
