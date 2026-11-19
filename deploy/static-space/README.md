---
title: finrag
emoji: 📄
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: IA documentaire sourcée sur 45 fiches service-public.gouv.fr
---

# finrag : IA documentaire sourcée (démonstration gratuite)

Une question de citoyen est cherchée dans 45 fiches officielles service-public.gouv.fr du
domaine de l'Intérieur (Licence Ouverte 2.0, DILA). Tout tourne dans le navigateur : le même
modèle d'embeddings que finrag, les vecteurs des 598 passages calculés par finrag, le même seuil
de preuve (0,61) et le même nombre de passages (3). La page montre les passages sources qu'un
modèle de langage recevrait, ou le refus.

Avant chaque publication, une vérification automatique compare la version navigateur du modèle
à finrag sur les 69 questions de ses deux suites d'évaluation : la précision retenue est la plus
légère qui donne exactement les mêmes décisions (répondre ou refuser) et dont l'écart de score
reste inférieur à la distance au seuil de la question la plus proche. L'écart mesuré est affiché
sur la page.

Code, tests et évaluations : https://github.com/HadiJamalAhmad/finrag
