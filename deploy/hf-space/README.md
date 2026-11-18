---
title: finrag
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: IA documentaire sourcée sur 45 fiches service-public.gouv.fr
---

# finrag : IA documentaire sourcée (démonstration gratuite)

Une question de citoyen passe par le vrai chemin de réponse de finrag (contrôle de périmètre,
recherche sémantique, seuil de preuve), sur 45 fiches officielles service-public.gouv.fr du
domaine de l'Intérieur (Licence Ouverte 2.0, DILA).

Ce Space gratuit tourne sur processeur, sans modèle de langage : quand la preuve suffit, il
montre les passages sources que le modèle recevrait, avec leurs liens ; sinon, il affiche le
refus et l'étape qui l'a décidé. Les réponses rédigées par le modèle (Llama 3.2 3B) tournent
sur Kubernetes, Modal ou un GPU.

Réglages figés : embeddings `paraphrase-multilingual-MiniLM-L12-v2`, seuil 0,61, 3 passages.
Code, tests et évaluations : https://github.com/HadiJamalAhmad/finrag
