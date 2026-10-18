# Adaptive Compute and RAG for Efficient & Reliable Local LLMs

**Allocation adaptative des ressources de calcul et de recherche documentaire pour des systèmes LLM locaux, économes et fiables**

Projet de recherche mené par **Alae Lahbichi**, étudiant en 5ᵉ année IA/Data (a.lahbichi2004@gmail.com).

## Problématique

Comment allouer automatiquement, et de manière mesurable, les ressources de calcul et de recherche documentaire à chaque requête, afin de réduire le coût d'un système LLM local sans dégrader significativement la qualité, la factualité et la fiabilité des réponses ?

Pour chaque question, le système doit choisir entre trois options : un modèle léger ou un modèle plus puissant, avec ou sans recherche documentaire (RAG), ou l'abstention si les preuves sont insuffisantes. Le compromis qualité / latence / mémoire / énergie est mesuré sur matériel local.

## Avancement

| Partie | Contenu | Statut |
|---|---|---|
| **I. État de l'art** | Revue de littérature, tableau de lecture, positionnement | ✅ v0.1 (pré-analyse à partir des résumés) |
| **II. Benchmark & protocole** | Jeu de questions, métriques, critères de succès, pilote | ✅ v1.0-pilote (gel après choix de la recherche) |
| III. Baselines | Grand modèle seul, RAG systématique, petit modèle seul | ⬜ à venir |
| IV. Politique adaptative | Routeur conjoint modèle × RAG × abstention | ⬜ à venir |
| V. Expériences & ablations | Comparaisons, analyse par catégorie | ⬜ à venir |
| VI. Article | Rédaction | ⬜ à venir |

## Partie I : État de l'art

```
livrable/
├── L1_etat_de_l_art.pdf     # Livrable L1 : état de l'art (12 p.)
├── L1_etat_de_l_art.tex     # Source LaTeX (compatible Overleaf)
└── references.bib           # Bibliographie (45 références)
resume/
├── tableau_de_lecture.csv   # Tableau de lecture : 43 articles, thème, priorité, méthode, limites…
├── synthese_par_theme.md    # Synthèse courte par thème
└── organiser_zotero.js      # Script de rangement de la bibliothèque Zotero
```

**Principaux constats de la revue :**
- Le **routage de modèles**, le **RAG adaptatif** et l'**abstention** sont presque toujours étudiés séparément.
- Le coût y est surtout exprimé en **prix d'API** ou en nombre d'appels, rarement en latence, mémoire ou énergie mesurées sur machine locale.
- Des travaux de 2026 (RAGRouter-Bench, Self-Routing RAG, RACER) réduisent cet écart : la contribution doit être délimitée précisément (voir section 10 du livrable).

> ⚠️ La version v0.1 a été rédigée à partir des résumés des articles. Les lectures complètes restent à faire avant toute réutilisation dans un article.

## Partie II : Benchmark et protocole

```
livrable/L2_cahier_experimental.pdf   # Livrable L2 : cahier expérimental (9 p.)
configs/protocole_v1.json             # Tous les paramètres figés du protocole
bench/
├── build_benchmark.py   # Construit le corpus et les questions dev/test depuis SQuAD 2.0 et HotpotQA
├── retrieval.py         # Recherche documentaire (embeddings MiniLM + similarité cosinus)
├── generation.py        # Génération locale avec MLX et prompts figés
├── metrics.py           # F1, exact match, abstention (couverture/risque), bootstrap
├── check_env.py         # Relevé du matériel, des paquets et des révisions de modèles
├── run_pilot.py         # Pilote : stratégies fixes A–D sur le jeu de développement
├── energie.py           # Mesure de l'énergie avec powermetrics (droits administrateur)
└── common.py            # Chemins et utilitaires partagés
data/
├── benchmark/           # corpus.jsonl (3 293 passages), dev.jsonl (180), test.jsonl (450), MANIFEST.json
└── LICENCE_DONNEES.md   # Données dérivées de SQuAD 2.0 et HotpotQA (CC BY-SA 4.0)
results/pilote/          # Sorties brutes et résumé du pilote
```

**Résultats du pilote (jeu de développement, 180 questions) :**

| Stratégie | Score global | Latence moyenne | Tokens | Énergie / question |
|---|---|---|---|---|
| A – grand modèle, sans RAG | 0,313 | 0,25 s | 62 | 3,8 J |
| B – grand modèle + RAG (référence) | 0,424 | 0,99 s | 579 | 22,8 J |
| C – petit modèle, sans RAG | 0,233 | 0,13 s | 84 | 1,9 J |
| D – petit modèle + RAG | 0,298 | 0,39 s | 601 | 8,6 J |
| *Oracle (meilleur choix par question)* | *0,623* | *0,30 s* | – | – |

Aucune stratégie fixe n'est la meilleure sur toutes les catégories : une politique adaptative a une marge réelle.

**Modèles utilisés** (exécutés localement avec MLX sur Apple M3 Pro, 18 Go) : Qwen2.5-1.5B-Instruct et Qwen3-4B-Instruct-2507, quantifiés en 4 bits.

## Reproduire

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bench/build_benchmark.py              # télécharge les sources, vérifie leur empreinte, construit le benchmark
HF_HUB_OFFLINE=1 python bench/run_pilot.py   # pilote sur le jeu de développement
cd livrable && latexmk -pdf L2_cahier_experimental.tex
```
