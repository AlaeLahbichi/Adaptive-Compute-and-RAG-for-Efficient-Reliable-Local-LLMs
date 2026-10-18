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
| **II. Benchmark & protocole** | Jeu de questions, métriques, critères de succès, pilote | ✅ gelé (`protocole-v1.0`) |
| **III. Méthodes de référence** | Choix de la recherche, stratégies fixes A–F, routeurs R1 et R2 | ✅ |
| **IV. Politique adaptative** | Routeur conjoint modèle × RAG × abstention | ✅ gelée (`politique-v1.0`) |
| **V. Évaluation finale** | Test unique, verdict, ablations, énergie | ✅ (critère principal non atteint) |
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

## Partie III : Méthodes de référence

```
livrable/L3_methodes_de_reference.pdf   # Livrable L3 (7 p.)
livrable/figures/                       # Figure qualité–coût
bench/
├── compare_retrieval.py   # Compare recherche dense, BM25 et hybride (rappel@k, temps)
├── run_strategies.py      # Exécute les 6 stratégies fixes A–F
├── routers.py             # Routeurs de référence R1 (question seule) et R2 (Adaptive-RAG simplifié)
└── plot_partie3.py        # Figure qualité–coût
results/partie3/           # Comparaison des recherches, sorties des stratégies, résultats des routeurs
```

**Résultats (jeu de développement, validation croisée) :**

| Méthode | Score global | Latence moyenne | Écart à B |
|---|---|---|---|
| B – grand modèle + 3 passages BM25 (référence) | 0,541 | 0,92 s | – |
| R1 – routeur « question seule », réglé 20 % plus rapide que B | 0,425 | 0,74 s | −11,6 pts |
| R2 – Adaptive-RAG simplifié | 0,522 | 0,85 s | −1,9 pts |
| *Oracle* | *0,762* | *0,40 s* | *+22,1 pts* |

- La recherche **BM25** remplace la recherche dense : meilleur rappel et 40 fois plus rapide. Le protocole est gelé (étiquette `protocole-v1.0`).
- Aucun routeur fondé sur la seule question ne satisfait le critère de succès : la méthode adaptative devra utiliser d'autres signaux (qualité des preuves, incertitude du modèle).

**Modèles utilisés** (exécutés localement avec MLX sur Apple M3 Pro, 18 Go) : Qwen2.5-1.5B-Instruct et Qwen3-4B-Instruct-2507, quantifiés en 4 bits.

## Partie IV : Politique adaptative

```
livrable/L4_politique_adaptative.pdf   # Livrable L4 (7 p.)
configs/politique_v1.json              # Réglage figé de la politique (seuils)
bench/
├── politique.py        # Signaux (preuves BM25, confiance du modèle), exécution, simulation, réglage, ablations
└── plot_partie4.py     # Figure qualité–coût (dev ou test)
results/partie4/        # Sorties dev avec confiance du modèle, réglage et ablations
```

**Principe :** abstention précoce si les preuves BM25 sont trop faibles → grand modèle avec 3 ou 6 passages selon la force des preuves → abstention finale si la confiance du modèle est trop basse. Réglée sur le jeu de développement sous la contrainte « latence ≤ 78 % de B ».

| Méthode (dev) | Score global | Latence moyenne | Écart à B |
|---|---|---|---|
| B – référence | 0,541 | 0,92 s | – |
| Politique adaptative (réglage figé) | 0,552 | 0,71 s (−23 %) | +1,1 pts |
| Politique, estimation par validation croisée | 0,520 | −22 % | −2,1 pts |

L'abstention est la composante clé : sans elle, le score tombe à 0,39.

## Partie V : Évaluation finale (jeu de test, exécuté une seule fois)

```
livrable/L5_evaluation_finale.pdf   # Livrable L5 (8 p.)
bench/
├── run_final.py        # 3 passes sur le test (ordre des méthodes aléatoire), mesure d'énergie
└── evaluate_final.py   # Verdict, routeurs R1/R2 et ablations appliqués au test
results/partie5/        # Sorties brutes des 3 passes, énergie, evaluation_finale.json
```

| Méthode (test, 450 questions) | Score global | Latence moyenne | Énergie / question |
|---|---|---|---|
| B – référence | 0,537 | 0,87 s | 19,3 J |
| **Politique adaptative** | **0,550** (+1,3 pt) | **0,79 s (−10 %)** | **17,5 J (−9 %)** |
| R1 – routeur « question seule » | 0,489 | 0,74 s | – |
| R2 – Adaptive-RAG simplifié | 0,517 | 0,82 s | – |
| *Oracle* | *0,797* | *0,36 s* | – |

**Verdict :** qualité au niveau de B (non-infériorité démontrée, IC 95 % de la perte [−4,1 ; +1,6] pts), moins d'erreurs, et meilleure que R1 et R2 ; mais la latence ne baisse que de 10 % au lieu de 20 % : **le critère de succès principal n'est pas atteint**. Cause : le seuil d'abstention précoce, réglé sur 180 questions, se déclenche sur 25 % des questions de dev mais 15 % du test.

## Reproduire

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bench/build_benchmark.py              # télécharge les sources, vérifie leur empreinte, construit le benchmark
HF_HUB_OFFLINE=1 python bench/run_pilot.py   # pilote sur le jeu de développement
HF_HUB_OFFLINE=1 python bench/compare_retrieval.py
HF_HUB_OFFLINE=1 python bench/run_strategies.py dev
python bench/routers.py dev && python bench/plot_partie3.py
cd livrable && latexmk -pdf L2_cahier_experimental.tex
```
