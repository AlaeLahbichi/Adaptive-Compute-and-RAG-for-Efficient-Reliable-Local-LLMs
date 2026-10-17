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
| II. Benchmark & protocole | Jeu de questions, métriques, critères de succès | ⬜ à venir |
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

### Recompiler le PDF

```bash
cd livrable
latexmk -pdf L1_etat_de_l_art.tex
```
