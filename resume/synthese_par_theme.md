# Synthèse de la littérature par thème

**Projet :** Allocation adaptative des ressources de calcul et de recherche documentaire pour des systèmes LLM locaux, économes et fiables
**Corpus :** 43 références (collection Zotero du projet), état au 14/10/2026
**Statut :** pré-analyse faite à partir des résumés. Elle est **à confirmer par la lecture** des articles (passes 2 et 3).

> Fichier associé : `tableau_de_lecture.csv` (s'ouvre dans Excel ou Google Sheets, séparateur `;`).
> Dans le tableau, un `?` signale une information de mémoire à vérifier dans l'article.

---

## Vue d'ensemble en une phrase

Le **routage** (quel modèle ?), le **RAG adaptatif** (faut-il chercher, et combien ?) et l'**abstention** (faut-il répondre ?) sont trois domaines actifs, mais **étudiés séparément**. Ils mesurent presque toujours le coût en **dollars d'API** ou en nombre d'appels, rarement en **énergie, mémoire ou latence sur machine locale**.

| # | Thème | Nb | Question du cahier des charges | Articles à lire en priorité |
|---|---|---|---|---|
| 1 | Routage & cascades de LLM | 11 | QR1, QR3 | Survey Varangot-Reille, FrugalGPT, Hybrid LLM, RouteLLM, AutoMix |
| 2 | RAG adaptatif | 12 | QR1, QR2 | **Moskvoretskii 2025**, **Adaptive-RAG**, Self-RAG, Corrective RAG, Mallen |
| 3 | Réduction du contexte RAG | 2 | QR3 | LeanContext |
| 4 | Incertitude & abstention | 6 | QR2 | Survey Wen (abstention), Kuhn (entropie sémantique) |
| 5 | Efficacité & énergie | 3 | QR3 | Samsi (From Words to Watts) |
| 6 | Benchmarks & données | 4 | QR4 | SQuAD 2.0, HotpotQA |
| 7 | Périphérique | 5 | — | à ne pas lire en détail |

---

## 1. Routage & cascades de LLM : « quel modèle pour quelle question ? »

**Idée commune :** inutile d'envoyer toutes les questions au plus gros modèle. On choisit le modèle selon la question.

**Trois familles :**
- **Routage avant génération** : on prédit la difficulté puis on choisit le modèle. *Hybrid LLM* (Ding 2024) et *RouteLLM* (Ong 2024) entraînent un classifieur qui décide entre un petit et un grand modèle. Résultats : jusqu'à 40 % d'appels en moins au grand modèle (Hybrid LLM), coût divisé par plus de 2 (RouteLLM), sans perte de qualité.
- **Cascade après génération** : on essaie le petit modèle et on escalade s'il paraît incertain. Exemples : *FrugalGPT* (Chen 2023, jusqu'à −98 % de coût), *AutoMix* (auto-vérification + POMDP, −50 %), *Mixture of Thoughts* (cohérence des réponses du petit modèle, 40 % du coût). Inconvénient : le coût du petit modèle est toujours payé.
- **Autres approches** : bandits (*MetaLLM*), routage par expertise (*Zooter*), benchmark standardisé (*RouterBench*).

**⚠️ Concurrent récent à lire en détail :** *Not All Tokens Are Equal* (Fu, août 2026). Il calcule un signal de difficulté **localement** avant exécution et route selon le **coût réel**. Il est très proche de ton sujet, mais ne semble pas traiter le RAG ni l'abstention : à confirmer.

**Ce qui manque (opportunité pour toi) :**
- Le coût est presque toujours exprimé en **$ d'API cloud**, pas en énergie, mémoire ou latence sur une **machine locale**. La survey de Varangot-Reille (2025) le signale elle-même : « accounting for non-financial costs ».
- Le routage choisit **seulement le modèle**. Il ne décide ni du RAG ni de l'abstention.

---

## 2. RAG adaptatif : « faut-il chercher dans les documents, et combien ? »

**Idée commune :** le RAG systématique coûte cher et peut ajouter du bruit. On ne cherche que lorsque c'est utile.

**Les signaux de décision existants :**
| Signal | Article | Remarque |
|---|---|---|
| Popularité de l'entité | Mallen 2022 | Article fondateur : le RAG aide surtout sur les connaissances rares |
| Complexité de la question (classifieur) | **Adaptive-RAG** (Jeong 2024) | Choisit entre pas de RAG, RAG simple et RAG itératif. **Code disponible** |
| Tokens de réflexion appris | Self-RAG (Asai 2023) | Nécessite d'entraîner le modèle |
| Qualité des documents récupérés | **Corrective RAG** (Yan 2024) | Évaluateur léger : utiliser, corriger ou ignorer |
| Confiance des tokens générés | FLARE, DRAGIN | Récupération pendant la génération |
| Connaissance de soi | SKR | Comparaison avec des questions déjà vues |

**⭐ Résultat clé pour ton projet :** *Moskvoretskii et al. (ACL 2025)* comparent 35 méthodes sur 6 jeux de données. Les **estimateurs d'incertitude simples font souvent aussi bien ou mieux que les pipelines complexes**, pour un coût moindre. Cela justifie de partir de **signaux simples et interprétables**, comme le demande le cahier des charges (Phase 4).

**Cas d'application :** HyPA-RAG (juridique) et LogicRAG (questions multi-étapes) adaptent les paramètres de récupération à la complexité. WeKnow-RAG ajoute une auto-évaluation.

**Ce qui manque :**
- Ces méthodes utilisent **un seul modèle**. Elles ne choisissent pas sa taille.
- L'**abstention** est rarement une sortie possible.
- L'efficacité est mesurée en **nombre d'appels ou de récupérations**, rarement en énergie ou mémoire réelles.

---

## 3. Réduction du contexte RAG : « combien de passages donner au modèle ? »

- *LeanContext* (Arefeen 2024) garde les k phrases clés, avec k choisi par apprentissage par renforcement. Résultat : **−37 à −68 % de coût pour −1,4 à −2,7 % de ROUGE**. C'est exactement le type de compromis chiffré visé par ton critère de succès (−20 % de coût pour moins de 2 points de qualité perdus).
- *Clustered RAG* (Akesson 2024) réduit de 46 à 90 % les tokens par regroupement. ⚠️ Il porte aussi le nom « CRAG », à ne pas confondre avec Corrective RAG.

**Pour toi :** le nombre de passages récupérés (top-k) peut devenir **une dimension de décision supplémentaire** de ta politique.

---

## 4. Incertitude & abstention : « le modèle sait-il qu'il ne sait pas ? »

**Mesurer l'incertitude :**
- **Probabilité interne** : *Kadavath 2022* propose P(True), la probabilité que la réponse soit vraie, et P(IK), la probabilité que le modèle connaisse la réponse. P(IK) **augmente quand des sources pertinentes sont fournies**, ce qui en fait un signal possible pour décider de chercher davantage.
- **Cohérence entre plusieurs réponses** : *Semantic Entropy* (Kuhn 2023) et *SelfCheckGPT* (Manakul 2023). Ces méthodes sont efficaces mais **coûteuses**, car elles exigent plusieurs générations par question.
- **Confiance exprimée en mots** : *Lin 2022*. Elle est calibrée seulement après fine-tuning. Le cahier des charges demande de ne pas s'y fier aveuglément.

**S'abstenir :**
- *Kamath 2020* définit le cadre **couverture / risque** : répondre au plus grand nombre de questions tout en gardant une précision élevée. C'est la métrique d'abstention à utiliser.
- *Wen 2024* propose une survey de l'abstention (méthodes, benchmarks, métriques). C'est la **référence centrale pour QR2**.

**Ce qui manque :** le **coût de l'estimation d'incertitude** elle-même est rarement intégré dans la décision. Échantillonner 5 réponses pour mesurer l'incertitude peut coûter plus cher que d'appeler directement le grand modèle. **C'est une question de recherche intéressante pour toi.**

---

## 5. Efficacité & énergie : « combien ça coûte vraiment ? »

- *Samsi 2023 (From Words to Watts)* mesure l'énergie de l'inférence LLaMA selon la taille du modèle sur GPU. C'est la **méthode de mesure à adapter** à ta machine locale.
- *Luccioni 2022 (BLOOM)* et *Luccioni 2023 (Power Hungry Processing, à ajouter)* fournissent la motivation « IA frugale » pour l'introduction.

**Ce qui manque :** ces travaux **mesurent** l'énergie mais **ne l'optimisent pas** par une décision par requête. Et ils portent sur du matériel de datacenter, pas sur un ordinateur personnel.

---

## 6. Benchmarks & données : « sur quoi tester ? »

Ils couvrent les 4 catégories exigées par le cahier des charges (Phase 2) :
| Catégorie de question | Jeu de données |
|---|---|
| Simple | SQuAD (partie avec réponse) |
| Complexe / multi-étapes | HotpotQA (avec passages justificatifs), MuSiQue |
| Non couverte par les documents | SQuAD 2.0, MuSiQue-Full |
| Types variés + pénalité d'hallucination | CRAG Benchmark (Meta) |

Adaptive-RAG utilise déjà SQuAD, HotpotQA et MuSiQue : **reprendre les mêmes jeux permet une comparaison directe** avec lui.

---

## 7. Périphérique (ne pas lire en détail)
ThunderServe (cloud), BudgetMLAgent (agents), Cambricon-LLM (matériel), EBRB-LLM (article de position), thèse ML Marketplaces. Une phrase dans le Related Work au maximum.

---

## 🎯 Le manque identifié et la contribution possible

**Ce que fait la littérature :**
```
Routage (modèle)        ─┐
RAG adaptatif (chercher) ├─  étudiés SÉPARÉMENT, coût en $ API ou nb d'appels
Abstention (répondre ?) ─┘
```

**Ce que ton projet pourrait faire :**
> **Une politique de décision unique qui choisit *conjointement* la taille du modèle, le recours et l'ampleur du RAG, et l'abstention, à partir de signaux d'incertitude peu coûteux, évaluée sur du *matériel local* en mesurant qualité, factualité, latence, mémoire et énergie.**

**Les travaux à battre ou dont il faut se distinguer clairement :**
1. **Adaptive-RAG** : décide du RAG selon la complexité, mais ne choisit pas le modèle et ne s'abstient pas.
2. **Moskvoretskii 2025** : signaux d'incertitude pour le RAG, avec un seul modèle et sans énergie.
3. **Hybrid LLM / RouteLLM** : choix du modèle, sans RAG ni abstention.
4. **Not All Tokens Are Equal (2026)** : signal local et coût réel. À lire en priorité pour vérifier qu'il ne couvre pas déjà ton idée.

## ⚠️ À faire avant de valider cette contribution
1. **Vérifier qu'aucun travail 2025-2026 ne combine déjà routage + RAG + abstention.** Recherches à lancer : `"joint model routing and retrieval"`, `"retrieval-aware LLM routing"`, `"RAG router small large model"`, `"cost-aware adaptive RAG abstention"`, `"on-device LLM energy routing"`.
2. Lire en détail (passe 3) : Moskvoretskii, Adaptive-RAG, Hybrid LLM, Not All Tokens Are Equal, Corrective RAG.
3. Compléter les colonnes « Limites » du tableau **après lecture**. La pré-analyse actuelle vient des résumés.
4. Faire valider la phrase de contribution par ton encadrant.
