# Licence des données du benchmark

Les fichiers de `data/benchmark/` sont dérivés de deux jeux de données publics, tous deux sous licence
**Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)**. Ils sont redistribués sous la même licence.

| Source | Référence | Licence |
|---|---|---|
| SQuAD 2.0 (dev) | Rajpurkar, Jia & Liang (2018), *Know What You Don't Know: Unanswerable Questions for SQuAD*, ACL | CC BY-SA 4.0 |
| HotpotQA (dev, distractor) | Yang et al. (2018), *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP | CC BY-SA 4.0 |

Modifications : sélection aléatoire de questions (graine 2026), regroupement des paragraphes en un corpus
unique, ajout des champs `categorie`, `comportement_attendu` et `passages_gold`.
Les fichiers sources ne sont pas versionnés ; `bench/build_benchmark.py` les télécharge et vérifie leur empreinte SHA-256.
