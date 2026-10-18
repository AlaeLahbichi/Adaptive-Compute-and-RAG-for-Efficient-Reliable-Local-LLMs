"""Routeurs de référence (Partie III), évalués par validation croisée sur le jeu de DÉVELOPPEMENT.

R1 – Routeur « question seule » (contrôle) : pour chaque stratégie A–F, une régression ridge prédit le
     score à partir du seul texte de la question ; le routeur choisit argmax(score prédit − λ × latence).
     Faire varier λ donne une courbe qualité–coût.
R2 – Adaptive-RAG simplifié (Jeong et al., 2024) : un classifieur de complexité choisit, avec le grand
     modèle, entre aucun passage (A), 3 passages (B) ou 6 passages (E). Étiquette d'une question : la
     stratégie la plus simple qui répond correctement ; à défaut, biais du jeu de données (SQuAD → B,
     HotpotQA → E), comme dans l'article original.

La génération étant déterministe, la sortie d'un routeur pour une question est celle de la stratégie
choisie (sorties pré-calculées par run_strategies.py). Le temps de calcul du routeur est mesuré et ajouté.

Usage :
  python bench/routers.py dev
"""
import json
import re
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from common import DONNEES, RESULTATS, charger_config, lire_jsonl
from metrics import bootstrap_difference, bootstrap_ic

STRATEGIES = ["A_grand_sans_rag", "B_grand_rag", "C_petit_sans_rag", "D_petit_rag", "E_grand_rag6", "F_petit_rag6"]
ADAPTIVE_RAG = ["A_grand_sans_rag", "B_grand_rag", "E_grand_rag6"]
REFERENCE = "B_grand_rag"
MOTS_QU = ["what", "who", "when", "where", "which", "how", "why", "whose", "is", "are", "did", "was"]
LAMBDAS = np.round(np.concatenate([[0.0], np.geomspace(0.01, 2.0, 25)]), 4)


class Caracteristiques:
    """Caractéristiques calculées à partir du seul texte de la question."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    @staticmethod
    def _manuelles(questions):
        lignes = []
        for q in questions:
            mots = q.split()
            premier = mots[0].lower() if mots else ""
            lignes.append([
                len(mots) / 20,
                sum(m[:1].isupper() for m in mots[1:]) / 5,
                float(bool(re.search(r"\d", q))),
                float(" and " in q.lower() or " both " in q.lower()),
                *[float(premier == m) for m in MOTS_QU],
            ])
        return csr_matrix(np.array(lignes))

    def fit_transform(self, questions):
        return hstack([self.tfidf.fit_transform(questions), self._manuelles(questions)]).tocsr()

    def transform(self, questions):
        return hstack([self.tfidf.transform(questions), self._manuelles(questions)]).tocsr()


def charger(split):
    questions = {q["id"]: q for q in lire_jsonl(DONNEES / f"{split}.jsonl")}
    sorties = {}
    for l in lire_jsonl(RESULTATS / "partie3" / split / "sorties.jsonl"):
        sorties.setdefault(l["id"], {})[l["strategie"]] = l
    ids = sorted(sorties)
    return [questions[i] for i in ids], [sorties[i] for i in ids]


def etiquette_adaptive_rag(question, sortie):
    for s in ADAPTIVE_RAG:
        if sortie[s]["correct"]:
            return s
    return "E_grand_rag6" if question["source"] == "hotpotqa" else "B_grand_rag"


def evaluer(choix, sorties, questions, surcout_s):
    """Résultats d'une politique (liste des stratégies choisies, une par question)."""
    lignes = [sorties[i][s] for i, s in enumerate(choix)]
    scores = np.array([l["score"] for l in lignes])
    latences = np.array([l["latence_s"] for l in lignes]) + surcout_s
    repondues = [l for l in lignes if not l["abstenu"]]
    bas, haut = bootstrap_ic(scores)
    ref = np.array([s[REFERENCE]["score"] for s in sorties])
    d_bas, d_haut = bootstrap_difference(scores, ref)
    cats = sorted({q["categorie"] for q in questions})
    return {
        "score_global": round(float(scores.mean()), 4), "ic95": [round(bas, 4), round(haut, 4)],
        "ecart_vs_B_points": round(100 * float(scores.mean() - ref.mean()), 2),
        "ecart_vs_B_ic95_points": [round(100 * d_bas, 2), round(100 * d_haut, 2)],
        "latence_moyenne_s": round(float(latences.mean()), 4),
        "latence_p95_s": round(float(np.quantile(latences, 0.95)), 4),
        "reduction_latence_vs_B": round(1 - float(latences.mean()) / float(np.mean(
            [s[REFERENCE]["latence_s"] for s in sorties])), 4),
        "tokens_moyens": round(float(np.mean([l["tokens_prompt"] + l["tokens_generes"] for l in lignes])), 1),
        "couverture": round(len(repondues) / len(lignes), 4),
        "risque": round(sum(not l["correct"] for l in repondues) / max(len(repondues), 1), 4),
        "par_categorie": {c: round(float(np.mean([scores[i] for i, q in enumerate(questions)
                                                   if q["categorie"] == c])), 4) for c in cats},
        "choix": {s: choix.count(s) for s in STRATEGIES if s in choix},
    }


def validation_croisee(questions, sorties, cfg, n_plis=5):
    """Prédictions hors pli des deux routeurs, et temps de calcul moyen par question."""
    textes = [q["question"] for q in questions]
    y_score = np.array([[s[st]["score"] for st in STRATEGIES] for s in sorties])
    y_ar = np.array([etiquette_adaptive_rag(q, s) for q, s in zip(questions, sorties)])
    strates = [q["categorie"] for q in questions]
    pred_scores = np.zeros_like(y_score)
    pred_ar = np.empty(len(questions), dtype=object)
    lat_strategies = np.zeros((n_plis, len(STRATEGIES)))
    temps_r1, temps_r2 = [], []
    plis = StratifiedKFold(n_plis, shuffle=True, random_state=cfg["graine"]).split(textes, strates)
    for p, (app, val) in enumerate(plis):
        feats = Caracteristiques()
        X_app = feats.fit_transform([textes[i] for i in app])
        ridges = [Ridge(alpha=1.0).fit(X_app, y_score[app, j]) for j in range(len(STRATEGIES))]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_app, y_ar[app])
        lat_strategies[p] = [np.mean([sorties[i][st]["latence_s"] for i in app]) for st in STRATEGIES]
        for i in val:
            debut = time.perf_counter()
            x = feats.transform([textes[i]])
            pred_scores[i] = [r.predict(x)[0] for r in ridges]
            temps_r1.append(time.perf_counter() - debut)
            debut = time.perf_counter()
            pred_ar[i] = clf.predict(feats.transform([textes[i]]))[0]
            temps_r2.append(time.perf_counter() - debut)
    return pred_scores, list(pred_ar), lat_strategies.mean(axis=0), float(np.mean(temps_r1)), float(np.mean(temps_r2)), y_ar


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if split != "dev":
        raise SystemExit("Les routeurs sont réglés sur le jeu de développement uniquement.")
    cfg = charger_config()
    questions, sorties = charger(split)
    pred_scores, pred_ar, lat_moy, t_r1, t_r2, y_ar = validation_croisee(questions, sorties, cfg)

    resultats = {"n_questions": len(questions), "surcout_routeur_ms": {"R1": round(1000 * t_r1, 3),
                                                                         "R2": round(1000 * t_r2, 3)},
                 "strategies_fixes": {}, "R1_question_seule": {"courbe": []}}
    for st in STRATEGIES:
        resultats["strategies_fixes"][st] = evaluer([st] * len(questions), sorties, questions, 0.0)

    meilleurs = []
    for i, s in enumerate(sorties):
        top = max(v["score"] for v in s.values())
        meilleurs.append(min((st for st in STRATEGIES if s[st]["score"] == top), key=lambda st: s[st]["latence_s"]))
    resultats["oracle"] = evaluer(meilleurs, sorties, questions, 0.0)

    for lam in LAMBDAS:
        choix = [STRATEGIES[int(np.argmax(p - lam * lat_moy))] for p in pred_scores]
        resultats["R1_question_seule"]["courbe"].append({"lambda": float(lam), **evaluer(choix, sorties, questions, t_r1)})

    # Point de fonctionnement rapporté : meilleur score parmi les réglages qui respectent la contrainte de
    # latence du critère principal (au moins 20 % plus rapide que B).
    admissibles = [p for p in resultats["R1_question_seule"]["courbe"]
                   if p["reduction_latence_vs_B"] >= cfg["criteres_de_succes"]["reduction_latence_moyenne_min"]]
    resultats["R1_question_seule"]["point_retenu"] = max(admissibles, key=lambda p: p["score_global"]) if admissibles else None

    resultats["R2_adaptive_rag"] = evaluer(pred_ar, sorties, questions, t_r2)
    resultats["R2_adaptive_rag"]["precision_classifieur"] = round(float(np.mean(np.array(pred_ar) == y_ar)), 4)
    resultats["R2_adaptive_rag"]["repartition_etiquettes"] = {s: int((y_ar == s).sum()) for s in ADAPTIVE_RAG}

    sortie = RESULTATS / "partie3" / "routeurs.json"
    sortie.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")

    def ligne(nom, r):
        print(f"{nom:24s} score={r['score_global']:.3f} [{r['ic95'][0]:.3f};{r['ic95'][1]:.3f}] "
              f"écart_B={r['ecart_vs_B_points']:+.1f}pts {r['ecart_vs_B_ic95_points']} "
              f"latence={r['latence_moyenne_s']:.3f}s (−{100 * r['reduction_latence_vs_B']:.0f}%) "
              f"couv={r['couverture']:.2f} risque={r['risque']:.2f}")
    for st, r in resultats["strategies_fixes"].items():
        ligne(st, r)
    ligne("Oracle", resultats["oracle"])
    if resultats["R1_question_seule"]["point_retenu"]:
        ligne("R1 question seule", resultats["R1_question_seule"]["point_retenu"])
        print("   choix R1 :", resultats["R1_question_seule"]["point_retenu"]["choix"])
    ligne("R2 Adaptive-RAG simplifié", resultats["R2_adaptive_rag"])
    print("   choix R2 :", resultats["R2_adaptive_rag"]["choix"], "précision :", resultats["R2_adaptive_rag"]["precision_classifieur"])
    print("surcoût routeurs (ms) :", resultats["surcout_routeur_ms"])


if __name__ == "__main__":
    main()
