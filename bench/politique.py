"""Politique adaptative conjointe (Partie IV) : taille du modèle × profondeur de recherche × abstention.

Déroulement pour une question :
  1. Recherche BM25 (6 passages) et signal de preuve (score du meilleur passage / score maximal).
     Si la preuve < seuil_preuve_abstention → abstention immédiate, sans aucune génération.
  2. (Si la cascade est active) le PETIT modèle répond avec les 3 meilleurs passages.
     S'il ne s'abstient pas et que sa confiance ≥ seuil_petit → on garde sa réponse.
  3. Sinon, escalade vers le GRAND modèle, avec 6 passages si la preuve < seuil_preuve, sinon 3.
  4. Si le grand modèle s'abstient ou que sa confiance < seuil_abstention → réponse UNANSWERABLE.

La confiance d'un modèle se calcule sur les probabilités des tokens qu'il génère (moyenne géométrique
« conf_moy » ou minimum « conf_min »). Tout le coût des signaux est dans la latence mesurée.

Réglage : sur le jeu de DÉVELOPPEMENT uniquement, en simulant la politique à partir des sorties
déterministes de chaque étape (stratégies D, B, E de run_strategies.py).

Usage :
  python bench/politique.py      # règle la politique, écrit configs/politique_v1.json et results/partie4/
"""
import itertools
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import CONFIG, DONNEES, RESULTATS, charger_config, lire_jsonl
from generation import construire_prompt
from metrics import MARQUEUR_ABSTENTION, bootstrap_difference, bootstrap_ic, noter

PETIT, GRAND3, GRAND6 = "D_petit_rag", "B_grand_rag", "E_grand_rag6"
REFERENCE = "B_grand_rag"
CONFIG_POLITIQUE = CONFIG.parent / "politique_v1.json"
LATENCE_RECHERCHE = 0.0002  # recherche BM25 seule (0,19 ms mesurés en Partie III)
MARGE_LATENCE = 0.78  # réglage visé : latence ≤ 78 % de B (le critère exige ≤ 80 %), marge contre le bruit


# ---------------------------------------------------------------------------------------------------
# Exécution réelle (utilisée en Partie V)
# ---------------------------------------------------------------------------------------------------
def executer_politique(q, generateurs, recherche, p, seuil_f1):
    """Exécute la politique sur une question et mesure sa latence de bout en bout."""
    debut = time.perf_counter()
    passages, preuve = recherche.chercher_avec_preuve(q["question"], 6)
    if preuve < p["seuil_preuve_abstention"]:
        latence = time.perf_counter() - debut
        return {"id": q["id"], "categorie": q["categorie"], "sortie": MARQUEUR_ABSTENTION,
                "etapes": ["abstention_precoce"], "preuve": round(preuve, 4), "latence_s": round(latence, 4),
                "tokens": 0, **noter(q, MARQUEUR_ABSTENTION, seuil_f1)}
    etapes, tokens, garde = [], 0, False
    if p["cascade"]:
        r = generateurs["petit"].repondre_avec_confiance(construire_prompt(q["question"], passages[:3]))
        etapes, tokens = ["petit_k3"], r["tokens_prompt"] + r["tokens_generes"]
        garde = MARQUEUR_ABSTENTION.lower() not in r["texte"].lower() and r[p["conf_petit"]] >= p["seuil_petit"]
    if not garde:
        k = 6 if preuve < p["seuil_preuve"] else 3
        r = generateurs["grand"].repondre_avec_confiance(construire_prompt(q["question"], passages[:k]))
        etapes.append(f"grand_k{k}")
        tokens += r["tokens_prompt"] + r["tokens_generes"]
        if MARQUEUR_ABSTENTION.lower() in r["texte"].lower() or r[p["conf_grand"]] < p["seuil_abstention"]:
            r = {**r, "texte": MARQUEUR_ABSTENTION}
            etapes.append("abstention")
    latence = time.perf_counter() - debut
    return {"id": q["id"], "categorie": q["categorie"], "sortie": r["texte"], "etapes": etapes,
            "preuve": round(preuve, 4), "latence_s": round(latence, 4), "tokens": tokens,
            **noter(q, r["texte"], seuil_f1)}


# ---------------------------------------------------------------------------------------------------
# Simulation à partir des sorties enregistrées de chaque étape
# ---------------------------------------------------------------------------------------------------
class Donnees:
    """Tableaux alignés par question : sorties des étapes D, B, E et signal de preuve."""

    def __init__(self, questions, sorties, preuves, seuil_f1):
        self.questions = questions
        self.n = len(questions)
        self.preuve = np.array(preuves)
        self.categories = np.array([q["categorie"] for q in questions])
        abst = [noter(q, MARQUEUR_ABSTENTION, seuil_f1) for q in questions]
        self.abst_score = np.array([a["score"] for a in abst])
        self.abst_correct = np.array([a["correct"] for a in abst])
        self.etape = {}
        for s in (PETIT, GRAND3, GRAND6, REFERENCE, "A_grand_sans_rag", "C_petit_sans_rag", "F_petit_rag6"):
            lignes = [x[s] for x in sorties]
            self.etape[s] = {c: np.array([l[c] for l in lignes]) for c in
                             ("score", "latence_s", "abstenu", "correct", "conf_moy", "conf_min")}
            self.etape[s]["tokens"] = np.array([l["tokens_prompt"] + l["tokens_generes"] for l in lignes])


def simuler(d, p, idx=None):
    """Résultats par question de la politique de paramètres p (sur le sous-ensemble idx)."""
    idx = np.arange(d.n) if idx is None else idx
    pe, g3, g6 = d.etape[PETIT], d.etape[GRAND3], d.etape[GRAND6]
    cascade = p["cascade"]
    precoce = d.preuve[idx] < p["seuil_preuve_abstention"]
    garde_petit = (~pe["abstenu"][idx]) & (pe[p["conf_petit"]][idx] >= p["seuil_petit"]) if cascade \
        else np.zeros(len(idx), dtype=bool)
    k6 = d.preuve[idx] < p["seuil_preuve"]
    def grand(c):
        return np.where(k6, g6[c][idx], g3[c][idx])
    abst = grand("abstenu") | (grand(p["conf_grand"]) < p["seuil_abstention"])
    score = np.where(garde_petit, pe["score"][idx], np.where(abst, d.abst_score[idx], grand("score")))
    correct = np.where(garde_petit, pe["correct"][idx], np.where(abst, d.abst_correct[idx], grand("correct")))
    abstenu = np.where(garde_petit, False, abst)
    latence = (pe["latence_s"][idx] if cascade else 0) + np.where(garde_petit, 0, grand("latence_s"))
    tokens = (pe["tokens"][idx] if cascade else 0) + np.where(garde_petit, 0, grand("tokens"))
    choix = np.where(garde_petit, "petit_k3", np.where(k6, "grand_k6", "grand_k3"))
    # Abstention précoce : aucune génération, seul le coût de la recherche BM25 est payé
    score = np.where(precoce, d.abst_score[idx], score)
    correct = np.where(precoce, d.abst_correct[idx], correct)
    abstenu = np.where(precoce, True, abstenu)
    latence = np.where(precoce, LATENCE_RECHERCHE, latence)
    tokens = np.where(precoce, 0, tokens)
    choix = np.where(precoce, "abstention_precoce", choix)
    return {"score": score.astype(float), "correct": correct.astype(bool), "abstenu": abstenu.astype(bool),
            "latence_s": latence.astype(float), "tokens": tokens.astype(float), "choix": choix}


def resumer(res, d, idx=None):
    idx = np.arange(d.n) if idx is None else idx
    ref = d.etape[REFERENCE]
    s, lat = res["score"], res["latence_s"]
    repondues = ~res["abstenu"]
    bas, haut = bootstrap_ic(s)
    d_bas, d_haut = bootstrap_difference(s, ref["score"][idx])
    return {
        "score_global": round(float(s.mean()), 4), "ic95": [round(bas, 4), round(haut, 4)],
        "ecart_vs_B_points": round(100 * float(s.mean() - ref["score"][idx].mean()), 2),
        "ecart_vs_B_ic95_points": [round(100 * d_bas, 2), round(100 * d_haut, 2)],
        "latence_moyenne_s": round(float(lat.mean()), 4),
        "latence_p95_s": round(float(np.quantile(lat, 0.95)), 4),
        "reduction_latence_vs_B": round(1 - float(lat.mean()) / float(ref["latence_s"][idx].mean()), 4),
        "tokens_moyens": round(float(res["tokens"].mean()), 1),
        "couverture": round(float(repondues.mean()), 4),
        "risque": round(float((~res["correct"][repondues]).mean()) if repondues.any() else 0.0, 4),
        "par_categorie": {c: round(float(s[d.categories[idx] == c].mean()), 4)
                          for c in sorted(set(d.categories[idx]))},
        "choix": {c: int((res["choix"] == c).sum()) for c in ("abstention_precoce", "petit_k3", "grand_k3", "grand_k6")},
        "abstentions_decidees": int(res["abstenu"].sum()),
    }


def grille(d):
    q = np.quantile(d.preuve, np.linspace(0.1, 0.9, 9))
    q0 = np.quantile(d.preuve, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    seuils_conf = np.round(np.arange(0, 1.0001, 0.025), 3)
    preuves_k6 = [-1.0, *q.round(4), 2.0]
    abstentions = [0.0, *np.round(np.arange(0.3, 0.96, 0.05), 2)]
    precoces = [-1.0, *q0.round(4)]
    avec = [dict(cascade=True, conf_petit=cp, seuil_petit=float(sp), seuil_preuve=float(se), conf_grand=cg,
                 seuil_abstention=float(sa), seuil_preuve_abstention=float(s0))
            for cp, sp, se, cg, sa, s0 in itertools.product(
                ("conf_moy", "conf_min"), seuils_conf, preuves_k6, ("conf_moy", "conf_min"), abstentions, precoces)]
    sans = [dict(cascade=False, conf_petit="conf_moy", seuil_petit=2.0, seuil_preuve=float(se), conf_grand=cg,
                 seuil_abstention=float(sa), seuil_preuve_abstention=float(s0))
            for se, cg, sa, s0 in itertools.product(preuves_k6, ("conf_moy", "conf_min"), abstentions, precoces)]
    return avec + sans


def regler(d, idx, candidats, fixe=None):
    """Meilleur score sous la contrainte de latence (≤ MARGE_LATENCE × B), sur les questions idx."""
    plafond = MARGE_LATENCE * d.etape[REFERENCE]["latence_s"][idx].mean()
    meilleur, meilleur_cle = None, None
    for p in candidats:
        if fixe:
            p = {**p, **fixe}
        r = simuler(d, p, idx)
        lat = r["latence_s"].mean()
        if lat > plafond:
            continue
        cle = (round(r["score"].mean(), 6), -lat)
        if meilleur_cle is None or cle > meilleur_cle:
            meilleur, meilleur_cle = p, cle
    return meilleur


def courbe(d, candidats, n_points=24):
    """Frontière qualité–coût de la politique : meilleur score pour chaque plafond de latence."""
    ref = d.etape[REFERENCE]["latence_s"].mean()
    res = [(simuler(d, p)["score"].mean(), simuler(d, p)["latence_s"].mean()) for p in candidats[::7]]
    points = []
    for plafond in np.linspace(0.35, 1.3, n_points) * ref:
        ok = [r for r in res if r[1] <= plafond]
        if ok:
            s, l = max(ok)
            points.append({"latence_moyenne_s": round(float(l), 4), "score_global": round(float(s), 4)})
    return points


def preuves_bm25(questions, cfg):
    from retrieval import Recherche
    recherche = Recherche(cfg["modeles"]["embeddings"], "bm25")
    return [recherche.chercher_avec_preuve(q["question"], 6)[1] for q in questions]


def charger(split, partie):
    questions = {q["id"]: q for q in lire_jsonl(DONNEES / f"{split}.jsonl")}
    sorties = {}
    for l in lire_jsonl(RESULTATS / partie / split / "sorties.jsonl"):
        sorties.setdefault(l["id"], {})[l["strategie"]] = l
    ids = sorted(sorties)
    return [questions[i] for i in ids], [sorties[i] for i in ids]


def main():
    cfg = charger_config()
    seuil_f1 = cfg["evaluation"]["seuil_f1_reponse_correcte"]
    questions, sorties = charger("dev", "partie4")
    d = Donnees(questions, sorties, preuves_bm25(questions, cfg), seuil_f1)
    candidats = grille(d)
    tous = np.arange(d.n)
    print(f"{len(candidats)} réglages candidats")

    # 1. Réglage sur tout le jeu de développement → politique figée
    p = regler(d, tous, candidats)
    res = resumer(simuler(d, p), d)

    # 2. Estimation honnête : réglage sur 4/5, application sur le 1/5 restant
    plis = StratifiedKFold(5, shuffle=True, random_state=cfg["graine"]).split(np.zeros(d.n), d.categories)
    parts = {k: np.zeros(d.n) for k in ("score", "latence_s", "tokens")}
    parts.update(correct=np.zeros(d.n, bool), abstenu=np.zeros(d.n, bool), choix=np.empty(d.n, dtype=object))
    for app, val in plis:
        pv = regler(d, app, candidats)
        rv = simuler(d, pv, val)
        for k in parts:
            parts[k][val] = rv[k]
    res_vc = resumer(parts, d)

    # 3. Ablations (chaque brique désactivée, réglage refait sous la même contrainte)
    ablations = {}
    for nom, fixe in (
        ("sans_cascade", {"cascade": False, "seuil_petit": 2.0}),
        ("sans_abstention_precoce", {"seuil_preuve_abstention": -1.0}),
        ("sans_recherche_adaptative", {"seuil_preuve": -1.0}),
        ("sans_abstention_finale", {"seuil_abstention": 0.0}),
        ("sans_aucune_abstention", {"seuil_abstention": 0.0, "seuil_preuve_abstention": -1.0}),
    ):
        pa = regler(d, tous, candidats, fixe=fixe)
        ablations[nom] = {"parametres": pa, **(resumer(simuler(d, pa), d) if pa else {"admissible": False})}

    resultats = {
        "contrainte": f"latence moyenne ≤ {MARGE_LATENCE} × latence de B",
        "politique": {"parametres": p, **res},
        "validation_croisee": res_vc,
        "ablations": ablations,
        "references": {s: resumer({"score": d.etape[s]["score"], "correct": d.etape[s]["correct"],
                                    "abstenu": d.etape[s]["abstenu"], "latence_s": d.etape[s]["latence_s"],
                                    "tokens": d.etape[s]["tokens"], "choix": np.array([s] * d.n)}, d)
                       for s in (REFERENCE, GRAND6, PETIT)},
        "courbe": courbe(d, candidats),
        "preuve_quantiles": {f"q{int(100 * x)}": round(float(np.quantile(d.preuve, x)), 4) for x in (0.1, 0.5, 0.9)},
    }
    sortie = RESULTATS / "partie4"
    (sortie / "politique_dev.json").write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    CONFIG_POLITIQUE.write_text(json.dumps({
        "version": "1.0", "date": "2026-10-18", "regle_sur": "jeu de developpement (180 questions)",
        "contrainte_de_reglage": resultats["contrainte"], "parametres": p,
        "etapes": "BM25 k=6 -> abstention precoce si preuve faible -> (cascade) petit modele k=3 -> si incertain : grand modele k=3 ou k=6 selon la preuve -> abstention si incertain",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def ligne(nom, r):
        print(f"{nom:48s} score={r['score_global']:.3f} écart_B={r['ecart_vs_B_points']:+.1f} {r['ecart_vs_B_ic95_points']} "
              f"latence={r['latence_moyenne_s']:.3f}s (−{100 * r['reduction_latence_vs_B']:.0f}%) "
              f"couv={r['couverture']:.2f} risque={r['risque']:.2f} choix={r['choix']}")
    print("paramètres :", p)
    ligne("Politique (réglée sur tout dev)", res)
    ligne("Politique (validation croisée)", res_vc)
    for nom, r in ablations.items():
        if "score_global" in r:
            ligne(nom, r)
        else:
            print(nom, ": aucun réglage admissible")
    for s, r in resultats["references"].items():
        ligne(s, r)


if __name__ == "__main__":
    main()
