"""Analyse de l'évaluation finale (Partie V) sur le jeu de TEST.

- Qualité : passe 1 (génération déterministe ; la stabilité entre passes est vérifiée).
- Latence : médiane des 3 passes pour chaque question.
- Politique adaptative : exécution réelle ; la simulation à partir des étapes sert de contrôle.
- Routeurs R1 et R2 : entraînés sur le jeu de développement, appliqués au test (réglages figés en Partie III).
- Ablations : réglages figés en Partie IV, appliqués au test par simulation.
- Verdict sur les critères de succès pré-enregistrés (L2, section 8).

Usage :
  python bench/evaluate_final.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from common import DONNEES, RESULTATS, charger_config, lire_jsonl
from metrics import bootstrap_difference
from politique import CONFIG_POLITIQUE, Donnees, REFERENCE, preuves_bm25, resumer, simuler
from routers import STRATEGIES, Caracteristiques, etiquette_adaptive_rag

POLITIQUE = "P_politique_adaptative"
LAMBDA_R1 = 0.113  # point de fonctionnement figé en Partie III


def charger_test():
    questions = {q["id"]: q for q in lire_jsonl(DONNEES / "test.jsonl")}
    lignes = lire_jsonl(RESULTATS / "partie5" / "test" / "sorties.jsonl")
    par = {}
    for l in lignes:
        par.setdefault(l["id"], {}).setdefault(l["strategie"], {})[l["passe"]] = l
    ids = sorted(par)
    stabilite, sorties = {}, []
    for i in ids:
        ligne = {}
        for s, passes in par[i].items():
            base = dict(passes[1])
            base["latence_s"] = float(np.median([p["latence_s"] for p in passes.values()]))
            ligne[s] = base
            stabilite.setdefault(s, []).append(len({p["sortie"] for p in passes.values()}) == 1)
        sorties.append(ligne)
    return [questions[i] for i in ids], sorties, {s: round(float(np.mean(v)), 4) for s, v in stabilite.items()}


def resultat_methode(d, sorties, nom):
    lignes = [s[nom] for s in sorties]
    res = {"score": np.array([l["score"] for l in lignes], float),
           "correct": np.array([l["correct"] for l in lignes], bool),
           "abstenu": np.array([l["abstenu"] for l in lignes], bool),
           "latence_s": np.array([l["latence_s"] for l in lignes], float),
           "tokens": np.array([l.get("tokens", l.get("tokens_prompt", 0) + l.get("tokens_generes", 0))
                               for l in lignes], float),
           "choix": np.array([("+".join(l["etapes"]) if "etapes" in l else nom) for l in lignes])}
    r = resumer(res, d)
    if nom == POLITIQUE:
        r["choix"] = {c: int((res["choix"] == c).sum()) for c in sorted(set(res["choix"]))}
    else:
        r.pop("choix")
    return r, res


def routeurs(questions_test, sorties_test, d_test):
    """R1 et R2 entraînés sur tout le jeu de développement, appliqués au test."""
    q_dev, s_dev = [], []
    par = {}
    for l in lire_jsonl(RESULTATS / "partie4" / "dev" / "sorties.jsonl"):
        par.setdefault(l["id"], {})[l["strategie"]] = l
    dev = {q["id"]: q for q in lire_jsonl(DONNEES / "dev.jsonl")}
    for i in sorted(par):
        q_dev.append(dev[i])
        s_dev.append(par[i])
    feats = Caracteristiques()
    X_dev = feats.fit_transform([q["question"] for q in q_dev])
    X_test = feats.transform([q["question"] for q in questions_test])
    lat_dev = np.array([np.mean([s[st]["latence_s"] for s in s_dev]) for st in STRATEGIES])
    pred = np.column_stack([Ridge(alpha=1.0).fit(X_dev, [s[st]["score"] for s in s_dev]).predict(X_test)
                            for st in STRATEGIES])
    choix_r1 = [STRATEGIES[int(np.argmax(p - LAMBDA_R1 * lat_dev))] for p in pred]
    y = [etiquette_adaptive_rag(q, s) for q, s in zip(q_dev, s_dev)]
    choix_r2 = list(LogisticRegression(max_iter=2000, C=1.0).fit(X_dev, y).predict(X_test))
    surcout = {"R1": 0.0005, "R2": 0.0003}  # mesurés en Partie III (0,5 ms et 0,3 ms)
    sortie = {}
    for nom, choix in (("R1_question_seule", choix_r1), ("R2_adaptive_rag", choix_r2)):
        lignes = [sorties_test[i][c] for i, c in enumerate(choix)]
        res = {"score": np.array([l["score"] for l in lignes], float),
               "correct": np.array([l["correct"] for l in lignes], bool),
               "abstenu": np.array([l["abstenu"] for l in lignes], bool),
               "latence_s": np.array([l["latence_s"] for l in lignes], float) + surcout[nom[:2]],
               "tokens": np.array([l["tokens_prompt"] + l["tokens_generes"] for l in lignes], float),
               "choix": np.array(choix)}
        r = resumer(res, d_test)
        r["choix"] = {c: choix.count(c) for c in STRATEGIES if c in choix}
        sortie[nom] = (r, res)
    return sortie


def main():
    cfg = charger_config()
    params = json.loads(CONFIG_POLITIQUE.read_text(encoding="utf-8"))["parametres"]
    questions, sorties, stabilite = charger_test()
    d = Donnees(questions, sorties, preuves_bm25(questions, cfg), cfg["evaluation"]["seuil_f1_reponse_correcte"])
    resultats = {"n_questions": len(questions), "stabilite_des_sorties_entre_passes": stabilite, "methodes": {}}
    bruts = {}
    for nom in [*STRATEGIES, POLITIQUE]:
        resultats["methodes"][nom], bruts[nom] = resultat_methode(d, sorties, nom)

    sim = simuler(d, params)
    resultats["controle_simulation_politique"] = {
        "accord_score": round(float(np.mean(np.isclose(sim["score"], bruts[POLITIQUE]["score"]))), 4),
        "score_simule": round(float(sim["score"].mean()), 4),
        "latence_simulee_s": round(float(sim["latence_s"].mean()), 4)}

    for nom, (r, res) in routeurs(questions, sorties, d).items():
        resultats["methodes"][nom], bruts[nom] = r, res

    meilleurs = []
    for s in sorties:
        top = max(s[st]["score"] for st in STRATEGIES)
        meilleurs.append(min((st for st in STRATEGIES if s[st]["score"] == top), key=lambda st: s[st]["latence_s"]))
    lignes = [sorties[i][c] for i, c in enumerate(meilleurs)]
    resultats["methodes"]["Oracle"] = resumer({
        "score": np.array([l["score"] for l in lignes], float), "correct": np.array([l["correct"] for l in lignes]),
        "abstenu": np.array([l["abstenu"] for l in lignes]), "latence_s": np.array([l["latence_s"] for l in lignes]),
        "tokens": np.array([l["tokens_prompt"] + l["tokens_generes"] for l in lignes], float),
        "choix": np.array(meilleurs)}, d)
    resultats["methodes"]["Oracle"].pop("choix")

    abl = json.loads((RESULTATS / "partie4" / "politique_dev.json").read_text(encoding="utf-8"))["ablations"]
    resultats["ablations"] = {}
    for nom, a in abl.items():
        if a.get("parametres"):
            resultats["ablations"][nom] = resumer(simuler(d, a["parametres"]), d)

    # Verdict sur les critères pré-enregistrés
    ref, pol = bruts[REFERENCE], bruts[POLITIQUE]
    perte = 100 * (ref["score"].mean() - pol["score"].mean())
    d_bas, d_haut = bootstrap_difference(ref["score"], pol["score"])
    reduction = 1 - pol["latence_s"].mean() / ref["latence_s"].mean()
    crit = cfg["criteres_de_succes"]
    pertes_cat = {c: round(100 * float(ref["score"][d.categories == c].mean() - pol["score"][d.categories == c].mean()), 2)
                  for c in sorted(set(d.categories))}
    resultats["verdict"] = {
        "reduction_latence_vs_B": round(float(reduction), 4),
        "perte_score_points": round(float(perte), 2),
        "perte_score_ic95_points": [round(100 * d_bas, 2), round(100 * d_haut, 2)],
        "critere_latence_atteint": bool(reduction >= crit["reduction_latence_moyenne_min"]),
        "critere_qualite_atteint_estimation_ponctuelle": bool(perte <= crit["perte_score_global_max_points"]),
        "non_inferiorite_demontree": bool(100 * d_haut <= crit["perte_score_global_max_points"]),
        "critere_principal_atteint": bool(reduction >= crit["reduction_latence_moyenne_min"]
                                          and perte <= crit["perte_score_global_max_points"]),
        "pertes_par_categorie_points": pertes_cat,
        "robustesse_aucune_categorie_au_dela_de_5_points": bool(max(pertes_cat.values()) <= 5),
        "bat_R1": bool(pol["score"].mean() > bruts["R1_question_seule"]["score"].mean()),
        "bat_R2": bool(pol["score"].mean() > bruts["R2_adaptive_rag"]["score"].mean()),
    }
    energie_chemin = RESULTATS / "partie5" / "test" / "energie.json"
    if energie_chemin.exists():
        e = json.loads(energie_chemin.read_text(encoding="utf-8"))
        resultats["energie_joules_par_question"] = {
            m: round(float(np.mean([x["joules_par_question"] for x in v])), 3) for m, v in e["methodes"].items()}
        resultats["puissance_moyenne_w"] = {
            m: round(float(np.mean([x["puissance_moyenne_w"] for x in v])), 2) for m, v in e["methodes"].items()}
    (RESULTATS / "partie5" / "evaluation_finale.json").write_text(json.dumps(resultats, indent=2, ensure_ascii=False),
                                                                 encoding="utf-8")

    for nom, r in resultats["methodes"].items():
        print(f"{nom:26s} score={r['score_global']:.3f} [{r['ic95'][0]:.3f};{r['ic95'][1]:.3f}] "
              f"écart_B={r['ecart_vs_B_points']:+.1f} {r['ecart_vs_B_ic95_points']} latence={r['latence_moyenne_s']:.3f}s "
              f"(−{100 * r['reduction_latence_vs_B']:.0f}%) couv={r['couverture']:.2f} risque={r['risque']:.2f} "
              f"cat={r['par_categorie']}")
    for nom, r in resultats["ablations"].items():
        print(f"ABLATION {nom:45s} score={r['score_global']:.3f} latence={r['latence_moyenne_s']:.3f}s")
    print(json.dumps({k: resultats[k] for k in ("verdict", "controle_simulation_politique",
                                                 "stabilite_des_sorties_entre_passes")}, indent=2, ensure_ascii=False))
    if "energie_joules_par_question" in resultats:
        print(json.dumps(resultats["energie_joules_par_question"], indent=2))


if __name__ == "__main__":
    main()
