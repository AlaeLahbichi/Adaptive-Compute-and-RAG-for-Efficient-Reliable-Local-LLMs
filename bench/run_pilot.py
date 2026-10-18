"""Pilote du protocole : exécute les stratégies fixes A–D sur le jeu de DÉVELOPPEMENT.

Le jeu de test n'est jamais lu par ce script.

Usage :
  HF_HUB_OFFLINE=1 python bench/run_pilot.py
"""
import json
import random
import time

import numpy as np

from check_env import releve
from common import DONNEES, RESULTATS, charger_config, ecrire_jsonl, lire_jsonl
from generation import Generateur, construire_prompt, memoire_pic_go, reinitialiser_memoire_pic
from metrics import bootstrap_ic, couverture_risque, noter, rappel_recherche
from retrieval import Recherche

SORTIE = RESULTATS / "pilote"


def analyse_recherche(recherche, questions, ks=(1, 3, 6, 10)):
    """Rappel de la recherche seule, par catégorie (questions avec passages justificatifs)."""
    res = {}
    for q in questions:
        ids = [p["id"] for p in recherche.chercher(q["question"], max(ks))]
        for k in ks:
            res.setdefault(q["categorie"], {}).setdefault(k, []).append(rappel_recherche(q, ids[:k]))
    return {c: {f"rappel@{k}": round(float(np.mean(v)), 3) for k, v in d.items()} for c, d in res.items()}


def executer_strategie(nom, params, generateurs, recherche, questions, cfg):
    gen = generateurs[params["modele"]]
    k = params["k"]
    for q in questions[: cfg["generation"]["requetes_de_chauffe"]]:  # chauffe, résultats ignorés
        gen.repondre(construire_prompt(q["question"], recherche.chercher(q["question"], k) if k else []))
    reinitialiser_memoire_pic()
    lignes = []
    for q in questions:
        debut = time.perf_counter()
        passages = recherche.chercher(q["question"], k) if k else []
        sortie, n_prompt, n_gen = gen.repondre(construire_prompt(q["question"], passages))
        latence = time.perf_counter() - debut
        note = noter(q, sortie, cfg["evaluation"]["seuil_f1_reponse_correcte"])
        lignes.append({
            "strategie": nom, "id": q["id"], "categorie": q["categorie"], "sortie": sortie,
            "passages": [p["id"] for p in passages],
            "rappel_recherche": rappel_recherche(q, [p["id"] for p in passages]) if k else None,
            "latence_s": round(latence, 4), "tokens_prompt": n_prompt, "tokens_generes": n_gen, **note,
        })
    return lignes, memoire_pic_go()


def resumer(lignes, cfg):
    scores = [l["score"] for l in lignes]
    lat = np.array([l["latence_s"] for l in lignes])
    couverture, risque = couverture_risque(lignes)
    bas, haut = bootstrap_ic(scores, cfg["evaluation"]["bootstrap_reechantillonnages"])
    return {
        "n": len(lignes),
        "score_global": round(float(np.mean(scores)), 4), "ic95": [round(bas, 4), round(haut, 4)],
        "em": round(float(np.mean([l["em"] for l in lignes])), 4),
        "couverture": round(couverture, 4), "risque": round(risque, 4),
        "latence_moyenne_s": round(float(lat.mean()), 4),
        "latence_mediane_s": round(float(np.median(lat)), 4),
        "latence_p95_s": round(float(np.quantile(lat, 0.95)), 4),
        "tokens_moyens": round(float(np.mean([l["tokens_prompt"] + l["tokens_generes"] for l in lignes])), 1),
    }


def main():
    cfg = charger_config()
    questions = lire_jsonl(DONNEES / "dev.jsonl")
    random.Random(cfg["graine"]).shuffle(questions)
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "environnement.json").write_text(json.dumps(releve(), indent=2, ensure_ascii=False), encoding="utf-8")

    debut_total = time.perf_counter()
    recherche = Recherche(cfg["modeles"]["embeddings"])
    rappel = analyse_recherche(recherche, questions)

    generateurs = {role: Generateur(cfg["modeles"][role], cfg["generation"]["max_tokens"])
                   for role in ("petit", "grand")}

    toutes, resume = [], {"recherche": rappel, "strategies": {}}
    for nom, params in cfg["strategies_fixes"].items():
        print(f"→ {nom}")
        lignes, memoire = executer_strategie(nom, params, generateurs, recherche, questions, cfg)
        toutes += lignes
        resume["strategies"][nom] = {
            "global": {**resumer(lignes, cfg), "memoire_pic_go": round(memoire, 2)},
            "par_categorie": {c: resumer([l for l in lignes if l["categorie"] == c], cfg)
                              for c in sorted({l["categorie"] for l in lignes})},
        }
    duree = time.perf_counter() - debut_total
    n_generations = len(toutes)
    resume["duree_pilote_s"] = round(duree, 1)
    resume["estimation_test_complet_min"] = round(
        duree / n_generations * 450 * len(cfg["strategies_fixes"]) / 60, 1)

    ecrire_jsonl(SORTIE / "sorties.jsonl", toutes)
    (SORTIE / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resume, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
