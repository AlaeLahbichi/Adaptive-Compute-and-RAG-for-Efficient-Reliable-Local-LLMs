"""Évaluation finale (Partie V) : exécution UNIQUE sur le jeu de TEST, après le gel de la politique.

Trois passes complètes (latence = médiane par question, conformément au protocole). Dans chaque passe,
l'ordre des méthodes est tiré au hasard (graine fixe) pour éviter un biais lié à l'échauffement de la
machine. Méthodes exécutées : stratégies fixes A–F et politique adaptative (exécution réelle).
Si la variable SUDO_MDP est fournie, powermetrics mesure l'énergie pendant toute l'évaluation.

Usage :
  SUDO_MDP=... HF_HUB_OFFLINE=1 python bench/run_final.py
"""
import json
import os
import random
import tempfile
import time

from check_env import releve
from common import DONNEES, RESULTATS, charger_config, ecrire_jsonl, lire_jsonl
from energie import energie_fenetre, lancer_powermetrics, lire_echantillons
from generation import Generateur
from politique import CONFIG_POLITIQUE, executer_politique
from retrieval import Recherche
from run_pilot import executer_strategie
from run_strategies import toutes_strategies

N_PASSES = 3
POLITIQUE = "P_politique_adaptative"


def executer_politique_complete(questions, generateurs, recherche, params, cfg):
    for q in questions[: cfg["generation"]["requetes_de_chauffe"]]:  # chauffe
        executer_politique(q, generateurs, recherche, params, cfg["evaluation"]["seuil_f1_reponse_correcte"])
    return [{"strategie": POLITIQUE, **executer_politique(q, generateurs, recherche, params,
                                                         cfg["evaluation"]["seuil_f1_reponse_correcte"])}
            for q in questions]


def main():
    cfg = charger_config()
    params = json.loads(CONFIG_POLITIQUE.read_text(encoding="utf-8"))["parametres"]
    questions = lire_jsonl(DONNEES / "test.jsonl")
    random.Random(cfg["graine"]).shuffle(questions)
    sortie = RESULTATS / "partie5" / "test"
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "environnement.json").write_text(json.dumps(releve(), indent=2, ensure_ascii=False), encoding="utf-8")

    recherche = Recherche(cfg["modeles"]["embeddings"], cfg["recherche"]["methode"])
    generateurs = {role: Generateur(cfg["modeles"][role], cfg["generation"]["max_tokens"])
                   for role in ("petit", "grand")}
    methodes = [*toutes_strategies(cfg), POLITIQUE]

    mot_de_passe = os.environ.pop("SUDO_MDP", None)
    proc, brut, debut_mesure = None, None, time.time()
    if mot_de_passe:
        brut = tempfile.NamedTemporaryFile(prefix="powermetrics_", suffix=".txt", delete=False).name
        proc = lancer_powermetrics(brut, mot_de_passe)
        del mot_de_passe
        time.sleep(2)
    fenetres, toutes = [], []
    try:
        t0 = time.time()
        time.sleep(20)
        fenetres.append({"methode": "repos", "passe": 0, "debut": t0, "fin": time.time(), "n": 0})
        rng = random.Random(cfg["graine"])
        for passe in range(1, N_PASSES + 1):
            ordre = methodes[:]
            rng.shuffle(ordre)
            for nom in ordre:
                print(f"passe {passe} → {nom}", flush=True)
                t0 = time.time()
                if nom == POLITIQUE:
                    lignes = executer_politique_complete(questions, generateurs, recherche, params, cfg)
                else:
                    lignes, _ = executer_strategie(nom, toutes_strategies(cfg)[nom], generateurs, recherche,
                                                   questions, cfg)
                fenetres.append({"methode": nom, "passe": passe, "debut": t0, "fin": time.time(),
                                 "n": len(lignes) + cfg["generation"]["requetes_de_chauffe"]})
                toutes += [{**l, "passe": passe} for l in lignes]
                ecrire_jsonl(sortie / "sorties.jsonl", toutes)  # sauvegarde au fil de l'eau
                time.sleep(3)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)

    (sortie / "fenetres.json").write_text(json.dumps(fenetres, indent=2), encoding="utf-8")
    if brut:
        echantillons = lire_echantillons(brut, debut_mesure)
        os.unlink(brut)
        repos = fenetres[0]
        _, p_repos = energie_fenetre(echantillons, repos["debut"], repos["fin"])
        energie = {"puissance_repos_w": round(p_repos, 3), "echantillons": len(echantillons), "methodes": {}}
        for f in fenetres[1:]:
            e_brute, p_moy = energie_fenetre(echantillons, f["debut"], f["fin"])
            e_nette = e_brute - p_repos * (f["fin"] - f["debut"])
            energie["methodes"].setdefault(f["methode"], []).append({
                "passe": f["passe"], "duree_s": round(f["fin"] - f["debut"], 2),
                "puissance_moyenne_w": round(p_moy, 3), "energie_nette_j": round(e_nette, 2),
                "joules_par_question": round(e_nette / f["n"], 4)})
        (sortie / "energie.json").write_text(json.dumps(energie, indent=2, ensure_ascii=False), encoding="utf-8")
    print("terminé")


if __name__ == "__main__":
    main()
