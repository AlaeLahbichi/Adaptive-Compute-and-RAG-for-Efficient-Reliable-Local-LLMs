"""Mesure de l'énergie des stratégies fixes avec powermetrics (macOS, droits administrateur requis).

powermetrics relève la puissance CPU + GPU + ANE toutes les 200 ms pendant l'exécution. L'énergie d'une
stratégie est l'intégrale de cette puissance sur sa fenêtre d'exécution, moins la consommation au repos
mesurée juste avant. Seul le jeu de DÉVELOPPEMENT est utilisé.

Usage (le mot de passe est lu depuis la variable d'environnement SUDO_MDP, jamais écrit sur disque) :
  SUDO_MDP=... HF_HUB_OFFLINE=1 python bench/energie.py
"""
import json
import os
import random
import re
import subprocess
import tempfile
import time

import numpy as np

from common import DONNEES, RESULTATS, charger_config, lire_jsonl
from generation import Generateur
from retrieval import Recherche
from run_pilot import executer_strategie

INTERVALLE_MS = 200
DUREE_REPOS_S = 20

RE_ECHANTILLON = re.compile(r"\*\*\* Sampled system activity .*?\(([\d.]+)ms elapsed\) \*\*\*")
RE_PUISSANCE = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW")


def lancer_powermetrics(chemin, mot_de_passe):
    proc = subprocess.Popen(
        ["sudo", "-S", "-p", "", "powermetrics", "-i", str(INTERVALLE_MS),
         "--samplers", "cpu_power,gpu_power,ane_power", "-o", chemin],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(mot_de_passe + "\n")
    proc.stdin.flush()
    return proc


def lire_echantillons(chemin, debut):
    """Renvoie une liste (instant_fin, duree_s, puissance_w) à partir de la sortie texte de powermetrics."""
    texte = open(chemin, encoding="utf-8", errors="ignore").read()
    blocs = RE_ECHANTILLON.split(texte)[1:]
    echantillons, t = [], debut
    for duree_ms, contenu in zip(blocs[0::2], blocs[1::2]):
        m = RE_PUISSANCE.search(contenu)
        duree = float(duree_ms) / 1000
        t += duree
        if m:
            echantillons.append((t, duree, int(m.group(1)) / 1000))
    return echantillons


def energie_fenetre(echantillons, debut, fin):
    """Énergie (J) et puissance moyenne (W) des échantillons qui se terminent dans [debut, fin]."""
    dans = [(d, p) for t, d, p in echantillons if debut < t <= fin]
    if not dans:
        return None, None
    energie = sum(d * p for d, p in dans)
    return energie, energie / sum(d for d, _ in dans)


def main():
    mot_de_passe = os.environ.pop("SUDO_MDP", None)
    if not mot_de_passe:
        raise SystemExit("Variable SUDO_MDP absente.")
    cfg = charger_config()
    questions = lire_jsonl(DONNEES / "dev.jsonl")
    random.Random(cfg["graine"]).shuffle(questions)

    recherche = Recherche(cfg["modeles"]["embeddings"])
    generateurs = {role: Generateur(cfg["modeles"][role], cfg["generation"]["max_tokens"])
                   for role in ("petit", "grand")}

    sortie_brute = tempfile.NamedTemporaryFile(prefix="powermetrics_", suffix=".txt", delete=False).name
    debut_mesure = time.time()
    proc = lancer_powermetrics(sortie_brute, mot_de_passe)
    del mot_de_passe
    fenetres = {}
    try:
        time.sleep(2)  # laisser powermetrics démarrer
        t0 = time.time()
        time.sleep(DUREE_REPOS_S)
        fenetres["repos"] = (t0, time.time())
        for nom, params in cfg["strategies_fixes"].items():
            print(f"→ {nom}")
            t0 = time.time()
            lignes, _ = executer_strategie(nom, params, generateurs, recherche, questions, cfg)
            fenetres[nom] = (t0, time.time(), len(lignes))
            time.sleep(3)  # séparer nettement les fenêtres
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    erreurs = proc.stderr.read().strip()
    echantillons = lire_echantillons(sortie_brute, debut_mesure)
    os.unlink(sortie_brute)
    if not echantillons:
        raise SystemExit(f"Aucun échantillon powermetrics lu. {erreurs}")

    _, p_repos = energie_fenetre(echantillons, *fenetres["repos"])
    resultats = {"intervalle_ms": INTERVALLE_MS, "puissance_repos_w": round(p_repos, 2),
                 "echantillons": len(echantillons), "strategies": {}}
    for nom in cfg["strategies_fixes"]:
        debut, fin, n = fenetres[nom]
        e_brute, p_moy = energie_fenetre(echantillons, debut, fin)
        e_nette = e_brute - p_repos * (fin - debut)
        resultats["strategies"][nom] = {
            "duree_s": round(fin - debut, 1),
            "puissance_moyenne_w": round(p_moy, 2),
            "energie_brute_j": round(e_brute, 1),
            "energie_nette_j": round(e_nette, 1),
            "joules_par_question": round(e_nette / n, 3),
        }
    chemin = RESULTATS / "pilote" / "energie.json"
    chemin.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
