"""Exécute toutes les stratégies fixes (A–F) avec la recherche figée du protocole.

Les sorties servent à entraîner et à évaluer les routeurs de référence (Partie III). Comme la génération
est déterministe, la sortie d'un routeur pour une question est celle de la stratégie qu'il choisit.

Usage :
  HF_HUB_OFFLINE=1 python bench/run_strategies.py dev     # jeu de développement
  HF_HUB_OFFLINE=1 python bench/run_strategies.py test    # évaluation finale uniquement, après le gel
"""
import json
import random
import sys

from check_env import releve
from common import DONNEES, RESULTATS, charger_config, ecrire_jsonl, lire_jsonl
from generation import Generateur
from retrieval import Recherche
from run_pilot import executer_strategie, resumer


def toutes_strategies(cfg):
    return {**cfg["strategies_fixes"], **cfg["strategies_complementaires"]}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if split not in ("dev", "test"):
        raise SystemExit("Usage : run_strategies.py dev|test")
    cfg = charger_config()
    questions = lire_jsonl(DONNEES / f"{split}.jsonl")
    random.Random(cfg["graine"]).shuffle(questions)

    sortie = RESULTATS / "partie3" / split
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "environnement.json").write_text(json.dumps(releve(), indent=2, ensure_ascii=False), encoding="utf-8")

    recherche = Recherche(cfg["modeles"]["embeddings"], cfg["recherche"]["methode"])
    generateurs = {role: Generateur(cfg["modeles"][role], cfg["generation"]["max_tokens"])
                   for role in ("petit", "grand")}
    toutes, resume = [], {}
    for nom, params in toutes_strategies(cfg).items():
        print(f"→ {nom}")
        lignes, memoire = executer_strategie(nom, params, generateurs, recherche, questions, cfg)
        toutes += lignes
        resume[nom] = {
            "global": {**resumer(lignes, cfg), "memoire_pic_go": round(memoire, 2)},
            "par_categorie": {c: resumer([l for l in lignes if l["categorie"] == c], cfg)
                              for c in sorted({l["categorie"] for l in lignes})},
        }
    ecrire_jsonl(sortie / "sorties.jsonl", toutes)
    (sortie / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")
    for nom, r in resume.items():
        g = r["global"]
        print(f"{nom:18s} score={g['score_global']:.3f} latence={g['latence_moyenne_s']:.3f}s "
              f"couverture={g['couverture']:.2f} risque={g['risque']:.2f} tokens={g['tokens_moyens']:.0f}")


if __name__ == "__main__":
    main()
