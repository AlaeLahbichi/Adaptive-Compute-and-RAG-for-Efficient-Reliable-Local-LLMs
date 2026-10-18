"""Compare les méthodes de recherche documentaire sur le jeu de DÉVELOPPEMENT (rappel@k et latence).

Usage :
  HF_HUB_OFFLINE=1 python bench/compare_retrieval.py
"""
import json
import time

import numpy as np

from common import DONNEES, RESULTATS, charger_config, lire_jsonl
from metrics import rappel_recherche
from retrieval import Recherche

KS = (1, 3, 6, 10)


def main():
    cfg = charger_config()
    questions = lire_jsonl(DONNEES / "dev.jsonl")
    resultats = {}
    for methode in Recherche.METHODES:
        recherche = Recherche(cfg["modeles"]["embeddings"], methode)
        recherche.chercher("warm-up", 3)
        rappels, latences = {}, []
        for q in questions:
            debut = time.perf_counter()
            ids = [p["id"] for p in recherche.chercher(q["question"], max(KS))]
            latences.append(time.perf_counter() - debut)
            for k in KS:
                rappels.setdefault(q["categorie"], {}).setdefault(k, []).append(rappel_recherche(q, ids[:k]))
        resultats[methode] = {
            "latence_moyenne_ms": round(1000 * float(np.mean(latences)), 2),
            "rappel": {c: {f"@{k}": round(float(np.mean(v)), 3) for k, v in d.items()} for c, d in rappels.items()},
            "rappel@3_moyen_simple_complexe": round(float(np.mean(
                [np.mean(rappels[c][3]) for c in ("simple", "complexe")])), 3),
        }
        print(methode, json.dumps(resultats[methode], ensure_ascii=False))
    sortie = RESULTATS / "partie3" / "comparaison_recherche.json"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
