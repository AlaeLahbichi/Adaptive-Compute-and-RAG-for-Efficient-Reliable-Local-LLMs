"""Chemins et utilitaires partagés par les scripts du benchmark."""
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CONFIG = RACINE / "configs" / "protocole_v1.json"
DONNEES = RACINE / "data" / "benchmark"
BRUTES = RACINE / "data" / "raw"
CACHE = RACINE / "data" / "cache"
RESULTATS = RACINE / "results"


def charger_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def lire_jsonl(chemin):
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def ecrire_jsonl(chemin, lignes):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        for ligne in lignes:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
