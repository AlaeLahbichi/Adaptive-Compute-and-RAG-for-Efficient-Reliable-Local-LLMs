"""Relève l'environnement matériel et logiciel (à joindre à chaque série d'expériences)."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import psutil

from common import RESULTATS, charger_config

PAQUETS = ["mlx", "mlx-lm", "transformers", "torch", "sentence-transformers", "numpy", "pandas", "psutil"]


def sysctl(cle):
    return subprocess.run(["sysctl", "-n", cle], capture_output=True, text=True).stdout.strip()


def revision_modele(nom):
    """Révision (commit Hugging Face) du modèle présent dans le cache local."""
    dossier = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{nom.replace('/', '--')}" / "snapshots"
    revisions = sorted(p.name for p in dossier.iterdir()) if dossier.exists() else []
    return revisions[0] if revisions else None


def releve():
    cfg = charger_config()
    versions = {}
    for p in PAQUETS:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "machine": {
            "puce": sysctl("machdep.cpu.brand_string"),
            "coeurs": psutil.cpu_count(),
            "memoire_gio": round(psutil.virtual_memory().total / 2**30, 1),
            "macos": platform.mac_ver()[0],
        },
        "python": sys.version.split()[0],
        "paquets": versions,
        "modeles": {role: {"nom": nom, "revision": revision_modele(nom)} for role, nom in cfg["modeles"].items()},
    }


if __name__ == "__main__":
    rapport = releve()
    sortie = RESULTATS / "pilote" / "environnement.json"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rapport, indent=2, ensure_ascii=False))
