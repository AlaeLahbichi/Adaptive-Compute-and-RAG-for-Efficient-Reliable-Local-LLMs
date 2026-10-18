"""Figures qualité–coût des Parties IV (développement) et V (test).

Usage :
  python bench/plot_partie4.py dev     # livrable/figures/partie4_qualite_cout.pdf
  python bench/plot_partie4.py test    # livrable/figures/partie5_qualite_cout.pdf
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RACINE, RESULTATS  # noqa: E402

SURFACE, ENCRE, ENCRE_2, ENCRE_3, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIE_1, SERIE_2, SERIE_3 = "#2a78d6", "#eb6834", "#1baf7a"
NOMS = {"A_grand_sans_rag": "A", "B_grand_rag": "B (réf.)", "C_petit_sans_rag": "C",
        "D_petit_rag": "D", "E_grand_rag6": "E", "F_petit_rag6": "F"}


def charger(split):
    if split == "dev":
        p4 = json.loads((RESULTATS / "partie4" / "politique_dev.json").read_text(encoding="utf-8"))
        p3 = json.loads((RESULTATS / "partie3" / "routeurs.json").read_text(encoding="utf-8"))
        fixes = p3["strategies_fixes"]
        return {"fixes": fixes, "politique": p4["politique"], "r1": p3["R1_question_seule"]["point_retenu"],
                "r2": p3["R2_adaptive_rag"], "oracle": p3["oracle"], "courbe": p4["courbe"]}
    r = json.loads((RESULTATS / "partie5" / "evaluation_finale.json").read_text(encoding="utf-8"))
    m = r["methodes"]
    return {"fixes": {k: v for k, v in m.items() if k in NOMS}, "politique": m["P_politique_adaptative"],
            "r1": m["R1_question_seule"], "r2": m["R2_adaptive_rag"], "oracle": m["Oracle"], "courbe": None}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    d = charger(split)
    ref = d["fixes"]["B_grand_rag"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between([0, 0.8 * ref["latence_moyenne_s"]], ref["score_global"] - 0.02, 1.0,
                    color=GRILLE, alpha=0.7, linewidth=0, zorder=0)
    ax.text(0.012, ref["score_global"] - 0.012, "zone du critère de succès", color=ENCRE_2, fontsize=7.5, va="top")
    for nom, s in d["fixes"].items():
        ax.scatter(s["latence_moyenne_s"], s["score_global"], s=36, color=ENCRE_3, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        ax.annotate(NOMS[nom], (s["latence_moyenne_s"], s["score_global"]), xytext=(5, 4),
                    textcoords="offset points", color=ENCRE_2, fontsize=8)
    if d["courbe"]:
        c = sorted(d["courbe"], key=lambda p: p["latence_moyenne_s"])
        ax.plot([p["latence_moyenne_s"] for p in c], [p["score_global"] for p in c], color=SERIE_1, linewidth=2,
                zorder=4, label="Politique adaptative : frontière selon le plafond de latence")
    pol = d["politique"]
    ax.scatter(pol["latence_moyenne_s"], pol["score_global"], s=80, marker="o", color=SERIE_1,
               edgecolor=SURFACE, linewidth=1.5, zorder=6, label="Politique adaptative (réglage figé)")
    ax.scatter(d["r1"]["latence_moyenne_s"], d["r1"]["score_global"], s=56, marker="s", color=SERIE_3,
               edgecolor=SURFACE, linewidth=1.5, zorder=5, label="R1 – routeur « question seule »")
    ax.scatter(d["r2"]["latence_moyenne_s"], d["r2"]["score_global"], s=64, marker="D", color=SERIE_2,
               edgecolor=SURFACE, linewidth=1.5, zorder=5, label="R2 – Adaptive-RAG simplifié")
    o = d["oracle"]
    ax.scatter(o["latence_moyenne_s"], o["score_global"], s=64, marker="*", facecolor="none", edgecolor=ENCRE,
               linewidth=1, label="Oracle (borne théorique)", zorder=5)
    ax.set_xlabel("Latence moyenne par question (s)", color=ENCRE_2)
    ax.set_ylabel("Score global", color=ENCRE_2)
    ax.set_xlim(0, max(s["latence_moyenne_s"] for s in d["fixes"].values()) * 1.12)
    ax.set_ylim(0, max(0.7, o["score_global"] + 0.05))
    ax.grid(True, color=GRILLE, linewidth=0.6)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(ENCRE_3)
    ax.tick_params(colors=ENCRE_2)
    ax.legend(frameon=False, loc="lower right", fontsize=7, labelcolor=ENCRE_2)
    fig.tight_layout()
    sortie = RACINE / "livrable" / "figures" / f"partie{'4' if split == 'dev' else '5'}_qualite_cout.pdf"
    fig.savefig(sortie, facecolor=SURFACE)
    fig.savefig(sortie.with_suffix(".png"), facecolor=SURFACE)
    print(sortie)


if __name__ == "__main__":
    main()
