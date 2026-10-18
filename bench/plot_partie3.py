"""Figure qualité–coût de la Partie III (score global en fonction de la latence moyenne).

Usage :
  python bench/plot_partie3.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RACINE, RESULTATS  # noqa: E402

# Palette de référence (thème clair) : deux séries colorées au plus, le reste en encre neutre.
SURFACE, ENCRE, ENCRE_2, ENCRE_3, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIE_1, SERIE_2 = "#2a78d6", "#eb6834"
NOMS = {"A_grand_sans_rag": "A", "B_grand_rag": "B (réf.)", "C_petit_sans_rag": "C",
        "D_petit_rag": "D", "E_grand_rag6": "E", "F_petit_rag6": "F"}


def main():
    r = json.loads((RESULTATS / "partie3" / "routeurs.json").read_text(encoding="utf-8"))
    ref = r["strategies_fixes"]["B_grand_rag"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Zone du critère principal : au moins 20 % plus rapide que B, au plus 2 points de score en moins.
    ax.fill_between([0, 0.8 * ref["latence_moyenne_s"]], ref["score_global"] - 0.02, 1.0,
                    color=GRILLE, alpha=0.7, linewidth=0, zorder=0)
    ax.text(0.012, ref["score_global"] - 0.012, "zone du critère de succès", color=ENCRE_2, fontsize=7.5,
            va="top")

    for nom, s in r["strategies_fixes"].items():
        ax.scatter(s["latence_moyenne_s"], s["score_global"], s=36, color=ENCRE_3, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        ax.annotate(NOMS[nom], (s["latence_moyenne_s"], s["score_global"]), xytext=(5, 4),
                    textcoords="offset points", color=ENCRE_2, fontsize=8)

    courbe = sorted(r["R1_question_seule"]["courbe"], key=lambda p: p["latence_moyenne_s"])
    ax.plot([p["latence_moyenne_s"] for p in courbe], [p["score_global"] for p in courbe], color=SERIE_1,
            linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1,
            label="R1 – routeur « question seule » (courbe selon λ)", zorder=4)
    r2 = r["R2_adaptive_rag"]
    ax.scatter(r2["latence_moyenne_s"], r2["score_global"], s=64, marker="D", color=SERIE_2,
               edgecolor=SURFACE, linewidth=1.5, label="R2 – Adaptive-RAG simplifié", zorder=5)
    o = r["oracle"]
    ax.scatter(o["latence_moyenne_s"], o["score_global"], s=64, marker="*", facecolor="none",
               edgecolor=ENCRE, linewidth=1, label="Oracle (borne théorique)", zorder=5)

    ax.set_xlabel("Latence moyenne par question (s)", color=ENCRE_2)
    ax.set_ylabel("Score global", color=ENCRE_2)
    ax.set_xlim(0, max(s["latence_moyenne_s"] for s in r["strategies_fixes"].values()) * 1.12)
    ax.set_ylim(0, max(0.7, o["score_global"] + 0.05))
    ax.grid(True, color=GRILLE, linewidth=0.6)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(ENCRE_3)
    ax.tick_params(colors=ENCRE_2)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5, labelcolor=ENCRE_2)
    fig.tight_layout()
    sortie = RACINE / "livrable" / "figures" / "partie3_qualite_cout.pdf"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sortie, facecolor=SURFACE)
    fig.savefig(sortie.with_suffix(".png"), facecolor=SURFACE)
    print(sortie)


if __name__ == "__main__":
    main()
