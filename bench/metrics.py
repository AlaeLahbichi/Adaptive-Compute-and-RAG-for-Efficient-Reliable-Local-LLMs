"""Métriques du protocole : exactitude/F1, abstention (couverture/risque), coût et intervalles de confiance."""
import collections
import re
import string

import numpy as np

MARQUEUR_ABSTENTION = "UNANSWERABLE"


def normaliser(texte):
    """Normalisation officielle SQuAD : minuscules, sans ponctuation, sans articles, espaces réduits."""
    texte = texte.lower()
    texte = "".join(c for c in texte if c not in set(string.punctuation))
    texte = re.sub(r"\b(a|an|the)\b", " ", texte)
    return " ".join(texte.split())


def f1(prediction, reference):
    p, r = normaliser(prediction).split(), normaliser(reference).split()
    communs = sum((collections.Counter(p) & collections.Counter(r)).values())
    if communs == 0:
        return 0.0
    precision, rappel = communs / len(p), communs / len(r)
    return 2 * precision * rappel / (precision + rappel)


def exact(prediction, reference):
    return float(normaliser(prediction) == normaliser(reference))


def est_abstention(sortie):
    return MARQUEUR_ABSTENTION.lower() in sortie.lower()


def noter(question, sortie, seuil_correct):
    """Note une sortie selon le comportement attendu de la question."""
    abstenu = est_abstention(sortie)
    if question["comportement_attendu"] == "s_abstenir":
        score = em = float(abstenu)
        correct = abstenu
    elif abstenu:
        score = em = 0.0
        correct = False
    else:
        score = max(f1(sortie, r) for r in question["reponses"])
        em = max(exact(sortie, r) for r in question["reponses"])
        correct = score >= seuil_correct
    return {"abstenu": abstenu, "score": score, "em": em, "correct": bool(correct)}


def rappel_recherche(question, ids_trouves):
    """1 si tous les passages justificatifs figurent dans les passages récupérés."""
    return float(set(question["passages_gold"]) <= set(ids_trouves))


def couverture_risque(lignes):
    repondues = [l for l in lignes if not l["abstenu"]]
    couverture = len(repondues) / len(lignes) if lignes else 0.0
    risque = (sum(not l["correct"] for l in repondues) / len(repondues)) if repondues else 0.0
    return couverture, risque


def bootstrap_ic(valeurs, n=10000, niveau=0.95, graine=2026):
    """Intervalle de confiance bootstrap de la moyenne."""
    valeurs = np.asarray(valeurs, dtype=float)
    rng = np.random.default_rng(graine)
    moyennes = valeurs[rng.integers(0, len(valeurs), size=(n, len(valeurs)))].mean(axis=1)
    a = (1 - niveau) / 2
    return float(np.quantile(moyennes, a)), float(np.quantile(moyennes, 1 - a))


def bootstrap_difference(a, b, n=10000, niveau=0.95, graine=2026):
    """IC bootstrap apparié de la différence moyenne (a - b), questions rééchantillonnées ensemble."""
    return bootstrap_ic(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n, niveau, graine)
