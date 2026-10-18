"""Recherche documentaire : dense (embeddings MiniLM), lexicale (BM25) ou hybride (fusion des rangs).

Pour ~3 300 passages, une recherche dense exacte en NumPy est aussi rapide qu'un index FAISS
(IndexFlatIP) et donne exactement les mêmes résultats, sans dépendance supplémentaire.
"""
import hashlib
import math
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

from common import CACHE, DONNEES, lire_jsonl

MOTS_VIDES = set("""a an the of in on at to for from by with and or but is are was were be been being
do does did what which who whom whose when where why how that this these those it its as into than then
there their they he she his her him i you your we our not no yes""".split())


def mots(texte):
    return [m for m in re.findall(r"\w+", texte.lower()) if m not in MOTS_VIDES]


class BM25:
    """Okapi BM25 (k1 = 1,5, b = 0,75), calculé avec une matrice creuse."""

    def __init__(self, textes, k1=1.5, b=0.75):
        self.k1 = k1
        docs = [Counter(mots(t)) for t in textes]
        self.vocab = {m: i for i, m in enumerate(sorted({m for d in docs for m in d}))}
        longueurs = np.array([sum(d.values()) for d in docs], dtype=float)
        df = Counter(m for d in docs for m in d)
        n = len(docs)
        self.idf = np.array([math.log(1 + (n - df[m] + 0.5) / (df[m] + 0.5)) for m in self.vocab])
        lignes, colonnes, valeurs = [], [], []
        for i, d in enumerate(docs):
            norme = k1 * (1 - b + b * longueurs[i] / longueurs.mean())
            for m, tf in d.items():
                lignes.append(i)
                colonnes.append(self.vocab[m])
                valeurs.append(tf * (k1 + 1) / (tf + norme))
        self.poids = csr_matrix((valeurs, (lignes, colonnes)), shape=(n, len(self.vocab)))

    def scores(self, question):
        q = np.zeros(len(self.vocab))
        for m in set(mots(question)):
            if m in self.vocab:
                q[self.vocab[m]] = self.idf[self.vocab[m]]
        return self.poids @ q

    def score_maximal(self, question):
        """Score qu'obtiendrait un passage contenant très souvent tous les mots de la question."""
        return sum(self.idf[self.vocab[m]] * (self.k1 + 1) for m in set(mots(question)) if m in self.vocab)


class Recherche:
    METHODES = ("dense", "bm25", "hybride")

    def __init__(self, nom_modele, methode="dense", rrf_k=60):
        if methode not in self.METHODES:
            raise ValueError(f"Méthode inconnue : {methode}")
        self.methode = methode
        self.rrf_k = rrf_k
        self.passages = lire_jsonl(DONNEES / "corpus.jsonl")
        textes = [f"{p['titre']}. {p['texte']}" for p in self.passages]
        self.encodeur = SentenceTransformer(nom_modele, device="cpu")
        self.matrice = self._embeddings_corpus(nom_modele, textes)
        self.bm25 = BM25(textes) if methode != "dense" else None

    def _embeddings_corpus(self, nom_modele, textes):
        cle = hashlib.sha1((nom_modele + "\n".join(p["id"] for p in self.passages)).encode()).hexdigest()[:12]
        chemin = CACHE / f"embeddings_{cle}.npy"
        if chemin.exists():
            return np.load(chemin)
        CACHE.mkdir(parents=True, exist_ok=True)
        matrice = self.encodeur.encode(textes, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(chemin, matrice.astype(np.float32))
        return matrice

    def embedding_question(self, question):
        return self.encodeur.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]

    def _classement(self, question):
        if self.methode == "bm25":
            return np.argsort(-self.bm25.scores(question))
        dense = np.argsort(-(self.matrice @ self.embedding_question(question)))
        if self.methode == "dense":
            return dense
        # Fusion des rangs (Reciprocal Rank Fusion) : score = somme des 1 / (rrf_k + rang)
        lexical = np.argsort(-self.bm25.scores(question))
        fusion = np.zeros(len(self.passages))
        for classement in (dense, lexical):
            fusion[classement] += 1.0 / (self.rrf_k + np.arange(1, len(classement) + 1))
        return np.argsort(-fusion)

    def chercher(self, question, k):
        """Renvoie les k passages les mieux classés."""
        return [self.passages[i] for i in self._classement(question)[:k]]

    def chercher_avec_preuve(self, question, k):
        """Recherche BM25 + signal de suffisance des preuves, en un seul calcul de scores.

        preuve = score BM25 du meilleur passage / score maximal possible pour la question (entre 0 et 1) :
        proche de 1 si un passage contient de nombreuses fois tous les mots importants de la question.
        """
        if self.methode != "bm25":
            raise ValueError("Le signal de preuve est défini pour la recherche BM25.")
        scores = self.bm25.scores(question)
        meilleurs = np.argsort(-scores)[:k]
        maximum = self.bm25.score_maximal(question)
        preuve = float(scores[meilleurs[0]] / maximum) if maximum > 0 else 0.0
        return [self.passages[i] for i in meilleurs], preuve
