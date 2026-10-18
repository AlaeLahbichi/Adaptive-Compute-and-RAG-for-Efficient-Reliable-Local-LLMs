"""Recherche documentaire dense : embeddings MiniLM + similarité cosinus exacte.

Pour ~3 300 passages, une recherche exacte en NumPy est aussi rapide qu'un index FAISS
(IndexFlatIP) et donne exactement les mêmes résultats, sans dépendance supplémentaire.
"""
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from common import CACHE, DONNEES, lire_jsonl


class Recherche:
    def __init__(self, nom_modele):
        self.passages = lire_jsonl(DONNEES / "corpus.jsonl")
        self.encodeur = SentenceTransformer(nom_modele, device="cpu")
        self.matrice = self._embeddings_corpus(nom_modele)

    def _embeddings_corpus(self, nom_modele):
        textes = [f"{p['titre']}. {p['texte']}" for p in self.passages]
        cle = hashlib.sha1((nom_modele + "\n".join(p["id"] for p in self.passages)).encode()).hexdigest()[:12]
        chemin = CACHE / f"embeddings_{cle}.npy"
        if chemin.exists():
            return np.load(chemin)
        CACHE.mkdir(parents=True, exist_ok=True)
        matrice = self.encodeur.encode(textes, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(chemin, matrice.astype(np.float32))
        return matrice

    def chercher(self, question, k):
        """Renvoie les k passages les plus proches (liste de dicts avec un champ 'score')."""
        q = self.encodeur.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.matrice @ q
        meilleurs = np.argsort(-scores)[:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in meilleurs]
