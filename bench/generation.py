"""Génération locale avec MLX (Apple Silicon) et prompts figés du protocole."""
import mlx.core as mx
import numpy as np
from mlx_lm import generate, load, stream_generate

from metrics import MARQUEUR_ABSTENTION

PROMPT_SANS_RAG = (
    "Answer the question with a short phrase only. If you do not know the answer or it cannot be "
    f"determined, reply exactly: {MARQUEUR_ABSTENTION}.\n\nQuestion: {{question}}\nAnswer:"
)

PROMPT_RAG = (
    "Answer the question using only the passages below, with a short phrase only. If the passages do "
    f"not contain the answer, reply exactly: {MARQUEUR_ABSTENTION}.\n\n{{passages}}\n\n"
    "Question: {question}\nAnswer:"
)


def construire_prompt(question, passages):
    if not passages:
        return PROMPT_SANS_RAG.format(question=question)
    bloc = "\n\n".join(f"[{i}] {p['titre']}: {p['texte']}" for i, p in enumerate(passages, 1))
    return PROMPT_RAG.format(question=question, passages=bloc)


class Generateur:
    def __init__(self, nom_modele, max_tokens):
        self.nom = nom_modele
        self.max_tokens = max_tokens
        self.modele, self.tokenizer = load(nom_modele)

    def repondre(self, contenu):
        """Renvoie (texte, nb_tokens_prompt, nb_tokens_generes)."""
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": contenu}], add_generation_prompt=True, tokenize=False)
        texte = generate(self.modele, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens)
        n_prompt = len(self.tokenizer.encode(prompt))
        n_gen = len(self.tokenizer.encode(texte, add_special_tokens=False))
        return texte.strip(), n_prompt, n_gen

    def repondre_avec_confiance(self, contenu):
        """Comme repondre, avec la confiance du modèle : probabilités des tokens générés (décodage glouton).

        Renvoie un dict : texte, tokens_prompt, tokens_generes, conf_moy (moyenne géométrique des
        probabilités des tokens) et conf_min (probabilité du token le moins sûr).
        """
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": contenu}], add_generation_prompt=True, tokenize=False)
        morceaux, logprobs, n_prompt = [], [], 0
        for r in stream_generate(self.modele, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens):
            morceaux.append(r.text)
            logprobs.append(r.logprobs[r.token].item())
            n_prompt = r.prompt_tokens
        lp = np.array(logprobs) if logprobs else np.array([0.0])
        return {
            "texte": "".join(morceaux).strip(),
            "tokens_prompt": n_prompt,
            "tokens_generes": len(logprobs),
            "conf_moy": float(np.exp(lp.mean())),
            "conf_min": float(np.exp(lp.min())),
        }


def memoire_pic_go():
    return mx.get_peak_memory() / 1e9


def reinitialiser_memoire_pic():
    mx.reset_peak_memory()
