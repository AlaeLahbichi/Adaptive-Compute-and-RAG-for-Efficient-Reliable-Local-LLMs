"""Génération locale avec MLX (Apple Silicon) et prompts figés du protocole."""
import mlx.core as mx
from mlx_lm import generate, load

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


def memoire_pic_go():
    return mx.get_peak_memory() / 1e9


def reinitialiser_memoire_pic():
    mx.reset_peak_memory()
