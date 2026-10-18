"""Construit le benchmark (corpus + questions dev/test) à partir de SQuAD 2.0 et HotpotQA.

Catégories :
  - simple       : question SQuAD 2.0 avec réponse (un seul passage suffit)
  - complexe     : question HotpotQA (raisonnement sur deux passages)
  - non_couverte : question SQuAD 2.0 sans réponse dans le passage (comportement attendu : s'abstenir)
La catégorie « ambigue » est annotée à la main (data/benchmark/ambigues_a_annoter.jsonl).

Usage :
  python bench/build_benchmark.py            # télécharge les sources si besoin, puis construit
"""
import hashlib
import json
import random
import urllib.request

import pandas as pd

from common import BRUTES, DONNEES, charger_config, ecrire_jsonl

FICHIERS = {"squad2_dev": "squad2_dev.json", "hotpotqa_dev_distractor": "hotpot_dev_distractor.parquet"}


def telecharger_sources(cfg):
    BRUTES.mkdir(parents=True, exist_ok=True)
    for nom, info in cfg["benchmark"]["sources"].items():
        chemin = BRUTES / FICHIERS[nom]
        if not chemin.exists():
            print(f"Téléchargement de {nom}…")
            urllib.request.urlretrieve(info["url"], chemin)
        empreinte = hashlib.sha256(chemin.read_bytes()).hexdigest()
        if empreinte != info["sha256"]:
            raise SystemExit(f"Empreinte SHA-256 inattendue pour {nom} : {empreinte}")


def passages_squad(squad):
    """Tous les paragraphes de SQuAD 2.0 dev, avec leurs questions."""
    corpus, questions = [], []
    for ia, article in enumerate(squad["data"]):
        titre = article["title"].replace("_", " ")
        for ip, para in enumerate(article["paragraphs"]):
            pid = f"squad:{ia}:{ip}"
            corpus.append({"id": pid, "titre": titre, "texte": para["context"], "source": "squad2"})
            for qa in para["qas"]:
                questions.append({
                    "id": f"squad:{qa['id']}",
                    "article": ia,
                    "question": qa["question"].strip(),
                    "reponses": sorted({a["text"] for a in qa["answers"]}),
                    "impossible": qa["is_impossible"],
                    "passages_gold": [pid],
                })
    return corpus, questions


def passage_hotpot(titre, phrases):
    texte = "".join(phrases).strip()
    cle = hashlib.sha1(f"{titre}\n{texte}".encode()).hexdigest()[:10]
    return {"id": f"hotpot:{cle}", "titre": titre, "texte": texte, "source": "hotpotqa"}


def main():
    cfg = charger_config()
    telecharger_sources(cfg)
    rng = random.Random(cfg["graine"])
    n_dev = cfg["benchmark"]["questions_par_categorie"]["dev"]
    n_test = cfg["benchmark"]["questions_par_categorie"]["test"]

    # --- SQuAD 2.0 : séparation dev/test par article pour éviter les fuites entre les deux jeux
    squad = json.loads((BRUTES / FICHIERS["squad2_dev"]).read_text(encoding="utf-8"))
    corpus, qs_squad = passages_squad(squad)
    articles = list(range(len(squad["data"])))
    rng.shuffle(articles)
    n_art_dev = round(len(articles) * cfg["benchmark"]["part_articles_squad_dev"])
    articles_dev = set(articles[:n_art_dev])

    def tirer(pool, n, etiquette):
        if len(pool) < n:
            raise SystemExit(f"Pas assez de questions pour {etiquette} ({len(pool)} < {n})")
        return rng.sample(pool, n)

    selection = {"dev": [], "test": []}
    for split, n in (("dev", n_dev), ("test", n_test)):
        dans_split = [q for q in qs_squad if (q["article"] in articles_dev) == (split == "dev")]
        for categorie, impossible in (("simple", False), ("non_couverte", True)):
            pool = [q for q in dans_split if q["impossible"] == impossible]
            for q in tirer(pool, n, f"{categorie}/{split}"):
                selection[split].append({
                    "id": q["id"], "split": split, "categorie": categorie, "source": "squad2",
                    "question": q["question"],
                    "reponses": [] if impossible else q["reponses"],
                    "passages_gold": q["passages_gold"],
                    "comportement_attendu": "s_abstenir" if impossible else "repondre",
                })

    # --- HotpotQA (distractor, niveau « hard ») : tirage aléatoire sans recouvrement dev/test
    hotpot = pd.read_parquet(BRUTES / FICHIERS["hotpotqa_dev_distractor"])
    indices = list(range(len(hotpot)))
    rng.shuffle(indices)
    tirages = {"dev": indices[:n_dev], "test": indices[n_dev:n_dev + n_test]}
    vus = {p["id"] for p in corpus}
    for split, idx in tirages.items():
        for i in idx:
            ligne = hotpot.iloc[i]
            par_titre = {}
            for titre, phrases in zip(ligne["context"]["title"], ligne["context"]["sentences"]):
                p = passage_hotpot(titre, list(phrases))
                par_titre[titre] = p["id"]
                if p["id"] not in vus:
                    corpus.append(p)
                    vus.add(p["id"])
            gold = sorted({par_titre[t] for t in ligne["supporting_facts"]["title"]})
            selection[split].append({
                "id": f"hotpot:{ligne['id']}", "split": split, "categorie": "complexe", "source": "hotpotqa",
                "question": ligne["question"].strip(),
                "reponses": [ligne["answer"]],
                "passages_gold": gold,
                "comportement_attendu": "repondre",
                "type_hotpot": ligne["type"],
            })

    for split in selection:
        rng.shuffle(selection[split])
        ecrire_jsonl(DONNEES / f"{split}.jsonl", selection[split])
    ecrire_jsonl(DONNEES / "corpus.jsonl", corpus)

    manifeste = {
        "graine": cfg["graine"],
        "corpus": {"passages": len(corpus),
                   "par_source": pd.Series([p["source"] for p in corpus]).value_counts().to_dict()},
        "questions": {s: pd.Series([q["categorie"] for q in selection[s]]).value_counts().to_dict()
                      for s in selection},
        "articles_squad": {"dev": sorted(articles_dev), "test": sorted(set(articles) - articles_dev)},
        "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted(DONNEES.glob("*.jsonl")) if f.name != "ambigues_a_annoter.jsonl"},
    }
    (DONNEES / "MANIFEST.json").write_text(json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifeste["corpus"], ensure_ascii=False))
    print(json.dumps(manifeste["questions"], ensure_ascii=False))


if __name__ == "__main__":
    main()
