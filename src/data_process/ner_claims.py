# /// script
# dependencies = [
#   "jsonlines",
#   "spacy[cuda12x]==3.7.4",
#   "scispacy>=0.5.5",
#   "en-core-sci-lg @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz",
#   "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"
# ]
# ///
"""Cache NER entities for all claims.

Reads   data/combined/step_1/*.jsonl
Writes  data/combined/cache/claim_ner_cache.npz

For each unique claim, runs en_core_sci_lg (biomedical NER) and
en_core_web_trf (general NER), then stores per claim:
  - sci_entities: entities from en_core_sci_lg
  - web_entities: entities from en_core_web_trf
  - all_entities: union of both

Run:  uv run python -m decomposer.data_process.ner_claims
"""

import glob
import json
import os

import jsonlines
import numpy as np
import spacy
from tqdm import tqdm

INPUT_DIR = "data/combined/step_1"
CACHE_PATH = "data/combined/cache/claim_ner_cache.npz"
BATCH_SIZE = 512
SAVE_EVERY = 1000


def load_all_claims():
    """Load all unique claims from step_1 jsonl files."""
    claims = {}
    fpath = os.path.join(INPUT_DIR, "train.jsonl")
    with jsonlines.open(fpath) as reader:
        for row in reader:
            key = row["claim"]
            claims[key] = key
    return claims


def load_cache():
    """Load cached NER results. Returns dict of claim -> {sci, web, all}."""
    if not os.path.exists(CACHE_PATH):
        return {}
    data = np.load(CACHE_PATH, allow_pickle=False)
    ids = data["ids"].tolist()
    sci = data["sci_entities"].tolist()
    web = data["web_entities"].tolist()
    all_ = data["all_entities"].tolist()
    cache = {}
    for i, id_ in enumerate(ids):
        cache[id_] = {
            "sci": json.loads(sci[i]),
            "web": json.loads(web[i]),
            "all": json.loads(all_[i]),
        }
    return cache


def save_cache(cache):
    """Save all NER results to a single .npz file atomically to prevent corruption."""
    if not cache:
        return
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    ids = list(cache.keys())
    sci = [json.dumps(cache[id_]["sci"]) for id_ in ids]
    web = [json.dumps(cache[id_]["web"]) for id_ in ids]
    all_ = [json.dumps(cache[id_]["all"]) for id_ in ids]
    tmp_path = CACHE_PATH + ".tmp.npz"
    np.savez(
        tmp_path,
        ids=np.array(ids),
        sci_entities=np.array(sci),
        web_entities=np.array(web),
        all_entities=np.array(all_),
    )
    os.replace(tmp_path, CACHE_PATH)


def main():
    claims = load_all_claims()
    print(f"Total unique claims: {len(claims)}")

    cache = load_cache()
    print(f"Already cached: {len(cache)}")

    missing_ids = [id_ for id_ in claims if id_ not in cache]
    print(f"Need to process: {len(missing_ids)}")

    if not missing_ids:
        print("All NER results already cached.")
        return

    print("Loading spacy models...")
    nlp_sci = spacy.load("en_core_sci_lg")
    nlp_web = spacy.load("en_core_web_trf")

    processed = 0

    for doc_sci, doc_web, claim in zip(
        tqdm(
            nlp_sci.pipe(missing_ids, batch_size=BATCH_SIZE),
            total=len(missing_ids),
            desc="NER",
        ),
        nlp_web.pipe(missing_ids, batch_size=BATCH_SIZE),
        missing_ids,
    ):
        sci_ents = sorted({e.text for e in doc_sci.ents})
        web_ents = sorted({e.text for e in doc_web.ents})
        all_ents = sorted(set(sci_ents) | set(web_ents))

        cache[claim] = {
            "sci": sci_ents,
            "web": web_ents,
            "all": all_ents,
        }

        processed += 1
        if processed % SAVE_EVERY == 0:
            save_cache(cache)
            print(f"  Saved cache ({len(cache)} entries)")

    save_cache(cache)
    print(f"\nDone. Total cached: {len(cache)}")


if __name__ == "__main__":
    main()
