import os
import glob
import numpy as np
import jsonlines
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

EMBEDDING_MODEL = "text-embedding-3-large"
COST_PER_M_TOKENS = 0.13  # $/1M tokens for text-embedding-3-large
INPUT_DIR = "data/combined/step_3"
CACHE_PATH = "data/combined/cache/claim_embeddings.npz"
BATCH_SIZE = 2048  # OpenAI max batch size for embedding API


def load_all_claims():
    """Load all unique claims from step_3 jsonl files, keyed by claim.lower()."""
    claims = {}
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
    for fpath in files:
        with jsonlines.open(fpath) as reader:
            for row in reader:
                key = row["claim"]
                claims[key] = key
    return claims


def load_cache():
    """Load cached embeddings. Returns dict of id -> embedding."""
    if not os.path.exists(CACHE_PATH):
        return {}
    data = np.load(CACHE_PATH, allow_pickle=False)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"]
    return {id_: embeddings[i] for i, id_ in enumerate(ids)}


def save_cache(cache):
    """Save all embeddings to a single .npz file atomically to prevent corruption."""
    if not cache:
        return
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    ids = list(cache.keys())
    embeddings = np.array([cache[id_] for id_ in ids], dtype=np.float32)
    tmp_path = CACHE_PATH + ".tmp.npz"
    np.savez(tmp_path, ids=np.array(ids), embeddings=embeddings)
    os.replace(tmp_path, CACHE_PATH)


def embed_batch_wrapper(batch_ids, batch_texts, client):
    """Embed a batch of texts using the OpenAI API."""
    response = client.embeddings.create(input=batch_texts, model=EMBEDDING_MODEL)
    embeddings = [item.embedding for item in response.data]
    tokens_used = response.usage.total_tokens
    return batch_ids, embeddings, tokens_used


def main():
    # Load all claims
    claims = load_all_claims()
    print(f"Total claims: {len(claims)}")

    # Load cache and find what's missing
    cache = load_cache()
    print(f"Already cached: {len(cache)}")

    missing_ids = [id_ for id_ in claims if id_ not in cache]
    print(f"Need to embed: {len(missing_ids)}")

    if not missing_ids:
        print("All embeddings already cached.")
        return

    # Prepare batches
    batches = []
    for i in range(0, len(missing_ids), BATCH_SIZE):
        batch_ids = missing_ids[i : i + BATCH_SIZE]
        batch_texts = [claims[id_] for id_ in batch_ids]
        batches.append((batch_ids, batch_texts))
    print(f"Batches to process: {len(batches)}")

    # Embed in parallel, save after each batch completes
    client = OpenAI()
    total_tokens = 0
    total_cost = 0.0
    pbar = tqdm(total=len(batches), desc="Embedding batches")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(embed_batch_wrapper, batch_ids, batch_texts, client): i
            for i, (batch_ids, batch_texts) in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_ids, embeddings, tokens_used = future.result()
            for id_, emb in zip(batch_ids, embeddings):
                cache[id_] = emb

            total_tokens += tokens_used
            batch_cost = tokens_used / 1_000_000 * COST_PER_M_TOKENS
            total_cost += batch_cost
            pbar.set_postfix(
                tokens=f"{total_tokens:,}",
                cost=f"${total_cost:.4f}",
                batch_cost=f"${batch_cost:.4f}",
            )
            pbar.update(1)

            save_cache(cache)

    pbar.close()
    print(f"\nDone. Total tokens: {total_tokens:,}, Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
