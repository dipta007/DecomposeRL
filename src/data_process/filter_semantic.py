"""Semantic-embedding filtering: dedup & decontamination.

Reads   data/combined/step_2/*.jsonl   (output of filter.py)
Writes  data/combined/step_3/train.jsonl  (overwritten with semantic-filtered data)

Requires claim embeddings to exist at EMBEDDING_CACHE_PATH.
Run embed_claims.py first to generate them.

Run:  uv run python -m decomposer.data_process.filter_semantic
"""

from collections import Counter
from pathlib import Path

import jsonlines
import numpy as np
import torch
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MINHASH_THRESHOLD = 0.7
MINHASH_NUM_PERM = 128
SEMANTIC_DEDUP_THRESHOLD = 0.70
EMBEDDING_CACHE_PATH = "data/combined/cache/claim_embeddings.npz"
DECONTAM_SEMANTIC_THRESHOLD = 0.90

INPUT_DIR = Path("data/combined/step_3")
OUTPUT_DIR = Path("data/combined/step_4")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_summary(
    data: list[dict],
    step_name: str,
    prev_data: list[dict] | None = None,
    params: dict[str, object] | None = None,
):
    """Print per-source count table after a filtering step, with per-source removals."""
    src_after = Counter(item["src"] for item in data)
    src_before = (
        Counter(item["src"] for item in prev_data) if prev_data is not None else None
    )

    all_srcs = sorted(
        (set(src_before) | set(src_after)) if src_before else set(src_after)
    )

    if src_before is not None:
        header = f"{'src':<25} {'Before':>8} {'Removed':>8} {'After':>8}"
    else:
        header = f"{'src':<25} {'Count':>8}"
    sep = "-" * len(header)

    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n=== {step_name} ({param_str}) ===")
    else:
        print(f"\n=== {step_name} ===")
    print(sep)
    print(header)
    print(sep)
    for src in all_srcs:
        after = src_after.get(src, 0)
        if src_before is not None:
            before = src_before.get(src, 0)
            removed = before - after
            print(f"{src:<25} {before:>8} {removed:>8} {after:>8}")
        else:
            print(f"{src:<25} {after:>8}")
    print(sep)

    total_after = len(data)
    if src_before is not None:
        total_before = len(prev_data)
        total_removed = total_before - total_after
        pct = (total_removed / total_before * 100) if total_before else 0
        print(
            f"{'TOTAL':<25} {total_before:>8} {total_removed:>8} {total_after:>8}  ({pct:.1f}% removed)"
        )
    else:
        print(f"{'TOTAL':<25} {total_after:>8}")
    print(sep)


def build_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Build a MinHash from word-level shingles of text."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for w in words:
        m.update(w.encode("utf-8"))
    return m


def load_embeddings():
    """Load the claim embedding cache. Returns (lookup dict, embedding matrix)."""
    data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=False)
    emb_ids = data["ids"].tolist()
    emb_matrix = data["embeddings"]
    emb_lookup = {id_: i for i, id_ in enumerate(emb_ids)}
    return emb_lookup, emb_matrix


def get_claim_embeddings(
    items: list[dict], emb_lookup: dict, emb_matrix: np.ndarray
) -> np.ndarray:
    """Return embedding matrix for items in order. Raises if any claim is missing."""
    indices = []
    for item in items:
        idx = emb_lookup.get(item["claim"])
        if idx is None:
            raise ValueError(f"Claim missing from embedding cache: {item['claim']!r}")
        indices.append(idx)
    return emb_matrix[indices].astype(np.float32)


def semantic_filter_against(
    source_items: list[dict],
    source_embs: np.ndarray,
    ref_embs: np.ndarray,
    threshold: float,
    batch_size: int = 4096,
) -> list[dict]:
    """Remove source items whose embedding is too similar to any reference embedding.
    Uses GPU-accelerated batched cosine similarity."""
    if len(source_embs) == 0 or len(ref_embs) == 0:
        return source_items

    src_t = torch.from_numpy(source_embs).to(DEVICE)
    ref_t = torch.from_numpy(ref_embs).to(DEVICE)
    src_t = torch.nn.functional.normalize(src_t, dim=1)
    ref_t = torch.nn.functional.normalize(ref_t, dim=1)

    keep = torch.ones(len(source_embs), dtype=torch.bool)
    for i in range(0, len(src_t), batch_size):
        sim = src_t[i : i + batch_size] @ ref_t.T
        max_sim = sim.max(dim=1).values.cpu()
        keep[i : i + batch_size] = max_sim < threshold

    keep = keep.numpy()
    return [item for item, k in zip(source_items, keep) if k]


def semantic_dedup_greedy(
    items: list[dict],
    embs: np.ndarray,
    threshold: float,
    batch_size: int = 4096,
) -> list[dict]:
    """Greedy self-dedup: keep first occurrence, remove later items that are
    too similar (>= threshold) to any already-kept item.

    Uses GPU for batched cosine similarity, processes rows in chunks.
    """
    if len(embs) == 0:
        return items

    n = len(items)
    removed = torch.zeros(n, dtype=torch.bool)

    # Normalize and move to GPU once
    embs_t = torch.from_numpy(embs).to(DEVICE)
    embs_t = torch.nn.functional.normalize(embs_t, dim=1)

    for i in tqdm(range(n), desc="Semantic dedup"):
        if removed[i]:
            continue
        # Compute similarity of item i against all j > i on GPU
        tail = embs_t[i + 1 :]
        if tail.shape[0] == 0:
            break
        sims = tail @ embs_t[i]  # (n - i - 1,)
        # Mark similar items as removed (offset by i+1)
        hits = (sims >= threshold).cpu()
        removed[i + 1 :] |= hits

    removed = removed.numpy()
    return [item for i, item in enumerate(items) if not removed[i]]


# ---------------------------------------------------------------------------
# Filtering steps
# ---------------------------------------------------------------------------


def step_semantic_dedup(
    data: list[dict], emb_lookup: dict, emb_matrix: np.ndarray
) -> list[dict]:
    """Semantic dedup using cached claim embeddings. Keep first per cluster."""
    embs = get_claim_embeddings(data, emb_lookup, emb_matrix)
    return semantic_dedup_greedy(data, embs, SEMANTIC_DEDUP_THRESHOLD)


def step_decontamination(
    train_data: list[dict],
    test_claims: list[str],
    emb_lookup: dict,
    emb_matrix: np.ndarray,
) -> list[dict]:
    """Three-pronged decontamination: exact, MinHash, semantic."""
    # --- a. Exact match ---
    test_claims_norm = {c.strip().lower() for c in test_claims}
    after_exact = [
        item
        for item in train_data
        if item["claim"].strip().lower() not in test_claims_norm
    ]
    n_exact = len(train_data) - len(after_exact)
    print(f"  a. Exact match removed: {n_exact}")

    # --- b. MinHash decontamination ---
    test_lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    for i, claim in enumerate(test_claims):
        mh = build_minhash(claim)
        test_lsh.insert(f"test_{i}", mh)

    after_minhash = []
    n_minhash = 0
    for item in tqdm(after_exact, desc="Decontam MinHash"):
        mh = build_minhash(item["claim"])
        if len(test_lsh.query(mh)) == 0:
            after_minhash.append(item)
        else:
            n_minhash += 1
    print(f"  b. MinHash decontam removed: {n_minhash}")

    # --- c. Semantic decontamination ---
    test_emb_indices = [emb_lookup[c] for c in test_claims if c in emb_lookup]
    if not test_emb_indices:
        print("  c. Semantic decontam: no test embeddings found, skipping")
        return after_minhash
    test_embs = emb_matrix[test_emb_indices].astype(np.float32)
    train_embs = get_claim_embeddings(after_minhash, emb_lookup, emb_matrix)
    kept = semantic_filter_against(
        after_minhash, train_embs, test_embs, DECONTAM_SEMANTIC_THRESHOLD
    )
    n_semantic = len(after_minhash) - len(kept)
    print(f"  c. Semantic decontam removed: {n_semantic}")
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load train data
    train_path = INPUT_DIR / "train.jsonl"
    with jsonlines.open(train_path, "r") as reader:
        train_data = list(reader)
    print(f"Loaded {len(train_data)} train rows from {train_path}")
    print_summary(train_data, "Initial")

    # Collect all test claims (for decontamination)
    test_files = sorted(f for f in INPUT_DIR.glob("*.jsonl") if f.name != "train.jsonl")
    test_claims = []
    for test_path in test_files:
        with jsonlines.open(test_path, "r") as reader:
            for item in reader:
                test_claims.append(item["claim"])
    print(f"\nCollected {len(test_claims)} test claims from {len(test_files)} files")

    # Load embeddings
    print(f"\nLoading embeddings from {EMBEDDING_CACHE_PATH}")
    emb_lookup, emb_matrix = load_embeddings()
    print(f"  {len(emb_lookup)} claim embeddings loaded")

    step_num = 0

    def run_step(step_name, fn, *args, params=None):
        nonlocal train_data, step_num
        step_num += 1
        prev_data = train_data
        train_data = fn(train_data, *args)
        print_summary(train_data, f"Step {step_num}: {step_name}", prev_data, params)

    run_step(
        "Semantic dedup",
        step_semantic_dedup,
        emb_lookup,
        emb_matrix,
        params={"SEMANTIC_DEDUP_THRESHOLD": SEMANTIC_DEDUP_THRESHOLD},
    )
    run_step(
        "Decontamination",
        step_decontamination,
        test_claims,
        emb_lookup,
        emb_matrix,
        params={
            "DECONTAM_SEMANTIC_THRESHOLD": DECONTAM_SEMANTIC_THRESHOLD,
            "MINHASH_THRESHOLD": MINHASH_THRESHOLD,
        },
    )

    # Overwrite filtered train
    train_out = OUTPUT_DIR / "train.jsonl"
    with jsonlines.open(train_out, "w") as writer:
        for item in train_data:
            writer.write(item)
    print(f"\nWrote {len(train_data)} rows to {train_out}")

    # Copy test files as-is
    for test_path in test_files:
        out_path = OUTPUT_DIR / test_path.name
        with (
            jsonlines.open(test_path, "r") as reader,
            jsonlines.open(out_path, "w") as writer,
        ):
            for item in reader:
                writer.write(item)
        print(f"Copied {test_path.name} to {out_path}")


if __name__ == "__main__":
    print("Starting semantic filtering...")
    main()
