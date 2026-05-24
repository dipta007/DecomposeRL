"""MiniCheck complexity filter — keep claims where evidence clearly supports/refutes.

Reads   data/combined/step_2/*.jsonl   (output of filter.py)
Writes  data/combined/step_3/train.jsonl  (filtered)
        data/combined/step_3/test_*.jsonl  (copied as-is)
Cache   data/combined/cache/minicheck_cache.npz

Run:  uv run python -m decomposer.data_process.complexity_minicheck
"""

import os
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import jsonlines
import numpy as np
from minicheck.minicheck import MiniCheck
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MINICHECK_MODEL = "Bespoke-MiniCheck-7B"
MINICHECK_CACHE_DIR = "./ckpts"
MINICHECK_MAX_MODEL_LEN = 16768
CORRECTNESS_LOWER = 0.3  # remove likely mislabeled data (MiniCheck strongly disagrees)
CORRECTNESS_UPPER = 0.8  # remove trivially easy data (decomposition unnecessary)

INPUT_DIR = Path("data/combined/step_2")
OUTPUT_DIR = Path("data/combined/step_3")
CACHE_PATH = Path("data/combined/cache/minicheck_cache.npz")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_key(evidence: str, claim: str) -> str:
    """Deterministic hash key for an (evidence, claim) pair."""
    h = hashlib.sha256(f"{evidence}\n---\n{claim}".encode()).hexdigest()
    return h


def load_cache() -> dict[str, float]:
    """Load cached MiniCheck raw probabilities. Returns {key: raw_prob}."""
    if not CACHE_PATH.exists():
        return {}
    data = np.load(CACHE_PATH, allow_pickle=False)
    keys = data["keys"].tolist()
    probs = data["probs"].tolist()
    return dict(zip(keys, probs))


def save_cache(cache: dict[str, float]) -> None:
    """Save MiniCheck raw probabilities to .npz atomically to prevent corruption."""
    if not cache:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cache.keys())
    probs = [cache[k] for k in keys]
    tmp_path = str(CACHE_PATH) + ".tmp.npz"
    np.savez(tmp_path, keys=np.array(keys), probs=np.array(probs, dtype=np.float64))
    os.replace(tmp_path, str(CACHE_PATH))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_summary(
    data: list[dict],
    prev_data: list[dict],
):
    """Print per-source before/after/removed counts."""
    src_before = Counter(item["src"] for item in prev_data)
    src_after = Counter(item["src"] for item in data)
    all_srcs = sorted(set(src_before) | set(src_after))

    header = f"{'src':<25} {'Before':>8} {'Removed':>8} {'After':>8}"
    sep = "-" * len(header)

    print(f"\n=== MiniCheck filter ({CORRECTNESS_LOWER} <= correctness <= {CORRECTNESS_UPPER}) ===")
    print(sep)
    print(header)
    print(sep)
    for src in all_srcs:
        before = src_before.get(src, 0)
        after = src_after.get(src, 0)
        removed = before - after
        print(f"{src:<25} {before:>8} {removed:>8} {after:>8}")
    print(sep)

    total_before = len(prev_data)
    total_after = len(data)
    total_removed = total_before - total_after
    pct = (total_removed / total_before * 100) if total_before else 0
    print(
        f"{'TOTAL':<25} {total_before:>8} {total_removed:>8} {total_after:>8}  ({pct:.1f}% removed)"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read train.jsonl from INPUT_DIR
    train_path = INPUT_DIR / "train.jsonl"
    with jsonlines.open(train_path, "r") as reader:
        train_data = list(reader)
    print(f"Loaded {len(train_data)} train rows from {train_path}")

    # 2. Load cache and find uncached items
    cache = load_cache()
    print(f"Already cached: {len(cache)}")

    keys = [_make_key(item["evidence"], item["claim"]) for item in train_data]
    uncached_indices = [i for i, k in enumerate(keys) if k not in cache]
    print(f"Need to score: {len(uncached_indices)}")

    # 3. Score only uncached (evidence, claim) pairs
    if uncached_indices:
        print(f"Loading MiniCheck model: {MINICHECK_MODEL}")
        scorer = MiniCheck(
            model_name=MINICHECK_MODEL,
            cache_dir=MINICHECK_CACHE_DIR,
            max_model_len=MINICHECK_MAX_MODEL_LEN,
        )

        docs = [train_data[i]["evidence"] for i in uncached_indices]
        claims = [train_data[i]["claim"] for i in uncached_indices]
        _, raw_probs, _, _ = scorer.score(docs=docs, claims=claims)

        for idx, prob in zip(uncached_indices, raw_probs):
            cache[keys[idx]] = float(prob)

        save_cache(cache)
        print(f"Saved cache ({len(cache)} entries)")
    else:
        print("All scores already cached — skipping model load.")

    # 4. Compute correctness from cached raw probs
    correctness_scores = []
    for item, key in zip(train_data, keys):
        prob = cache[key]
        if item["label"] == "Supported":
            correctness = prob
        else:
            correctness = 1.0 - prob
        correctness_scores.append(correctness)

    # 5. Filter by threshold (remove mislabeled + trivially easy)
    filtered = [
        item
        for item, score in zip(train_data, correctness_scores)
        if CORRECTNESS_LOWER <= score <= CORRECTNESS_UPPER
    ]

    # 6. Print summary
    print_summary(filtered, train_data)

    # 7. Write filtered train to OUTPUT_DIR
    train_out = OUTPUT_DIR / "train.jsonl"
    with jsonlines.open(train_out, "w") as writer:
        for item in filtered:
            writer.write(item)
    print(f"\nWrote {len(filtered)} rows to {train_out}")

    # 8. Copy test files as-is
    test_files = sorted(f for f in INPUT_DIR.glob("*.jsonl") if f.name != "train.jsonl")
    for test_path in test_files:
        shutil.copy2(test_path, OUTPUT_DIR / test_path.name)
        print(f"Copied {test_path.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
