"""Filter by NER entity count using cached NER results.

Reads   data/combined/step_1/*.jsonl
Writes  data/combined/step_2/train.jsonl  (filtered)
        data/combined/step_2/test_*.jsonl  (copied as-is)

Requires NER cache at CACHE_PATH. Run ner_claims.py first:
  uv run python -m decomposer.data_process.ner_claims

Run:  uv run python -m decomposer.data_process.filter_num_of_ner
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import jsonlines
import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MIN_NER_ENTITIES = 2
NER_CACHE_PATH = "data/combined/cache/claim_ner_cache.npz"

INPUT_DIR = Path("data/combined/step_1")
OUTPUT_DIR = Path("data/combined/step_2")

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


def load_ner_cache() -> dict[str, list[str]]:
    """Load NER cache. Returns dict of claim -> list of all entities.

    Requires ner_claims.py to have been run first.
    """
    if not os.path.exists(NER_CACHE_PATH):
        raise FileNotFoundError(
            f"NER cache not found at {NER_CACHE_PATH}. "
            "Run `uv run python -m decomposer.data_process.ner_claims` first."
        )
    data = np.load(NER_CACHE_PATH, allow_pickle=False)
    ids = data["ids"].tolist()
    all_ents = data["all_entities"].tolist()
    return {id_: json.loads(ents) for id_, ents in zip(ids, all_ents)}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def step_ner_entity_count(
    data: list[dict], ner_cache: dict[str, list[str]]
) -> list[dict]:
    """Remove claims with < MIN_NER_ENTITIES named entities.

    Uses cached NER results (union of en_core_sci_lg + en_core_web_trf).
    Raises if any claim is missing from the cache.
    """
    kept = []
    distribution = Counter()
    for item in data:
        claim = item["claim"]
        if claim not in ner_cache:
            raise ValueError(
                f"Claim missing from NER cache: {claim!r}. "
                "Re-run `uv run python -m decomposer.data_process.ner_claims`."
            )
        ents = ner_cache[claim]
        distribution[len(ents)] += 1
        if len(ents) >= MIN_NER_ENTITIES:
            kept.append(item)

    print("  Entity count distribution:")
    for entity_count, count in sorted(distribution.items()):
        print(f"    {entity_count} entities: {count} rows")

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

    # Load NER cache
    print(f"\nLoading NER cache from {NER_CACHE_PATH}")
    ner_cache = load_ner_cache()
    print(f"  {len(ner_cache)} claims in NER cache")

    # Filter
    prev_data = train_data
    train_data = step_ner_entity_count(train_data, ner_cache)
    print_summary(
        train_data,
        "NER entity count",
        prev_data,
        params={"MIN_NER_ENTITIES": MIN_NER_ENTITIES},
    )

    # Write filtered train
    train_out = OUTPUT_DIR / "train.jsonl"
    with jsonlines.open(train_out, "w") as writer:
        for item in train_data:
            writer.write(item)
    print(f"\nWrote {len(train_data)} rows to {train_out}")

    # Copy test files as-is
    test_files = sorted(f for f in INPUT_DIR.glob("*.jsonl") if f.name != "train.jsonl")
    for test_path in test_files:
        shutil.copy2(test_path, OUTPUT_DIR / test_path.name)
        print(f"Copied {test_path.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
