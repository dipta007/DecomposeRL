"""Diversity-based train set sampling (final curation step).

Reads   data/combined/step_6/*.jsonl
Writes  data/combined/step_7_submod/train.jsonl  (diverse subset)
        data/combined/step_7_submod/test_*.jsonl  (copied as-is)

Uses submodular optimization (FacilityLocation via submodlib) for
embedding-space diversity, sqrt-proportional source allocation, and
label-balanced stratification.

Run:  uv run python -m decomposer.data_process.diversity
"""

import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_BUDGET = 5_000
LABEL_RATIO = {"Supported": 0.5, "Refuted": 0.5}
EMBEDDING_CACHE_PATH = "data/combined/cache/claim_embeddings.npz"
FACILITY_LOCATION_METRIC = "cosine"
INPUT_DIR = Path("data/combined/step_6")
OUTPUT_DIR = Path("data/combined/step_7_submod")

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


# ---------------------------------------------------------------------------
# Submodular selection
# ---------------------------------------------------------------------------


def select_diverse(embeddings: np.ndarray, budget: int) -> list[int]:
    """Select diverse subset via FacilityLocation."""
    from submodlib import FacilityLocationFunction

    if budget >= len(embeddings):
        return list(range(len(embeddings)))
    obj = FacilityLocationFunction(
        n=len(embeddings),
        data=embeddings,
        mode="dense",
        metric=FACILITY_LOCATION_METRIC,
    )
    greedy_list = obj.maximize(budget=budget, optimizer="LazyGreedy")
    return [idx for idx, _ in greedy_list]


def compute_source_budgets(
    source_counts: dict[str, int], total_label_budget: int
) -> dict[str, int]:
    """Compute per-source budgets using sqrt-proportional allocation."""
    sqrt_counts = {src: math.sqrt(count) for src, count in source_counts.items()}
    sqrt_total = sum(sqrt_counts.values())

    budgets = {}
    for src, count in source_counts.items():
        raw_budget = total_label_budget * sqrt_counts[src] / sqrt_total
        # Cap at actual group size
        budgets[src] = min(round(raw_budget), count)

    # Adjust rounding: distribute any leftover to sources with remaining capacity
    allocated = sum(budgets.values())
    deficit = total_label_budget - allocated
    if deficit > 0:
        for src in sorted(source_counts, key=lambda s: source_counts[s], reverse=True):
            if budgets[src] < source_counts[src]:
                add = min(deficit, source_counts[src] - budgets[src])
                budgets[src] += add
                deficit -= add
                if deficit <= 0:
                    break

    return budgets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load train data
    train_path = INPUT_DIR / "train.jsonl"
    with jsonlines.open(train_path, "r") as reader:
        train_data = list(reader)
    print(f"Loaded {len(train_data)} train rows from {train_path}")
    print_summary(train_data, "Initial")

    # Print label distribution
    label_counts = Counter(item["label"] for item in train_data)
    print(f"\nLabel distribution: {dict(label_counts)}")

    # 2. Load embeddings
    print(f"\nLoading embeddings from {EMBEDDING_CACHE_PATH}")
    emb_lookup, emb_matrix = load_embeddings()
    print(f"  {len(emb_lookup)} claim embeddings loaded")

    # 3. Group items by (label, source)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, item in enumerate(train_data):
        groups[(item["label"], item["src"])].append(i)

    # 4. Collect all (label, source) tasks and prepare embeddings
    tasks = []  # (label, src, group_indices, group_embs, src_budget)

    for label, ratio in LABEL_RATIO.items():
        label_budget = round(TOTAL_BUDGET * ratio)

        # Get per-source counts for this label
        source_counts = {}
        for (lbl, src), idxs in groups.items():
            if lbl == label:
                source_counts[src] = len(idxs)

        if not source_counts:
            print(f"\nWarning: no items with label={label!r}, skipping")
            continue

        # Compute sqrt-proportional budgets
        budgets = compute_source_budgets(source_counts, label_budget)

        print(f"\n--- Label: {label} (budget={label_budget}) ---")
        for src in sorted(budgets):
            print(
                f"  {src:<25} available={source_counts[src]:>6}  budget={budgets[src]:>6}"
            )

        for src, src_budget in tqdm(
            budgets.items(), desc=f"Preparing embeddings ({label})"
        ):
            group_indices = groups[(label, src)]
            group_items = [train_data[i] for i in group_indices]
            group_embs = get_claim_embeddings(group_items, emb_lookup, emb_matrix)
            tasks.append((label, src, group_indices, group_embs, src_budget))

    # 5. Run FacilityLocation selection in parallel across all groups
    print(f"\nRunning {len(tasks)} diversity selection tasks in parallel...")
    results = Parallel(n_jobs=4, return_as="generator")(
        delayed(select_diverse)(embs, budget) for _, _, _, embs, budget in tasks
    )
    results = list(tqdm(results, total=len(tasks), desc="FacilityLocation selection"))

    # 6. Map results back to global indices
    selected_indices: list[int] = []
    for (label, src, group_indices, _, _), diverse_local in zip(tasks, results):
        for local_idx in diverse_local:
            selected_indices.append(group_indices[local_idx])
        print(f"  {src:<25} selected {len(diverse_local):>6} / {len(group_indices):>6}")

    # 7. Collect selected items
    selected_indices = sorted(set(selected_indices))
    selected_data = [train_data[i] for i in selected_indices]

    # 8. Print summary
    print_summary(
        selected_data,
        "Diversity selection",
        train_data,
        params={
            "TOTAL_BUDGET": TOTAL_BUDGET,
            "LABEL_RATIO": LABEL_RATIO,
            "METRIC": FACILITY_LOCATION_METRIC,
        },
    )

    # Final label distribution
    final_label_counts = Counter(item["label"] for item in selected_data)
    print(f"\nFinal label distribution: {dict(final_label_counts)}")
    for lbl, cnt in sorted(final_label_counts.items()):
        pct = cnt / len(selected_data) * 100 if selected_data else 0
        print(f"  {lbl}: {cnt} ({pct:.1f}%)")

    # 7. Write output
    train_out = OUTPUT_DIR / "train.jsonl"
    with jsonlines.open(train_out, "w") as writer:
        for item in selected_data:
            writer.write(item)
    print(f"\nWrote {len(selected_data)} rows to {train_out}")

    # 8. Copy test files as-is
    test_files = sorted(f for f in INPUT_DIR.glob("*.jsonl") if f.name != "train.jsonl")
    for test_path in test_files:
        shutil.copy2(test_path, OUTPUT_DIR / test_path.name)
        print(f"Copied {test_path.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
