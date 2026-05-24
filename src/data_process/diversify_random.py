"""Random train set sampling baseline for the diversity step.

Mirror of `diversify_submod.py` but selects items by uniform random
sampling within each label bucket instead of FacilityLocation submodular
optimization. Used as the ablation baseline for the data-curation claim.

Reads   data/combined/step_6/*.jsonl
Writes  data/combined/step_7_random/train.jsonl  (random subset)
        data/combined/step_7_random/test_*.jsonl  (copied as-is)

Run:  uv run python -m decomposer.data_process.diversify_random
"""

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_BUDGET = 5000
LABEL_RATIO = {"Supported": 0.5, "Refuted": 0.5}
RANDOM_SEED = 42
INPUT_DIR = Path("data/combined/step_6")
OUTPUT_DIR = Path("data/combined/step_7_random")

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    # 1. Load train data
    train_path = INPUT_DIR / "train.jsonl"
    with jsonlines.open(train_path, "r") as reader:
        train_data = list(reader)
    print(f"Loaded {len(train_data)} train rows from {train_path}")
    print_summary(train_data, "Initial")

    label_counts = Counter(item["label"] for item in train_data)
    print(f"\nLabel distribution: {dict(label_counts)}")

    # 2. Group items by label
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(train_data):
        by_label[item["label"]].append(i)

    # 3. Uniform random sample within each label bucket up to its budget
    selected_indices: list[int] = []
    for label, ratio in LABEL_RATIO.items():
        label_budget = round(TOTAL_BUDGET * ratio)
        available = by_label.get(label, [])
        if not available:
            print(f"\nWarning: no items with label={label!r}, skipping")
            continue

        take = min(label_budget, len(available))
        chosen = rng.sample(available, take)
        selected_indices.extend(chosen)
        print(
            f"\n--- Label: {label} (budget={label_budget}) ---"
            f"\n  available={len(available)}  sampled={take}"
        )

    # 4. Collect selected items
    selected_indices = sorted(set(selected_indices))
    selected_data = [train_data[i] for i in selected_indices]

    # 5. Print summary
    print_summary(
        selected_data,
        "Random selection",
        train_data,
        params={
            "TOTAL_BUDGET": TOTAL_BUDGET,
            "LABEL_RATIO": LABEL_RATIO,
            "RANDOM_SEED": RANDOM_SEED,
        },
    )

    final_label_counts = Counter(item["label"] for item in selected_data)
    print(f"\nFinal label distribution: {dict(final_label_counts)}")
    for lbl, cnt in sorted(final_label_counts.items()):
        pct = cnt / len(selected_data) * 100 if selected_data else 0
        print(f"  {lbl}: {cnt} ({pct:.1f}%)")

    # 6. Write output
    train_out = OUTPUT_DIR / "train.jsonl"
    with jsonlines.open(train_out, "w") as writer:
        for item in selected_data:
            writer.write(item)
    print(f"\nWrote {len(selected_data)} rows to {train_out}")

    # 7. Copy test files as-is
    test_files = sorted(f for f in INPUT_DIR.glob("*.jsonl") if f.name != "train.jsonl")
    for test_path in test_files:
        shutil.copy2(test_path, OUTPUT_DIR / test_path.name)
        print(f"Copied {test_path.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
