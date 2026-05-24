"""Augment training data with long-evidence samples (step 8).

Reads   data/combined/step_7_farthest/train.jsonl   (curated short-evidence set)
        data/combined/step_2/train.jsonl             (pre-diversity pool with all evidence lengths)
Writes  data/combined/step_8/train.jsonl             (step_7 + long-evidence samples from step_2)
        data/combined/step_8/test_*.jsonl            (copied from step_7)

Keeps all step_7 training data, then adds samples from step_2 whose evidence
exceeds a token-length threshold and that are not already in step_7.
This addresses the distribution gap between training data (median ~500 tokens)
and long-evidence test sets like CoverBench (median ~1200, p90 ~6600 tokens).

Run:  PYTHONPATH=. uv run decomposer/data_process/augment_long_evidence.py
"""

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import jsonlines
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STEP7_DIR = Path("data/combined/step_7_submod")
STEP4_DIR = Path("data/combined/step_2")
OUTPUT_DIR = Path("data/combined/step_8")
TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"

MIN_EVIDENCE_TOKENS = 3000  # only add samples with evidence >= this many tokens
BATCH_SIZE = 512  # tokenizer batch size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_summary(
    data: list[dict], step_name: str, prev_data: list[dict] | None = None
):
    """Print per-source count table."""
    src_after = Counter(item["src"] for item in data)
    src_before = (
        Counter(item["src"] for item in prev_data) if prev_data is not None else None
    )
    all_srcs = sorted(
        (set(src_before) | set(src_after)) if src_before else set(src_after)
    )

    if src_before is not None:
        header = f"{'src':<25} {'Before':>8} {'After':>8} {'Added':>8}"
    else:
        header = f"{'src':<25} {'Count':>8}"
    sep = "-" * len(header)

    print(f"\n=== {step_name} ===")
    print(sep)
    print(header)
    print(sep)
    for src in all_srcs:
        after = src_after.get(src, 0)
        if src_before is not None:
            before = src_before.get(src, 0)
            added = after - before
            print(f"{src:<25} {before:>8} {after:>8} {added:>+8}")
        else:
            print(f"{src:<25} {after:>8}")
    print(sep)
    total_after = len(data)
    if src_before is not None:
        total_before = len(prev_data)
        total_added = total_after - total_before
        print(f"{'TOTAL':<25} {total_before:>8} {total_after:>8} {total_added:>+8}")
    else:
        print(f"{'TOTAL':<25} {total_after:>8}")
    print(sep)


def tokenize_batched(tok, texts: list[str], desc: str = "Tokenizing") -> list[int]:
    """Tokenize texts in batches with a progress bar. Returns token counts."""
    token_lens = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tok(batch, add_special_tokens=False)
        token_lens.extend(len(ids) for ids in encoded["input_ids"])
    return token_lens


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizer
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # 2. Load step_7 train (keep all)
    step7_path = STEP7_DIR / "train.jsonl"
    print(f"\nLoading step_7 train from {step7_path}...")
    with jsonlines.open(step7_path, "r") as reader:
        step7_data = list(tqdm(reader, desc="Loading step_7"))
    print(f"Loaded {len(step7_data)} rows")
    print_summary(step7_data, "Step 7 (base)")

    label_counts = Counter(item["label"] for item in step7_data)
    print(f"Label distribution: {dict(label_counts)}")

    # 3. Build ID set of step_7 samples (to avoid duplicates)
    step7_ids = {item["id"] for item in step7_data}

    # 4. Load step_2 train
    step4_path = STEP4_DIR / "train.jsonl"
    print(f"\nLoading step_2 train from {step4_path}...")
    with jsonlines.open(step4_path, "r") as reader:
        step4_data = list(tqdm(reader, desc="Loading step_2"))
    print(f"Loaded {len(step4_data)} rows")

    # 5. Tokenize evidence and filter for long samples
    print(f"\nTokenizing step_2 evidence (batch_size={BATCH_SIZE})...")
    evidences = [item.get("evidence", "") for item in step4_data]
    token_lens = tokenize_batched(tok, evidences, desc="Tokenizing step_2 evidence")

    print(f"\nFiltering for evidence >= {MIN_EVIDENCE_TOKENS} tokens...")
    long_candidates = []
    skipped_dup = 0
    for item, tl in tqdm(
        zip(step4_data, token_lens), total=len(step4_data), desc="Filtering"
    ):
        if tl >= MIN_EVIDENCE_TOKENS:
            if item["id"] not in step7_ids:
                long_candidates.append(item)
            else:
                skipped_dup += 1

    print(f"Found {len(long_candidates)} long-evidence samples not in step_7")
    print(f"Skipped {skipped_dup} duplicates (already in step_7)")

    if long_candidates:
        print_summary(long_candidates, "Long-evidence candidates")
        cand_labels = Counter(item["label"] for item in long_candidates)
        print(f"Label distribution: {dict(cand_labels)}")

        # Token stats for the candidates
        cand_evidences = [item.get("evidence", "") for item in long_candidates]
        cand_lens = np.array(
            tokenize_batched(tok, cand_evidences, desc="Tokenizing candidates")
        )
        print(
            f"Evidence token stats: min={int(cand_lens.min())}, "
            f"mean={cand_lens.mean():.0f}, median={np.median(cand_lens):.0f}, "
            f"max={int(cand_lens.max())}"
        )

    # 6. Combine step_7 + long-evidence candidates
    combined = step7_data + long_candidates
    print_summary(combined, "Step 8 (combined)", step7_data)

    final_labels = Counter(item["label"] for item in combined)
    print(f"\nFinal label distribution: {dict(final_labels)}")
    for lbl, cnt in sorted(final_labels.items()):
        pct = cnt / len(combined) * 100 if combined else 0
        print(f"  {lbl}: {cnt} ({pct:.1f}%)")

    # 7. Write train
    train_out = OUTPUT_DIR / "train.jsonl"
    print(f"\nWriting {len(combined)} rows to {train_out}...")
    with jsonlines.open(train_out, "w") as writer:
        for item in tqdm(combined, desc="Writing train"):
            writer.write(item)
    print(f"Done.")

    # 8. Copy test files from step_7
    test_files = sorted(f for f in STEP7_DIR.glob("*.jsonl") if f.name != "train.jsonl")
    print(f"\nCopying {len(test_files)} test files from {STEP7_DIR}...")
    for test_path in tqdm(test_files, desc="Copying test files"):
        shutil.copy2(test_path, OUTPUT_DIR / test_path.name)
    print("Done.")


if __name__ == "__main__":
    main()
