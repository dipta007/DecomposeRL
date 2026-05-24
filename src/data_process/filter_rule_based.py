"""Unified train-set filtering pipeline.

Reads   data/combined/step_0/*.jsonl
Writes  data/combined/step_1/train.jsonl  (filtered)
        data/combined/step_1/test_*.jsonl  (copied as-is)

Run:  uv run python -m decomposer.data_process.filter_v1
"""

import re
import shutil
from collections import Counter
from pathlib import Path

import jsonlines
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from transformers import AutoTokenizer

from decomposer.prompts import USER_PROMPT_2WAY_TEMPLATE

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MIN_EVIDENCE_SENTENCES = 3
MIN_EVIDENCE_TOKENS = 200
MAX_EVIDENCE_TOKENS = 10000
MAX_PROMPT_TOKENS = 11500
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
MINHASH_THRESHOLD = 0.7
MINHASH_NUM_PERM = 128
MAX_CLAIM_EVIDENCE_RATIO = 0.15
MAX_LEXICAL_OVERLAP = 1.0  # disabled, as the terminology overlap is often high and doesn't necessarily indicate low quality

INPUT_DIR = Path("data/combined/step_0")
OUTPUT_DIR = Path("data/combined/step_1")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return len([s for s in sentences if s.strip()])


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


# ---------------------------------------------------------------------------
# Filtering steps
# ---------------------------------------------------------------------------


def step_evidence_sentence_count(data: list[dict]) -> list[dict]:
    """Remove rows with < MIN_EVIDENCE_SENTENCES sentences in evidence."""
    return [
        item
        for item in tqdm(data, desc="Evidence sentence count")
        if count_sentences(item["evidence"]) >= MIN_EVIDENCE_SENTENCES
        or item["src"]
        in [
            "pubhealthtab",
            "scitab",
        ]
    ]


def step_evidence_token_length(data: list[dict], tokenizer) -> list[dict]:
    """Remove rows where evidence has < MIN or > MAX tokens."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    kept = []
    token_counts_by_src: dict[str, list[int]] = defaultdict(list)
    for item in tqdm(data, desc="Evidence token length"):
        token_ids = tokenizer.encode(item["evidence"], add_special_tokens=False)
        n_tokens = len(token_ids)
        token_counts_by_src[item["src"]].append(n_tokens)
        if MIN_EVIDENCE_TOKENS <= n_tokens <= MAX_EVIDENCE_TOKENS:
            kept.append(item)

        if n_tokens > MAX_EVIDENCE_TOKENS:
            print(f"Skipping {item['src']} with {n_tokens} evidence tokens")

    # Save bar chart of evidence token distribution per source
    plot_dir = Path("data/combined/plots/filter_rule_based")
    plot_dir.mkdir(parents=True, exist_ok=True)

    srcs = sorted(token_counts_by_src.keys())
    n_srcs = len(srcs)
    fig, axes = plt.subplots(n_srcs, 1, figsize=(10, 3 * n_srcs))
    if n_srcs == 1:
        axes = [axes]
    for ax, src in zip(axes, srcs):
        counts = token_counts_by_src[src]
        ax.hist(counts, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(
            MIN_EVIDENCE_TOKENS,
            color="red",
            linestyle="--",
            label=f"min={MIN_EVIDENCE_TOKENS}",
        )
        ax.axvline(
            MAX_EVIDENCE_TOKENS,
            color="red",
            linestyle="--",
            label=f"max={MAX_EVIDENCE_TOKENS}",
        )
        ax.set_title(
            f"{src} (n={len(counts)}, median={sorted(counts)[len(counts) // 2]})"
        )
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Evidence Token Count")
    fig.tight_layout()
    out_path = plot_dir / "evidence_token_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return kept


def step_prompt_token_length(data: list[dict], tokenizer) -> list[dict]:
    """Remove rows where the full prompt exceeds MAX_PROMPT_TOKENS.

    Uses the chat template from filter_long.py to compute actual prompt length.
    """
    kept = []
    for item in tqdm(data, desc="Prompt token length"):
        user_content = USER_PROMPT_2WAY_TEMPLATE.format(
            evidence_doc=item["evidence"],
            claim=item["claim"],
        )
        messages = [{"role": "user", "content": user_content}]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        length = len(token_ids)
        item["prompt_token_length"] = length
        if length < MAX_PROMPT_TOKENS:
            kept.append(item)
    return kept


def step_claim_evidence_ratio(data: list[dict]) -> list[dict]:
    """Remove rows where claim_words / evidence_words > MAX_CLAIM_EVIDENCE_RATIO."""
    kept = []
    for item in tqdm(data, desc="Claim-evidence ratio"):
        claim_words = len(item["claim"].split())
        evidence_words = len(item["evidence"].split())
        if evidence_words == 0:
            continue
        ratio = claim_words / evidence_words
        max_ratio = MAX_CLAIM_EVIDENCE_RATIO
        if item["src"] in ["pubhealthtab", "scitab", "pubmedclaim"]:
            max_ratio = 0.8  # tabular evidence is more concise, so allow higher ratio
        if item["src"] in [
            "claimdecomp",
            "healthver",
            "uphill",
            "matter_of_fact",
            "feverous",
        ]:
            max_ratio = (
                0.4  # these datasets often have shorter evidence, so allow higher ratio
            )
        # if item["src"] in ["pubmedclaim", "faviq_r_set"]:
        #     max_ratio = 0.05

        if ratio <= max_ratio:
            kept.append(item)
    return kept


def step_lexical_overlap(data: list[dict]) -> list[dict]:
    """Remove rows where word-level overlap > MAX_LEXICAL_OVERLAP."""
    if MAX_LEXICAL_OVERLAP >= 1.0:
        return data  # skip if no filtering
    kept = []
    for item in tqdm(data, desc="Lexical overlap"):
        claim_words = set(item["claim"].lower().split())
        evidence_words = set(item["evidence"].lower().split())
        if not claim_words:
            continue
        overlap = len(claim_words & evidence_words) / len(claim_words)
        if overlap <= MAX_LEXICAL_OVERLAP:
            kept.append(item)
    return kept


def step_minhash_dedup(data: list[dict]) -> list[dict]:
    """MinHash LSH dedup on claim text. Keep first occurrence."""
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    kept = []
    for i, item in enumerate(tqdm(data, desc="MinHash dedup")):
        mh = build_minhash(item["claim"])
        key = f"train_{i}"
        results = lsh.query(mh)
        if len(results) == 0:
            lsh.insert(key, mh)
            kept.append(item)
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

    # Load tokenizer once
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    step_num = 0

    def run_step(step_name, fn, *args, params=None):
        nonlocal train_data, step_num
        step_num += 1
        prev_data = train_data
        train_data = fn(train_data, *args)
        print_summary(train_data, f"Step {step_num}: {step_name}", prev_data, params)

    # --- Cheap filters (string/word ops only) ---
    run_step(
        "Claim-evidence ratio",
        step_claim_evidence_ratio,
        params={"MAX_CLAIM_EVIDENCE_RATIO": MAX_CLAIM_EVIDENCE_RATIO},
    )
    run_step(
        "Lexical overlap",
        step_lexical_overlap,
        params={"MAX_LEXICAL_OVERLAP": MAX_LEXICAL_OVERLAP},
    )
    run_step(
        "Evidence sentence count",
        step_evidence_sentence_count,
        params={"MIN_EVIDENCE_SENTENCES": MIN_EVIDENCE_SENTENCES},
    )

    # --- Moderate cost (tokenizer) ---
    run_step(
        "MinHash dedup",
        step_minhash_dedup,
        params={
            "MINHASH_THRESHOLD": MINHASH_THRESHOLD,
            "MINHASH_NUM_PERM": MINHASH_NUM_PERM,
        },
    )
    run_step(
        "Evidence token length",
        step_evidence_token_length,
        tokenizer,
        params={
            "MIN_EVIDENCE_TOKENS": MIN_EVIDENCE_TOKENS,
            "MAX_EVIDENCE_TOKENS": MAX_EVIDENCE_TOKENS,
        },
    )
    run_step(
        "Prompt token length",
        step_prompt_token_length,
        tokenizer,
        params={"MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS, "TOKENIZER": TOKENIZER_NAME},
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
