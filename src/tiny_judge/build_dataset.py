#!/usr/bin/env python3
"""Build HF dataset for tiny-judge distillation from cached LLM judge responses.

Walks `.cache/<run>/<func>/`, applies a strict prompt-validity check against the
current templates in `decomposer.prompts`, dedups by text-hash, splits 80/10/10
by claim/document hash, builds balanced variants, and pushes each subset to
`dipta007/decomposeRL-tiny-judge` (private) one at a time so partial progress
survives a crash.

Usage:
    uv run -m decomposer.tiny_judge.build_dataset --push
    uv run -m decomposer.tiny_judge.build_dataset --subsets coverage --limit 5000
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Value
from tqdm.auto import tqdm

from decomposer.prompts import (
    ANSWER_CHECKER_PROMPT_TEMPLATE,
    ATOMICITY_CHECKLIST_PROMPT_TEMPLATE,
    COVERAGE_PROMPT_TEMPLATE,
    QUESTION_CHECKER_PROMPT_TEMPLATE,
)

logger = logging.getLogger("tiny_judge.build")

JUDGE_MODEL_ID = "Qwen/Qwen3-32B"
HUB_REPO = "dipta007/decomposeRL-tiny-judge"

ATOMICITY_CRITERIA = (
    "is_question",
    "single_focus",
    "no_conjunctions",
    "verifiable",
    "grounded",
)
COVERAGE_LABEL_MAP = {"supported": 0, "refuted": 1, "not_enough_information": 2}


@dataclass(frozen=True)
class SubsetConfig:
    name: str
    cache_dirname: str
    required_input_fields: tuple
    label_dtype: str  # "float" or "int"
    split_field: str  # field used for claim_hash bucketing
    # For atomicity sub-checklists: name of the single criterion to extract as a
    # binary 0/1 label (one of ATOMICITY_CRITERIA). None for the aggregate task
    # and for non-atomicity tasks.
    criterion: str | None = None


SUBSETS = (
    SubsetConfig(
        name="atomicity_checklist",
        cache_dirname="atomicity_checklist",
        required_input_fields=("claim", "question"),
        label_dtype="float",
        split_field="claim",
    ),
    # 5 per-criterion binary subsets, each derived from the same atomicity cache
    # files (identical text, different label = single yes/no for one criterion).
    # Same prompt-validity gate as the aggregate, so they share rows 1:1.
    *[
        SubsetConfig(
            name=f"atomicity_{c}",
            cache_dirname="atomicity_checklist",
            required_input_fields=("claim", "question"),
            label_dtype="int",
            split_field="claim",
            criterion=c,
        )
        for c in ATOMICITY_CRITERIA
    ],
    SubsetConfig(
        name="question_answerable",
        cache_dirname="question_answerable",
        required_input_fields=("document", "question"),
        label_dtype="int",
        split_field="document",
    ),
    SubsetConfig(
        name="answer_correctness",
        cache_dirname="answer_correctness",
        required_input_fields=("document", "question", "answer"),
        label_dtype="int",
        split_field="document",
    ),
    SubsetConfig(
        name="coverage",
        cache_dirname="coverage",
        required_input_fields=("claim", "answers"),
        label_dtype="int",
        split_field="claim",
    ),
)


def _render_current_prompt(subset_name: str, cached: dict) -> str:
    # All atomicity_* subsets (aggregate + 5 per-criterion) share the same prompt.
    if subset_name.startswith("atomicity_"):
        return ATOMICITY_CHECKLIST_PROMPT_TEMPLATE.format(
            claim=cached["claim"], question=cached["question"]
        )
    if subset_name == "question_answerable":
        return QUESTION_CHECKER_PROMPT_TEMPLATE.format(
            document=cached["document"], question=cached["question"]
        )
    if subset_name == "answer_correctness":
        sentence = f"Q: {cached['question']}\nA: {cached['answer']}"
        return ANSWER_CHECKER_PROMPT_TEMPLATE.format(
            document=cached["document"], sentence=sentence
        )
    if subset_name == "coverage":
        return COVERAGE_PROMPT_TEMPLATE.format(
            claim=cached["claim"], answers=cached["answers"]
        )
    raise ValueError(f"Unknown subset: {subset_name}")


def _format_text(subset_name: str, cached: dict) -> str:
    # All atomicity_* subsets share the same text representation.
    if subset_name.startswith("atomicity_"):
        return f"Claim: {cached['claim']}\nQuestion: {cached['question']}"
    if subset_name == "question_answerable":
        return f"Document: {cached['document']}\nQuestion: {cached['question']}"
    if subset_name == "answer_correctness":
        return (
            f"Document: {cached['document']}\n"
            f"Question: {cached['question']}\n"
            f"Answer: {cached['answer']}"
        )
    if subset_name == "coverage":
        return f"Claim: {cached['claim']}\nAnswers:\n{cached['answers']}"
    raise ValueError(f"Unknown subset: {subset_name}")


def _parse_atomicity_per_criterion(er: str) -> dict | None:
    """Return {criterion_name: 0|1} for a valid response, or None if any of the
    5 criteria is missing (same strict gate the aggregate uses)."""
    results: dict[str, int] = {}
    for line in er.splitlines():
        line_l = line.strip().lower()
        for c in ATOMICITY_CRITERIA:
            if c in results:
                continue
            if line_l.startswith(c):
                results[c] = 1 if "yes" in line_l else 0
                break
    if len(results) < len(ATOMICITY_CRITERIA):
        return None
    return results


def _parse_label(subset_name: str, extracted_response: str):
    if not extracted_response:
        return None
    er = extracted_response.strip()

    if subset_name == "atomicity_checklist":
        per_crit = _parse_atomicity_per_criterion(er)
        if per_crit is None:
            return None
        return sum(per_crit.values()) / len(ATOMICITY_CRITERIA)

    if subset_name.startswith("atomicity_"):
        per_crit = _parse_atomicity_per_criterion(er)
        if per_crit is None:
            return None
        criterion = _SUBSETS_BY_NAME[subset_name].criterion
        return per_crit.get(criterion)  # 0 or 1; should always be present after the strict gate

    if subset_name in ("question_answerable", "answer_correctness"):
        try:
            v = int(er)
        except ValueError:
            return None
        return v if v in (0, 1) else None

    if subset_name == "coverage":
        low = er.lower()
        if "supported" in low:
            return COVERAGE_LABEL_MAP["supported"]
        if "refuted" in low:
            return COVERAGE_LABEL_MAP["refuted"]
        if "not enough" in low or "not_enough" in low:
            return COVERAGE_LABEL_MAP["not_enough_information"]
        return None

    raise ValueError(f"Unknown subset: {subset_name}")


_SUBSETS_BY_NAME = {s.name: s for s in SUBSETS}


# Module-level flag mutated by main() before workers spawn so each worker sees it.
_MINIMAL_COLS = False


def _process_file(args):
    """Worker: parse one cache file and emit a row dict (or rejection reason)."""
    file_path, subset_name, source_run = args
    try:
        with open(file_path, "r") as f:
            cached = json.load(f)
    except Exception:
        return ("json_err", None)

    if cached.get("judge_model_id") != JUDGE_MODEL_ID:
        return ("wrong_judge_model", None)

    sub = _SUBSETS_BY_NAME[subset_name]
    for fld in sub.required_input_fields:
        v = cached.get(fld, "")
        if not (isinstance(v, str) and v.strip()):
            return ("missing_input", None)

    expected_prompt = _render_current_prompt(subset_name, cached)
    if cached.get("prompt") != expected_prompt:
        return ("prompt_mismatch", None)

    label = _parse_label(subset_name, cached.get("extracted_response", ""))
    if label is None:
        return ("unparseable_response", None)

    text = _format_text(subset_name, cached)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    split_val = cached[sub.split_field]
    claim_hash = hashlib.sha256(split_val.encode("utf-8")).hexdigest()[:16]

    if _MINIMAL_COLS:
        # Drop the bulky `prompt` and `raw_response` fields. Cuts per-row size
        # from ~7 KB to ~1.5 KB. Use this when total in-memory rows would
        # otherwise exceed available RAM.
        return (
            "ok",
            {
                "text": text,
                "label": label,
                "text_hash": text_hash,
                "claim_hash": claim_hash,
                "extracted_response": cached.get("extracted_response", ""),
                "source_run": source_run,
            },
        )

    try:
        raw = cached["response"]["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ("missing_response", None)

    return (
        "ok",
        {
            "text": text,
            "label": label,
            "text_hash": text_hash,
            "claim_hash": claim_hash,
            "prompt": cached["prompt"],
            "raw_response": raw,
            "extracted_response": cached.get("extracted_response", ""),
            "source_run": source_run,
        },
    )


def _enumerate_files(cache_root: str, cache_dirname: str):
    """Yield (file_path, run_name) lazily across all runs."""
    if not os.path.isdir(cache_root):
        return
    for run in sorted(os.listdir(cache_root)):
        sub_path = os.path.join(cache_root, run, cache_dirname)
        if not os.path.isdir(sub_path):
            continue
        with os.scandir(sub_path) as it:
            for entry in it:
                if entry.is_file():
                    yield entry.path, run


def _count_files(cache_root: str, cache_dirname: str) -> int:
    """Sum of file counts across runs (for tqdm total). Reads dir entries only."""
    total = 0
    if not os.path.isdir(cache_root):
        return 0
    for run in os.listdir(cache_root):
        sub_path = os.path.join(cache_root, run, cache_dirname)
        if not os.path.isdir(sub_path):
            continue
        try:
            total += sum(1 for _ in os.scandir(sub_path))
        except OSError:
            pass
    return total


def _build_subset_streaming(
    subset: SubsetConfig,
    cache_root: str,
    workers: int,
    limit,
    schema: pa.Schema,
    out_path: Path,
    flush_every: int = 10_000,
):
    """Stream rows from workers directly to a parquet file on disk.

    Memory usage is bounded to: seen_hashes set (~24 B/hash × N_unique ≈ 21 MB
    for 1M rows) + one in-flight buffer of `flush_every` row dicts (~70 MB
    for 10k rows × 7KB), regardless of total subset size.
    """
    logger.info(
        f"[{subset.name}] enumerating {cache_root}/<run>/{subset.cache_dirname}/ "
        f"(this may take 1-2 min on networked FS for large subsets) …"
    )
    t0 = time.time()
    total = _count_files(cache_root, subset.cache_dirname)
    enumerate_secs = time.time() - t0
    if limit:
        total = min(total, limit)
    logger.info(
        f"[{subset.name}] enumerated {total} files in {enumerate_secs:.1f}s; "
        f"spawning {workers} workers; streaming → {out_path}"
    )

    rejected = Counter()
    seen_hashes: set = set()
    buffer: list = []
    n_processed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")

    def _flush(force: bool = False):
        if not buffer:
            return
        if not force and len(buffer) < flush_every:
            return
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table)
        buffer.clear()

    def _arg_iter():
        for i, (path, run) in enumerate(
            _enumerate_files(cache_root, subset.cache_dirname)
        ):
            if limit and i >= limit:
                break
            yield path, subset.name, run

    pbar = tqdm(
        total=total,
        desc=f"[{subset.name}] read",
        unit="file",
        smoothing=0.05,
        mininterval=1.0,
    )
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for status, row in ex.map(_process_file, _arg_iter(), chunksize=64):
                n_processed += 1
                pbar.update(1)
                if status != "ok":
                    rejected[status] += 1
                    continue
                h = row["text_hash"]
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                buffer.append(row)
                if len(buffer) >= flush_every:
                    _flush()
                if n_processed % 100_000 == 0:
                    pbar.set_postfix(unique=len(seen_hashes), refresh=False)
        _flush(force=True)
    finally:
        writer.close()
        pbar.close()

    n_kept = len(seen_hashes)
    seen_hashes.clear()
    gc.collect()

    elapsed = time.time() - t0
    logger.info(
        f"[{subset.name}] processed {n_processed} files in {elapsed:.1f}s; "
        f"unique rows kept = {n_kept}; rejected breakdown = {dict(rejected)}"
    )
    return n_kept


_PA_TYPE_MAP = {
    "string": pa.string(),
    "int32": pa.int32(),
    "float32": pa.float32(),
}


def _features_to_pa_schema(features: Features) -> pa.Schema:
    fields = [
        pa.field(name, _PA_TYPE_MAP.get(ftype.dtype, pa.string()))
        for name, ftype in features.items()
    ]
    return pa.schema(fields)


def _features_for_subset(subset: SubsetConfig) -> Features:
    label_value = Value("float32" if subset.label_dtype == "float" else "int32")
    base_fields = {
        "text": Value("string"),
        "label": label_value,
        "text_hash": Value("string"),
        "claim_hash": Value("string"),
        "extracted_response": Value("string"),
        "source_run": Value("string"),
    }
    if not _MINIMAL_COLS:
        base_fields["prompt"] = Value("string")
        base_fields["raw_response"] = Value("string")
    return Features(base_fields)


def _split_and_balance_from_parquet(
    all_unique_path: Path,
    subset: SubsetConfig,
    parquet_dir: Path,
    features: Features,
    seed: int = 0,
) -> DatasetDict:
    """Read the streamed `_all_unique.parquet`, materialize 6 final split parquets
    via memory-mapped Dataset operations (no row data held in RAM)."""
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lazy-load (memory-mapped) the all-unique parquet via HF Datasets.
    logger.info(f"[{subset.name}] loading streamed parquet (memory-mapped) …")
    full = Dataset.from_parquet(str(all_unique_path))
    full = full.cast(features)
    n_total = len(full)
    logger.info(f"[{subset.name}] loaded {n_total} unique rows for split assignment")

    # 2. Compute split assignment vector. Reading just the claim_hash column.
    claim_hashes = full["claim_hash"]  # python list, ~16 B × N rows ≈ small
    split_idx = {"train": [], "validation": [], "test": []}
    for i, h in enumerate(
        tqdm(claim_hashes, desc=f"[{subset.name}] assign split", unit="row",
             mininterval=1.0, smoothing=0.1)
    ):
        b = int(h, 16) % 100
        if b < 90:
            split_idx["train"].append(i)
        elif b < 95:
            split_idx["validation"].append(i)
        else:
            split_idx["test"].append(i)
    del claim_hashes
    gc.collect()

    logger.info(
        f"[{subset.name}] natural splits: "
        f"train={len(split_idx['train'])}, "
        f"validation={len(split_idx['validation'])}, "
        f"test={len(split_idx['test'])}"
    )

    # 3. For each natural split: select rows by index → write to parquet (streaming).
    natural_paths = {}
    for k, idxs in tqdm(
        list(split_idx.items()),
        desc=f"[{subset.name}] write natural",
        unit="split",
        mininterval=0.5,
    ):
        path = parquet_dir / f"{k}.parquet"
        logger.info(f"[{subset.name}] writing natural {k} ({len(idxs)} rows) → {path.name}")
        ds_split = full.select(idxs, keep_in_memory=False)
        ds_split.to_parquet(str(path))
        natural_paths[k] = path
        del ds_split
        gc.collect()

    # 4. For each natural split: compute balanced indices (label-balanced
    #    downsample to min class count), select, write balanced parquet.
    rng = random.Random(seed)
    balanced_paths = {}
    for k, idxs in tqdm(
        list(split_idx.items()),
        desc=f"[{subset.name}] write balanced",
        unit="split",
        mininterval=0.5,
    ):
        ds_natural = Dataset.from_parquet(str(natural_paths[k]))
        labels = ds_natural["label"]  # column read; small
        by_label: dict = {}
        for i, lbl in enumerate(labels):
            by_label.setdefault(lbl, []).append(i)
        if not by_label:
            kept = []
        else:
            n_min = min(len(v) for v in by_label.values())
            kept = []
            for lbl in sorted(by_label.keys()):
                local = list(by_label[lbl])
                rng.shuffle(local)
                kept.extend(local[:n_min])
            rng.shuffle(kept)
        bk = f"{k}_balanced"
        bpath = parquet_dir / f"{bk}.parquet"
        logger.info(f"[{subset.name}] writing balanced {bk} ({len(kept)} rows) → {bpath.name}")
        if kept:
            ds_balanced = ds_natural.select(kept, keep_in_memory=False)
        else:
            # Empty natural split → empty balanced
            ds_balanced = ds_natural
        ds_balanced.to_parquet(str(bpath))
        balanced_paths[bk] = bpath
        del ds_natural, ds_balanced, labels, by_label
        gc.collect()

    del full
    gc.collect()

    # 5. Build final DatasetDict by lazy-loading from the 6 parquet files.
    all_split_names = (
        "train",
        "validation",
        "test",
        "train_balanced",
        "validation_balanced",
        "test_balanced",
    )
    paths = {**natural_paths, **balanced_paths}
    dd = DatasetDict(
        {k: Dataset.from_parquet(str(paths[k])) for k in all_split_names}
    )
    dd = DatasetDict({k: ds.cast(features) for k, ds in dd.items()})
    return dd


def _log_split_stats(name: str, dd: DatasetDict):
    import statistics

    for split, ds in dd.items():
        if len(ds) == 0:
            logger.info(f"[{name}] {split}: n=0")
            continue

        labels = ds["label"]
        # Round float labels for clean display (regression atomicity stores
        # passed/5 as float32, which prints as 0.20000000298023224 etc).
        if labels and isinstance(labels[0], float):
            cnt = Counter(round(x, 2) for x in labels)
        else:
            cnt = Counter(labels)
        label_dist = dict(sorted(cnt.items()))

        lengths = sorted(len(t) for t in ds["text"])
        n = len(lengths)
        mean = statistics.fmean(lengths)
        std = statistics.pstdev(lengths) if n > 1 else 0.0

        def pct(p):
            i = min(n - 1, max(0, int(round((p / 100) * (n - 1)))))
            return lengths[i]

        unique_claims = len(set(ds["claim_hash"]))

        logger.info(
            f"[{name}] {split}: n={n}, label_dist={label_dist}, "
            f"unique_claims={unique_claims}, "
            f"text_len(min/p25/p50/p75/p95/p99/max)="
            f"{lengths[0]}/{pct(25)}/{pct(50)}/{pct(75)}/{pct(95)}/{pct(99)}/{lengths[-1]}, "
            f"text_len(mean/std)={mean:.1f}/{std:.1f}"
        )


def _check_hub_auth() -> bool:
    try:
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        logger.info(f"hub auth ok: user={who.get('name')}")
        return True
    except Exception as e:
        logger.error(
            f"hub auth check failed: {e}. "
            "Set HF_TOKEN env var or run `huggingface-cli login` before --push."
        )
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-root",
        default=".cache",
        help="Root containing per-run cache subdirs (default: .cache)",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/tiny_judge/.dataset_cache",
        help="Local arrow cache dir before push",
    )
    ap.add_argument(
        "--hub-repo",
        default=HUB_REPO,
        help=f"Hub repo id (default: {HUB_REPO})",
    )
    ap.add_argument(
        "--push", action="store_true", help="Push each subset to hub after build"
    )
    ap.add_argument(
        "--no-private", action="store_true", help="Push as public (default: private)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug: cap files per subset",
    )
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        choices=[s.name for s in SUBSETS],
        help="Only build these subsets (default: all)",
    )
    ap.add_argument(
        "--minimal-cols",
        action="store_true",
        help="Drop the bulky `prompt` and `raw_response` columns from rows. "
        "Cuts per-row size ~4x. Use if you hit RAM pressure on large subsets.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    # Set module-level flag BEFORE spawning workers so children inherit it.
    global _MINIMAL_COLS
    _MINIMAL_COLS = args.minimal_cols
    if _MINIMAL_COLS:
        # We're about to read this in subprocess workers via fork/spawn — set
        # via environment so spawned workers can pick it up too. (fork inherits
        # the global; spawn re-imports the module with this flag re-applied.)
        os.environ["TINY_JUDGE_MINIMAL_COLS"] = "1"
    elif os.environ.get("TINY_JUDGE_MINIMAL_COLS") == "1":
        _MINIMAL_COLS = True

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.push and not _check_hub_auth():
        raise SystemExit(1)

    selected = (
        SUBSETS
        if not args.subsets
        else tuple(s for s in SUBSETS if s.name in args.subsets)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    import shutil
    from datasets import load_from_disk

    for subset in selected:
        t0 = time.time()
        logger.info(f"========== Subset: {subset.name} ==========")

        # Streaming pipeline: workers → all_unique.parquet (only hashes in RAM) →
        # split + balance via on-disk Dataset ops → final 6 parquets → save_to_disk.
        parquet_tmp = output_dir / f"_pq_{subset.name}"
        parquet_tmp.mkdir(parents=True, exist_ok=True)
        all_unique_path = parquet_tmp / "_all_unique.parquet"

        features = _features_for_subset(subset)
        schema = _features_to_pa_schema(features)

        n_kept = _build_subset_streaming(
            subset, args.cache_root, args.workers, args.limit,
            schema, all_unique_path,
        )
        if n_kept == 0:
            logger.warning(f"[{subset.name}] no rows kept; skipping")
            shutil.rmtree(parquet_tmp, ignore_errors=True)
            continue

        dd = _split_and_balance_from_parquet(
            all_unique_path, subset, parquet_tmp, features, seed=0
        )

        local_path = output_dir / subset.name
        logger.info(f"[{subset.name}] writing final DatasetDict → {local_path} …")
        save_t0 = time.time()
        dd.save_to_disk(str(local_path))
        logger.info(
            f"[{subset.name}] save_to_disk done in {time.time() - save_t0:.1f}s"
        )
        # Drop file handles before deleting the temp dir.
        del dd
        gc.collect()
        shutil.rmtree(parquet_tmp, ignore_errors=True)
        # Reload from the canonical local path for the push step.
        dd = load_from_disk(str(local_path))
        logger.info(f"[{subset.name}] saved locally to {local_path}")

        _log_split_stats(subset.name, dd)

        if args.push:
            private = not args.no_private
            logger.info(
                f"[{subset.name}] pushing to {args.hub_repo} "
                f"(config={subset.name}, private={private})"
            )
            dd.push_to_hub(
                args.hub_repo,
                subset.name,
                private=private,
                commit_message=f"Add config: {subset.name}",
            )
            logger.info(f"[{subset.name}] push complete")

        elapsed = time.time() - t0
        total_rows = sum(len(s) for s in dd.values())
        logger.info(f"[{subset.name}] done in {elapsed:.1f}s ({total_rows} rows total)")
        summary.append((subset.name, total_rows))

    logger.info("========== Summary ==========")
    for name, total in summary:
        logger.info(f"  {name}: {total} rows across all 6 splits")


if __name__ == "__main__":
    main()
