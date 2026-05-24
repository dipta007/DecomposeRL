"""Shared building blocks for the three encoder training scripts.

Anything that's identical across `train_encoder{,_balanced,_lora}.py` lives
here so we don't have three copies that drift apart.

Specifically:
- `WeightedCETrainer`: HF Trainer override that applies class weights and label
  smoothing to CE loss. Used by the full-FT and LoRA-on-natural scripts.
- `tokenize_dataset` / `prepare_for_trainer`: identical preprocessing across all
  three encoder scripts.
- `auto_eval_steps`: target-N-evals-per-run cadence formula.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import Value
from transformers import Trainer


class WeightedCETrainer(Trainer):
    """HF Trainer override that applies class weights and label smoothing to CE loss."""

    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=(
                self._class_weights.to(logits.device)
                if self._class_weights is not None
                else None
            ),
            label_smoothing=self._label_smoothing,
        )
        loss = loss_fct(logits, labels.long())
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(ds, tokenizer, max_length: int):
    """Tokenize a `text` column with truncation; keep `text` + `label` only."""

    def fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_length, padding=False
        )

    keep = {"text", "label"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.map(fn, batched=True, remove_columns=drop, desc="tokenize")


def prepare_for_trainer(ds, problem_type: str):
    """Cast `label` → `labels` with the right dtype, drop the original `text` column."""
    ds = ds.remove_columns(["text"])
    if problem_type == "regression":
        ds = ds.cast_column("label", Value("float32"))
    else:
        ds = ds.cast_column("label", Value("int64"))
    return ds.rename_column("label", "labels")


def stratified_eval_subset(ds, target: int = 5000, seed: int = 42):
    """Return a stratified subset of `ds` with EXACTLY min(len(ds), target) rows.

    Waterfilling algorithm: process classes from smallest to largest. At each
    class, the remaining target is split equally across the remaining classes;
    the actual take is min(that quota, available rows in this class). The
    leftover from any under-quota class spills into the larger classes that
    follow, so the total is always exactly `target` whenever `len(ds) >= target`.

    Example (atomicity_checklist val, target=5000, class counts [18,23,474,2402,2441,41060]):
        b0 takes 18 (smaller than 833 quota), pool drops to 4982 across 5 classes
        b1 takes 23 (smaller than 996), pool drops to 4959 across 4 classes
        b2 takes 474 (smaller than 1239), pool drops to 4485 across 3 classes
        b3,b4,b5 each take 1495 → total 18+23+474+1495+1495+1495 = 5000

    The shuffle is seeded so the same `seed` reproduces the same val subset.
    """
    if len(ds) == 0:
        return ds
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i, lbl in enumerate(ds["label"]):
        # Guard against regression labels: a float that isn't a whole number
        # would be silently rounded by int() below and conflate distinct values
        # into the same bucket.
        if isinstance(lbl, float) and not lbl.is_integer():
            raise ValueError(
                "stratified_eval_subset requires integer/categorical labels; "
                f"got float label {lbl!r} at row {i}. Reframe the task as "
                "classification or use a different subsampler for regression."
            )
        by_label.setdefault(int(lbl), []).append(i)
    if not by_label:
        return ds.select([])

    target = min(target, len(ds))
    sorted_labels = sorted(by_label, key=lambda l: len(by_label[l]))
    chosen: list[int] = []
    remaining_target = target
    for k, lbl in enumerate(sorted_labels):
        remaining_classes = len(sorted_labels) - k
        # Last class absorbs the leftover so rounding can't leave the total < target.
        quota = remaining_target if remaining_classes == 1 else remaining_target // remaining_classes
        take = min(quota, len(by_label[lbl]))
        idxs = list(by_label[lbl])
        rng.shuffle(idxs)
        chosen.extend(idxs[:take])
        remaining_target -= take

    rng.shuffle(chosen)
    return ds.select(chosen, keep_in_memory=False)


def sort_by_length(ds):
    """Sort a tokenized dataset by `input_ids` length (ascending).

    HF Trainer's `group_by_length=True` only applies to the train sampler — eval
    iterates the dataset in stored order. Pre-sorting eval by length means each
    batch contains similar-length sequences, so the per-batch padding ceiling
    drops from "longest in entire eval set" to "longest in this batch". On the
    big tasks (val ~5k with max_length 8192) this typically cuts eval wall-time
    by 3-5x. Output is a permutation, so metrics are unchanged.
    """
    if len(ds) == 0:
        return ds
    lengths = [len(x) for x in ds["input_ids"]]
    order = sorted(range(len(ds)), key=lambda i: lengths[i])
    return ds.select(order, keep_in_memory=False)


def auto_eval_steps(
    n_train: int, eff_batch: int, epochs: int, target_evals: int = 30
) -> int:
    """Pick eval_steps so a run yields ~target_evals evaluations, regardless of size.

    Train sizes vary across tasks by orders of magnitude (1.3k → 3.5M rows), so a
    single fixed `eval_steps` either over- or under-evals at the extremes. This
    formula gives the same val-curve resolution either way.
    """
    total_steps = max(1, (n_train // max(1, eff_batch)) * max(1, epochs))
    return max(1, total_steps // max(1, target_evals))
