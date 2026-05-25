"""Shared utilities for tiny-judge training scripts.

Defines per-task configuration (label space, repo names), the canonical
imbalance-robust evaluation metrics (macro-F1 / MAE) used to pick best
checkpoints, dataset loading helpers, and class-weight computation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

DATASET_REPO = "anonymous/anonymous-tiny-judge"
WANDB_ENTITY = "anonymous"
WANDB_PROJECT = "anonymous-judge"
ENCODER_BACKBONE = "unsloth/ModernBERT-large"

# Bump this when you change training recipe / data / hyperparams in a way you
# want to filter on later. Emitted as the `version:vN` wandb tag.
TINY_JUDGE_VERSION = "v6"

# Label maps (mirrors src/train/tiny_judge/build_dataset.py)
COVERAGE_LABEL_MAP = {"supported": 0, "refuted": 1, "not_enough_information": 2}
COVERAGE_LABEL_NAMES = ["supported", "refuted", "not_enough_information"]
BINARY_LABEL_NAMES = ["no", "yes"]

# Atomicity is reframed from regression on [0, 1] to 6-way ordinal classification
# over equally-spaced bin centers, because MSE on the skewed [0, 1] distribution
# collapses to predicting-the-mean (loss/MAE drop, bin accuracy drops with them).
ATOMICITY_BIN_VALUES = np.linspace(0.0, 1.0, 6)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ATOMICITY_BIN_NAMES = [f"{v:.1f}" for v in ATOMICITY_BIN_VALUES]


# 5 binary sub-checklists derived from the same atomicity cache files as the
# aggregate. Order mirrors `ATOMICITY_CRITERIA` in build_dataset.py.
ATOMICITY_SUBCHECKLISTS = (
    "is_question",
    "single_focus",
    "no_conjunctions",
    "verifiable",
    "grounded",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskConfig:
    """All task-specific knobs in one place."""

    name: str  # subset name in HF dataset and config name
    num_labels: int  # 1 for regression, K for K-way classification
    problem_type: str  # "single_label_classification" | "regression"
    repo_short: str  # used to build {org}/{repo_short}-judge[-suffix]
    label_names: list  # human-readable; only used for logging classification metrics


TASKS = {
    "atomicity_checklist": TaskConfig(
        name="atomicity_checklist",
        num_labels=len(ATOMICITY_BIN_VALUES),
        problem_type="single_label_classification",
        repo_short="atomicity",
        label_names=ATOMICITY_BIN_NAMES,
    ),
    # 5 binary sub-checklists; each is a 1-vs-0 head trained on the same source
    # rows as the aggregate, just with a per-criterion label. At inference the
    # 5 yes/no predictions sum/5 reproduces the aggregate atomicity score.
    **{
        f"atomicity_{c}": TaskConfig(
            name=f"atomicity_{c}",
            num_labels=2,
            problem_type="single_label_classification",
            repo_short=f"atomicity-{c.replace('_', '-')}",
            label_names=BINARY_LABEL_NAMES,
        )
        for c in ATOMICITY_SUBCHECKLISTS
    },
    "question_answerable": TaskConfig(
        name="question_answerable",
        num_labels=2,
        problem_type="single_label_classification",
        repo_short="question",
        label_names=BINARY_LABEL_NAMES,
    ),
    "answer_correctness": TaskConfig(
        name="answer_correctness",
        num_labels=2,
        problem_type="single_label_classification",
        repo_short="answer",
        label_names=BINARY_LABEL_NAMES,
    ),
    "coverage": TaskConfig(
        name="coverage",
        num_labels=3,
        problem_type="single_label_classification",
        repo_short="coverage",
        label_names=COVERAGE_LABEL_NAMES,
    ),
}

ALL_TASK_NAMES = tuple(TASKS.keys())


def repo_name_for(task: TaskConfig, suffix: str = "") -> str:
    base = f"anonymous/{task.repo_short}-judge"
    return f"{base}-{suffix}" if suffix else base


def load_task_dataset(
    task: TaskConfig,
    train_split: str,
    limit: int | None = None,
    eval_limit: int | None = None,
):
    """Load the 6-split DatasetDict for a task; return train, val, test, test_balanced.

    `train_split` should be either "train" or "train_balanced".
    Validation and test are ALWAYS the natural splits (deployment realism).
    test_balanced is also returned for symmetric comparison.

    `limit` caps train rows; `eval_limit` caps val + test + test_balanced rows
    (for CPU smoke tests where the full ~45k val split would take hours).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_REPO, task.name)
    logger.info(
        f"[{task.name}] loaded: {{ {', '.join(f'{k}={len(v)}' for k, v in ds.items())} }}"
    )

    train = ds[train_split]
    val = ds["validation"]
    test = ds["test"]
    test_balanced = ds["test_balanced"]

    if limit:
        train = train.select(range(min(limit, len(train))))
        logger.info(f"[{task.name}] --limit applied: train={len(train)}")
    if eval_limit:
        val = val.select(range(min(eval_limit, len(val))))
        test = test.select(range(min(eval_limit, len(test))))
        test_balanced = test_balanced.select(range(min(eval_limit, len(test_balanced))))
        logger.info(
            f"[{task.name}] --eval-limit applied: val={len(val)}, test={len(test)}, "
            f"test_balanced={len(test_balanced)}"
        )

    if task.name == "atomicity_checklist":

        def _bin_batch(examples):
            return {
                "label": [
                    int(np.argmin(np.abs(float(s) - ATOMICITY_BIN_VALUES)))
                    for s in examples["label"]
                ]
            }

        train = train.map(_bin_batch, batched=True, desc="bin atomicity (train)")
        val = val.map(_bin_batch, batched=True, desc="bin atomicity (val)")
        test = test.map(_bin_batch, batched=True, desc="bin atomicity (test)")
        test_balanced = test_balanced.map(
            _bin_batch, batched=True, desc="bin atomicity (test_balanced)"
        )

    return train, val, test, test_balanced


def compute_classification_metrics(predictions, labels, task: TaskConfig) -> dict:
    """Returns a dict of metrics. macro_f1 is the canonical selection metric."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    preds = np.asarray(predictions)
    if preds.ndim == 2:
        preds = preds.argmax(axis=-1)
    labels = np.asarray(labels).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }
    if task.num_labels == 2:
        metrics["mcc"] = float(matthews_corrcoef(labels, preds))

    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(task.num_labels)), zero_division=0
    )
    for i in range(task.num_labels):
        cls = task.label_names[i] if i < len(task.label_names) else str(i)
        metrics[f"f1_{cls}"] = float(f[i])
        metrics[f"precision_{cls}"] = float(p[i])
        metrics[f"recall_{cls}"] = float(r[i])
    return metrics


def metric_for_best(task: TaskConfig) -> tuple[str, bool]:
    """(name, greater_is_better) pair for HF Trainer's metric_for_best_model."""
    if task.problem_type == "regression":
        return "mae", False
    return "macro_f1", True


def setup_wandb(
    run_kind: str,
    task: TaskConfig,
    enabled: bool,
    train_split: str,
    backbone: str,
):
    """Start a fresh wandb run for this task with explicit tags. No-op if disabled.

    `train_split` should be "natural" or "balanced" (a label, not the HF split
    name). `backbone` is a short tag like "modernbert-large" or "tfidf-lr".

    We explicitly call `wandb.init(tags=[...])` instead of relying on the
    `WANDB_TAGS` env var, because HF Trainer's WandbCallback does NOT forward
    that env var into its own `wandb.init` call, and wandb's Settings layer
    caches env-var pickup across re-inits in the same process — so a `--task
    all` loop ends up tagging every run with the first task's tags. By
    pre-initing here, the callback sees `wandb.run is not None` and reuses
    our run with our tags.
    """
    if not enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return

    import wandb

    # Make sure no run is left over from the previous task (defensive — main()
    # also calls finish_wandb_run between tasks).
    if wandb.run is not None:
        wandb.finish()

    family = task.name.split("_", 1)[0]  # atomicity_*/question_*/answer_*/coverage
    tags = [
        f"kind:{run_kind}",
        f"task:{task.name}",
        f"family:{family}",
        f"split:{train_split}",
        f"backbone:{backbone}",
        f"version:{TINY_JUDGE_VERSION}",
    ]

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=run_kind,
        name=f"{run_kind}__{task.name}",
        tags=tags,
        reinit=True,
    )


def finish_wandb_run():
    """Close the current wandb run so the next task starts a fresh run/ID."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
    # Clear sticky identifiers so the next task's init can't accidentally resume
    # the just-finished run (which would inherit its name + ID).
    for k in ("WANDB_RUN_ID", "WANDB_RESUME"):
        os.environ.pop(k, None)
