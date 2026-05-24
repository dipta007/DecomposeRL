from typing import Dict, List

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

HF_DATASET = "dipta007/DecomposeRL"
HF_SUBSET = "5000"


def load_test_data(dataset_name: str) -> list[dict]:
    split = f"test_{dataset_name}"
    ds = load_dataset(HF_DATASET, HF_SUBSET, split=split)
    print(f"Loaded {len(ds)} samples from {HF_DATASET}/{HF_SUBSET} split={split}")
    return [dict(row) for row in ds]


def load_train_data() -> list[dict]:
    ds = load_dataset(HF_DATASET, HF_SUBSET, split="train")
    print(f"Loaded {len(ds)} samples from {HF_DATASET}/{HF_SUBSET} split=train")
    return [dict(row) for row in ds]


def compute_classification_metrics(
    gt_labels: List[str], pred_labels: List[str]
) -> Dict[str, float]:
    gt_labels_lower = [l.lower() for l in gt_labels]
    pred_labels_lower = [l.lower() if l else "refuted" for l in pred_labels]

    return {
        "accuracy": accuracy_score(gt_labels_lower, pred_labels_lower),
        "balanced_accuracy": balanced_accuracy_score(gt_labels_lower, pred_labels_lower),
        "precision": precision_score(
            gt_labels_lower, pred_labels_lower, pos_label="supported", zero_division=0
        ),
        "recall": recall_score(
            gt_labels_lower, pred_labels_lower, pos_label="supported", zero_division=0
        ),
        "f1": f1_score(
            gt_labels_lower, pred_labels_lower, pos_label="supported", zero_division=0
        ),
    }
