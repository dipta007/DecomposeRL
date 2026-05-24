"""
Baseline evaluation using MiniCheck (NLI-based fact verification).

Instead of prompting an LLM, this script uses MiniCheck to directly score
whether the evidence supports or refutes each claim. The raw probability
from MiniCheck is thresholded at 0.5 to produce a binary label.

Usage:
    PYTHONPATH=. python decomposer/eval/baseline_minicheck.py \
        -d data/combined/step_9/test_pubmedclaim.jsonl \
        -o outputs/baseline_minicheck

    # With a custom threshold
    PYTHONPATH=. python decomposer/eval/baseline_minicheck.py \
        -d data/combined/step_9/test_pubmedclaim.jsonl \
        -o outputs/baseline_minicheck \
        --threshold 0.5

    # With a different MiniCheck model
    PYTHONPATH=. python decomposer/eval/baseline_minicheck.py \
        -d data/combined/step_9/test_pubmedclaim.jsonl \
        -o outputs/baseline_minicheck \
        --minicheck_model Bespoke-MiniCheck-7B
"""

import os

import argparse
import json
from typing import Dict, List

import jsonlines
import numpy as np
from minicheck.minicheck import MiniCheck

from decomposer.eval.utils import compute_classification_metrics

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_dataset_name(test_data_path: str) -> str:
    """Extract dataset name from the test data path."""
    filename = os.path.basename(test_data_path)
    if filename.startswith("test_") and filename.endswith(".jsonl"):
        return filename[len("test_") : -len(".jsonl")]
    raise ValueError(f"Unexpected test data filename format: {filename}")


def load_test_data(test_data_path: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with jsonlines.open(test_data_path) as reader:
        for line in reader:
            data.append(line)
    print(f"Loaded {len(data)} samples from {test_data_path}")
    return data


# ---------------------------------------------------------------------------
# MiniCheck scoring
# ---------------------------------------------------------------------------


def score_with_minicheck(
    test_data: List[Dict],
    minicheck_model: str,
    cache_dir: str,
    max_model_len: int,
) -> List[float]:
    """Score all (evidence, claim) pairs with MiniCheck.

    Returns a list of raw probabilities (one per sample). Higher probability
    means the evidence is more likely to support the claim.
    """
    print(f"Loading MiniCheck model: {minicheck_model}")
    scorer = MiniCheck(
        model_name=minicheck_model,
        cache_dir=cache_dir,
        max_model_len=max_model_len,
    )

    docs = [item["evidence"] for item in test_data]
    claims = [item["claim"] for item in test_data]

    print(f"Scoring {len(test_data)} samples...")
    _, raw_probs, _, _ = scorer.score(docs=docs, claims=claims)

    return [float(p) for p in raw_probs]


# ---------------------------------------------------------------------------
# Results processing
# ---------------------------------------------------------------------------


def process_results(
    test_data: List[Dict],
    raw_probs: List[float],
    threshold: float,
) -> List[Dict]:
    """Convert MiniCheck probabilities to predicted labels and build result dicts.

    Each result includes:
    - minicheck_prob: raw probability that evidence supports the claim
    - pred_confidence: confidence for the predicted label (prob if pred=Supported,
      1-prob if pred=Refuted). Analogous to pred_confidence from logprobs in test.py.
    - correctness_confidence: how confident MiniCheck is about the ground-truth
      label (prob if gt=Supported, 1-prob if gt=Refuted). Higher = more confident
      in the correct direction.
    - pred_label: thresholded binary prediction
    """
    results = []
    for sample, prob in zip(test_data, raw_probs):
        pred_label = "supported" if prob >= threshold else "refuted"

        # Pred confidence: how confident MiniCheck is in its predicted label
        pred_confidence = prob if pred_label == "supported" else 1.0 - prob

        # Correctness confidence: how aligned MiniCheck's score is with the GT
        gt_label = sample.get("label", "").lower()
        if gt_label == "supported":
            correctness_confidence = prob
        else:
            correctness_confidence = 1.0 - prob

        result = {
            "id": sample.get("id"),
            "claim": sample.get("claim"),
            "gt_label": sample.get("label"),
            "dataset": sample.get("src"),
            "minicheck_prob": prob,
            "pred_label": pred_label,
            "pred_confidence": pred_confidence,
            "correctness_confidence": correctness_confidence,
        }
        results.append(result)

    return results


def compute_metrics(
    results: List[Dict],
    threshold: float,
) -> Dict:
    """Compute classification metrics plus MiniCheck confidence statistics.

    Follows the same aggregated metrics pattern as test.py's compute_aggregated_metrics.
    """
    gt_labels = [r["gt_label"] for r in results]
    pred_labels = [r["pred_label"] for r in results]

    # Classification metrics (same as test.py)
    metrics = compute_classification_metrics(gt_labels, pred_labels)

    # Raw MiniCheck probability stats
    probs = [r["minicheck_prob"] for r in results]
    metrics.update(
        {
            "mean_minicheck_prob": float(np.mean(probs)),
            "std_minicheck_prob": float(np.std(probs)),
            "median_minicheck_prob": float(np.median(probs)),
        }
    )

    # Pred confidence stats (analogous to test.py's pred_confidence from logprobs)
    pred_confidences = [r["pred_confidence"] for r in results]
    metrics.update(
        {
            "mean_pred_confidence": float(np.mean(pred_confidences)),
            "std_pred_confidence": float(np.std(pred_confidences)),
            "min_pred_confidence": float(np.min(pred_confidences)),
            "max_pred_confidence": float(np.max(pred_confidences)),
            "num_samples_with_confidence": len(pred_confidences),
        }
    )

    # Correctness confidence stats (how confident MiniCheck is about the GT label)
    correctness = [r["correctness_confidence"] for r in results]
    metrics.update(
        {
            "mean_correctness_confidence": float(np.mean(correctness)),
            "std_correctness_confidence": float(np.std(correctness)),
            "median_correctness_confidence": float(np.median(correctness)),
        }
    )

    # Per-label correctness confidence breakdown
    supported_conf = [
        r["correctness_confidence"]
        for r in results
        if r["gt_label"].lower() == "supported"
    ]
    refuted_conf = [
        r["correctness_confidence"]
        for r in results
        if r["gt_label"].lower() == "refuted"
    ]

    if supported_conf:
        metrics["mean_correctness_confidence_supported"] = float(
            np.mean(supported_conf)
        )
    if refuted_conf:
        metrics["mean_correctness_confidence_refuted"] = float(np.mean(refuted_conf))

    metrics.update(
        {
            "threshold": threshold,
            "total_samples": len(results),
        }
    )

    return metrics


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
    dataset_name: str,
):
    """Save results and metrics to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save per-sample results as JSONL
    jsonl_path = os.path.join(
        output_dir, f"{dataset_name}_baseline_minicheck_results.jsonl"
    )
    with jsonlines.open(jsonl_path, "w") as writer:
        for result in results:
            writer.write(result)
    print(f"Saved per-sample results to {jsonl_path}")

    # Save aggregated metrics as JSON
    json_path = os.path.join(
        output_dir, f"{dataset_name}_baseline_minicheck_metrics.json"
    )
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved aggregated metrics to {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation using MiniCheck (NLI-based fact verification)"
    )
    parser.add_argument(
        "--test_data",
        "-d",
        type=str,
        default="data/combined/step_9/test_pubmedclaim.jsonl",
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--minicheck_model",
        type=str,
        default="Bespoke-MiniCheck-7B",
        help="MiniCheck model name (default: Bespoke-MiniCheck-7B)",
    )
    parser.add_argument(
        "--minicheck_cache_dir",
        type=str,
        default="./ckpts",
        help="Directory for MiniCheck model weights cache",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16768,
        help="Maximum model context length for MiniCheck (default: 16768)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for Supported vs Refuted (default: 0.5)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-evaluation even if results already exist",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MiniCheck Baseline Configuration:")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Check if already evaluated before loading the model
    dataset_name = get_dataset_name(args.test_data)
    print(f"Dataset name: {dataset_name}")

    metrics_path = os.path.join(
        args.output_dir, f"{dataset_name}_baseline_minicheck_metrics.json"
    )

    if not args.force and os.path.exists(metrics_path):
        print(f"\n[SKIP] Already evaluated: {metrics_path}")
        print("Use --force to re-evaluate.")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print(f"\n{'=' * 60}")
        print("Existing Metrics:")
        print(f"{'=' * 60}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"{'=' * 60}")
        print("\nDone!")
        return

    # Load test data
    test_data = load_test_data(args.test_data)

    # Score with MiniCheck
    raw_probs = score_with_minicheck(
        test_data=test_data,
        minicheck_model=args.minicheck_model,
        cache_dir=args.minicheck_cache_dir,
        max_model_len=args.max_model_len,
    )

    # Process results
    results = process_results(test_data, raw_probs, args.threshold)
    metrics = compute_metrics(results, args.threshold)

    # Add metadata
    metrics["model"] = args.minicheck_model
    metrics["method"] = "minicheck"

    print(f"\n{'=' * 60}")
    print("Aggregated Metrics:")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    # Save results
    save_results(results, metrics, args.output_dir, dataset_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
