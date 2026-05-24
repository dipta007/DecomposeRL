"""
Find the best checkpoint by pooled (micro) balanced accuracy across 9 dev
datasets, then evaluate that checkpoint on the held-out datasets
(coverbench, llmaggrefact).

Usage:
    PYTHONPATH=. python decomposer/eval/eval_best_checkpoint.py -v 66
    PYTHONPATH=. python decomposer/eval/eval_best_checkpoint.py -v 61 62 63
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import jsonlines
from sklearn.metrics import balanced_accuracy_score

DEV_DATASETS = [
    "fever",
    "claimdecomp",
    "hover",
    "feverous",
    "wice",
    "ex_fever",
    "pubhealthfact",
    "fool_me_twice",
    "pubmedclaim",
]
HELD_OUT_DATASETS = ["coverbench", "llmaggrefact"]


def find_exp_dir(outputs_root: str, version: int) -> str:
    pattern = os.path.join(outputs_root, f"2way_*_v{version}")
    matches = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not matches:
        raise FileNotFoundError(
            f"No experiment directory matching {pattern}. "
            f"Expected exactly one of outputs/2way_<size>_v{version}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple experiment directories matched {pattern}: {matches}. "
            "Disambiguate manually."
        )
    return matches[0]


def list_checkpoints(exp_dir: str) -> List[Tuple[str, int]]:
    checkpoints = []
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if not (os.path.isdir(item_path) and item.startswith("checkpoint-")):
            continue
        if not os.path.exists(os.path.join(item_path, "adapter_config.json")):
            continue
        try:
            idx = int(item.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((item_path, idx))
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def load_pooled_labels(
    checkpoint_path: str, datasets: List[str]
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Pool gt/pred labels across the given datasets for one checkpoint.

    Uses the same `pred_label or 'refuted'` fallback as
    compute_classification_metrics so micro balanced accuracy here is
    comparable to the per-dataset balanced_accuracy in *_test_metrics_v2.json.
    """
    gt_all: List[str] = []
    pred_all: List[str] = []
    per_dataset_counts: Dict[str, int] = {}

    for ds in datasets:
        results_path = os.path.join(
            checkpoint_path, f"{ds}_test_results_v2.jsonl"
        )
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"Missing required dev-set results file: {results_path}. "
                f"Run test.py on {ds} for this checkpoint before selecting the best one."
            )
        count = 0
        with jsonlines.open(results_path) as reader:
            for row in reader:
                gt = row.get("gt_label")
                if not gt:
                    raise ValueError(
                        f"Missing gt_label in row of {results_path}"
                    )
                pred = row.get("pred_label")
                pred = pred.lower() if pred else "refuted"
                gt_all.append(gt.lower())
                pred_all.append(pred)
                count += 1
        per_dataset_counts[ds] = count

    return gt_all, pred_all, per_dataset_counts


def compute_micro_balanced_accuracy(
    checkpoint_path: str, datasets: List[str]
) -> Dict:
    gt_all, pred_all, per_dataset_counts = load_pooled_labels(
        checkpoint_path, datasets
    )
    if not gt_all:
        raise RuntimeError(
            f"No samples found across datasets for checkpoint {checkpoint_path}."
        )
    score = balanced_accuracy_score(gt_all, pred_all)
    return {
        "micro_balanced_accuracy": float(score),
        "total_samples": len(gt_all),
        "per_dataset_counts": per_dataset_counts,
    }


def run_test_py(test_data: str, checkpoint_path: str, force: bool) -> int:
    cmd = [
        sys.executable,
        "decomposer/eval/test.py",
        "-d",
        test_data,
        "-c",
        checkpoint_path,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def write_exp_summary(
    exp_dir: str, dataset: str, checkpoint_path: str
) -> Optional[str]:
    metrics_path = os.path.join(
        checkpoint_path, f"{dataset}_test_metrics_v2.json"
    )
    if not os.path.exists(metrics_path):
        print(
            f"[WARN] Expected metrics file not found after test.py: {metrics_path}. "
            "Skipping exp-dir summary."
        )
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    summary_path = os.path.join(exp_dir, f"{dataset}_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump([metrics], f, indent=2)
    print(f"Wrote exp-dir summary: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select best checkpoint by pooled (micro) balanced accuracy over "
            "9 dev datasets, then run held-out evals on it."
        )
    )
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="+",
        required=True,
        help="One or more experiment version numbers, e.g. -v 66 or -v 61 62 63",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="outputs",
        help="Root dir containing experiment folders (default: outputs)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/combined_5k/step_9",
        help="Dir containing test_<dataset>.jsonl files",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-evaluation of held-out datasets even if metrics exist",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Only pick the best checkpoint; skip running held-out evals.",
    )
    args = parser.parse_args()

    versions = list(dict.fromkeys(args.versions))
    print(f"Processing versions: {versions}")

    failures: List[Tuple[int, str]] = []
    for version in versions:
        print("\n" + "#" * 60)
        print(f"# Version {version}")
        print("#" * 60)
        try:
            process_version(
                version=version,
                outputs_root=args.outputs_root,
                data_dir=args.data_dir,
                force=args.force,
                skip_eval=args.skip_eval,
            )
        except Exception as e:
            print(f"[ERROR] v{version} failed: {e}")
            failures.append((version, str(e)))

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for v, msg in failures:
            print(f"  v{v}: {msg}")
        sys.exit(1)
    print(f"All {len(versions)} version(s) processed successfully.")
    print("=" * 60)


def process_version(
    version: int,
    outputs_root: str,
    data_dir: str,
    force: bool,
    skip_eval: bool,
) -> None:
    exp_dir = find_exp_dir(outputs_root, version)
    print(f"Experiment dir: {exp_dir}")

    checkpoints = list_checkpoints(exp_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* subdirs found under {exp_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s).")

    per_checkpoint = []
    for cp_path, cp_idx in checkpoints:
        info = compute_micro_balanced_accuracy(cp_path, DEV_DATASETS)
        info["checkpoint_index"] = cp_idx
        info["checkpoint_path"] = cp_path
        per_checkpoint.append(info)
        print(
            f"  [{cp_idx:>5}] micro_balanced_accuracy="
            f"{info['micro_balanced_accuracy']:.6f} "
            f"(n={info['total_samples']})"
        )

    best = max(per_checkpoint, key=lambda x: x["micro_balanced_accuracy"])
    print("\n" + "=" * 60)
    print(
        f"v{version} best checkpoint: {best['checkpoint_path']} "
        f"(index={best['checkpoint_index']}, "
        f"micro_balanced_accuracy={best['micro_balanced_accuracy']:.6f})"
    )
    print("=" * 60)

    selection = {
        "version": version,
        "exp_dir": exp_dir,
        "datasets_used": DEV_DATASETS,
        "best_checkpoint_path": best["checkpoint_path"],
        "best_checkpoint_index": best["checkpoint_index"],
        "best_micro_balanced_accuracy": best["micro_balanced_accuracy"],
        "per_checkpoint": per_checkpoint,
    }
    selection_path = os.path.join(exp_dir, "best_checkpoint_selection.json")
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"Wrote selection: {selection_path}")

    if skip_eval:
        print("--skip_eval set; not running held-out evals.")
        return

    if best["checkpoint_index"] % 50 != 0:
        raise RuntimeError(
            f"Best checkpoint index {best['checkpoint_index']} is not a "
            "multiple of 50; test.py would skip it."
        )

    for ds in HELD_OUT_DATASETS:
        test_data = os.path.join(data_dir, f"test_{ds}.jsonl")
        if not os.path.exists(test_data):
            raise FileNotFoundError(f"Test data file not found: {test_data}")
        rc = run_test_py(test_data, best["checkpoint_path"], force)
        if rc != 0:
            raise RuntimeError(
                f"test.py exited with code {rc} on dataset {ds}."
            )
        write_exp_summary(exp_dir, ds, best["checkpoint_path"])


if __name__ == "__main__":
    main()
