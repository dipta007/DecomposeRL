"""Re-run the rule-based aggregator on existing ClaimDecomp result JSONLs.

When `aggregate_claimdecomp` changes (e.g. all-Unknown is reclassified from
None to Refuted, or the answer-line regex gets loosened), we want existing
rows to pick up the new policy WITHOUT spending API tokens. This script reads
every cached `generation` and re-derives:
  - parsed_answers
  - frac_yes_decisive
  - pred_label

It then rewrites the JSONL in place and refreshes the sibling metrics JSON.
Idempotent: re-running produces no further changes once a file is updated.

Only ClaimDecomp baseline files are supported (the script refuses other
methods because their rules are different).

Usage:
    # one file
    PYTHONPATH=. python decomposer/baselines/reaggregate_in_place.py \\
        --results outputs/baseline_14b_prompted/fever_baseline_results_claimdecomp.jsonl

    # bulk: every claimdecomp results JSONL under a dir
    PYTHONPATH=. python decomposer/baselines/reaggregate_in_place.py \\
        --root outputs

    # dry-run (count would-be changes; no writes)
    PYTHONPATH=. python decomposer/baselines/reaggregate_in_place.py \\
        --root outputs --dry-run
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import jsonlines

from decomposer.baselines.prompts import aggregate_claimdecomp
from decomposer.eval.utils import compute_classification_metrics


CLAIMDECOMP_FILE_RE = re.compile(r"^(.+?)_baseline_results_claimdecomp\.jsonl$")


def _is_claimdecomp_file(path: str) -> bool:
    return CLAIMDECOMP_FILE_RE.match(os.path.basename(path)) is not None


def reaggregate_one(path: str, dry_run: bool) -> Tuple[int, int]:
    """Re-derive pred_label/parsed_answers/frac_yes_decisive for one file.

    Returns (n_changed, n_total).
    """
    rows: List[Dict] = []
    with jsonlines.open(path) as r:
        for line in r:
            rows.append(line)
    n_changed = 0
    for row in rows:
        gen = row.get("generation") or ""
        new_verdict, new_answers = aggregate_claimdecomp(gen)
        decisive = [a for a in new_answers if a in ("yes", "no")]
        new_frac = (
            sum(1 for a in decisive if a == "yes") / len(decisive)
            if decisive
            else None
        )
        if (
            row.get("pred_label") != new_verdict
            or row.get("parsed_answers") != new_answers
            or row.get("frac_yes_decisive") != new_frac
        ):
            n_changed += 1
            row["pred_label"] = new_verdict
            row["parsed_answers"] = new_answers
            row["frac_yes_decisive"] = new_frac
    if dry_run or n_changed == 0:
        return n_changed, len(rows)

    # Write JSONL back.
    with jsonlines.open(path, "w") as w:
        for row in rows:
            w.write(row)

    # Refresh metrics JSON; preserve existing metadata keys.
    metrics_path = path.replace("_results_", "_metrics_").replace(".jsonl", ".json")
    gt = [r["gt_label"] for r in rows]
    pred = [r["pred_label"] for r in rows]
    metrics = compute_classification_metrics(gt, pred)
    metrics["total_samples"] = len(rows)
    metrics["unparsed"] = sum(1 for r in rows if r["pred_label"] is None)
    n_subqs = [r.get("num_subquestions", len(r.get("subquestions", []))) for r in rows]
    if n_subqs:
        metrics["mean_num_subquestions"] = sum(n_subqs) / len(n_subqs)
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                old = json.load(f)
            for k in ("prompt_mode", "method", "backend", "model", "provider", "decomposer"):
                if k in old and k not in metrics:
                    metrics[k] = old[k]
        except Exception:
            pass
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return n_changed, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--results",
        help="Single *_baseline_results_claimdecomp.jsonl to process",
    )
    grp.add_argument(
        "--root",
        help="Walk this directory and re-aggregate every claimdecomp JSONL under it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count changes but don't write JSONL or metrics back",
    )
    args = parser.parse_args()

    if args.results:
        paths = [args.results]
    else:
        paths = sorted(
            p
            for p in glob.glob(
                os.path.join(args.root, "**", "*_baseline_results_claimdecomp.jsonl"),
                recursive=True,
            )
        )

    if not paths:
        raise SystemExit("No claimdecomp results JSONL found.")

    print(f"[reaggregate] {len(paths)} file(s) to process")
    total_changed = total_rows = 0
    for path in paths:
        if not _is_claimdecomp_file(path):
            print(f"  [skip] not a claimdecomp results file: {path}")
            continue
        n_changed, n_total = reaggregate_one(path, args.dry_run)
        total_changed += n_changed
        total_rows += n_total
        marker = "(dry)" if args.dry_run else "(written)" if n_changed else "(no change)"
        print(f"  {n_changed:>5}/{n_total:<5} changed {marker}  {path}")
    print(
        f"[reaggregate] total: {total_changed} rows changed across {total_rows} rows."
    )
    if args.dry_run:
        print("[reaggregate] dry-run — no files written.")


if __name__ == "__main__":
    main()
