"""Retry only the [API_ERROR] rows in an existing baseline results JSONL.

Use this after a budget-killed UMBC / OpenAI / Anthropic sweep where some rows
fall through to the `[API_ERROR]` sentinel but the per-dataset metrics JSON was
still written. Surviving rows are kept untouched; only failed ones are re-issued.

What it does:
1. Loads <prefix>_baseline_results_<method>.jsonl.
2. Finds rows whose `generation` contains `[API_ERROR]`.
3. Re-issues those rows through the API using the same prompts.py template.
4. Writes the patched rows back to the same JSONL (in place).
5. Recomputes <prefix>_baseline_metrics_<method>.json from the updated rows.

Supported method families (auto-detected from filename or row):
- Standard prompted (self_ask, decomposed_prompting, hiss, folk, programfc,
  chen_complex) — uses prompts.py::build_prompted_prompt.
- ClaimDecomp — uses cached `subquestions` from the row + the aggregator
  prompt; verdict is recomputed via the rule-based aggregator. T5 is NOT
  re-run (it's deterministic, so cached subquestions are reused).
- QACheck — NOT supported (multi-turn state). Re-run with FORCE=1 if needed.

Idempotent: re-running on the same file will retry whatever rows still hold
[API_ERROR] after the previous attempt.

Usage:
    PYTHONPATH=. python decomposer/baselines/retry_api_errors.py \\
        --results outputs/baseline_api_umbc_prompted/fever_baseline_results_self_ask.jsonl \\
        --provider umbc

    # Bulk: every JSONL in a dir
    for f in outputs/baseline_api_umbc_prompted/*_baseline_results_*.jsonl; do
        PYTHONPATH=. python decomposer/baselines/retry_api_errors.py --results "$f" --provider umbc
    done

    # Dry-run: just report how many rows would be retried
    PYTHONPATH=. python decomposer/baselines/retry_api_errors.py \\
        --results <path> --dry-run
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import jsonlines

from decomposer.baselines.prompts import (
    PROMPTED_TEMPLATES,
    aggregate_claimdecomp,
    build_claimdecomp_aggregator_prompt,
    build_prompted_prompt,
    extract_verdict_tag,
)
from decomposer.eval.utils import compute_classification_metrics

# E2E baselines (simple/cot/iterative) live in decomposer/eval/baseline.py and
# use their own templates + verdict extractors. Import them here so we can
# rebuild prompts of cells produced by that runner too.
from decomposer.eval.baseline import (
    COT_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE,
    extract_cot_label,
    extract_iterative_label,
    extract_simple_label,
)
from decomposer.prompts import USER_PROMPT_2WAY_TEMPLATE


E2E_MODES = {"simple", "cot", "iterative"}
SUPPORTED_METHODS = set(PROMPTED_TEMPLATES.keys()) | {"claimdecomp"} | E2E_MODES


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------
def parse_results_filename(path: str) -> tuple[str, str]:
    """Return (dataset, method) from a `<dataset>_baseline_results_<method>.jsonl` path."""
    fname = os.path.basename(path)
    m = re.match(r"(.+?)_baseline_results_(\w+)\.jsonl$", fname)
    if not m:
        raise ValueError(
            f"Cannot parse dataset/method from filename: {fname}. "
            f"Expected <dataset>_baseline_results_<method>.jsonl"
        )
    return m.group(1), m.group(2)


def metrics_path_for(results_path: str) -> str:
    return results_path.replace("_results_", "_metrics_").replace(".jsonl", ".json")


# --------------------------------------------------------------------------
# Row classification
# --------------------------------------------------------------------------
def is_error_row(row: Dict) -> bool:
    gen = row.get("generation") or ""
    return "[API_ERROR]" in gen


# --------------------------------------------------------------------------
# Prompt rebuilding per method family
# --------------------------------------------------------------------------
def build_prompt_for_row(method: str, row: Dict, test_sample: Dict) -> Optional[str]:
    """Reconstruct the prompt that was originally issued for `row`.

    `test_sample` comes from the original test JSONL (keyed by `id`); we use it
    only for the evidence field, which isn't saved in the result row.
    """
    claim = test_sample.get("claim") or row.get("claim")
    evidence = test_sample.get("evidence")
    if claim is None or evidence is None:
        return None

    if method in PROMPTED_TEMPLATES:
        return build_prompted_prompt(method, claim, evidence)

    if method == "claimdecomp":
        # Reuse the T5 sub-questions cached in the original row — T5 is
        # deterministic, so this avoids re-loading a 3B model just to retry.
        subqs = row.get("subquestions") or []
        return build_claimdecomp_aggregator_prompt(
            claim=claim, evidence=evidence, subquestions=subqs
        )

    # E2E baselines from decomposer/eval/baseline.py.
    if method == "simple":
        return SIMPLE_PROMPT_TEMPLATE.format(evidence=evidence, claim=claim)
    if method == "cot":
        return COT_PROMPT_TEMPLATE.format(evidence=evidence, claim=claim)
    if method == "iterative":
        # Note the kwarg is `evidence_doc`, not `evidence`, in the policy template.
        return USER_PROMPT_2WAY_TEMPLATE.format(claim=claim, evidence_doc=evidence)

    return None


def reparse_row(method: str, row: Dict, new_generation: str) -> Dict:
    """Replace the generation + recompute pred_label and method-specific fields."""
    row["generation"] = new_generation
    if "[API_ERROR]" in new_generation:
        row["pred_label"] = None
        if method == "claimdecomp":
            row["parsed_answers"] = []
            row["frac_yes_decisive"] = None
        return row

    if method in PROMPTED_TEMPLATES:
        row["pred_label"] = extract_verdict_tag(new_generation)
        return row

    if method == "claimdecomp":
        verdict, parsed_answers = aggregate_claimdecomp(new_generation)
        decisive = [a for a in parsed_answers if a in ("yes", "no")]
        frac_yes = (
            sum(1 for a in decisive if a == "yes") / len(decisive)
            if decisive
            else None
        )
        row["parsed_answers"] = parsed_answers
        row["frac_yes_decisive"] = frac_yes
        row["pred_label"] = verdict
        return row

    # E2E modes: use the same extractors that decomposer/eval/baseline.py uses
    # so verdicts produced by retry are bit-identical to those produced by a
    # fresh sweep.
    if method == "simple":
        row["pred_label"] = extract_simple_label(new_generation)
        return row
    if method == "cot":
        row["pred_label"] = extract_cot_label(new_generation)
        # The CoT runner also stores a `reasoning` field; refresh it for the
        # retried row using the same extraction logic as baseline.py.
        try:
            from decomposer.eval.baseline import extract_cot_reasoning

            row["reasoning"] = extract_cot_reasoning(new_generation)
        except Exception:
            pass
        return row
    if method == "iterative":
        row["pred_label"] = extract_iterative_label(new_generation)
        return row

    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a *_baseline_results_*.jsonl to patch in place",
    )
    parser.add_argument(
        "--test_data",
        default=None,
        help="Path to the source test JSONL (for evidence lookup). "
        "If omitted, derived from --data_dir + dataset.",
    )
    parser.add_argument(
        "--data_dir",
        default="data/combined_5k/step_9",
        help="Directory containing test_<dataset>.jsonl files",
    )
    # API plumbing — mirrors run.py.
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "custom", "anthropic_native", "umbc"],
        default="umbc",
    )
    parser.add_argument("--api_model", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=16768)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be retried and exit without API calls",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"Results file not found: {args.results}")
    dataset, method = parse_results_filename(args.results)

    if method == "qacheck":
        raise SystemExit(
            "QACheck has multi-turn state that can't be cheaply patched. "
            "Re-run the cell with FORCE=1 instead."
        )
    if method not in SUPPORTED_METHODS:
        raise SystemExit(
            f"Unsupported method '{method}'. Supported: {sorted(SUPPORTED_METHODS)}"
        )
    is_e2e = method in E2E_MODES

    # --- Load existing results JSONL ---
    rows: List[Dict] = []
    with jsonlines.open(args.results) as r:
        for line in r:
            rows.append(line)
    error_indices = [i for i, row in enumerate(rows) if is_error_row(row)]
    n_total = len(rows)
    n_err = len(error_indices)
    print(f"[retry] {args.results}")
    print(f"        method={method} dataset={dataset}")
    print(f"        total rows: {n_total}, [API_ERROR] rows: {n_err}")
    if n_err == 0:
        print("        nothing to retry — exiting cleanly.")
        return
    if args.dry_run:
        print("        (--dry-run) no API calls made.")
        return

    # --- Load test JSONL to recover the evidence field ---
    test_path = args.test_data or os.path.join(args.data_dir, f"test_{dataset}.jsonl")
    if not os.path.exists(test_path):
        raise SystemExit(
            f"Test data not found at {test_path}. Pass --test_data explicitly."
        )
    test_by_id: Dict[str, Dict] = {}
    with jsonlines.open(test_path) as r:
        for sample in r:
            test_by_id[str(sample.get("id"))] = sample
    print(f"        test_data: {test_path} ({len(test_by_id)} samples indexed by id)")

    # --- Rebuild prompts for the error rows ---
    prompts: List[str] = []
    retry_indices: List[int] = []
    n_unrebuildable = 0
    for i in error_indices:
        rid = str(rows[i].get("id"))
        sample = test_by_id.get(rid)
        if sample is None:
            n_unrebuildable += 1
            continue
        prompt = build_prompt_for_row(method, rows[i], sample)
        if prompt is None:
            n_unrebuildable += 1
            continue
        prompts.append(prompt)
        retry_indices.append(i)
    if n_unrebuildable:
        print(f"        WARN: {n_unrebuildable} error rows could not be rebuilt")
    print(f"        retrying {len(prompts)} row(s) via provider={args.provider} ...")

    # --- Issue API calls (import here to keep dry-run path fast) ---
    from decomposer.baselines.api import build_config, run_api_inference

    cfg = build_config(
        provider=args.provider,
        model=args.api_model,
        base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )
    generations = run_api_inference(cfg, prompts)

    # --- Patch rows in place ---
    n_recovered = n_still_error = 0
    for i, gen in zip(retry_indices, generations):
        rows[i] = reparse_row(method, rows[i], gen)
        if is_error_row(rows[i]):
            n_still_error += 1
        else:
            n_recovered += 1
    print(f"        recovered: {n_recovered}, still error: {n_still_error}")

    # --- Write patched JSONL back ---
    with jsonlines.open(args.results, "w") as w:
        for row in rows:
            w.write(row)
    print(f"        wrote {args.results}")

    # --- Recompute metrics, preserving existing metadata keys ---
    gt = [r["gt_label"] for r in rows]
    pred = [r["pred_label"] for r in rows]
    metrics = compute_classification_metrics(gt, pred)
    metrics["total_samples"] = len(rows)
    metrics["unparsed"] = sum(1 for r in rows if r["pred_label"] is None)
    if method == "claimdecomp":
        # Recompute the claimdecomp-specific summary fields.
        n_subqs = [r.get("num_subquestions", len(r.get("subquestions", []))) for r in rows]
        metrics["mean_num_subquestions"] = (
            sum(n_subqs) / max(len(n_subqs), 1)
        )
    m_path = metrics_path_for(args.results)
    if os.path.exists(m_path):
        try:
            with open(m_path) as f:
                old = json.load(f)
            for k in ("prompt_mode", "method", "backend", "model", "provider", "decomposer"):
                if k in old and k not in metrics:
                    metrics[k] = old[k]
        except Exception as e:
            print(f"        WARN: could not read old metrics JSON ({e}); writing fresh")
    with open(m_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"        wrote {m_path}")
    print("[retry] done.")


if __name__ == "__main__":
    main()
