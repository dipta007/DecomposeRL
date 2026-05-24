"""Diagnose unparsed rows in a baseline results JSONL.

Samples N rows whose `pred_label` is None (or whose generation contains
`[API_ERROR]`) and prints them so you can see what the model actually wrote
before deciding whether to (a) loosen the parser, (b) bump max_tokens and
re-run, or (c) re-issue the API call.

For ClaimDecomp, the printed payload includes the T5-generated sub-questions
that were fed into the aggregator — that's usually the key to telling
"model went off-format" from "input was degenerate".

Usage:
    PYTHONPATH=. python decomposer/baselines/diagnose_unparsed.py \\
        --results outputs/baseline_14b_prompted/fever_baseline_results_claimdecomp.jsonl

    # More / fewer samples (default 5)
    PYTHONPATH=. python decomposer/baselines/diagnose_unparsed.py \\
        --results <path> --n 10

    # Cap how much of each generation to print (default 1500 chars)
    PYTHONPATH=. python decomposer/baselines/diagnose_unparsed.py \\
        --results <path> --max_chars 4000

    # Save the sampled rows to a JSONL for offline inspection
    PYTHONPATH=. python decomposer/baselines/diagnose_unparsed.py \\
        --results <path> --out unparsed_samples.jsonl
"""

import argparse
import json
import os
import random
from typing import Dict, List

import jsonlines


def _is_unparsed(row: Dict) -> bool:
    """True if the row has no pred_label, or its generation is an API_ERROR sentinel."""
    if row.get("pred_label") is None:
        return True
    gen = row.get("generation") or ""
    return "[API_ERROR]" in gen


def _truncate(s: str, n: int) -> str:
    if s is None:
        return "<none>"
    s = str(s)
    if len(s) <= n:
        return s
    cut = s[:n]
    return cut + f"\n[…truncated, {len(s) - n} more chars omitted]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", required=True, help="Path to *_baseline_results_*.jsonl"
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of unparsed rows to sample"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1500,
        help="Max chars of `generation` to print per row",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for sampling (deterministic so repeat invocations show same rows)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="If given, also write the sampled rows to this JSONL",
    )
    parser.add_argument(
        "--include_api_error",
        action="store_true",
        help="Include rows whose generation is an [API_ERROR] sentinel "
        "(by default they're shown only if pred_label is None too)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"Not found: {args.results}")

    rows: List[Dict] = []
    with jsonlines.open(args.results) as r:
        for line in r:
            rows.append(line)
    unparsed = [row for row in rows if _is_unparsed(row)]
    if not args.include_api_error:
        unparsed = [r for r in unparsed if "[API_ERROR]" not in (r.get("generation") or "")]

    print(f"=== {args.results} ===")
    print(f"    total rows: {len(rows)}")
    n_pred_none = sum(1 for r in rows if r.get("pred_label") is None)
    n_apierr = sum(1 for r in rows if "[API_ERROR]" in (r.get("generation") or ""))
    print(f"    pred_label=None: {n_pred_none}")
    print(f"    [API_ERROR] in generation: {n_apierr}")
    print(f"    inspectable (unparsed, not API error): {len(unparsed)}")

    if not unparsed:
        print("    nothing to show.")
        return

    rng = random.Random(args.seed)
    sample = unparsed if len(unparsed) <= args.n else rng.sample(unparsed, args.n)

    for i, row in enumerate(sample, 1):
        print()
        print(f"--- sample {i}/{len(sample)} ---")
        print(f"id          : {row.get('id')}")
        print(f"dataset     : {row.get('dataset')}")
        print(f"gt_label    : {row.get('gt_label')}")
        print(f"pred_label  : {row.get('pred_label')!r}")
        # Method-specific trace fields (present only for some baselines).
        if "subquestions" in row:
            sqs = row.get("subquestions") or []
            print(f"sub-questions ({len(sqs)}): {sqs}")
        if "parsed_answers" in row:
            print(f"parsed_answers: {row.get('parsed_answers')}")
        if "frac_yes_decisive" in row:
            print(f"frac_yes_decisive: {row.get('frac_yes_decisive')}")
        if "qa_history" in row:
            qa = row.get("qa_history") or []
            print(f"qa_history ({len(qa)} turns):")
            for j, (q, a) in enumerate(qa[:3], 1):
                print(f"  Q{j}: {_truncate(q, 200)}")
                print(f"  A{j}: {_truncate(a, 200)}")
            if len(qa) > 3:
                print(f"  ... ({len(qa) - 3} more turn(s) hidden)")
        # Claim — useful when the model rambled about a different topic.
        print(f"claim       : {_truncate(row.get('claim'), 400)}")
        # The main payload.
        print("generation  :")
        print("    " + _truncate(row.get("generation"), args.max_chars).replace("\n", "\n    "))

    if args.out:
        with jsonlines.open(args.out, "w") as w:
            for row in sample:
                w.write(row)
        print(f"\nSampled rows saved to {args.out}")


if __name__ == "__main__":
    main()
