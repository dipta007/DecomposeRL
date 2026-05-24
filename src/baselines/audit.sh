#!/usr/bin/env bash
# audit.sh — Scan baseline outputs for unparsed verdicts and [API_ERROR] rows.
#
# Reports two failure modes that survive a sweep silently:
#   1. unparsed > 0    : the model output didn't yield a <verdict> tag for some
#                        rows (usually max_tokens truncation or sentinel-baked
#                        API errors). Counted in the metrics JSON.
#   2. [API_ERROR] ... : per-sample JSONL has the sentinel rows from a
#                        non-retryable API failure (e.g. budget exceeded).
#
# Usage:
#   bash decomposer/baselines/audit.sh
#   ROOT=outputs/baseline_api_umbc_prompted bash decomposer/baselines/audit.sh
#   LIMIT=200 bash decomposer/baselines/audit.sh     # show up to 200 rows per category
#   SHOW_RETRY=1 bash decomposer/baselines/audit.sh  # emit ready-to-run retry commands

set -uo pipefail

: "${ROOT:=outputs}"
: "${LIMIT:=30}"
: "${SHOW_RETRY:=0}"

if [[ ! -d "$ROOT" ]]; then
  echo "audit: ROOT=$ROOT not found" >&2
  exit 1
fi

echo "=========================================================="
echo "Baseline audit — scanning $ROOT"
echo "=========================================================="

PYTHONPATH=. python3 - "$ROOT" "$LIMIT" "$SHOW_RETRY" <<'PY'
import glob, json, os, re, sys

root = sys.argv[1]
limit = int(sys.argv[2])
show_retry = sys.argv[3] == "1"

warn_unparsed = []  # (metrics_path, n_unparsed, n_total)
warn_apierr = []    # (results_path, n_err)

for path in sorted(
    glob.glob(os.path.join(root, "**", "*_baseline_metrics_*.json"), recursive=True)
):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        continue
    n_unparsed = d.get("unparsed", 0) or 0
    n_total = d.get("total_samples", 0) or 0
    if n_unparsed:
        warn_unparsed.append((path, n_unparsed, n_total))
    jsonl = path.replace("_metrics_", "_results_").replace(".json", ".jsonl")
    if os.path.exists(jsonl):
        n_err = 0
        with open(jsonl) as fh:
            for line in fh:
                if "[API_ERROR]" in line:
                    n_err += 1
        if n_err:
            warn_apierr.append((jsonl, n_err))


def _print_truncated(rows, fmt):
    for r in rows[:limit]:
        print(fmt(r))
    if len(rows) > limit:
        print(f"    ... and {len(rows) - limit} more (raise LIMIT to see all)")


if warn_unparsed:
    print(f"\n  unparsed > 0 in {len(warn_unparsed)} metrics file(s):")
    _print_truncated(
        warn_unparsed,
        lambda r: f"    {r[1]:>5}/{r[2]:<5} ({100.0 * r[1] / max(r[2], 1):5.1f}%)  {r[0]}",
    )

if warn_apierr:
    print(f"\n  [API_ERROR] in {len(warn_apierr)} results file(s):")
    _print_truncated(warn_apierr, lambda r: f"    {r[1]:>5}  {r[0]}")

    if show_retry:
        print("\n  Suggested retry commands (one per file):")
        for path, _ in warn_apierr:
            # Try to infer the provider from the parent dir name, e.g.
            #   outputs/baseline_api_umbc_prompted -> umbc
            parent = os.path.basename(os.path.dirname(path))
            m = re.match(r"baseline_api_([a-zA-Z0-9_]+)_(?:prompted|e2e)$", parent)
            provider = m.group(1) if m else "<provider>"
            print(
                f"    PYTHONPATH=. python decomposer/baselines/retry_api_errors.py "
                f"--results {path} --provider {provider}"
            )

if not warn_unparsed and not warn_apierr:
    print("  clean — all metrics have unparsed=0 and no API errors in results")
PY

echo "=========================================================="
