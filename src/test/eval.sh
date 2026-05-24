#!/usr/bin/env bash
# Evaluate a checkpoint on all test datasets.
#
# Usage:
#     bash src/test/eval.sh outputs/2way_7b/checkpoint-100

set -uo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: $0 <checkpoint_path>"
  echo "Example: $0 outputs/2way_7b/checkpoint-100"
  exit 1
fi

CHECKPOINT="$1"
DATASETS=(
  fever claimdecomp hover wice feverous
  ex_fever pubhealthfact coverbench fool_me_twice
  pubmedclaim llmaggrefact
)

echo "Checkpoint: $CHECKPOINT"
echo "Datasets: ${DATASETS[*]}"

for dataset in "${DATASETS[@]}"; do
  echo "--- $dataset ---"
  PYTHONPATH=. uv run python src/test/test.py -c "$CHECKPOINT" -d "$dataset"
done

echo "Done."
