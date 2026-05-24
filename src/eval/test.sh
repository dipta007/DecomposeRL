# Datasets
DATASETS=(
  # Must-have baselines
  "fever"
  "claimdecomp"
  "hover"
  # Strongly recommended
  "wice"
  "feverous"
  # Breadth / domain transfer
  "ex_fever"
  "pubhealthfact"
  # "coverbench"
  # "coverbench_small"
  "fool_me_twice"
  "pubmedclaim"
  # "llmaggrefact"
  # Not needed
  # "matter_of_fact"
  # "healthver"
  # "covidfact"
  # "pubhealthtab"
  # "faviq_a_set"
  # "faviq_r_set"
  # "ambifc"
)
SIZES=("3b" "7b" "14b" "32b")

# =============================================================================
# Baseline models
# =============================================================================
# PROMPT_MODES=("simple" "cot" "iterative")
# PROMPT_MODES=("simple" "cot")
#
# for dataset in "${DATASETS[@]}"; do
#   for size in "${SIZES[@]}"; do
#     for mode in "${PROMPT_MODES[@]}"; do
#       PYTHONPATH=. python decomposer/eval/baseline.py \
#         -m "Qwen/Qwen2.5-${size^^}-Instruct" \
#         -o "outputs/baseline_${size}" \
#         --prompt_mode "$mode" \
#         -d "data/combined/step_9/test_${dataset}.jsonl"
#     done
#   done
#   PYTHONPATH=. python decomposer/eval/baseline_minicheck.py \
#     -d "data/combined/step_9/test_${dataset}.jsonl" \
#     -o outputs/baseline_minicheck_7b --force
# done

# exit 0

# =============================================================================
# Test with trained models
# =============================================================================

# Parse version numbers from arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 <version_numbers...>"
  echo "Example: $0 43 44 45"
  exit 1
fi
VERSIONS=("$@")

# random shuffle datasets
for i in "${!DATASETS[@]}"; do
  j=$((RANDOM % (i + 1)))
  temp="${DATASETS[i]}"
  DATASETS[i]="${DATASETS[j]}"
  DATASETS[j]="$temp"
done

# random shuffle versions
for i in "${!VERSIONS[@]}"; do
  j=$((RANDOM % (i + 1)))
  temp="${VERSIONS[i]}"
  VERSIONS[i]="${VERSIONS[j]}"
  VERSIONS[j]="$temp"
done

echo "Evaluating versions: ${VERSIONS[*]}"
echo "Datasets (random order): ${DATASETS[*]}"

while true; do
  echo "=== Starting evaluation run at $(date) ==="

  for dataset in "${DATASETS[@]}"; do
    # for ((i = ${#DATASETS[@]} - 1; i >= 0; i--)); do
    #   dataset="${DATASETS[$i]}"

    for v in "${VERSIONS[@]}"; do
      for size in "${SIZES[@]}"; do
        checkpoint_dir="outputs/2way_${size}_v${v}"
        [ -d "$checkpoint_dir" ] || continue
        echo "Evaluating dataset '$dataset' on checkpoint dir '$checkpoint_dir'..."
        PYTHONPATH=. python decomposer/eval/test.py -d "data/combined_5k/step_9/test_${dataset}.jsonl" -c "$checkpoint_dir"
      done
    done
  done

  echo "=== Finished evaluation run at $(date). Starting again... ==="
  # exit 0
done
