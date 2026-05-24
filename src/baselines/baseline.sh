#!/usr/bin/env bash
# Run all baselines over evaluation datasets.
#
# Usage:
#     bash src/baselines/baseline.sh
#
# Override via env vars:
#     DATASETS="pubmedclaim wice" SIZES="7b" bash src/baselines/baseline.sh
#
# For API baselines, set OPENAI_API_KEY and/or ANTHROPIC_API_KEY.

set -uo pipefail

: "${OUT_ROOT:=outputs}"
: "${DATASETS:=fever claimdecomp hover feverous wice ex_fever pubhealthfact fool_me_twice pubmedclaim coverbench}"
: "${SIZES:=7b}"
: "${METHODS:=self_ask decomposed_prompting hiss folk programfc chen_complex}"
: "${FORCE:=0}"

[[ "$FORCE" == "1" ]] && FORCE_FLAG="--force" || FORCE_FLAG=""

read -ra DATASET_ARR <<< "$DATASETS"
read -ra SIZE_ARR <<< "$SIZES"
read -ra METHOD_ARR <<< "$METHODS"

echo "Datasets: ${DATASET_ARR[*]}"
echo "Sizes:    ${SIZE_ARR[*]}"
echo "Methods:  ${METHOD_ARR[*]}"

# Simple + CoT baselines
for dataset in "${DATASET_ARR[@]}"; do
  for size in "${SIZE_ARR[@]}"; do
    for mode in simple cot; do
      PYTHONPATH=. uv run python src/baselines/direct.py \
        -m "Qwen/Qwen2.5-${size^^}-Instruct" \
        -o "$OUT_ROOT/baseline_${size}" \
        --mode "$mode" \
        -d "$dataset" \
        $FORCE_FLAG
    done
  done
done

# MiniCheck
for dataset in "${DATASET_ARR[@]}"; do
  PYTHONPATH=. uv run python src/baselines/nli.py \
    -d "$dataset" \
    -o "$OUT_ROOT/baseline_minicheck_7b" \
    $FORCE_FLAG
done

# Prompted decomposition baselines (vLLM)
for dataset in "${DATASET_ARR[@]}"; do
  for size in "${SIZE_ARR[@]}"; do
    for method in "${METHOD_ARR[@]}"; do
      PYTHONPATH=. uv run python src/baselines/run.py \
        --method "$method" \
        --backend vllm \
        --model "Qwen/Qwen2.5-${size^^}-Instruct" \
        --dataset "$dataset" \
        --output_dir "$OUT_ROOT/baseline_${size}_prompted" \
        $FORCE_FLAG
    done
  done
done

# ClaimDecomp (T5 decomposer + LLM aggregator)
for dataset in "${DATASET_ARR[@]}"; do
  for size in "${SIZE_ARR[@]}"; do
    PYTHONPATH=. uv run python src/baselines/claimdecomp.py \
      --backend vllm \
      --model "Qwen/Qwen2.5-${size^^}-Instruct" \
      --dataset "$dataset" \
      --output_dir "$OUT_ROOT/baseline_${size}_prompted" \
      $FORCE_FLAG
  done
done

# QACheck (multi-turn QA loop)
for dataset in "${DATASET_ARR[@]}"; do
  for size in "${SIZE_ARR[@]}"; do
    PYTHONPATH=. uv run python src/baselines/qacheck.py \
      --backend vllm \
      --model "Qwen/Qwen2.5-${size^^}-Instruct" \
      --dataset "$dataset" \
      --output_dir "$OUT_ROOT/baseline_${size}_prompted" \
      $FORCE_FLAG
  done
done

# API baselines (only if keys are set)
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  for dataset in "${DATASET_ARR[@]}"; do
    for method in "${METHOD_ARR[@]}"; do
      PYTHONPATH=. uv run python src/baselines/run.py \
        --method "$method" \
        --backend api --provider openai \
        --dataset "$dataset" \
        --output_dir "$OUT_ROOT/baseline_api_openai_prompted" \
        $FORCE_FLAG
    done
  done
fi

echo "Done."
