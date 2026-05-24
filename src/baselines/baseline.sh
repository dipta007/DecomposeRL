#!/usr/bin/env bash
# Unified baseline sweep.
#
# Runs every baseline (existing simple/cot/iterative end-to-end + MiniCheck +
# the 6 prompted decomposition baselines) over every eval dataset and every
# model. Output paths mirror decomposer/eval/baseline.py so existing analysis
# scripts keep working:
#     outputs/baseline_<variant>/<dataset>_baseline_results_<mode>.jsonl
#     outputs/baseline_<variant>/<dataset>_baseline_metrics_<mode>.json
#
# Usage:
#     bash decomposer/baselines/baseline.sh
#
# Override defaults via env vars, e.g.:
#     DATASETS="pubmedclaim wice" METHODS="hiss folk" SIZES="7b" \
#         bash decomposer/baselines/baseline.sh
#
# Required env (only if running API baselines):
#     OPENAI_API_KEY      for provider=openai    (default model: gpt-4.1-mini)
#     ANTHROPIC_API_KEY   for provider=anthropic (default model: claude-haiku-4-5)

set -uo pipefail

# -----------------------------------------------------------------------------
# Config (override via env vars).
# -----------------------------------------------------------------------------
: "${DATA_DIR:=data/combined_5k/step_9}"
: "${OUT_ROOT:=outputs}"

# Match decomposer/eval/test.sh dataset list.
: "${DATASETS:=fever claimdecomp hover feverous wice ex_fever pubhealthfact fool_me_twice pubmedclaim coverbench}"

# Qwen2.5 variants for local vLLM rows.
: "${SIZES:=3b 7b 14b 32b}"

# Prompted baselines (new — decomposer/baselines/run.py).
: "${METHODS:=self_ask decomposed_prompting hiss folk programfc chen_complex}"

# Existing end-to-end modes (decomposer/eval/baseline.py).
: "${E2E_MODES:=simple cot iterative}"

# API providers to sweep. Empty string disables API rows.
: "${API_PROVIDERS:=openai}"

# Toggle individual blocks. ALL DEFAULT TO 0 — enable explicitly per run.
# Example: RUN_PROMPTED_VLLM=1 SIZES="7b" bash decomposer/baselines/baseline.sh
: "${RUN_E2E:=0}"           # simple / cot / iterative end-to-end Qwen rows (vLLM)
: "${RUN_E2E_API:=0}"       # simple / cot / iterative end-to-end via frontier APIs
: "${RUN_MINICHECK:=0}"     # MiniCheck-7B
: "${RUN_PROMPTED_VLLM:=0}" # 6 prompted methods over Qwen2.5
: "${RUN_PROMPTED_API:=0}"  # 6 prompted methods over gpt-4.1-mini + claude-haiku-4-5
: "${RUN_CLAIMDECOMP:=0}"   # T5-decomposer + Qwen/API aggregator (Chen 2022)
: "${RUN_QACHECK:=0}"       # multi-turn QACheck (Pan EMNLP 2023)

# Force-recompute every (dataset, method) pair, ignoring existing metrics JSONs.
# Default (FORCE=0) preserves the dataset-level skip semantics of each runner.
: "${FORCE:=0}"
if [[ "$FORCE" == "1" ]]; then
  FORCE_FLAG="--force"
else
  FORCE_FLAG=""
fi

mkdir -p "$OUT_ROOT"

# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------
declare -a DATASET_ARR=($DATASETS)
declare -a SIZE_ARR=($SIZES)
declare -a METHOD_ARR=($METHODS)
declare -a E2E_ARR=($E2E_MODES)
declare -a PROV_ARR=($API_PROVIDERS)

echo "=========================================================="
echo "Baseline sweep"
echo "=========================================================="
echo "  Datasets       : ${DATASET_ARR[*]}"
echo "  Qwen sizes     : ${SIZE_ARR[*]}"
echo "  Prompted modes : ${METHOD_ARR[*]}"
echo "  E2E modes      : ${E2E_ARR[*]}"
echo "  API providers  : ${PROV_ARR[*]:-<disabled>}"
echo "  Toggles        : E2E=$RUN_E2E E2E_API=$RUN_E2E_API MINICHECK=$RUN_MINICHECK"
echo "                   PROMPTED_VLLM=$RUN_PROMPTED_VLLM PROMPTED_API=$RUN_PROMPTED_API"
echo "                   CLAIMDECOMP=$RUN_CLAIMDECOMP QACHECK=$RUN_QACHECK"
echo "  Force recompute: $FORCE"
echo "=========================================================="

# -----------------------------------------------------------------------------
# 1) Existing end-to-end Qwen baselines (simple / cot / iterative).
#    Output dir: outputs/baseline_<size>/
# -----------------------------------------------------------------------------
if [[ "$RUN_E2E" == "1" ]]; then
  echo "--- [1/4] End-to-end Qwen baselines (simple/cot/iterative) ---"
  for dataset in "${DATASET_ARR[@]}"; do
    for size in "${SIZE_ARR[@]}"; do
      for mode in "${E2E_ARR[@]}"; do
        PYTHONPATH=. python decomposer/eval/baseline.py \
          -m "Qwen/Qwen2.5-${size^^}-Instruct" \
          -o "$OUT_ROOT/baseline_${size}" \
          --prompt_mode "$mode" \
          -d "$DATA_DIR/test_${dataset}.jsonl" \
          $FORCE_FLAG
      done
    done
  done
fi

# -----------------------------------------------------------------------------
# 1b) End-to-end frontier-API baselines (simple / cot / iterative).
#     Mirrors block 1 but routes inference through decomposer/baselines/api.py
#     instead of vLLM. Output dir: outputs/baseline_api_<provider>_e2e/
# -----------------------------------------------------------------------------
if [[ "$RUN_E2E_API" == "1" && ${#PROV_ARR[@]} -gt 0 ]]; then
  echo "--- [1b] End-to-end frontier-API baselines (simple/cot/iterative) ---"
  for provider in "${PROV_ARR[@]}"; do
    case "$provider" in
    openai)
      if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "[skip] e2e_api openai: OPENAI_API_KEY not set"
        continue
      fi
      ;;
    anthropic)
      if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "[skip] e2e_api anthropic: ANTHROPIC_API_KEY not set"
        continue
      fi
      ;;
    umbc | anthropic_native)
      if [[ -z "${UMBC_GATEWAY_KEY:-}" ]]; then
        echo "[skip] e2e_api ${provider}: UMBC_GATEWAY_KEY not set"
        continue
      fi
      ;;
    esac
    for dataset in "${DATASET_ARR[@]}"; do
      for mode in "${E2E_ARR[@]}"; do
        PYTHONPATH=. python decomposer/eval/baseline.py \
          --backend api --provider "$provider" \
          -o "$OUT_ROOT/baseline_api_${provider}_e2e" \
          --prompt_mode "$mode" \
          -d "$DATA_DIR/test_${dataset}.jsonl" \
          $FORCE_FLAG
      done
    done
  done
fi

# -----------------------------------------------------------------------------
# 2) MiniCheck-7B end-to-end classifier.
#    Output dir: outputs/baseline_minicheck_7b/
# -----------------------------------------------------------------------------
if [[ "$RUN_MINICHECK" == "1" ]]; then
  echo "--- [2/4] MiniCheck-7B ---"
  for dataset in "${DATASET_ARR[@]}"; do
    PYTHONPATH=. python decomposer/eval/baseline_minicheck.py \
      -d "$DATA_DIR/test_${dataset}.jsonl" \
      -o "$OUT_ROOT/baseline_minicheck_7b" \
      $FORCE_FLAG
  done
fi

# -----------------------------------------------------------------------------
# 3) Prompted decomposition baselines on Qwen2.5 via vLLM.
#    Output dir: outputs/baseline_<size>_prompted/
# -----------------------------------------------------------------------------
if [[ "$RUN_PROMPTED_VLLM" == "1" ]]; then
  echo "--- [3/4] Prompted decomposition baselines on Qwen via vLLM ---"
  for dataset in "${DATASET_ARR[@]}"; do
    for size in "${SIZE_ARR[@]}"; do
      for method in "${METHOD_ARR[@]}"; do
        PYTHONPATH=. python decomposer/baselines/run.py \
          --method "$method" \
          --backend vllm \
          --model "Qwen/Qwen2.5-${size^^}-Instruct" \
          --test_data "$DATA_DIR/test_${dataset}.jsonl" \
          --output_dir "$OUT_ROOT/baseline_${size}_prompted" \
          $FORCE_FLAG
      done
    done
  done
fi

# -----------------------------------------------------------------------------
# 4) Prompted decomposition baselines on frontier APIs (OpenAI + Anthropic).
#    Output dir: outputs/baseline_api_<provider>_prompted/
# -----------------------------------------------------------------------------
if [[ "$RUN_PROMPTED_API" == "1" && ${#PROV_ARR[@]} -gt 0 ]]; then
  echo "--- [4/4] Prompted decomposition baselines on frontier APIs ---"
  for provider in "${PROV_ARR[@]}"; do
    case "$provider" in
    openai)
      if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "[skip] provider=openai: OPENAI_API_KEY not set"
        continue
      fi
      ;;
    anthropic)
      if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "[skip] provider=anthropic: ANTHROPIC_API_KEY not set"
        continue
      fi
      ;;
    umbc | anthropic_native)
      if [[ -z "${UMBC_GATEWAY_KEY:-}" ]]; then
        echo "[skip] provider=${provider}: UMBC_GATEWAY_KEY not set"
        continue
      fi
      ;;
    esac
    for dataset in "${DATASET_ARR[@]}"; do
      for method in "${METHOD_ARR[@]}"; do
        PYTHONPATH=. python decomposer/baselines/run.py \
          --method "$method" \
          --backend api --provider "$provider" \
          --test_data "$DATA_DIR/test_${dataset}.jsonl" \
          --output_dir "$OUT_ROOT/baseline_api_${provider}_prompted" \
          $FORCE_FLAG
      done
    done
  done
fi

# -----------------------------------------------------------------------------
# 5) ClaimDecomp (Chen 2022): T5-3B-ClaimDecomp decomposes each claim into
#    sub-questions; aggregator LLM (Qwen vLLM or frontier API) answers from
#    the same evidence and emits a verdict. Output paths reuse the
#    prompted-vLLM / prompted-API layouts so the analysis loaders pick it up
#    as another "method".
#
#    Output dirs:
#      outputs/baseline_<size>_prompted/<dataset>_baseline_metrics_claimdecomp.json
#      outputs/baseline_api_<provider>_prompted/<dataset>_baseline_metrics_claimdecomp.json
# -----------------------------------------------------------------------------
if [[ "$RUN_CLAIMDECOMP" == "1" ]]; then
  echo "--- [5/5] ClaimDecomp (T5 decomposer + LLM aggregator) ---"
  # Qwen vLLM aggregators.
  for dataset in "${DATASET_ARR[@]}"; do
    for size in "${SIZE_ARR[@]}"; do
      PYTHONPATH=. python decomposer/baselines/claimdecomp.py \
        --backend vllm \
        --model "Qwen/Qwen2.5-${size^^}-Instruct" \
        --test_data "$DATA_DIR/test_${dataset}.jsonl" \
        --output_dir "$OUT_ROOT/baseline_${size}_prompted" \
        $FORCE_FLAG
    done
  done
  # API aggregators (skip when key missing — same logic as the prompted-API block).
  if [[ ${#PROV_ARR[@]} -gt 0 ]]; then
    for provider in "${PROV_ARR[@]}"; do
      case "$provider" in
      openai)
        [[ -z "${OPENAI_API_KEY:-}" ]] && {
          echo "[skip] claimdecomp openai: OPENAI_API_KEY unset"
          continue
        }
        ;;
      anthropic)
        [[ -z "${ANTHROPIC_API_KEY:-}" ]] && {
          echo "[skip] claimdecomp anthropic: ANTHROPIC_API_KEY unset"
          continue
        }
        ;;
      umbc | anthropic_native)
        [[ -z "${UMBC_GATEWAY_KEY:-}" ]] && {
          echo "[skip] claimdecomp ${provider}: UMBC_GATEWAY_KEY unset"
          continue
        }
        ;;
      esac
      for dataset in "${DATASET_ARR[@]}"; do
        PYTHONPATH=. python decomposer/baselines/claimdecomp.py \
          --backend api --provider "$provider" \
          --test_data "$DATA_DIR/test_${dataset}.jsonl" \
          --output_dir "$OUT_ROOT/baseline_api_${provider}_prompted" \
          $FORCE_FLAG
      done
    done
  fi
fi

# -----------------------------------------------------------------------------
# 6) QACheck (Pan EMNLP 2023): multi-turn sufficiency check → next question →
#    answer → repeat → final verdict, batched across claims per turn.
#    Output paths reuse the prompted-vLLM / prompted-API layouts.
#
#    Output dirs:
#      outputs/baseline_<size>_prompted/<dataset>_baseline_metrics_qacheck.json
#      outputs/baseline_api_<provider>_prompted/<dataset>_baseline_metrics_qacheck.json
# -----------------------------------------------------------------------------
if [[ "$RUN_QACHECK" == "1" ]]; then
  echo "--- [6/6] QACheck (multi-turn QA loop) ---"
  # Qwen vLLM aggregators.
  for dataset in "${DATASET_ARR[@]}"; do
    for size in "${SIZE_ARR[@]}"; do
      PYTHONPATH=. python decomposer/baselines/qacheck.py \
        --backend vllm \
        --model "Qwen/Qwen2.5-${size^^}-Instruct" \
        --test_data "$DATA_DIR/test_${dataset}.jsonl" \
        --output_dir "$OUT_ROOT/baseline_${size}_prompted" \
        $FORCE_FLAG
    done
  done
  # API aggregators (skip when key missing).
  if [[ ${#PROV_ARR[@]} -gt 0 ]]; then
    for provider in "${PROV_ARR[@]}"; do
      case "$provider" in
      openai)
        [[ -z "${OPENAI_API_KEY:-}" ]] && {
          echo "[skip] qacheck openai: OPENAI_API_KEY unset"
          continue
        }
        ;;
      anthropic)
        [[ -z "${ANTHROPIC_API_KEY:-}" ]] && {
          echo "[skip] qacheck anthropic: ANTHROPIC_API_KEY unset"
          continue
        }
        ;;
      umbc | anthropic_native)
        [[ -z "${UMBC_GATEWAY_KEY:-}" ]] && {
          echo "[skip] qacheck ${provider}: UMBC_GATEWAY_KEY unset"
          continue
        }
        ;;
      esac
      for dataset in "${DATASET_ARR[@]}"; do
        PYTHONPATH=. python decomposer/baselines/qacheck.py \
          --backend api --provider "$provider" \
          --test_data "$DATA_DIR/test_${dataset}.jsonl" \
          --output_dir "$OUT_ROOT/baseline_api_${provider}_prompted" \
          $FORCE_FLAG
      done
    done
  fi
fi

# -----------------------------------------------------------------------------
# Post-sweep audit: defer to the standalone script so it can also be run
# on demand without re-launching a full sweep.
# Override scope via ROOT=outputs/<subdir>; show retry commands with SHOW_RETRY=1.
# -----------------------------------------------------------------------------
ROOT="$OUT_ROOT" SHOW_RETRY="${SHOW_RETRY:-0}" \
  bash "$(dirname "$0")/audit.sh"

echo "Done."
