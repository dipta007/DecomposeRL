#!/usr/bin/env bash
# GRPO training for DecomposeRL.
#
# Prerequisites:
#   - For REWARD_BACKEND=llm-judge (default): run `bash scripts/serve_judge.sh` first
#   - For REWARD_BACKEND=tiny-judge: no server needed (local classifiers)
#
# Usage:
#     bash scripts/train_grpo.sh --run_name my_run
#     bash scripts/train_grpo.sh --run_name my_run --model_name unsloth/Qwen2.5-3B-Instruct
#
# ── CLI Arguments ────────────────────────────────────────────────────────────
#   --run_name, -r        (required) W&B run name and output dir name
#   --model_name          Base model              [unsloth/Qwen2.5-7B-Instruct]
#   --lora_rank           LoRA rank               [64]
#   --num_of_generations  GRPO generations/prompt  [8]
#   --num_train_epochs    Training epochs          [2]
#   --max_seq_length      Max sequence length      [16016]
#   --loss_type           GRPO loss variant        [bnpo]
#   --random_seed         Random seed              [42]
#
# ── Environment Variables ────────────────────────────────────────────────────
#
#   Reward backend:
#     REWARD_BACKEND          llm-judge | tiny-judge          [llm-judge]
#     JUDGE_PORT              vLLM judge server port          [8000]
#
#   Reward toggles (1=enabled, 0=disabled):
#     NECESSITY_SALIENCY_REWARD   Leave-one-out saliency      [1]
#     JOINT_QUALITY_REWARD        Per-question quality         [1]
#     DIVERSITY_REWARD_ENABLED    Embedding diversity          [1]
#     COVERAGE_REWARD             Coverage judge               [1]
#     GOOD_NUM_Q_REWARD           Question count penalty       [1]
#
#   Reward config:
#     DIVERSITY_REWARD        mmr | vendi                     [mmr]
#     NECESSITY_AGGREGATION   mean | min                      [mean]
#     SUPERVISION_RATE        0.0-1.0 (fraction using GT)     [1.0]
#
# ── Examples ─────────────────────────────────────────────────────────────────
#
#   # Default training (LLM judge, all rewards on)
#   bash scripts/train_grpo.sh --run_name v1
#
#   # Tiny-judge backend (no judge server needed)
#   REWARD_BACKEND=tiny-judge bash scripts/train_grpo.sh --run_name v1_tiny
#
#   # Semi-supervised (30% GT labels, 70% self-consistency)
#   SUPERVISION_RATE=0.3 bash scripts/train_grpo.sh --run_name v1_semi
#
#   # Ablation: disable coverage reward
#   COVERAGE_REWARD=0 bash scripts/train_grpo.sh --run_name v1_no_cov
#
#   # Ablation: vendi diversity instead of MMR
#   DIVERSITY_REWARD=vendi bash scripts/train_grpo.sh --run_name v1_vendi
#
#   # 3B model, 1 epoch
#   bash scripts/train_grpo.sh --run_name v1_3b \
#       --model_name unsloth/Qwen2.5-3B-Instruct --num_train_epochs 1

set -uo pipefail

PYTHONPATH=. uv run python src/train/decomposerl/train.py "$@"
