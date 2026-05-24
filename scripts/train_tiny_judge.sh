#!/usr/bin/env bash
# Train tiny-judge classifiers (ModernBERT-large + LoRA) for fast reward computation.
#
# Usage:
#     bash scripts/train_tiny_judge.sh --task all
#     bash scripts/train_tiny_judge.sh --task coverage --push
#     bash scripts/train_tiny_judge.sh --task coverage --limit 100 --no-wandb
#
# Key arguments:
#   --task          all | atomicity_checklist | question_answerable | answer_correctness | coverage  [all]
#   --batch-size    Per-device batch size    [64]
#   --grad-accum    Gradient accumulation    [1]
#   --limit         Cap training rows (for debugging)
#   --push          Merge LoRA + push to HF Hub
#   --no-wandb      Disable W&B logging
#
# Run with -h for all options.

set -uo pipefail

PYTHONPATH=. uv run python src/train/tiny_judge/train_encoder_balanced.py "$@"
