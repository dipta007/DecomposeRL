#!/usr/bin/env bash
# Start vLLM servers for the LLM judge (Qwen3-32B) and embedding model (Qwen3-Embedding-8B).
# Required for GRPO training with REWARD_BACKEND=llm-judge (the default).
#
# Usage:
#     bash scripts/serve_judge.sh
#
# Override GPU count or ports via env vars:
#     JUDGE_TP=2 JUDGE_PORT=8000 EMB_PORT=8004 bash scripts/serve_judge.sh

set -uo pipefail

: "${JUDGE_MODEL:=Qwen/Qwen3-32B}"
: "${JUDGE_TP:=4}"
: "${JUDGE_PORT:=8000}"
: "${EMB_MODEL:=Qwen/Qwen3-Embedding-8B}"
: "${EMB_PORT:=8004}"

echo "Starting judge server: $JUDGE_MODEL (TP=$JUDGE_TP, port=$JUDGE_PORT)"
uvx vllm serve "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --tensor-parallel-size "$JUDGE_TP" \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256 &

echo "Starting embedding server: $EMB_MODEL (port=$EMB_PORT)"
uvx vllm serve "$EMB_MODEL" \
    --port "$EMB_PORT" \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 &

wait
