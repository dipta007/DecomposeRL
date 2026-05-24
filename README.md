# DecomposeRL: Traceable Claim Verification via RL-Trained Decomposition

[![Paper](https://img.shields.io/badge/arXiv-0000.00000-red)](https://arxiv.org/abs/0000.00000)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/dipta007/decomposeRL-7b)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/dipta007/DecomposeRL)
[![Collection](https://img.shields.io/badge/HuggingFace-Collection-blueviolet)](https://huggingface.co/collections/dipta007/decomposerl)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"DecomposeRL: Traceable Claim Verification via RL-Trained Decomposition"**

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
  - [Baselines](#baselines)
  - [Training](#training)
  - [Tiny Judge](#tiny-judge)
  - [Evaluation](#evaluation)
- [Reward Functions](#reward-functions)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

DecomposeRL trains language models to verify factual claims by decomposing them into sub-questions, answering each from evidence, and aggregating into a verdict. The model learns this decomposition strategy through GRPO (Group Relative Policy Optimization) with a multi-component reward system that evaluates question quality, answer correctness, and coverage.

Key contributions:
- **Iterative decomposition**: the model generates `<question>` / `<answer>` pairs in a thinking loop, then outputs a `<verification>` verdict
- **Multi-reward GRPO training**: 7 reward signals (format, verification, atomicity, diversity, answerability, correctness, coverage) guide the policy
- **Tiny judges**: distilled ModernBERT classifiers replace the LLM judge at training time for 100x faster reward computation
- **10 evaluation datasets**: FEVER, HoVer, WiCE, ClaimDecomp, ExFEVER, PubHealthFact, FoolMeTwice, PubMedClaim, CoverBench, LLMAggreFactt

## Installation

### Requirements
- Python 3.12
- CUDA 12.8+
- GPU with at least 24GB VRAM (for training)

### Setup

```bash
git clone https://github.com/dipta007/DecomposeRL.git
cd DecomposeRL

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make setup
```

## Dataset

The dataset is hosted on HuggingFace:

| Name | Description |
|------|-------------|
| [dipta007/DecomposeRL](https://huggingface.co/datasets/dipta007/DecomposeRL) | Train + 11 test splits across 10 verification datasets |

```python
from datasets import load_dataset

# Load training data
train = load_dataset("dipta007/DecomposeRL", "5000", split="train")

# Load a test split
test = load_dataset("dipta007/DecomposeRL", "5000", split="test_pubmedclaim")
```

Each example contains:
- `claim`: the factual claim to verify
- `evidence`: the evidence document
- `label`: `Supported` or `Refuted`
- `decomposed_questions`: ground-truth sub-questions (for training)

### Test Splits

| Split | Dataset | Samples |
|-------|---------|---------|
| `test_fever` | FEVER | - |
| `test_claimdecomp` | ClaimDecomp | - |
| `test_hover` | HoVer | - |
| `test_feverous` | FEVEROUS | - |
| `test_wice` | WiCE | - |
| `test_ex_fever` | ExFEVER | - |
| `test_pubhealthfact` | PubHealthFact | - |
| `test_fool_me_twice` | FoolMeTwice | - |
| `test_pubmedclaim` | PubMedClaim | - |
| `test_coverbench` | CoverBench | - |
| `test_llmaggrefact` | LLMAggreFactt | - |

## Models

| Model | Base | HuggingFace |
|-------|------|-------------|
| DecomposeRL-7B | Qwen2.5-7B-Instruct | [dipta007/decomposeRL-7b](https://huggingface.co/dipta007/decomposeRL-7b) |
| Tiny Judge (8 classifiers) | ModernBERT-large | [Collection](https://huggingface.co/collections/dipta007/decomposerl) |

All models and resources are collected in the [HuggingFace Collection](https://huggingface.co/collections/dipta007/decomposerl).

## Usage

### Baselines

Run prompted baselines (6 methods: Self-Ask, Decomposed Prompting, HiSS, FOLK, ProgramFC, Chen-Complex):

```bash
# Single method + dataset
PYTHONPATH=. uv run python src/baselines/run.py \
    --method self_ask --dataset pubmedclaim \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/baseline_7b_prompted

# Simple / Chain-of-Thought baselines
PYTHONPATH=. uv run python src/baselines/direct.py \
    -d pubmedclaim --mode cot \
    -o outputs/baseline_7b

# MiniCheck (NLI-based)
PYTHONPATH=. uv run python src/baselines/nli.py \
    -d pubmedclaim -o outputs/baseline_minicheck

# Run all baselines across all datasets
bash src/baselines/baseline.sh
```

API-backed baselines (OpenAI / Anthropic):

```bash
OPENAI_API_KEY=... PYTHONPATH=. uv run python src/baselines/run.py \
    --method hiss --backend api --provider openai \
    --dataset pubmedclaim --output_dir outputs/baseline_api
```

### Training

#### 1. Start the judge server (Qwen3-32B via vLLM)

```bash
uvx vllm serve Qwen/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92

uvx vllm serve Qwen/Qwen3-Embedding-8B \
    --port 8004 \
    --max-model-len 8192 \
    --trust-remote-code
```

#### 2. Run GRPO training

```bash
PYTHONPATH=. uv run python src/train/decomposerl/train.py \
    --model_name unsloth/Qwen2.5-7B-Instruct \
    --run_name my_run
```

Environment variables for reward ablations:

| Variable | Default | Description |
|----------|---------|-------------|
| `REWARD_BACKEND` | `llm-judge` | `llm-judge` or `tiny-judge` |
| `DIVERSITY_REWARD` | `mmr` | `mmr` or `vendi` |
| `NECESSITY_AGGREGATION` | `mean` | `mean` or `min` |
| `SUPERVISION_RATE` | `1.0` | fraction of claims using GT labels (0.0-1.0) |
| `COVERAGE_REWARD` | `1` | `0` to disable |
| `GOOD_NUM_Q_REWARD` | `1` | `0` to disable |

### Tiny Judge

The LLM judge (Qwen3-32B) is accurate but slow. We distill it into small ModernBERT-large classifiers ("tiny judges") — one per reward criterion — that run locally on a single GPU and are ~100x faster.

**Pre-trained tiny judges** are available on HuggingFace:

| Task | Model |
|------|-------|
| Atomicity (5 criteria) | [dipta007/atomicity-*-judge-balanced](https://huggingface.co/collections/dipta007/decomposerl) |
| Question Answerable | [dipta007/question-judge-balanced](https://huggingface.co/dipta007/question-judge-balanced) |
| Answer Correctness | [dipta007/answer-judge-balanced](https://huggingface.co/dipta007/answer-judge-balanced) |
| Coverage | [dipta007/coverage-judge-balanced](https://huggingface.co/dipta007/coverage-judge-balanced) |

**Use pre-trained tiny judges** for GRPO training (no judge server needed):

```bash
REWARD_BACKEND=tiny-judge bash scripts/train_grpo.sh --run_name my_run
```

**Train your own** from LLM judge cache:

```bash
# Train all tasks
bash scripts/train_tiny_judge.sh --task all

# Train a single task and push to HF Hub
bash scripts/train_tiny_judge.sh --task coverage --push
```

### Evaluation

Evaluate a trained checkpoint:

```bash
# Single dataset
PYTHONPATH=. uv run python src/test/test.py \
    -c outputs/my_run/checkpoint-100 \
    -d pubmedclaim

# All datasets
bash src/test/eval.sh outputs/my_run/checkpoint-100
```

## Reward Functions

| Reward | Description |
|--------|-------------|
| **Format** | Validates `<think>`, `<question>`, `<answer>`, `<verification>` tag structure |
| **Verification** | Binary match of predicted vs ground-truth label |
| **Atomicity** | 5-criterion checklist per question (is_question, single_focus, no_conjunctions, verifiable, grounded) |
| **Diversity** | MMR or Vendi Score over question embeddings |
| **Answerability** | Whether each question can be answered from the evidence |
| **Answer Correctness** | Whether each answer is factually correct given the evidence |
| **Coverage** | Whether the decomposed answers lead to the correct verdict |
| **Necessity** | Leave-one-out saliency: removing a question should change the verdict |

## Project Structure

```
DecomposeRL/
├── src/
│   ├── baselines/          # 8 baseline methods (direct, CoT, MiniCheck, 6 prompted)
│   ├── train/
│   │   ├── decomposerl/    # GRPO training + reward functions
│   │   └── tiny_judge/     # Distilled judge classifier training
│   ├── test/               # Checkpoint evaluation
│   └── data_process/       # Dataset curation pipeline
├── scripts/                # Shell scripts (train, eval)
├── Makefile
└── pyproject.toml
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{dipta2025decomposerl,
  title={DecomposeRL: Traceable Claim Verification via RL-Trained Decomposition},
  author={Shubhashis Roy Dipta and Ankur Padia and Francis Ferraro},
  year={2025},
  eprint={0000.00000},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/0000.00000},
}
```

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [TRL](https://github.com/huggingface/trl) for GRPO training
- [Qwen](https://github.com/QwenLM/Qwen) for base models

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/dipta007/DecomposeRL/issues).
