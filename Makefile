# Basic Makefile for DecomposeRL Project

# Variables
PYTHON = python3
PIP = pip3
PROJECT_NAME = DecomposeRL
VENV_NAME = .venv
PYTHON_VENV = $(VENV_NAME)/bin/python
PIP_VENV = $(VENV_NAME)/bin/pip

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@echo "=================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup targets
.PHONY: setup
setup: ## Set up the development environment
	@echo "Setting up development environment..."
	uv sync

.PHONY: format
format: ## Format code
	@echo "Formatting code..."
	uvx ruff format

# Cleanup targets
.PHONY: clean
clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true

.PHONY: clean-venv
clean-venv: ## Remove virtual environment
	@echo "Removing virtual environment..."
	rm -rf $(VENV_NAME)

.PHONY: clean-all
clean-all: clean clean-venv ## Clean everything including virtual environment

.PHONY: overleaf
overleaf: ## Pull the latest Overleaf submodule contents
	@echo "Updating Overleaf submodule..."
	git submodule update --remote --merge overleaf


.PHONY: plot
plot: ## Generate all analysis plots (outputs/analysis/)
	@echo "Generating analysis plots..."
	uv run -m decomposer.analysis.plot_checkpoints
	uv run -m decomposer.analysis.plot_avg
	uv run -m decomposer.analysis.plot_avg_weighted
	uv run -m decomposer.analysis.plot_micro
	# uv run -m decomposer.analysis.plot_coverbench_domains

.PHONY: dashboard
dashboard: ## Refresh strict-micro cache and launch the dashboard. Usage: make dashboard [PORT=8501]
	@echo "==> Pre-computing strict-micro aggregates (skipping entries already cached) ..."
	uv run python -m dashboard.precompute_aggregates --skip-existing
	@echo "==> Launching dashboard at http://localhost:$(or $(PORT),8501) ..."
	uv run streamlit run dashboard/app.py \
		--server.address 0.0.0.0 \
		--server.port $(or $(PORT),8501) \
		--browser.gatherUsageStats false

.PHONY: compare
compare: ## Compare versions (independent best per cell, with baselines). Usage: make compare V="61 62 63"
	@if [ -z "$(V)" ]; then \
		echo "Usage: make compare V=\"61 62 63\""; exit 1; \
	fi
	uv run -m decomposer.analysis.compare_versions --baseline $(V)

.PHONY: compare-joint
compare-joint: ## Compare versions (one checkpoint per version, with baselines). Usage: make compare-joint V="61 62 63"
	@if [ -z "$(V)" ]; then \
		echo "Usage: make compare-joint V=\"61 62 63\""; exit 1; \
	fi
	uv run -m decomposer.analysis.compare_versions --joint --baseline $(V)

.PHONY: eval-ood
eval-ood: ## Pick best checkpoint by micro-balanced-acc on 9 dev sets, then eval on coverbench+llmaggrefact. Usage: make eval-ood V="66" or V="61 62 63"
	@if [ -z "$(V)" ]; then \
		echo "Usage: make eval-ood V=\"66\" or V=\"61 62 63\""; exit 1; \
	fi
	PYTHONPATH=. uv run decomposer/eval/eval_best_checkpoint.py -v $(V)

.PHONY: yolo
yolo: ## Stage all, squash into one commit, and force push (destroys remote history)
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	MSG=$${m:-"update"}; \
	git add -A && \
	git commit -m "$$MSG" --allow-empty && \
	git reset --soft $$(git rev-list --max-parents=0 HEAD) && \
	git commit --amend -m "$$MSG" && \
	git push --force origin "$$BRANCH"

.PHONY: run-7b
run-7b: ## Run baseline experiments
	@echo "Running baseline experiments..."
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode iterative -d data/pubmedclaim/test_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode simple -d data/pubmedclaim/test_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode cot -d data/pubmedclaim/test_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode iterative -d data/pubmedclaim/test_2way.jsonl

	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode iterative -d data/coverbench/coverbench_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode simple -d data/coverbench/coverbench_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode cot -d data/coverbench/coverbench_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-7B-Instruct -o outputs/baseline_7b --prompt_mode iterative -d data/coverbench/coverbench_2way.jsonl

.PHONY: run-3b
run-3b: ## Run baseline experiments
	@echo "Running baseline experiments..."
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode iterative -d data/pubmedclaim/test_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode simple -d data/pubmedclaim/test_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode cot -d data/pubmedclaim/test_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode iterative -d data/pubmedclaim/test_2way.jsonl


	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode iterative -d data/coverbench/coverbench_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode simple -d data/coverbench/coverbench_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode cot -d data/coverbench/coverbench_2way.jsonl
	PYTHONPATH=. python decomposer/unsloth/baseline.py -m Qwen/Qwen2.5-3B-Instruct -o outputs/baseline_3b --prompt_mode iterative -d data/coverbench/coverbench_2way.jsonl
