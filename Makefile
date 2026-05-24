PYTHON = python3
PIP = pip3
PROJECT_NAME = DecomposeRL
VENV_NAME = .venv
PYTHON_VENV = $(VENV_NAME)/bin/python
PIP_VENV = $(VENV_NAME)/bin/pip

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@echo "=================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Set up the development environment
	@echo "Setting up development environment..."
	uv sync

.PHONY: format
format: ## Format code
	@echo "Formatting code..."
	uvx ruff format

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

.PHONY: yolo
yolo: ## Stage all, squash into one commit, and force push (destroys remote history)
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	MSG=$${m:-"update"}; \
	git add -A && \
	git commit -m "$$MSG" --allow-empty && \
	git reset --soft $$(git rev-list --max-parents=0 HEAD) && \
	git commit --amend -m "$$MSG" && \
	git push --force origin "$$BRANCH"
