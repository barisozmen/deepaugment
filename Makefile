# DeepAugment - Bayesian Optimization for Data Augmentation
# Minimal, elegant automation (Norvig style)

.DEFAULT_GOAL := help
SHELL := /bin/bash

##@ Core

install: ## Install dependencies and setup environment
	uv sync

dev: ## Install with dev dependencies
	uv sync --all-extras

test: ## Run tests
	uv run pytest tests/ -v

##@ Development

setup: ## Setup development environment (git hooks)
	@echo "Setting up git hooks..."
	@mkdir -p .git/hooks
	@echo '#!/bin/bash' > .git/hooks/pre-commit
	@echo 'uv run bin/bump_patch_version.py' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Git pre-commit hook installed"

version: ## Show current version
	@grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'

bump: ## Manually bump patch version
	uv run python bin/bump_patch_version.py

##@ Cleanup

clean: ## Remove generated files
	rm -rf .pytest_cache .ruff_cache __pycache__ .venv
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data: ## Remove experiment data
	rm -rf experiments/* reports/*

##@ Help

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: install dev test setup version bump clean clean-data help
