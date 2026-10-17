# tau-telecom-sft — every target is a thin wrapper over `uv run tausft …`.
# M0's targets are live; the rest of the plan's targets (§9) join at the milestone that needs them.
.DEFAULT_GOAL := help
MLFLOW_TRACKING_URI ?= http://localhost:5000
PLATFORM := ../nmp-central-ai

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## submodule at the pin + uv sync
	git submodule update --init vendor/tau2-bench
	uv sync

platform-up: ## start the central MLflow (nmp-central-ai on :5000)
	$(MAKE) -C $(PLATFORM) up

platform-status: ## preflight: the central MLflow must answer /health
	@curl -fsS $(MLFLOW_TRACKING_URI)/health >/dev/null || (echo "central MLflow down at $(MLFLOW_TRACKING_URI): run make platform-up"; exit 1)

smoke: platform-status ## zero-cost: build a dataset from the fixtures, log it to tau-telecom-sft/data
	uv run tausft smoke

split-check: ## the split's sha256 matches the manifest and tau2-loop's split v3
	uv run tausft split-check

leak-check: ## no test task, or test twin, in data/sft/ or configs/
	uv run tausft leak-check

fixtures: ## rewrite tests/fixtures/board_mini.json (only when the fixture script changes)
	uv run python tests/fixtures/make_fixtures.py

test: ## pytest, offline: split, leak, agent view, filters, converter, card
	uv run pytest

lint: ## ruff + mypy strict
	uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

check: split-check leak-check test lint ## everything M0 promises, offline

.PHONY: help setup platform-up platform-status smoke split-check leak-check fixtures test lint fmt check
