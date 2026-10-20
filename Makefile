# tau-telecom-sft — every target is a thin wrapper over `uv run tausft …`.
# M0, M1 and M2's targets are live; the EC2 targets (§9) join at M3 and M4.
.DEFAULT_GOAL := help
MLFLOW_TRACKING_URI ?= http://localhost:5000
PLATFORM := ../nmp-central-ai

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## submodule at the pin + uv sync (with the trainer and the Mac's MLX) + the viewer's npm ci
	git submodule update --init vendor/tau2-bench
	uv sync --all-extras
	cd frontend && npm ci

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

# ── M1: dataset v1 ──────────────────────────────────────────────────────────────────────────
download: ## fetch the board's telecom files S1 and S2 (128 MB) into raw/board/, once
	uv run tausft download

sources-check: ## S1, S2 and a tau2-loop telecom run load with one loader and have one shape
	uv run tausft sources-check

data: download platform-status ## build dataset v1 (agent view, F1-F10, rows, card) and log it
	uv run tausft build --version v1

# ── M2: the Mac proof of concept (Qwen3-0.6B; its scores are not results) ───────────────────
mask: ## print one row's loss mask as the trainer applies it
	uv run tausft mask --config configs/smoke.yaml

train-smoke: platform-status ## LoRA SFT on 16 of v1's rows on the Mac (MPS), logged to tau-telecom-sft/train
	uv run tausft train --config configs/smoke.yaml

# ── the viewer (plan §10): API :8085, UI :5177, 127.0.0.1 only ───────────────────────────────
VIEWER_PORT ?= 8085
dev: ## the viewer API on 127.0.0.1:8085, reloading (UI: cd frontend && npm run dev -> :5177)
	uv run python -m tau_telecom_sft.serving --port $(VIEWER_PORT) --reload

viewer: ## build the frontend and serve it with the API on 127.0.0.1:8085
	cd frontend && npm run build
	uv run python -m tau_telecom_sft.serving --port $(VIEWER_PORT)

ui-lint: ## frontend typecheck + design lint + vitest
	cd frontend && npm run typecheck && npm run lint:design && npm test

api-fixtures: ## rewrite tests/fixtures/runs/ and frontend/src/test/api.json
	uv run python tests/fixtures/make_run_fixtures.py && uv run python tests/fixtures/make_api_fixtures.py

fixtures: ## rewrite tests/fixtures/board_mini.json (only when the fixture script changes)
	uv run python tests/fixtures/make_fixtures.py

modal-serve: ## M3, M5: deploy the Student (vLLM, one Modal L4); it scales to zero 15 min after its last request
	uv run modal deploy -m tau_telecom_sft.cloud.modal_app

serve-modal: ## the recording proxy on :8020 in front of the Student on Modal (NAME=student-qwen3-4b)
	uv run tausft serve --remote $(or $(NAME),student-qwen3-4b)

modal-stop: ## stop the Modal app now; its Volumes (weights, runs) stay
	uv run modal app stop tau-telecom-sft --yes

modal-data: leak-check ## M4: dataset v1's train.jsonl and stats.json to the data Volume, once (no test task: leak-check)
	-uv run modal volume create tau-telecom-sft-data
	uv run modal volume put tau-telecom-sft-data data/sft/v1/train.jsonl v1/train.jsonl
	uv run modal volume put tau-telecom-sft-data data/sft/v1/stats.json v1/stats.json

modal-train: ## M4: LoRA SFT on one Modal L4, detached; MAX_STEPS=1 LONGEST=16 is the measured step
	uv run modal run --detach -m tau_telecom_sft.cloud.modal_app::train_job \
	  --config $(or $(CONFIG),configs/sft_a.yaml) \
	  $(if $(MAX_STEPS),--max-steps $(MAX_STEPS)) $(if $(LONGEST),--longest $(LONGEST))

sync-runs: platform-status ## M4: a training run from the runs Volume into runs/, logged to MLflow (RUN=<id>, WATCH=60)
	uv run tausft sync-runs $(RUN) $(if $(WATCH),--watch $(WATCH))

test: ## pytest, offline: split, leak, agent view, filters, converter, card
	uv run pytest

lint: ## ruff + mypy strict, and the frontend's checks when it is installed
	uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy
	@if [ -d frontend/node_modules ]; then $(MAKE) ui-lint; fi

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

check: split-check leak-check test lint ## everything M0 promises, offline

.PHONY: help setup platform-up platform-status smoke split-check leak-check fixtures test lint fmt check \
	download sources-check data mask train-smoke dev viewer ui-lint api-fixtures \
	modal-serve serve-modal modal-stop
