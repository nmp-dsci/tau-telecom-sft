# CLAUDE.md — tau-telecom-sft

> Read [`AGENTS.md`](./AGENTS.md) first: what this is, the rules, the decisions, the layout.
> This file is a pointer plus the rules that bite.

## Quick reference

- `make smoke` · `make split-check` · `make leak-check` · `make check`
- `uv run pytest` · `make lint`
- MLflow: central (`make platform-up` → `make -C ../nmp-central-ai up`) → http://localhost:5000
- The plan: `.lavish/s00_tau-airline-sft-plan.html` (telecom, despite the name)

## Rules

- **No test task, and no test task's twin, in `data/sft/`, `configs/` or anything the trainer
  reads.** A twin has a test task's exact set of root issues under any persona.
- **Never train on Claude's output.** Agent turns come from open-weights models only.
- **Never edit `vendor/tau2-bench` or `data/splits/telecom.json`, and never change tau2-loop
  from here.**
- **Paid runs** (teacher batches, GPU jobs) only after one measured conversation or step, and
  only when asked.
- Central MLflow only, experiments by name; keys from `~/.env`; never add `.lavish/` to
  `.gitignore`.
