# AGENTS.md — tau-telecom-sft

The source of truth for how this project is built and why. `CLAUDE.md` points here. The plan
is `.lavish/s00_tau-airline-sft-plan.html` (never gitignored). Its filename keeps "airline"
from its first draft, but it plans telecom. It scopes use case UC6 of
`ai-engineer-fit/.lavish/s63_fine-tuning-playbook.html` §9, and the same page is s64 there.

## 1 · What it is

This repo repeats the SFT half of Kai Wei Tan's TMLS talk ("Fine-Tuning LLMs for Real-World
Tool Calling: Lessons from tau2-bench") on the same domain, tau2 telecom. The talk took
Qwen3-30B from 0% to 75% pass^3. Here, an open student is fine-tuned with LoRA on *clean* passing
conversations from stronger open models. It is then scored in tau2-loop on that repo's telecom
split v3: the same 40 test tasks, the same Haiku 5.5 customer and the same gate as the Claude
telecom v2 agent (24/40). RL is out of scope.

Steps: sources (the board's GLM-5 and Qwen3.5-397B conversations) → the agent's view → F1 to
F10 → chat-and-tools JSONL → LoRA SFT in TRL (Qwen3-4B on an L4, then Qwen3-30B-A3B on an
80 GB A100) → vLLM on the same GPU → tau2-loop plays the 40 test tasks × 4 trials from the Mac.

## 2 · Rules

1. **Test task ids, and any test task's exact set of root issues, never enter `data/sft/`,
   `configs/` or anything the trainer reads.** `tests/test_no_leak.py` enforces it and
   `make leak-check` scans. A twin is the same root issues under any persona or symptom.
2. **Never train on Claude's output.** Agent turns come from open-weights models only.
   tau2-loop's Claude runs (S5) may be read for comparison, never trained on (Consumer Terms).
3. **Never edit `vendor/tau2-bench`, and never change tau2-loop from this repo.** The three
   tau2-loop changes M2 needs (D5) are made in tau2-loop, by its own rules.
4. **Immutable data.** `raw/` never changes. Dataset versions (`data/sft/vN/`) and run folders
   are append-only: a new rule means a new version.
5. **The split is a byte copy.** `data/splits/telecom.json` is tau2-loop's split v3, and
   `data/splits/manifest.json` records its sha256. Never edit it here.
6. **Keys:** `OPENROUTER_API_KEY` and `HF_TOKEN` come from `~/.env` on the Mac and from the
   cloud's secret store elsewhere. Never put them in the image or the repo.
7. **Tracking:** the central MLflow only, with the tags `project`, `git_sha`, `env` and
   `billing`. Experiments are referenced by name, never by id, and there is no local store.
8. **Paid runs:** a teacher batch or a GPU job only after one measured conversation or step,
   and only when the person asks for it.
9. Never add `.lavish/` to `.gitignore`.

## 3 · Decisions, and the reasons

The plan's §12 decisions:

| | Decision | Status | Choice |
|---|---|---|---|
| D1 | Split | settled 17 Oct | tau2-loop's telecom split v3: 74 train, 40 test, seed 300. All 74 are trained on, none held out for dev; the epoch count is fixed in advance. |
| D2 | Teacher data | settled 16 Oct | The board files S1 (GLM-5) and S2 (Qwen3.5-397B), 4 trials each: 592 conversations on the 74 tasks. GLM-5.1 runs (S3) only if M7's trigger fires. |
| D3 | Student | open; plan default | Qwen3-4B-Instruct-2507 on one L4, then Qwen3-30B-A3B (the talk's model) on one 80 GB A100 |
| D4 | Where training runs | open; plan default | EC2 g6.2xlarge (L4) for the 4B; RunPod 80 GB A100 for the 30B |
| D5 | Where the student is scored | open; plan default | tau2-loop, with three changes made there |
| D6 | Which tasks are trained on | open; plan default | The 74 training tasks only (`allow_pool=False`) |
| D7 | Where the student is served | open; plan default | vLLM on the training GPU, reached through a port forward |

Choices made while building M0:

| Decision | Choice | Why |
|---|---|---|
| Agent view | tau2's own `is_valid_agent_history_message`, with each `MultiToolMessage` split into its results | the agent never sees the customer's 30 phone tools; one rule, tau2's, not a copy |
| Row format | `messages` + `tools` + `meta`. Tool-call arguments are JSON strings (OpenAI's shape), and `content` is `""` on a tool-call turn | the shape tau2 sends at serving time; Qwen's template takes either a string or a dict |
| System prompt | tau2's stock `SYSTEM_PROMPT` with `get_policy()` at the pin. The card records its sha256 and the tools' sha256 | the prompt-bytes tests in M2 (Mac) and M3 (vLLM) compare the served bytes with these |
| F6 median | over the task's F1 passes, across all sources, computed before any rule runs | stable whatever order the rules run in |
| F9 order | keeps the first three in (source, task, trial, simulation id) order | deterministic, so a rebuild gives the same rows |
| F10 counter | characters ÷ 4 in the smoke build only. M1 counts with the student's tokenizer and chat template | the smoke must not download a tokenizer |
| Dropped test tasks | written as `WITHHELD` in `drops.jsonl` | every file in a build folder is a trainer input, so none may name a test task |
| Smoke | 17 handwritten fixture conversations, built with tau2's own models (`tests/fixtures/make_fixtures.py`). Each is a clean pass or breaks exactly one rule. | zero cost, and every rule is exercised. F8 can't fire while D6 is off, so a unit test covers it. |
| `websockets` dependency | declared here | tau2's data model imports it at the pin, though tau2 lists it only in its `voice` extra |
| House UI hue | none yet: registered without a `ui` block | with P9 at 15, no hue is left 30° from the ten others (D28). The only artifact is the plan, which keeps its inline CSS (D30). |
| The greeting | `Hi! How can I help you today?` stays in each row as a trained assistant turn | tau2's orchestrator writes it, so the student never has to. It's harmless, and M2 can mask it if the loss table shows it matters. |

## 4 · Layout

What exists after M0, and where later milestones add to it:

```
AGENTS.md  CLAUDE.md  README.md      rules · pointer · status and results tables
ai_specs/s00_project_plan.md         the milestone checklist (tick with date + evidence)
skills/  opensrc/                    from ai-project-template (opensrc/sources.json starts empty)
vendor/tau2-bench/                   submodule at 2174a60, the same pin as tau2-loop; never edited
data/splits/telecom.json             tau2-loop's split v3, byte for byte
data/splits/manifest.json            its sha256, source commit and copy date
data/sft/vN/                         M1: card.md, stats.json, drops.jsonl committed; train.jsonl in S3
src/tau_telecom_sft/
  config.py                          paths, pins, experiment names, settings (the only os.environ reader)
  split.py                           load, sha256 check, task-id parsing, test root-issue sets
  leak.py                            scan trainer inputs for test ids and test twins
  sources/results.py                 one loader: tau2's Results model (board files, tau2-loop runs)
  build/agent_view.py                tau2's agent-history rule
  build/filters.py                   F1 to F10, pure functions, first failure charged
  build/convert.py                   agent view → chat + tools row; stock prompt; prompt fingerprint
  build/dataset.py                   one build: train.jsonl, card.md, stats.json, drops.jsonl
  build/card.py                      the dataset card
  track.py                           central MLflow: preflight, required tags, log_build
  cli.py                             `tausft`: smoke, split-check, leak-check
tests/                               split, no-leak, build (agent view, filters, row, card)
tests/fixtures/                      board_mini.json and the script that writes it
```

Still to come, from the plan's §9 tree: `sources/board.py` (M1), `train/sft.py`, `configs/` and
`serve/mlx.py` (M2, the Mac proof of concept), `serve/vllm.py` (M3), `score/` (M2, M3), the
`Dockerfile` and `infra/terraform/` (M4), `runs/` (M2 on),
and `sources/teacher.py` (M7, only if needed).

## 5 · Commands

```
make setup          submodule at the pin + uv sync
make smoke          zero-cost dataset build from the fixtures, logged to tau-telecom-sft/data
make split-check    sha256 against the manifest and tau2-loop's split v3
make leak-check     no test task or twin in data/sft/ or configs/
make test · lint    offline: pytest; ruff + mypy strict
make check          split-check + leak-check + test + lint
```

Platform: `make -C ../nmp-central-ai check ARGS="--only P10"` runs `make smoke` and confirms
the run reached `tau-telecom-sft/data`.

## 6 · Tracking

| Experiment | One run per | Kinds |
|---|---|---|
| `tau-telecom-sft/data` | dataset build | `smoke`, `build` |
| `tau-telecom-sft/train` | training job (M2 on) | |
| `tau-telecom-sft/eval` | scored model (M2 on) | |

The registered model is `tau-telecom-sft-student` (M4 on). Every run carries `project`,
`git_sha`, `env` and `billing`. A build run also carries the dataset's `train_sha256`, the
split's sha256, and the system prompt's and tools' sha256.
