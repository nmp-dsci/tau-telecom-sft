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
| D4 | Where training runs | settled 19 Oct (s02) | Modal, one L4, for the 4B (M4); the 30B (M6) decided at M6. EC2 was the plan's default; s02 is the comparison. |
| D5 | Where the student is scored | open; plan default | tau2-loop, with three changes made there |
| D6 | Which tasks are trained on | open; plan default | The 74 training tasks only (`allow_pool=False`) |
| D7 | Where the student is served | settled 19 Oct (s02) | vLLM on one Modal L4 (`cloud/modal_app.py`, the `Student` server), behind Modal's proxy token; the recording proxy on the Mac forwards to it |
| D8 | The viewer's hue | open; plan default | lower the platform's 30° separation to 25° and take 228 (needs a change in nmp-central-ai) |
| D9 | Who can open the viewer | open; plan default | the Mac only (127.0.0.1), built to the portfolio site's contract (nmp-central-ai s06) so it can join later |
| D10 | Where sign-offs live | open; plan default | `data/sft/vN/review.jsonl`, append-only and committed |

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

Choices made while building M1 and M2 (17 Oct 2026):

| Decision | Choice | Why |
|---|---|---|
| Board files | S1 and S2 fetched once from `sierra-tau-bench-public` into `raw/board/`, size and sha256 pinned in `sources/board.py`; a file whose bytes differ is refused | raw/ never changes (rule 4) |
| One loader | S1, S2 and tau2-loop's `20261016T124631Z_telecom_v2_train` all load with tau2's `Results` at the pin (`tausft sources-check`); tau2-loop's file also has `simulation_index`, added to tau2 after February, which the loader does not need | §4's "one format pays off" |
| F10 | the student's tokenizer and chat template (Qwen3-4B-Instruct-2507), limit kept at 24,576: it drops none. Kept rows: median 10,054 tokens, max 16,869; the prompt and tools alone 7,134 | the board's teachers are shorter than tau2-loop's Haiku runs; the L4 only has to fit 17K |
| Text beside calls | an assistant turn with both text and tool calls (955 in S1, 249 in S2) is kept as tau2 stored it | it is the history the teacher's next turn saw; the stock prompt's "not both" is a request, not a rule tau2 enforces |
| Review sample | 10 kept, 10 dropped, seed 300, drops drawn one rule at a time, committed as `review_sample.json`; a test task's drop is never drawn | every rule is read at least once; the person signs off in the viewer |
| Smoke tokenizer | Qwen3-0.6B's weights with Qwen3-4B-Instruct-2507's tokenizer and template (identical vocabulary) | the Mac trains, masks and serves the text format M4 will |
| The mask | TRL's own training template (`qwen3_instruct_2507_training.jinja`); the check reads the trainer's `labels` back for row 0 before step 1. Trained: each agent turn's text and calls, its `<|im_end|>` and the newline after it | the check runs on what the loss sees, not on a re-rendering |
| Mac attention | `train/attention.py`: SDPA in 1,024-query blocks with checkpointing (`model_init.attn_implementation: chunked_sdpa`, smoke only) | MPS's SDPA builds the whole table: 11 GB at 8K tokens, which swapped the 32 GB Mac at 33 GB |
| Smoke size | 16 rows that fit 12K whole, one step each, lr 2e-4 | one measured step (2 rows) took 272 s; 16 steps is about half an hour |
| Mac serving | merge (PEFT) → `mlx_lm.convert` bf16 → `mlx_lm.server` on :8120 behind `serve/proxy.py` on :8020 | :8000 is held by another app on this Mac; the proxy records hashes of every served prompt and tool set, and whether calls came back structured |
| Students in tau2-loop | versions `agents/telecom/sN/` (s1 untrained 0.6B, s2 smoke-tuned 0.6B) on branch `feat/open-student`: `core.STUDENTS`, `AgentConfig` sampling, `factory.student_route`, no clock note | not `vN`, so no loop, optimiser or history reads them; every Claude version's bytes and route are unchanged |

## 4 · Layout

What exists after M2, and where later milestones add to it:

```
AGENTS.md  CLAUDE.md  README.md      rules · pointer · status and results tables
ai_specs/s00_project_plan.md         the milestone checklist (tick with date + evidence)
docs/run_folders.md                  the run-folder contract the trainer, scorer and viewer share
skills/  opensrc/                    from ai-project-template (opensrc/sources.json starts empty)
vendor/tau2-bench/                   submodule at 2174a60, the same pin as tau2-loop; never edited
data/splits/telecom.json             tau2-loop's split v3, byte for byte (+ manifest.json)
data/sft/v1/                         card.md, stats.json, drops.jsonl, review_sample.json committed;
                                     review.jsonl (sign-offs) committed; train.jsonl git-ignored
configs/smoke.yaml · sft_a.yaml      M2's Mac run; M4's 4B run (same script)
raw/board/                           S1 and S2 as downloaded (git-ignored, immutable)
runs/<id>/                           train: run.json, metrics.jsonl, mask.json (+ adapter/, ignored)
                                     eval: run.json, served.jsonl, tau2_loop/ (a copy, read only)
outputs/                             MLX weights and the proxy's logs (git-ignored)
src/tau_telecom_sft/
  config.py                          paths, pins, experiment names, settings (the only os.environ reader)
  split.py · leak.py                 the split and its checks; the leak scan
  sources/results.py · board.py      one loader (tau2's Results); the board files, pinned
  sources/compare.py                 S1, S2 and a tau2-loop run have one shape
  build/                             agent_view · filters (F1–F10) · convert · tokens (F10's counter)
                                     · dataset · card · review (the sample and the sign-off log)
  train/sft.py · mask.py             TRL LoRA SFT; the loss-mask check before step 1
  train/callbacks.py · attention.py  steps to the run folder and MLflow; the Mac's block attention
  serve/mlx.py · proxy.py            merge + MLX convert + mlx_lm.server; the recording proxy
  serve/remote.py                    the Student on Modal: its served names, its record, its URL
  cloud/modal_app.py                 Modal: vLLM on one L4 (M3, M5); the trainer joins at M4
  score/pull.py                      a tau2-loop run → runs/<id>/ with pass^k, checks, sign test
  serving/                           the viewer's API (FastAPI): app · store · views
  track.py                           central MLflow: data, train and eval runs
  cli.py                             `tausft`
frontend/                            the viewer (React 18, Vite, TS, react-router 7, house UI)
tests/                               split, leak, build, train, score, api; fixtures/ (board_mini, runs)
```

Still to come: the Modal trainer and its Volumes' `make modal-data` and `make sync-runs` (M4),
and `sources/teacher.py` (M7, only if needed). EC2's Dockerfile and Terraform are not needed (s02).
`.claude/skills/modal/` is Modal's own agent skill (v1.6.2, the `cloud` extra's pin).

## 5 · Commands

```
make setup          submodule at the pin + uv sync --all-extras + npm ci
make smoke          zero-cost dataset build from the fixtures, logged to tau-telecom-sft/data
make split-check    sha256 against the manifest and tau2-loop's split v3
make leak-check     no test task or twin in data/sft/ or configs/
make test · lint    offline: pytest; ruff + mypy strict + the frontend's checks
make check          split-check + leak-check + test + lint
make download       S1 and S2 into raw/board/, once          (M1)
make sources-check  one loader, one shape                     (M1)
make data           build dataset v1 and log it               (M1)
make mask           one row's loss mask, as the trainer sees it (M2)
make train-smoke    LoRA SFT on 16 rows on the Mac            (M2)
uv run tausft serve-prepare --model base|<train run>          (M2) merge + MLX bf16
uv run tausft serve --model base|<train run>                  (M2) mlx_lm.server behind the proxy, :8020
uv run tausft serve-check                                     (M2) a served call comes back as tool_calls
uv run tausft score-pull --loop-run <tau2-loop run> --model … --label … [--vs <eval run>]   (M2)
make modal-serve    deploy the Student: vLLM on one Modal L4  (M3, M5) scales to zero after 15 min
make serve-modal    the proxy on :8020 in front of it (NAME=student-qwen3-4b) (M3, M5)
uv run tausft serve-check --name student-qwen3-4b             (M3) vLLM returns tool_calls
uv run tausft score-pull --loop-run … --served-name student-qwen3-4b --label … --purpose result --results-row R0
make modal-stop     stop the Modal app now; the Volumes stay
make dev · viewer   the viewer on 127.0.0.1:8085 (dev UI :5177)
```

Scoring runs in tau2-loop (branch `feat/open-student` until merged):
`STUDENT_URL=http://127.0.0.1:8020 make eval DOMAIN=telecom AGENT=s3 SPLIT=test TRIALS=4`.
`s3` is the untrained 4B (M3); its tuned twin is added at M5 under the adapter's served name.

Platform: `make -C ../nmp-central-ai check ARGS="--only P10"` runs `make smoke` and confirms
the run reached `tau-telecom-sft/data`.

## 6 · Tracking

| Experiment | One run per | Kinds |
|---|---|---|
| `tau-telecom-sft/data` | dataset build | `smoke`, `build` |
| `tau-telecom-sft/train` | training job (M2 on) | `smoke`, `result` |
| `tau-telecom-sft/eval` | scored model (M2 on) | `smoke`, `result` |

The registered model is `tau-telecom-sft-student` (M4 on). Every run carries `project`,
`git_sha`, `env` and `billing`. A build run also carries the dataset's `train_sha256`, the
split's sha256, and the system prompt's and tools' sha256. A train run logs every step (loss,
gradient norm, learning rate, tokens per second, memory) and the system's metrics, its config as
params, and its run folder and adapter as artifacts; `run_folder` names its folder. An eval run
logs pass^k, the prompt-bytes and tool-call checks and, against a base, fixed, broke and p; its
tags name the tau2-loop run, the adapter and the train run. The run folder is the record:
a step MLflow missed is still in `metrics.jsonl`.
