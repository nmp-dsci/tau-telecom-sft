# tau-telecom-sft

Fine-tune an open model on clean tau2 telecom conversations, and score it with the same harness
as the Claude agents. This repeats the SFT half of Kai Wei Tan's TMLS talk, which took Qwen3-30B
from 0% to 75% pass^3 on telecom, using tau2-loop's split v3: 74 training tasks and 40 test
tasks. The student faces the same 40 test tasks as the Claude telecom v2 agent (24/40).

- The plan: [`.lavish/s00_tau-airline-sft-plan.html`](.lavish/s00_tau-airline-sft-plan.html)
- The rules and decisions: [`AGENTS.md`](AGENTS.md)
- The checklist: [`ai_specs/s00_project_plan.md`](ai_specs/s00_project_plan.md)
- The build of M1, M2 and the viewer, step by step: [`.lavish/s01_m1-m2-viewer-walkthrough.html`](.lavish/s01_m1-m2-viewer-walkthrough.html)
- The run-folder contract: [`docs/run_folders.md`](docs/run_folders.md)

## Status

| Milestone | Status | Date | Evidence |
|---|---|---|---|
| M0 · Create the repo and connect it | done | 2026-10-17 | split v3 sha256 `45d48101…` matches tau2-loop's; leak test runs; `make -C ../nmp-central-ai check ARGS="--only P10"` PASS |
| M1 · Build dataset v1 | done | 2026-10-19 | 518 of 912 kept, all reward 1.0; 74/74 training tasks with 4–8 each; all 19 test root issues in training; `data/sft/v1/card.md`; train.jsonl sha256 `75718a64…`; MLflow data `08d798c5`; signed off on the summary in `review.jsonl` (sample not read, 0/20) |
| M2 · Prove the pipeline on the Mac, end to end (0.6B) | done (smoke) | 2026-10-17 | mask 11/11 agent turns; loss 1.47 → 0.87 in 16 steps; prompt bytes ok on 223 served requests; 0/3 → 0/3 on training tasks; own-tool calls 37% → 95%; tau2-loop branch `feat/open-student` |
| M3 · Serve the untrained 4B on Modal (R0) | done (1 trial) | 2026-10-20 | vLLM 0.21.0 on one Modal L4 behind the recording proxy; R0 6/40 test (`runs/20261019T223319Z_eval_r0-4b-test`), 8/74 train (`runs/20261020T004651Z_eval_r0-4b-train`); tool calls structured, prompt bytes ok on all 4,337 requests; 41% of tool calls named phone tools; Modal bill ≈ $3.50; walkthrough `.lavish/s03_m3-baseline-and-m5-swap.html` |
| M4 · SFT the 4B on v1 | done | 2026-10-21 | LoRA r16 on all linear layers, 3 epochs = 99 steps on one Modal L4, on v1.1 (v1's 518 conversations, agent text stripped of surrounding whitespace; train.jsonl sha256 `3e95fed9…`); loss 1.21 → 0.364 (epoch means 0.69, 0.45, 0.40); 15.7M tokens at ~715 tokens/s in 6.1 h, peak 16.0 GB, no preemption; mask ok on 518/518 rows; adapter sha256 `8956c198…` on the runs Volume and in `runs/20261020T204721Z_train_sft-a-4b`; MLflow train `640880b1`; Modal bill ≈ $6.84, plus ≈ $0.25 for the two measured steps; plan `.lavish/s04_m4-sft-plan.html` |
| M5 · Score it on test (R1) | | | |
| M6 · The 30B shadow (R2, R3) | | | |
| M7 · GLM-5.1 data, only if needed (R4) | | | |
| M8 · Write it up (R5) | | | |

## Results

Every row is the 40 test tasks of split v3 × 4 trials (160 conversations; for now 1 trial each,
the person's call on 2026-10-20 until the pipeline is validated), played by tau2-loop
with rules tau2_loop/2 and a Haiku 5.5 customer at high effort, and paired by task with a sign
test against its own untrained base. Counts sit beside percentages: one task is 2.5 points.

| Row | Model | pass^1 | pass^3 | vs base (p) | Cost / conv |
|---|---|---|---|---|---|
| R0 | Qwen3-4B-Instruct-2507, untrained | 6/40, 1 trial | needs trials | n/a | $0.03 (L4 list × wall) |
| R1 | Qwen3-4B + SFT on v1 | M5 | M5 | M5 | M5 |
| R2 | Qwen3-30B-A3B, untrained | M6 | M6 | n/a | M6 |
| R3 | Qwen3-30B-A3B + SFT on v1 | M6 | M6 | M6 | M6 |
| R4 | the student that missed + SFT on v2 | if M7 | if M7 | if M7 | if M7 |
| R5 | Claude telecom v2 (Haiku 5.5, high) | 24/40, 1 trial | M8 | n/a | M8 |
| ref | the talk: Qwen3 30B (a different 40; 17 shared) | n/a | 0 → 75 → 77.5% | n/a | n/a |
| ref | GPT-4.1 mini on the board (all 114, tau2 0.1.3) | 48.9% | 31.2% | n/a | n/a |

### The Mac proof of concept (M2): smoke runs, not results

Qwen3-0.6B, 3 training tasks × 1 trial, scored in tau2-loop with the Haiku 5.5 customer.

| Run | Model | Passed | Calls on its own 13 tools | Prompt bytes | Run folder |
|---|---|---|---|---|---|
| s1 | Qwen3-0.6B, untrained | 0/3 (all ended by tool errors) | 15 of 41 (36.6%) | ok, 57 requests | `runs/20261017T124445Z_eval_smoke-0.6b-base` |
| s2 | + LoRA on 16 v1 rows | 0/3 (2 tool errors, 1 step cap) | 104 of 110 (94.5%) | ok, 166 requests | `runs/20261017T124447Z_eval_smoke-0.6b-tuned` |

## Quick start

```bash
make setup                                  # submodule at 2174a60 + uv sync --all-extras + npm ci
make check                                  # split, leak, tests, lint — offline
make smoke                                  # fixture build logged to MLflow tau-telecom-sft/data
make -C ../nmp-central-ai check ARGS="--only P10"

make download && make data                  # M1: the board files → dataset v1 (+ MLflow data)
make viewer                                 # the viewer on http://127.0.0.1:8085 (sign off /data/v1/review)
make train-smoke                            # M2: LoRA SFT on the Mac (≈30 min) → runs/<id>/ + MLflow train
uv run tausft serve --model <train run>     # M2: MLX behind the recording proxy on :8020
```

The central MLflow must be up (`make platform-up`). There is no local store. Scoring runs in
tau2-loop (branch `feat/open-student`): `STUDENT_URL=http://127.0.0.1:8020 make eval
DOMAIN=telecom AGENT=s2 SPLIT=train TASKS="…"`, then `uv run tausft score-pull …`.
