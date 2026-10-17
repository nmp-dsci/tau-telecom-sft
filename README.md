# tau-telecom-sft

Fine-tune an open model on clean tau2 telecom conversations, and score it with the same harness
as the Claude agents. This repeats the SFT half of Kai Wei Tan's TMLS talk, which took Qwen3-30B
from 0% to 75% pass^3 on telecom, using tau2-loop's split v3: 74 training tasks and 40 test
tasks. The student faces the same 40 test tasks as the Claude telecom v2 agent (24/40).

- The plan: [`.lavish/s00_tau-airline-sft-plan.html`](.lavish/s00_tau-airline-sft-plan.html)
- The rules and decisions: [`AGENTS.md`](AGENTS.md)
- The checklist: [`ai_specs/s00_project_plan.md`](ai_specs/s00_project_plan.md)

## Status

| Milestone | Status | Date | Evidence |
|---|---|---|---|
| M0 · Create the repo and connect it | done | 2026-10-17 | split v3 sha256 `45d48101…` matches tau2-loop's; leak test runs; `make -C ../nmp-central-ai check ARGS="--only P10"` PASS |
| M1 · Build dataset v1 | next | | |
| M2 · Prove the pipeline on the Mac, end to end (0.6B) | | | |
| M3 · Serve the untrained 4B on EC2 (R0) | | | |
| M4 · SFT the 4B on v1 | | | |
| M5 · Score it on test (R1) | | | |
| M6 · The 30B shadow (R2, R3) | | | |
| M7 · GLM-5.1 data, only if needed (R4) | | | |
| M8 · Write it up (R5) | | | |

## Results

Every row is the 40 test tasks of split v3 × 4 trials (160 conversations), played by tau2-loop
with rules tau2_loop/2 and a Haiku 5.5 customer at high effort, and paired by task with a sign
test against its own untrained base. Counts sit beside percentages: one task is 2.5 points.

| Row | Model | pass^1 | pass^3 | vs base (p) | Cost / conv |
|---|---|---|---|---|---|
| R0 | Qwen3-4B-Instruct-2507, untrained | M3 | M3 | n/a | M3 |
| R1 | Qwen3-4B + SFT on v1 | M5 | M5 | M5 | M5 |
| R2 | Qwen3-30B-A3B, untrained | M6 | M6 | n/a | M6 |
| R3 | Qwen3-30B-A3B + SFT on v1 | M6 | M6 | M6 | M6 |
| R4 | the student that missed + SFT on v2 | if M7 | if M7 | if M7 | if M7 |
| R5 | Claude telecom v2 (Haiku 5.5, high) | 24/40, 1 trial | M8 | n/a | M8 |
| ref | the talk: Qwen3 30B (a different 40; 17 shared) | n/a | 0 → 75 → 77.5% | n/a | n/a |
| ref | GPT-4.1 mini on the board (all 114, tau2 0.1.3) | 48.9% | 31.2% | n/a | n/a |

## Quick start

```bash
make setup                                  # tau2-bench submodule at 2174a60 + uv sync
make check                                  # split, leak, tests, lint — offline
make smoke                                  # fixture build logged to MLflow tau-telecom-sft/data
make -C ../nmp-central-ai check ARGS="--only P10"
```

The central MLflow must be up (`make platform-up`). There is no local store.
