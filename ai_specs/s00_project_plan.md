# s00 — tau-telecom-sft: the milestone checklist

The plan of record is `.lavish/s00_tau-airline-sft-plan.html` §10. Tick a box when its "done
when" holds, and add the date and the evidence (a run id, a data hash or a commit).

Success criteria, fixed before M4: the floor is that the tuned student beats the untrained one
on the 40 test tasks (one-sided sign test on the tasks that changed, p < 0.05). The targets are
pass^3 ≥ 50% for the 4B and ≥ 70% for the 30B shadow (the talk: 75%). Always report counts:
each of the 40 tasks is worth 2.5 points.

- [x] **M0 · Create the repo and connect it** (Mac). 2026-10-17.
  - [x] Template files (ai_specs/, skills/, opensrc/), tau2-bench submodule at `2174a60`
  - [x] tau2-loop's split v3 copied; sha256 `45d481019e630fba8a5a1e0b55f3fc789ba3118858300e8688c37b6c2fc4f90a`
        (tau2-loop commit `cfa1d4f`)
  - [x] AGENTS.md and the leak test (test ids and test root-issue sets)
  - [x] Registered in nmp-central-ai as P10 (branch `feat/register-tau-telecom-sft`, commit
        `161d946`); experiments tau-telecom-sft/data, /train, /eval created by `make mlflow-init`
  - [x] Smoke check: `make -C ../nmp-central-ai check ARGS="--only P10"` PASS (smoke=0 runs=1);
        run `0f37ee02c2394942a29f052685d7f06e` in tau-telecom-sft/data at code `0096417`,
        17 fixture conversations, 8 kept
  - Done when: the platform's smoke check passes, the split's sha matches tau2-loop's split v3,
    and the leak test runs. Evidence: README status · git
  - Open after M0: the registry branch needs merging (PR) like P9's; a house-UI hue (none is left
    30° from the others); D3–D7 still stand at the plan's recommended options.
- [x] **M1 · Build dataset v1** (Mac). 2026-10-19. Download S1 and S2 (128 MB) and keep the 74 training
  tasks. Strip the customer's phone calls, run F1 to F10 and convert. Measure token lengths with
  the student's tokenizer and set the F10 limit. Load S1, S2 and one tau2-loop telecom run with
  the same loader and test that they match. Start the viewer's data side (plan §10): `/data`,
  the rows in the chat and raw lenses, and `/data/v1/review`. Read 10 kept and 10 dropped
  conversations there and sign off.
  - Done when: the card's counts add up, the leak test passes, and the person has signed off on
    the sample in the viewer. Evidence: `data/sft/v1/card.md` · `review.jsonl` · MLflow data
  - Built 2026-10-17, signed off 2026-10-19:
    - [x] S1, S2 downloaded and pinned (sha256 `18e41808…`, `4038e532…`); `make sources-check` passes on
          S1, S2 and tau2-loop's `20261016T124631Z_telecom_v2_train`
    - [x] v1: 518 of 912 kept, 74/74 training tasks; F1 70 · F2 298 · F3 1 · F4 7 · F5 4 · F6 6 · F9 8;
          counts add up; leak check passes; train.jsonl sha256 `75718a64…`; MLflow data `08d798c5`
    - [x] F10 with Qwen3-4B-Instruct-2507's tokenizer and template: median 10,054, max 16,869; 24,576
          drops none, so it stays
    - [x] The viewer's data side: `/data/v1` (+ coverage), rows in chat, tokens and raw, `/data/v1/review`
    - [x] Signed off 2026-10-19, in chat, on the pass and coverage summary instead of the 20-row read
          (`review.jsonl`, `unread: 20`): all 518 kept scored reward 1.0 on the board; all 74 training
          tasks keep 4 to 8 (Easy 25 tasks / 183, Hard 23 / 160, None 26 / 175); all 19 root issues in
          test appear in training. The sample stays at `/data/v1/review` and can still be read.
    - Not done, and offered: re-score the 518 at the pin `2174a60` (the board's tau2 commit is unknown)
- [x] **M2 · Prove the pipeline on the Mac, end to end** (Mac · tau2-loop). 2026-10-17. A proof of concept
  with Qwen3-0.6B before anything runs on EC2; its scores are not results. (Reordered on
  2026-10-17 at the person's request: this was M3's smoke-train, now extended to serve and score.)
  - Train: `make train-smoke` runs the same TRL code as the EC2 job (`train/sft.py`, LoRA,
    assistant-only loss) for a few steps on v1's rows at 12K tokens. The system prompt and tools
    take about 7.6K tokens (characters ÷ 4), so 2K would hold no agent turn. It prints the loss
    mask for one row.
  - Serve: merge the adapter, convert it to MLX in bf16, and serve it with `mlx_lm.server` on
    `localhost:8000` with tool calls parsed. llama.cpp's `llama-server --jinja` is the fallback.
  - Score: make the three tau2-loop changes; play 3 training tasks × 1 trial for the untrained
    and the tuned 0.6B, with the Haiku 5.5 customer. Training tasks only.
  - See it: the viewer's `/train` and `/eval` pages and the rows' tokens lens, fed by the smoke run.
  - Done when: the loss mask is exactly the agent's turns; a served tool call comes back as
    `tool_calls`; the prompt-bytes test passes; tau2-loop records both runs; the viewer shows the
    chain from the smoke dataset to both scored runs. Evidence: tests
    assert the mask and the prompt bytes · MLflow train and eval (kind=smoke) · tau2-loop run ids
  - [x] Train: `runs/20261017T120150Z_train_smoke-0.6b`, 16 rows × 1 step, loss 1.469 → 0.872, 29 min,
        21.2 GB peak; MLflow train `b52a4b02`; adapter sha256 `4666b94f…`. One measured step first
        (`…-step1`: 2 rows, 272 s). Mask check 11/11 agent turns, 1,530 of 10,081 tokens.
  - [x] Serve: PEFT merge → MLX bf16 → mlx_lm.server :8120 behind `serve/proxy.py` :8020 (:8000 is
        another app's); `tausft serve-check` got a structured `tool_calls` reply
  - [x] tau2-loop's three changes on branch `feat/open-student` (`d31e8db`, worktree
        `../tau2-loop-student`): `core.STUDENTS`, versions `s1`/`s2`, `factory.student_route`; its
        full suite passes. One measured conversation first (`20261017T121015Z_telecom_s1_custom`).
  - [x] Score: s1 `20261017T123132Z_telecom_s1_custom` 0/3, s2 `20261017T123152Z_telecom_s2_custom`
        0/3 (fixed 0, broke 0). Prompt bytes ok on 57 + 166 requests; 150 structured replies, 0
        written as text; calls on the agent's own tools 36.6% → 94.5%. MLflow eval `24fceb79`,
        `d307e9ff`. Not results: a 0.6B on 3 training tasks.
  - [x] The viewer shows the chain: `/`, `/train/<run>`, `/eval/<run>?vs=` (s01 has screenshots)
- [x] **M3 · Serve the untrained 4B on EC2 and record the before row** (EC2 L4 · tau2-loop).
  Done 2026-10-20 on a Modal L4 instead (s02), at 1 trial per task (the person's call): R0 6/40 test,
  8/74 train; walkthrough s03.
  Start vLLM on the L4 with vLLM's own image, point M2's tau2-loop student version at it through
  the port forward, and play the 40 test tasks × 4 trials.
  - Done when: R0 is filled in, tool calls come back as `tool_calls` from vLLM too, and the
    prompt-bytes test passes on the served bytes. Evidence: MLflow eval · tau2-loop run ids
- [x] **M4 · SFT the 4B on v1** (EC2 L4). Done 2026-10-21 on a Modal L4 instead (s02, s04), on
  v1.1, v1 with the agent's text stripped (the mask check caught v1's leading newlines): 99 steps,
  loss 1.21 → 0.364, adapter on the runs Volume, not S3, with its sha256 in run.json.
  `configs/sft_a.yaml` on all 74 training tasks for a
  fixed 3 epochs. The final checkpoint is the model, and earlier epochs are kept for the record.
  Fix the success criteria and the epoch count before the run starts.
  - Done when: the final adapter is in S3, with its config, data hash and image tag logged.
    Evidence: MLflow train · runs/
- [x] **M5 · Score it on test, once** (EC2 L4 · tau2-loop). Done 2026-10-21 on the Modal L4 at
  1 trial per task: R1 17/40 test against R0's 6/40, 13 fixed and 2 broke, sign test p = 0.0037;
  33/74 on the training tasks, in-sample. pass^3 waits for more trials.
  40 tasks × 4 trials, with the sign
  test against R0. Optionally play the 74 training tasks once, labelled in-sample.
  - Done when: R1 is filled in, with its counts. Evidence: MLflow eval · README table
- [ ] **M6 · Train the shadow: Qwen3-30B-A3B on v1** (A100 80 GB · tau2-loop). Rent one 80 GB
  A100 and run the same image. Score the untrained base (R2) and the tuned one (R3).
  - Done when: R2 and R3 are filled in, beside the talk's numbers. Evidence: MLflow · README
- [ ] **M7 · Only if needed: add the GLM-5.1 data (S3).** Trigger: R1 or R3 misses its target,
  and its failed test tasks share root issues that are thin in v1's card. Measure one
  conversation's cost first, then run up to about 822 with an open customer. Build v2,
  retrain and score (R4).
  - Done when: R4 is filled in with its cost report, or the README records that the trigger
    didn't fire. Evidence: `data/sft/v2/card.md` · MLflow
- [ ] **M8 · Write it up** (Mac). A model card, and cost per conversation against Haiku and
  Sonnet. R5: replay Claude v2 on the 40 test tasks × 4 trials. The viewer's `/method` page
  finished. A Lavish results page set against the talk, and the adapters on the HF Hub, kept private.
  - Done when: published, with every number traced to a run id. Evidence: README · .lavish/
