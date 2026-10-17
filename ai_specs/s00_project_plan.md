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
  - [x] Registered in nmp-central-ai as P10; smoke check added
  - Done when: the platform's smoke check passes, the split's sha matches tau2-loop's split v3,
    and the leak test runs. Evidence: README status · git
- [ ] **M1 · Build dataset v1** (Mac). Download S1 and S2 (128 MB) and keep the 74 training
  tasks. Strip the customer's phone calls, run F1 to F10 and convert. Measure token lengths with
  the student's tokenizer and set the F10 limit. Load S1, S2 and one tau2-loop telecom run with
  the same loader and test that they match. Read 10 kept and 10 dropped conversations.
  - Done when: the card's counts add up, the leak test passes, and the person has signed off on
    the sample. Evidence: `data/sft/v1/card.md` · MLflow data
- [ ] **M2 · Serve the untrained 4B and record the before row** (EC2 L4 · tau2-loop). Start
  vLLM on the L4 with vLLM's own image, make the three tau2-loop changes, and play the 40 test
  tasks × 4 trials.
  - Done when: R0 is filled in, tool calls come back as `tool_calls`, and the prompt-bytes test
    passes. Evidence: MLflow eval · tau2-loop run ids
- [ ] **M3 · Smoke-train on the Mac.** 5 steps on Qwen3-0.6B at 2K tokens; print the loss mask
  for one example.
  - Done when: the tokens with loss are exactly the agent's turns, with nothing from the
    customer or the tools. Evidence: a test asserts the mask
- [ ] **M4 · SFT the 4B on v1** (EC2 L4). `configs/sft_a.yaml` on all 74 training tasks for a
  fixed 3 epochs. The final checkpoint is the model, and earlier epochs are kept for the record.
  Fix the success criteria and the epoch count before the run starts.
  - Done when: the final adapter is in S3, with its config, data hash and image tag logged.
    Evidence: MLflow train · runs/
- [ ] **M5 · Score it on test, once** (EC2 L4 · tau2-loop). 40 tasks × 4 trials, with the sign
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
  Sonnet. R5: replay Claude v2 on the 40 test tasks × 4 trials. A Lavish results page set
  against the talk, and the adapters on the HF Hub, kept private.
  - Done when: published, with every number traced to a run id. Evidence: README · .lavish/
