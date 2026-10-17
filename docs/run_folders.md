# Run folders: what a training job and a scored model leave behind

`runs/<run_id>/` is the record (plan §10, "Committed files are the record; MLflow is a link").
The viewer draws only what these files hold; MLflow indexes them. A folder is append-only:
once a run finishes, nothing in it changes. Big binaries (the adapter, merged weights, MLX
weights) are git-ignored; their sha256 is in `run.json`.

Run ids: `<UTC yyyymmddThhmmssZ>_<kind>_<label>`, e.g. `20261017T120000Z_train_smoke-0.6b`.

## A training job (`kind: train`)

```
runs/<id>/
  run.json        the job: config, data, where it ran, how long, cost, the adapter's hash
  metrics.jsonl   one line per logged optimiser step
  mask.json       the loss-mask check on one row, and that row rendered for the tokens lens
  adapter/        the LoRA adapter (git-ignored)
```

`run.json`

| key | meaning |
|---|---|
| `run_id`, `kind` (`train`), `label` | identity; `label` is e.g. `smoke-0.6b`, `sft-a-4b` |
| `purpose` | `smoke` (M2, not a result) or `result` |
| `status` | `running`, `finished`, `failed` |
| `created`, `finished`, `duration_s` | UTC ISO times; wall-clock seconds |
| `code_sha`, `image_tag` | the repo commit; the Docker image (null on the Mac) |
| `config_file`, `config` | the YAML the job read, and its parsed values (SFTConfig + LoRA) |
| `base_model`, `tokenizer` | Hugging Face ids |
| `dataset`, `train_sha256`, `n_rows`, `rows_selected` | the version, its hash, how many rows trained, and how they were chosen |
| `device`, `instance`, `cost_usd` | `mps` or `cuda`; e.g. `MacBook Pro M1 Pro 32 GB`, `g6.2xlarge`; dollars (0 on the Mac) |
| `steps`, `epochs`, `final_loss`, `peak_mem_gb`, `tokens_trained` | totals |
| `adapter` | `{path, sha256, bytes}` |
| `mask_check` | `{ok, row_sim_id, assistant_turns, trained_turns, problems: []}` |
| `mlflow_run_id`, `mlflow_experiment` | the link |

`metrics.jsonl`: `{step, epoch, loss, grad_norm, learning_rate, tokens, tokens_per_s, mem_gb, elapsed_s}`.

`mask.json`: `{row_sim_id, task_id, n_tokens, n_trained, segments: [{text, trained}], check}`.
Segments are the rendered row cut where the mask changes, so the tokens lens shows trained
text exactly as the trainer saw it.

## A scored model (`kind: eval`)

```
runs/<id>/
  run.json        what was played, by which model, and the scores
  served.jsonl    one line per model call the scorer made, recorded by serve/proxy.py
  tau2_loop/      a copy of the tau2-loop run folder: run.json, results.jsonl, tau2_results.json
```

`run.json`

| key | meaning |
|---|---|
| `run_id`, `kind` (`eval`), `label`, `purpose` | identity; `purpose` `smoke` for M2 |
| `results_row` | `R0`…`R5` for a `purpose: result` run; the overview fills that row from it |
| `model` | `{name, base_model, train_run (or null for an untrained base), adapter_sha256}` |
| `serving` | `{server, precision, endpoint, sampling: {temperature, top_p, top_k}, max_tokens}` |
| `tau2_loop_run`, `tau2_loop_version` | the run and agent version in tau2-loop |
| `split`, `task_ids`, `trials` | what was played (`task_ids` only for train-split smoke runs) |
| `scores` | `{n_tasks, trials, passed, pass_hat: {"1": …, "2": …}, by_task: {task: [0/1 per trial]}}` |
| `prompt_bytes` | `{ok, system_prompt_sha256, tools_sha256, expected_system_prompt_sha256, expected_tools_sha256, requests}` |
| `tool_calls` | `{ok, requests_with_tool_calls, text_tool_calls, errors, calls, agent_tool_calls, phone_tool_calls, invented_tool_calls, agent_tool_share, most_called}`: served calls came back structured, and which tools they named |
| `vs` | the base it is compared with: `{run_id, tasks, fixed, broke, same, p}`, paired by task, tau2-loop's one-sided sign test |
| `customer`, `sim_rules`, `tau2_loop_mlflow_run_id` | who played the customer, tau2-loop's rules, and its MLflow run |
| `mlflow_run_id`, `mlflow_experiment` | the link |

`served.jsonl`: `{t, system_prompt_sha256, tools_sha256, n_messages, finish_reason,
tool_calls, text_tool_call, prompt_tokens, completion_tokens, latency_s}`.
