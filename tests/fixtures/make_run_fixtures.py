"""Write tests/fixtures/runs/: one training run and two scored runs, in docs/run_folders.md's shape.

They feed the viewer's API tests offline. Everything is handwritten or copied from the
handwritten board_mini.json, and every scored task is a training task, so no fixture names a
test task. The numbers are made up for the tests: they are not results.

  <ts>_train_fixture-smoke   a finished smoke job: config, 6 logged steps, a mask check
  <ts>_eval_fixture-base     the untrained base on 3 training tasks × 1 trial: 1 passes
  <ts>_eval_fixture-tuned    the fixture job's adapter on the same 3: 2 pass (fixed 1, broke 0)

Run: uv run python tests/fixtures/make_run_fixtures.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tau_telecom_sft import config, split

HERE = Path(__file__).parent
OUT = HERE / "runs"
BOARD = HERE / "board_mini.json"
TRAIN = "20261017T000000Z_train_fixture-smoke"
BASE = "20261017T010000Z_eval_fixture-base"
TUNED = "20261017T020000Z_eval_fixture-tuned"
ADAPTER = "a" * 64
SYSTEM_SHA = "b" * 64
TOOLS_SHA = "c" * 64


def _write(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def train_run() -> None:
    d = OUT / TRAIN
    cfg = {
        "label": "fixture-smoke",
        "purpose": "smoke",
        "base_model": "Qwen/Qwen3-0.6B",
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset": "smoke",
        "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear"},
        "sft": {
            "max_length": 12288,
            "assistant_only_loss": True,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 2,
            "gradient_accumulation_steps": 2,
        },
    }
    _write(
        d / "run.json",
        {
            "run_id": TRAIN,
            "kind": "train",
            "label": "fixture-smoke",
            "purpose": "smoke",
            "status": "finished",
            "created": "2026-10-17T00:00:00Z",
            "finished": "2026-10-17T00:06:00Z",
            "duration_s": 360.0,
            "code_sha": "fixture",
            "image_tag": None,
            "config_file": "configs/smoke.yaml",
            "config": cfg,
            "base_model": "Qwen/Qwen3-0.6B",
            "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
            "dataset": "smoke",
            "train_sha256": "d" * 64,
            "n_rows": 8,
            "rows_selected": "8 of 8 rows",
            "device": "mps",
            "instance": "fixture Mac",
            "cost_usd": 0.0,
            "steps": 6,
            "epochs": 2,
            "final_loss": 0.61,
            "peak_mem_gb": 9.5,
            "tokens_trained": 2400,
            "adapter": {"path": f"runs/{TRAIN}/adapter", "sha256": ADAPTER, "bytes": 40_000_000},
            "mask_check": {
                "ok": True,
                "row_sim_id": "fixture-row",
                "assistant_turns": 2,
                "trained_turns": 2,
                "problems": [],
            },
            "mlflow_run_id": None,
            "mlflow_experiment": config.EXPERIMENT_TRAIN,
        },
    )
    losses = [1.9, 1.5, 1.2, 0.9, 0.75, 0.61]
    _jsonl(
        d / "metrics.jsonl",
        [
            {
                "step": i + 1,
                "epoch": round((i + 1) / 3, 3),
                "loss": loss,
                "grad_norm": 2.0 - i * 0.2,
                "learning_rate": 1e-4 * (1 - i / 6),
                "tokens": 400 * (i + 1),
                "tokens_per_s": 900.0 + i,
                "mem_gb": 9.0 + i * 0.1,
                "elapsed_s": 60.0 * (i + 1),
            }
            for i, loss in enumerate(losses)
        ],
    )
    _write(
        d / "mask.json",
        {
            "row_sim_id": "fixture-row",
            "task_id": "fixture task",
            "n_tokens": 30,
            "n_trained": 10,
            "segments": [
                {
                    "text": "<|im_start|>system\nYou are an agent.<|im_end|>\n<|im_start|>assistant\n",
                    "trained": False,
                },
                {"text": "Hi! How can I help you today?<|im_end|>\n", "trained": True},
                {
                    "text": "<|im_start|>user\nMy data is off.<|im_end|>\n<|im_start|>assistant\n",
                    "trained": False,
                },
                {"text": "Let me check your line.<|im_end|>\n", "trained": True},
            ],
            "check": {"ok": True, "assistant_turns": 2, "trained_turns": 2, "problems": []},
        },
    )


def eval_run(run_id: str, label: str, train_run: str | None, passes: dict[str, int]) -> None:
    board = json.loads(BOARD.read_text())
    train = split.current().train_set
    picked: dict[str, dict[str, Any]] = {}
    for sim in board["simulations"]:
        if sim["task_id"] in passes and sim["task_id"] in train and sim["task_id"] not in picked:
            picked[sim["task_id"]] = {**sim, "trial": 0}
    assert set(picked) == set(passes), (
        "every fixture task must be a training task in the board fixture"
    )
    d = OUT / run_id
    loop_run = f"{run_id.split('_')[0]}_telecom_student-0.6b_custom"
    tasks = sorted(passes)
    results = [
        {
            "task_id": t,
            "trial": 1,
            "reward": float(passes[t]),
            "correct": bool(passes[t]),
            "termination_reason": "user_stop",
            "n_agent_turns": 6,
            "n_tool_calls": 2,
            "n_tool_errors": 0,
            "duration_ms": 120_000,
            "error": None,
        }
        for t in tasks
    ]
    sims = []
    for t in tasks:
        s = dict(picked[t])
        s["reward_info"] = {**(s.get("reward_info") or {}), "reward": float(passes[t])}
        sims.append(s)
    _write(
        d / "tau2_loop" / "tau2_results.json",
        {
            "timestamp": board["timestamp"],
            "info": board["info"],
            "tasks": [t for t in board["tasks"] if t["id"] in passes],
            "simulations": sims,
        },
    )
    _jsonl(d / "tau2_loop" / "results.jsonl", results)
    _write(
        d / "tau2_loop" / "run.json",
        {"run_id": loop_run, "domain": "telecom", "agent": "student-0.6b", "split": "custom"},
    )
    _jsonl(
        d / "served.jsonl",
        [
            {
                "t": f"2026-10-17T0{i}:00:00Z",
                "system_prompt_sha256": SYSTEM_SHA,
                "tools_sha256": TOOLS_SHA,
                "n_messages": 3 + i,
                "finish_reason": "tool_calls" if i % 2 else "stop",
                "tool_calls": 1 if i % 2 else 0,
                "text_tool_call": False,
                "prompt_tokens": 7200 + 100 * i,
                "completion_tokens": 40,
                "latency_s": 1.5 + i / 10,
            }
            for i in range(4)
        ],
    )
    n_pass = sum(passes.values())
    _write(
        d / "run.json",
        {
            "run_id": run_id,
            "kind": "eval",
            "label": label,
            "purpose": "smoke",
            "created": f"2026-10-17T0{1 if train_run is None else 2}:00:00Z",
            "model": {
                "name": "Qwen3-0.6B" + (" + fixture-smoke" if train_run else ""),
                "base_model": "Qwen/Qwen3-0.6B",
                "train_run": train_run,
                "adapter_sha256": ADAPTER if train_run else None,
            },
            "serving": {
                "server": "mlx_lm.server",
                "precision": "bf16",
                "endpoint": "http://127.0.0.1:8000/v1",
                "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                "max_tokens": 32768,
            },
            "tau2_loop_run": loop_run,
            "tau2_loop_version": "telecom/student-0.6b",
            "split": "train",
            "task_ids": tasks,
            "trials": 1,
            "scores": {
                "n_tasks": len(tasks),
                "trials": 1,
                "passed": n_pass,
                "pass_hat": {"1": n_pass / len(tasks)},
                "by_task": {t: [passes[t]] for t in tasks},
            },
            "prompt_bytes": {
                "ok": True,
                "system_prompt_sha256": SYSTEM_SHA,
                "tools_sha256": TOOLS_SHA,
                "expected_system_prompt_sha256": SYSTEM_SHA,
                "expected_tools_sha256": TOOLS_SHA,
                "requests": 4,
            },
            "tool_calls": {"ok": True, "requests_with_tool_calls": 2, "text_tool_calls": 0},
            "vs": {"run_id": BASE} if train_run else None,
            "mlflow_run_id": None,
            "mlflow_experiment": config.EXPERIMENT_EVAL,
        },
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    board = json.loads(BOARD.read_text())
    train = split.current().train_set
    tasks: list[str] = []
    for sim in board["simulations"]:
        if sim["task_id"] in train and sim["task_id"] not in tasks:
            tasks.append(sim["task_id"])
    a, b, c = tasks[:3]
    train_run()
    eval_run(BASE, "fixture base 0.6B", None, {a: 1, b: 0, c: 0})
    eval_run(TUNED, "fixture tuned 0.6B", TRAIN, {a: 1, b: 1, c: 0})
    print(f"wrote {OUT.relative_to(config.ROOT)}: {TRAIN}, {BASE}, {TUNED}")


if __name__ == "__main__":
    main()
