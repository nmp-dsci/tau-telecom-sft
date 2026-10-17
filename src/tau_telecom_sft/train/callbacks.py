"""The trainer's one callback: every logged step to metrics.jsonl, and to MLflow when it can.

The run folder is the record. An MLflow call that fails (the tunnel from EC2 dropped, the Mac
asleep) is noted once and skipped, and training carries on; `make sync-runs` logs the folder
afterwards. Never the other way round: a job never dies because tracking did.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

STEP_KEYS = ("loss", "grad_norm", "learning_rate", "epoch")


def _mem_gb(device: str) -> float | None:
    import torch

    if device == "cuda":
        return round(float(torch.cuda.max_memory_allocated()) / 2**30, 2)
    if device == "mps":
        return round(float(torch.mps.driver_allocated_memory()) / 2**30, 2)
    return None


class RunFolder(TrainerCallback):  # type: ignore[misc]
    def __init__(self, run_dir: Path, device: str) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.device = device
        self.mlflow_run_id: str | None = None
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_tokens = 0
        self.tokens = 0
        self.last_loss: float | None = None
        self.peak_mem_gb: float | None = None
        self.mlflow_failed = False

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        logs = dict(logs or {})
        if "loss" not in logs:
            return  # the end-of-training summary line, not a step
        now = time.time()
        tokens = int(logs.get("num_tokens") or state.num_input_tokens_seen or 0)
        rate = (tokens - self.last_tokens) / max(now - self.last_t, 1e-6) if tokens else None
        mem = _mem_gb(self.device)
        self.peak_mem_gb = max(m for m in (mem, self.peak_mem_gb) if m is not None) if mem else None
        row: dict[str, Any] = {
            "step": int(state.global_step),
            **{k: logs.get(k) for k in STEP_KEYS},
            "tokens": tokens or None,
            "tokens_per_s": round(rate, 1) if rate else None,
            "mem_gb": mem,
            "elapsed_s": round(now - self.t0, 1),
        }
        self.last_t, self.last_tokens, self.tokens = now, tokens, tokens
        self.last_loss = logs.get("loss")
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self._mlflow(row)

    def _mlflow(self, row: dict[str, Any]) -> None:
        if self.mlflow_run_id is None or self.mlflow_failed:
            return
        try:
            from mlflow import MlflowClient

            client = MlflowClient()
            ts = int(time.time() * 1000)
            for k, v in row.items():
                if k != "step" and isinstance(v, int | float):
                    client.log_metric(self.mlflow_run_id, k, float(v), ts, row["step"])
        except Exception as e:  # noqa: BLE001 - tracking never stops a job
            self.mlflow_failed = True
            print(f"MLflow unreachable ({e}); steps stay in {self.path}, sync them later")
