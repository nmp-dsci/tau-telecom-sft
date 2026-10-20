"""`tausft sync-runs` (M4): a training run's folder from Modal's runs Volume into runs/, then
into the central MLflow.

Nothing on Modal reaches MLflow, which listens on this Mac, so the Mac logs the run from its
folder as the trainer would have: the run with its params and tags, every step in
metrics.jsonl, and, once it has finished, its files and the adapter as artifacts. The adapter's
sha256 must equal the one in run.json, so the adapter served at M5 is the one trained. The
checkpoints stay on the Volume; those at each epoch's end are kept for the record.

With --watch N it syncs every N seconds until the run ends, so the viewer's /train/<run> and
MLflow's curve stay under a minute behind the job. After a measured step it projects the full
run's hours and dollars from the step's tokens per second.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tau_telecom_sft import config, track
from tau_telecom_sft.build import tokens
from tau_telecom_sft.cloud.modal_app import RUNS_VOLUME, TRAIN_USD_PER_S
from tau_telecom_sft.train.sft import RUNS_DIR, _sha256_dir, load_config, select_rows, write_json

FILES = ("run.json", "metrics.jsonl", "mask.json")


def newest(volume: Any) -> str:
    names = sorted(str(e.path) for e in volume.listdir("/") if "_train_" in e.path)
    if not names:
        raise RuntimeError(f"no training run on the {RUNS_VOLUME} Volume yet")
    return names[-1]


def pull(volume: Any, run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    """The folder's files, and the adapter once the run has finished (checked against its
    sha256). The local run.json keeps the Mac's MLflow link, which the Volume's copy lacks."""
    dest = runs_dir / run_id
    before = json.loads((dest / "run.json").read_text()) if (dest / "run.json").exists() else {}
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        try:
            data = b"".join(volume.read_file(f"{run_id}/{name}"))
        except FileNotFoundError:
            continue  # mask.json and metrics.jsonl come a few minutes after run.json
        (dest / name).write_bytes(data)
    meta: dict[str, Any] = json.loads((dest / "run.json").read_text())
    for key in ("mlflow_run_id", "mlflow_experiment"):
        if before.get(key):
            meta[key] = before[key]
    if meta["status"] == "finished" and not (dest / "adapter").exists():
        for entry in volume.listdir(f"{run_id}/adapter", recursive=True):
            if entry.type.name == "FILE":
                out = runs_dir / entry.path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"".join(volume.read_file(entry.path)))
        sha, _ = _sha256_dir(dest / "adapter")
        if sha != meta["adapter"]["sha256"]:
            raise RuntimeError(
                f"{run_id}: the adapter copied home has sha256 {sha[:8]}…, not the job's "
                f"{meta['adapter']['sha256'][:8]}…"
            )
    write_json(dest / "run.json", meta)
    return meta


def steps(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def log(meta: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """MLflow: open the run once (its id is kept in the local run.json), log every step not yet
    logged, and close it with the folder and adapter as artifacts when the job is over."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities import Metric

    if not meta.get("mlflow_run_id"):
        meta["mlflow_run_id"] = track.start_train(meta, system_metrics=False)
        meta["mlflow_experiment"] = config.EXPERIMENT_TRAIN if meta["mlflow_run_id"] else None
        write_json(run_dir / "run.json", meta)
    rid = meta["mlflow_run_id"]
    if rid is None:
        return meta  # MLflow is down: the folder is the record, and the next sync tries again
    client = MlflowClient()
    done = max((m.step for m in client.get_metric_history(rid, "loss")), default=0)
    ts = int(time.time() * 1000)
    new = [
        Metric(k, float(v), ts, row["step"])
        for row in steps(run_dir)
        if row["step"] > done
        for k, v in row.items()
        if k != "step" and isinstance(v, int | float)
    ]
    for i in range(0, len(new), 1000):
        client.log_batch(rid, metrics=new[i : i + 1000])
    if meta["status"] in ("finished", "failed"):
        if mlflow.active_run() is None:
            mlflow.start_run(run_id=rid)
        finished = meta["status"] == "finished"
        track.end_train(
            rid,
            run_dir,
            status="FINISHED" if finished else "FAILED",
            summary=meta if finished else None,
        )
    return meta


def projection(meta: dict[str, Any], run_dir: Path) -> str | None:
    """After a measured step: the full run at the step's tokens per second. The step trains the
    longest rows, whose attention costs most per token, so this errs long."""
    rows = steps(run_dir)
    if meta["purpose"] != "smoke" or meta["status"] != "finished" or not rows:
        return None
    rate = rows[-1].get("tokens_per_s")
    if not rate:
        return None
    cfg = load_config(config.ROOT / meta["config_file"])
    data, _ = select_rows(cfg)
    total = sum(tokens.count(r, cfg["tokenizer"]) for r in data) * cfg["sft"]["num_train_epochs"]
    hours = total / rate / 3600
    return (
        f"the full run: {total:,} tokens at {rate:,.0f} tokens/s is about {hours:.1f} h and "
        f"${hours * 3600 * TRAIN_USD_PER_S:.2f} on Modal, plus a few minutes to load; "
        f"peak memory {meta.get('peak_mem_gb')} GB of the L4's 24"
    )


def run(run_id: str | None, watch: int = 0, console: Any = None) -> dict[str, Any]:
    import modal

    say = console.print if console is not None else print
    volume = modal.Volume.from_name(RUNS_VOLUME)
    run_id = run_id or newest(volume)
    while True:
        meta = log(pull(volume, run_id), RUNS_DIR / run_id)
        last = (steps(RUNS_DIR / run_id) or [{}])[-1]
        say(
            f"{run_id}: {meta['status']}, step {last.get('step', 0)}, loss {last.get('loss')}, "
            f"{last.get('tokens_per_s')} tokens/s, {last.get('mem_gb')} GB"
        )
        if meta["status"] != "running" or not watch:
            break
        time.sleep(watch)
    note = projection(meta, RUNS_DIR / run_id)
    if note:
        say(note)
    return meta
