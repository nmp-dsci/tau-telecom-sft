"""Log to the central MLflow (nmp-central-ai, http://localhost:5000), and nowhere else.

Three experiments, created by `make -C ../nmp-central-ai mlflow-init` and named, never
numbered, here: tau-telecom-sft/data (one run per dataset build), /train (one per job) and
/eval (one per scored model). A job in the cloud cannot reach localhost:5000; it writes
metrics.jsonl to the bucket and `make sync-mlflow` (M4) logs it from the Mac. Never falls back
to a local store (PLATFORM.md rule 1).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from tau_telecom_sft import config
from tau_telecom_sft.build.dataset import Built


class TrackingDownError(RuntimeError):
    """The central MLflow is not reachable; the remedy is `make platform-up`."""


def preflight() -> str:
    uri = config.settings().mlflow_tracking_uri.rstrip("/")
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=3) as r:  # noqa: S310 - platform URL
            if r.status != 200:
                raise TrackingDownError(f"{uri}/health returned {r.status}")
    except (urllib.error.URLError, OSError) as e:
        raise TrackingDownError(
            f"MLflow at {uri} is not reachable ({e}). Start the platform: `make platform-up`"
        ) from e
    return uri


def required_tags(billing: str = "none") -> dict[str, str]:
    """The tags PLATFORM.md requires on every run."""
    s = config.settings()
    return {"project": config.PROJECT, "git_sha": s.code_sha, "env": s.env, "billing": billing}


def _mlflow(experiment: str) -> Any:
    import mlflow

    preflight()
    mlflow.set_tracking_uri(config.settings().mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        raise TrackingDownError(
            f"experiment {experiment!r} is missing: run `make -C ../nmp-central-ai mlflow-init`"
        )
    mlflow.set_experiment(experiment)
    return mlflow


def log_build(built: Built, kind: str, log_rows: bool = False) -> str:
    """One run in tau-telecom-sft/data: the card, the counts, and the drop table."""
    mlflow = _mlflow(config.EXPERIMENT_DATA)
    st = built.stats
    with mlflow.start_run(run_name=f"{kind}-{st['dataset']}") as run:
        mlflow.set_tags(
            {
                **required_tags(),
                "kind": kind,
                "dataset": st["dataset"],
                "train_sha256": st["train_sha256"],
                "split_sha256": st["split_sha256"],
                "system_prompt_sha256": st["system_prompt_sha256"],
                "tools_sha256": st["tools_sha256"],
            }
        )
        mlflow.log_params(
            {
                "dataset": st["dataset"],
                "split_version": st["split_version"],
                "tau2_pin": st["tau2_pin"],
                "sources": ",".join(s["id"] for s in st["sources"]),
                "token_counter": st["token_counter"],
                "max_tokens": st["max_tokens"],
            }
        )
        mlflow.log_metrics(
            {
                "n_loaded": st["n_loaded"],
                "n_kept": st["n_kept"],
                "n_dropped": st["n_dropped"],
                "training_tasks_covered": st["training_tasks_covered"],
                **{f"dropped_{f['rule']}": f["dropped"] for f in st["filters"]},
                **{f"tokens_{k}": v for k, v in st["tokens"].items()},
            }
        )
        for name in ("card.md", "stats.json", "drops.jsonl", "review_sample.json"):
            if (built.out_dir / name).exists():
                mlflow.log_artifact(str(built.out_dir / name))
        if log_rows:
            mlflow.log_artifact(str(built.out_dir / "train.jsonl"))
        rid: str = run.info.run_id
    return rid


def _params(meta: dict[str, Any]) -> dict[str, Any]:
    cfg = meta["config"]
    out: dict[str, Any] = {
        "base_model": meta["base_model"],
        "tokenizer": meta["tokenizer"],
        "dataset": meta["dataset"],
        "n_rows": meta["n_rows"],
        "device": meta["device"],
        "purpose": meta["purpose"],
    }
    for section in ("lora", "sft", "rows"):
        out.update({f"{section}.{k}": v for k, v in (cfg.get(section) or {}).items()})
    return {k: str(v)[:500] for k, v in out.items()}


def start_train(meta: dict[str, Any], system_metrics: bool = True) -> str | None:
    """Open the job's run in tau-telecom-sft/train; None (and the job carries on) if MLflow is
    down, so a job writes its folder first and `make sync-runs` logs it later. A sync from
    Modal's Volume logs no system metrics: they would be the Mac's, not the GPU's."""
    try:
        mlflow = _mlflow(config.EXPERIMENT_TRAIN)
    except Exception as e:  # noqa: BLE001 - tracking never stops a job
        print(f"MLflow not reachable ({e}); the run folder is the record")
        return None
    run = mlflow.start_run(run_name=meta["run_id"], log_system_metrics=system_metrics)
    mlflow.set_tags(
        {
            **required_tags(billing=meta.get("billing", "none")),
            "kind": meta["purpose"],
            "run_folder": meta["run_id"],
            "dataset": meta["dataset"],
            "train_sha256": meta["train_sha256"],
            "device": meta["device"],
            "instance": meta["instance"],
            "image_tag": str(meta.get("image_tag")),
        }
    )
    mlflow.log_params(_params(meta))
    if meta.get("mask_check"):
        mlflow.log_metrics(
            {
                "mask_ok": float(meta["mask_check"]["ok"]),
                "mask_trained_tokens": meta["mask_check"]["trained_tokens"],
            }
        )
    rid: str = run.info.run_id
    return rid


def end_train(
    run_id: str | None, run_dir: Any, status: str, summary: dict[str, Any] | None = None
) -> None:
    if run_id is None:
        return
    try:
        import mlflow

        if summary:
            mlflow.log_metrics(
                {
                    k: float(summary[k])
                    for k in ("duration_s", "steps", "train_loss", "peak_mem_gb", "tokens_trained")
                    if isinstance(summary.get(k), int | float)
                }
            )
            mlflow.set_tag("adapter_sha256", summary["adapter"]["sha256"])
        for name in ("run.json", "metrics.jsonl", "mask.json"):
            if (run_dir / name).exists():
                mlflow.log_artifact(str(run_dir / name))
        if (run_dir / "adapter").exists():
            mlflow.log_artifacts(str(run_dir / "adapter"), artifact_path="adapter")
        mlflow.end_run(status=status)
    except Exception as e:  # noqa: BLE001 - the folder is the record
        print(f"MLflow logging at the end failed ({e}); the run folder is complete")


def log_eval(record: dict[str, Any]) -> str | None:
    """One run in tau-telecom-sft/eval per scored model: pass^k, the checks, the links."""
    try:
        mlflow = _mlflow(config.EXPERIMENT_EVAL)
    except Exception as e:  # noqa: BLE001 - the run folder is the record
        print(f"MLflow not reachable ({e}); the run folder is the record")
        return None
    sc = record["scores"]
    with mlflow.start_run(run_name=record["run_id"]) as run:
        mlflow.set_tags(
            {
                **required_tags(billing="subscription"),  # the customer runs on the subscription
                "kind": record["purpose"],
                "run_folder": record["run_id"],
                "tau2_loop_run": record["tau2_loop_run"],
                "tau2_loop_version": record["tau2_loop_version"],
                "model": record["model"]["name"],
                "train_run": record["model"].get("train_run") or "none",
                "adapter_sha256": record["model"].get("adapter_sha256") or "none",
                "split": record["split"],
            }
        )
        mlflow.log_params(
            {
                "server": record["serving"].get("server"),
                "precision": record["serving"].get("precision"),
                "sampling": json.dumps(record["serving"].get("sampling")),
                "customer": record["customer"]["model"],
                "trials": record["trials"],
                "n_tasks": sc["n_tasks"],
            }
        )
        metrics = {f"pass_hat_{k}": v for k, v in sc["pass_hat"].items()}
        metrics |= {
            "passed": sc["passed"],
            "prompt_bytes_ok": float(record["prompt_bytes"]["ok"]),
            "served_requests": record["prompt_bytes"]["requests"],
            "requests_with_tool_calls": record["tool_calls"]["requests_with_tool_calls"],
            "text_tool_calls": record["tool_calls"]["text_tool_calls"],
            "phone_tool_calls": record["tool_calls"].get("phone_tool_calls", 0),
        }
        if record["tool_calls"].get("agent_tool_share") is not None:
            metrics["agent_tool_share"] = record["tool_calls"]["agent_tool_share"]
        if record.get("vs"):
            metrics |= {f"vs_{k}": float(record["vs"][k]) for k in ("fixed", "broke", "same", "p")}
        mlflow.log_metrics(metrics)
        rid: str = run.info.run_id
    return rid
