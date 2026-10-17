"""Log to the central MLflow (nmp-central-ai, http://localhost:5000), and nowhere else.

Three experiments, created by `make -C ../nmp-central-ai mlflow-init` and named, never
numbered, here: tau-telecom-sft/data (one run per dataset build), /train (one per job) and
/eval (one per scored model). A job in the cloud cannot reach localhost:5000; it writes
metrics.jsonl to the bucket and `make sync-mlflow` (M4) logs it from the Mac. Never falls back
to a local store (PLATFORM.md rule 1).
"""

from __future__ import annotations

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
        for name in ("card.md", "stats.json", "drops.jsonl"):
            mlflow.log_artifact(str(built.out_dir / name))
        if log_rows:
            mlflow.log_artifact(str(built.out_dir / "train.jsonl"))
        rid: str = run.info.run_id
    return rid
