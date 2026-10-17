"""A scored model's record: read a tau2-loop run, never write to it (plan §10, D5).

tau2-loop plays the student and keeps its own run folder. `pull()` copies the three files the
viewer reads (run.json, results.jsonl, tau2_results.json) into `runs/<id>/tau2_loop/`, takes the
proxy's lines for the run's time window into `served.jsonl`, and writes `run.json` with pass^k,
the prompt-bytes and tool-call checks, and, against a base, the tasks fixed and broken with the
same one-sided sign test as tau2-loop's gate. Then one run in tau-telecom-sft/eval.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau_telecom_sft import config, track
from tau_telecom_sft.build.convert import agent_tool_names, phone_tool_names
from tau_telecom_sft.serve import proxy

RUNS_DIR = config.ROOT / "runs"
COPIED = ("run.json", "results.jsonl", "tau2_results.json")


def _parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def pass_hat(by_task: dict[str, list[int]]) -> dict[str, float]:
    """tau2's pass^k: the chance that k trials drawn from a task's n all pass, averaged."""
    n = min((len(v) for v in by_task.values()), default=0)
    out: dict[str, float] = {}
    for k in range(1, n + 1):
        vals = [math.comb(sum(v), k) / math.comb(len(v), k) for v in by_task.values()]
        out[str(k)] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out


def sign_test(fixed: int, broke: int) -> float:
    """One-sided exact sign test on the tasks that changed: P(X >= fixed), X ~ Bin(n, 1/2)."""
    n = fixed + broke
    if n == 0:
        return 1.0
    return round(sum(math.comb(n, i) for i in range(fixed, n + 1)) / float(1 << n), 4)


def compare(mine: dict[str, list[int]], base: dict[str, list[int]]) -> dict[str, Any]:
    """Paired by task on its pass fraction, as tau2-loop's gate pairs them."""
    shared = sorted(set(mine) & set(base))
    frac = {t: (sum(mine[t]) / len(mine[t]), sum(base[t]) / len(base[t])) for t in shared}
    fixed = [t for t, (a, b) in frac.items() if a > b]
    broke = [t for t, (a, b) in frac.items() if a < b]
    return {
        "tasks": len(shared),
        "fixed": len(fixed),
        "broke": len(broke),
        "same": len(shared) - len(fixed) - len(broke),
        "p": sign_test(len(fixed), len(broke)),
    }


def pull(
    loop_run: Path,
    *,
    label: str,
    served_log: Path | None,
    model: dict[str, Any],
    serving: dict[str, Any],
    purpose: str = "smoke",
    vs: str | None = None,
    results_row: str | None = None,
    use_mlflow: bool = True,
) -> Path:
    meta = json.loads((loop_run / "run.json").read_text())
    rows = [json.loads(x) for x in (loop_run / "results.jsonl").read_text().splitlines() if x]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_eval_{label}"
    out = RUNS_DIR / run_id
    (out / "tau2_loop").mkdir(parents=True)
    for name in COPIED:
        shutil.copy2(loop_run / name, out / "tau2_loop" / name)

    lines: list[dict[str, Any]] = []
    if served_log and served_log.exists():
        t0, t1 = _parse_ts(meta["started_at"]), _parse_ts(meta["finished_at"] or meta["started_at"])
        for x in served_log.read_text().splitlines():
            rec = json.loads(x)
            if t0 - 5 <= rec["t"] <= t1 + 5:
                lines.append(rec)
        (out / "served.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    stats = json.loads((config.SFT_DIR / "v1" / "stats.json").read_text())
    checks = proxy.summarise(lines, stats, agent_tool_names(), phone_tool_names())

    by_task: dict[str, list[int]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["task_id"], r.get("trial", 1))):
        by_task[r["task_id"]].append(int(bool(r.get("correct"))))
    record: dict[str, Any] = {
        "run_id": run_id,
        "kind": "eval",
        "label": label,
        "purpose": purpose,
        "results_row": results_row,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": config.settings().code_sha,
        "model": model,
        "serving": serving,
        "tau2_loop_run": meta["run_id"],
        "tau2_loop_version": f"{meta['domain']}/{meta['agent']}",
        "tau2_loop_mlflow_run_id": meta.get("mlflow_run_id"),
        "customer": {"model": meta.get("user_model"), "effort": meta.get("user_effort")},
        "sim_rules": meta.get("sim_rules"),
        "split": meta["split"],
        "task_ids": meta.get("task_ids") if purpose == "smoke" else None,
        "trials": meta["trials"],
        "scores": {
            "n_tasks": len(by_task),
            "trials": meta["trials"],
            "passed": sum(1 for v in by_task.values() if all(v)),
            "pass_hat": pass_hat(by_task),
            "by_task": dict(by_task),
        },
        **checks,
        "duration_s": round(_parse_ts(meta["finished_at"]) - _parse_ts(meta["started_at"]), 1)
        if meta.get("finished_at")
        else None,
        "cost_usd": 0.0,
    }
    if vs:
        base = json.loads((RUNS_DIR / vs / "run.json").read_text())
        record["vs"] = {"run_id": vs, **compare(by_task, base["scores"]["by_task"])}
    if use_mlflow:
        record["mlflow_run_id"] = track.log_eval(record)
        record["mlflow_experiment"] = config.EXPERIMENT_EVAL
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return out
