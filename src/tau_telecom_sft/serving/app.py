"""The viewer's API (plan §10), and the built frontend from the same origin.

The chain it makes inspectable: dataset version → training run → adapter → scored model.
Every GET reads files only (`store`); no number is read from MLflow, which is only linked. The
one write route appends a person's verdict or sign-off to `data/sft/<v>/review.jsonl` (D10),
and only from this machine. Test task ids never leave the API: a dropped test task is
WITHHELD in drops.jsonl, and the coverage lens returns how many test tasks need each root
issue, never which.

Run it: `uv run python -m tau_telecom_sft.serving` (127.0.0.1:8085), or
`uv run uvicorn --factory tau_telecom_sft.serving.app:create_app --port 8085 --reload`.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tau_telecom_sft import config, split
from tau_telecom_sft.build import review
from tau_telecom_sft.serving import views
from tau_telecom_sft.serving.store import Paths, Store

LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
THIN_ROWS = (
    10  # a root issue with fewer kept rows than this is flagged thin (M7's trigger reads it)
)
LENSES = ("chat", "tokens", "raw")
EXPERIMENTS = (config.EXPERIMENT_DATA, config.EXPERIMENT_TRAIN, config.EXPERIMENT_EVAL)

# The results table this repo fills in (plan §8). An eval run fills a row only when its
# run.json says `purpose: result` and `results_row: R<n>`; M2's smoke runs never do.
RESULT_ROWS = (
    ("R0", "Qwen3-4B, untrained", "M3"),
    ("R1", "Qwen3-4B + SFT on v1", "M5"),
    ("R2", "Qwen3-30B-A3B, untrained", "M6"),
    ("R3", "Qwen3-30B-A3B + SFT on v1", "M6"),
    ("R4", "the student that missed + SFT on v2", "M7, only if needed"),
    ("R5", "Claude telecom v2 (Haiku 5.5), 4 trials", "M8"),
)


class ReviewBody(BaseModel):
    kind: str = "verdict"
    sim_id: str | None = None
    verdict: str | None = None
    note: str = ""


class _Mlflow:
    """Whether the central MLflow answers, and each experiment's id for links; cached 30 s."""

    def __init__(self) -> None:
        self._at = 0.0
        self._up = False
        self._ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def _get(self, path: str, parse: bool = True) -> Any:
        base = config.settings().mlflow_tracking_uri.rstrip("/")
        with urllib.request.urlopen(f"{base}{path}", timeout=1.5) as r:  # noqa: S310 - platform URL
            body = r.read()
            return json.loads(body or b"{}") if parse else body

    def refresh(self) -> None:
        with self._lock:
            if time.time() - self._at < 30:
                return
            self._at = time.time()
            try:
                self._get("/health", parse=False)
                self._up = True
                for name in EXPERIMENTS:
                    if name not in self._ids:
                        q = urllib.parse.quote(name)
                        doc = self._get(
                            f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={q}"
                        )
                        if doc:
                            self._ids[name] = str(doc["experiment"]["experiment_id"])
            except (urllib.error.URLError, OSError, ValueError, KeyError):
                self._up = False

    @property
    def up(self) -> bool:
        self.refresh()
        return self._up

    def url(self, run_id: str | None, experiment: str | None) -> str | None:
        if not run_id or not experiment or not self.up or experiment not in self._ids:
            return None
        base = config.settings().mlflow_tracking_uri.rstrip("/")
        return f"{base}/#/experiments/{self._ids[experiment]}/runs/{run_id}"


SETTINGS = (
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "r",
    "lora_alpha",
    "max_length",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
)


def flat_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """A job's nested config (`sft:`, `lora:`, `rows:`) as dotted keys, for the table and diff."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat_config(v, f"{key}."))
        else:
            out[key] = v
    return out


def settings_of(cfg: dict[str, Any]) -> dict[str, Any]:
    """The settings that move a score, whatever section of the config they sit in."""
    flat = flat_config(cfg)
    out: dict[str, Any] = {}
    for name in SETTINGS:
        hit = next((v for k, v in flat.items() if k == name or k.endswith(f".{name}")), None)
        if hit is not None:
            out["lora_r" if name == "r" else name] = hit
    return out


def _median(xs: Iterable[float]) -> float | None:
    v = list(xs)
    return float(statistics.median(v)) if v else None


def create_app(
    paths: Paths | None = None, allowed_hosts: frozenset[str] = LOCAL_HOSTS, mlflow: bool = True
) -> FastAPI:
    app = FastAPI(title="tau-telecom-sft viewer", version="0.1.0")
    store = Store(paths)
    tracker = _Mlflow()
    tokens_cache: dict[tuple[str, str], dict[str, Any]] = {}
    tokens_lock = threading.Lock()

    def link(doc: dict[str, Any], default_experiment: str) -> str | None:
        if not mlflow:
            return None
        return tracker.url(
            doc.get("mlflow_run_id"), doc.get("mlflow_experiment") or default_experiment
        )

    def stats_or_404(v: str) -> dict[str, Any]:
        try:
            return store.stats(v)
        except FileNotFoundError as e:
            raise HTTPException(404, f"no dataset {v}") from e

    def run_or_404(run_id: str, kind: str) -> dict[str, Any]:
        doc = store.run(run_id)
        if doc is None or doc.get("kind") != kind:
            raise HTTPException(404, f"no {kind} run {run_id}")
        return doc

    # ---- shapes shared by several routes ----------------------------------------------

    def dataset_summary(v: str) -> dict[str, Any]:
        st = store.stats(v)
        d = store.version_dir(v)
        rv = review.status(d) if d else {"reviewed": 0, "total": 0, "signed_off": None}
        return {
            "version": v,
            "built": st.get("built"),
            "code_sha": st.get("code_sha"),
            "n_loaded": st.get("n_loaded"),
            "n_kept": st.get("n_kept"),
            "n_dropped": st.get("n_dropped"),
            "training_tasks_covered": st.get("training_tasks_covered"),
            "training_tasks_total": st.get("training_tasks_total"),
            "train_sha256": st.get("train_sha256"),
            "sources": [f"{s['id']} {s['label']}" for s in st.get("sources", [])],
            "tokens": st.get("tokens"),
            "max_tokens": st.get("max_tokens"),
            "token_counter": st.get("token_counter"),
            "rows_here": store.has_rows(v),
            "review": {
                "reviewed": rv["reviewed"],
                "total": rv["total"],
                "signed_off": rv["signed_off"],
            },
        }

    def scores_of(doc: dict[str, Any]) -> dict[str, list[int]]:
        rows = store.loop_results(doc["run_id"])
        if rows:
            return views.by_task(rows)
        bt = (doc.get("scores") or {}).get("by_task") or {}
        return {str(t): [int(x) for x in v] for t, v in bt.items()}

    def eval_summary(doc: dict[str, Any]) -> dict[str, Any]:
        sc = doc.get("scores") or {}
        model = doc.get("model") or {}
        return {
            "run_id": doc["run_id"],
            "label": doc.get("label") or doc["run_id"],
            "purpose": doc.get("purpose", "smoke"),
            "results_row": doc.get("results_row"),
            "created": doc.get("created"),
            "model": model,
            "split": doc.get("split"),
            "trials": doc.get("trials"),
            "tau2_loop_run": doc.get("tau2_loop_run"),
            "n_tasks": sc.get("n_tasks"),
            "passed": sc.get("passed"),
            "pass_hat": sc.get("pass_hat") or {},
            "prompt_bytes_ok": (doc.get("prompt_bytes") or {}).get("ok"),
            "tool_calls_ok": (doc.get("tool_calls") or {}).get("ok"),
            "vs": doc.get("vs"),
            "mlflow_url": link(doc, config.EXPERIMENT_EVAL),
        }

    def train_summary(doc: dict[str, Any], evals: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = doc.get("config") or {}
        scored = [
            eval_summary(e)
            for e in evals
            if (e.get("model") or {}).get("train_run") == doc["run_id"]
        ]
        return {
            "run_id": doc["run_id"],
            "label": doc.get("label") or doc["run_id"],
            "purpose": doc.get("purpose", "smoke"),
            "status": doc.get("status"),
            "created": doc.get("created"),
            "base_model": doc.get("base_model"),
            "dataset": doc.get("dataset"),
            "train_sha256": doc.get("train_sha256"),
            "n_rows": doc.get("n_rows"),
            "device": doc.get("device"),
            "instance": doc.get("instance"),
            "duration_s": doc.get("duration_s"),
            "cost_usd": doc.get("cost_usd"),
            "steps": doc.get("steps"),
            "final_loss": doc.get("final_loss"),
            "peak_mem_gb": doc.get("peak_mem_gb"),
            "settings": settings_of(cfg),
            "adapter": doc.get("adapter"),
            "mask_ok": (doc.get("mask_check") or {}).get("ok"),
            "scored": scored,
            "mlflow_url": link(doc, config.EXPERIMENT_TRAIN),
        }

    # ---- health and overview -----------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "code_sha": config.settings().code_sha,
            "mlflow": "up" if mlflow and tracker.up else "down",
        }

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        datasets = [dataset_summary(v) for v in store.versions()]
        evals = store.runs("eval")
        trains = store.runs("train")
        train_rows = [train_summary(t, evals) for t in trains]
        results = []
        for rid, model, when in RESULT_ROWS:
            hit = next(
                (e for e in evals if e.get("purpose") == "result" and e.get("results_row") == rid),
                None,
            )
            results.append(
                {
                    "row": rid,
                    "model": model,
                    "milestone": when,
                    "run": eval_summary(hit) if hit else None,
                }
            )
        chains = []
        for t in trains:
            chains.append(
                {
                    "dataset": t.get("dataset"),
                    "train_run": t["run_id"],
                    "train_label": t.get("label"),
                    "purpose": t.get("purpose", "smoke"),
                    "adapter_sha256": (t.get("adapter") or {}).get("sha256"),
                    "evals": [
                        eval_summary(e)
                        for e in evals
                        if (e.get("model") or {}).get("train_run") == t["run_id"]
                    ],
                    "bases": [
                        eval_summary(e)
                        for e in evals
                        if not (e.get("model") or {}).get("train_run")
                        and (e.get("model") or {}).get("base_model") == t.get("base_model")
                    ],
                }
            )
        return {
            "project": config.PROJECT,
            "code_sha": config.settings().code_sha,
            "milestones": store.milestones(),
            "datasets": datasets,
            "train_runs": train_rows,
            "eval_runs": [eval_summary(e) for e in evals],
            "results": results,
            "smoke": [eval_summary(e) for e in evals if e.get("purpose", "smoke") != "result"],
            "chains": chains,
        }

    # ---- datasets ----------------------------------------------------------------------

    @app.get("/api/data")
    def data() -> dict[str, Any]:
        return {"versions": [dataset_summary(v) for v in store.versions()]}

    @app.get("/api/data/{v}")
    def data_version(v: str) -> dict[str, Any]:
        st = stats_or_404(v)
        drops = store.drops(v)
        return {
            "summary": dataset_summary(v),
            "stats": st,
            "card": store.card(v),
            "withheld": sum(1 for d in drops if not d.get("sim_id")),
        }

    @app.get("/api/data/{v}/coverage")
    def coverage(v: str) -> dict[str, Any]:
        st = stats_or_404(v)
        s = split.current()
        train_need: Counter[str] = Counter(
            i for t in s.train for i in split.parse_task_id(t).issues
        )
        test_need: Counter[str] = Counter(i for t in s.test for i in split.parse_task_id(t).issues)
        kept: dict[str, int] = st.get("kept_by_root_issue", {})
        issues = sorted(set(kept) | set(train_need) | set(test_need))
        issues.sort(key=lambda i: (-test_need[i], int(kept.get(i, 0)), i))
        rows = [
            {
                "issue": i,
                "kept_rows": int(kept.get(i, 0)),
                "train_tasks": train_need[i],
                "test_tasks": test_need[i],
                "thin": test_need[i] > 0 and int(kept.get(i, 0)) < THIN_ROWS,
            }
            for i in issues
        ]
        symptoms_test = Counter(split.parse_task_id(t).symptom for t in s.test)
        return {
            "version": v,
            "thin_below": THIN_ROWS,
            "issues": rows,
            "symptoms": [
                {"symptom": k, "kept_rows": n, "test_tasks": symptoms_test.get(k, 0)}
                for k, n in sorted(st.get("kept_by_symptom", {}).items())
            ],
            "tasks_covered": st.get("training_tasks_covered"),
            "tasks_total": st.get("training_tasks_total"),
            "note": "test tasks are counted, never named",
        }

    def all_rows(v: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sim_id, (_, meta) in store.kept_index(v).items():
            rows.append(
                {
                    "sim_id": sim_id,
                    "status": "kept",
                    "task_id": meta.get("task_id"),
                    "symptom": meta.get("symptom"),
                    "source": meta.get("source"),
                    "trial": meta.get("trial"),
                    "agent_turns": meta.get("agent_turns"),
                    "rule": None,
                    "reason": None,
                }
            )
        for d in store.drops(v):
            if not d.get("sim_id"):
                continue  # a test task or twin: counted, never listed
            tid = str(d.get("task_id") or "")
            try:
                symptom: str | None = split.parse_task_id(tid).symptom
            except ValueError:
                symptom = None
            rows.append(
                {
                    "sim_id": d["sim_id"],
                    "status": "dropped",
                    "task_id": tid,
                    "symptom": symptom,
                    "source": d.get("source"),
                    "trial": d.get("trial"),
                    "agent_turns": d.get("agent_turns"),
                    "rule": d.get("rule"),
                    "reason": d.get("reason"),
                }
            )
        rows.sort(key=lambda r: (str(r["task_id"]), str(r["source"]), r["trial"] or 0, r["sim_id"]))
        return rows

    @app.get("/api/data/{v}/rows")
    def rows(
        v: str,
        status: str = Query("all", pattern="^(all|kept|dropped)$"),
        rule: str | None = None,
        task: str | None = None,
        symptom: str | None = None,
        source: str | None = None,
        q: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        st = stats_or_404(v)
        items = all_rows(v)
        counts = Counter(r["status"] for r in items)
        if status != "all":
            items = [r for r in items if r["status"] == status]
        if rule:
            items = [r for r in items if r["rule"] == rule]
        if task:
            items = [r for r in items if r["task_id"] == task]
        if symptom:
            items = [r for r in items if r["symptom"] == symptom]
        if source:
            items = [r for r in items if str(r["source"] or "").startswith(source)]
        if q:
            ql = q.lower()
            items = [
                r
                for r in items
                if ql in str(r["task_id"]).lower()
                or ql in r["sim_id"].lower()
                or ql in str(r["reason"] or "").lower()
            ]
        withheld = sum(1 for d in store.drops(v) if not d.get("sim_id"))
        return {
            "version": v,
            "rows_here": store.has_rows(v),
            "total": len(items),
            "counts": {"kept": counts["kept"], "dropped": counts["dropped"], "withheld": withheld},
            "n_kept": st.get("n_kept"),
            "rules": [f["rule"] for f in st.get("filters", [])],
            "sources": [f"{s['id']} {s['label']}" for s in st.get("sources", [])],
            "symptoms": sorted(st.get("kept_by_symptom", {})),
            "items": items[offset : offset + limit],
        }

    @app.get("/api/data/{v}/rows/{sim_id}")
    def row(v: str, sim_id: str, lens: str = Query("chat")) -> dict[str, Any]:
        stats_or_404(v)
        if lens not in LENSES:
            raise HTTPException(422, f"lens must be one of {LENSES}")
        kept = store.kept_row(v, sim_id)
        drop = next((d for d in store.drops(v) if d.get("sim_id") == sim_id), None)
        if kept is None and drop is None:
            raise HTTPException(404, f"no row {sim_id} in {v}")
        meta = (
            dict(kept["meta"])
            if kept
            else {
                "task_id": drop["task_id"] if drop else None,
                "source": drop.get("source") if drop else None,
                "trial": drop.get("trial") if drop else None,
                "agent_turns": drop.get("agent_turns") if drop else None,
                "sim_id": sim_id,
            }
        )
        d = store.version_dir(v)
        sample = review.status(d)["sample"] if d else {"kept": [], "dropped": []}
        out: dict[str, Any] = {
            "version": v,
            "sim_id": sim_id,
            "status": "kept" if kept else "dropped",
            "meta": meta,
            "drop": {"rule": drop["rule"], "reason": drop["reason"]} if drop else None,
            "in_review_sample": sim_id in set(sample["kept"]) | set(sample["dropped"]),
            "lens": lens,
            "raw_here": False,
        }
        raw = store.raw_sim(v, str(meta.get("source") or ""), sim_id)
        out["raw_here"] = raw is not None
        conv = views.conversation(str(meta.get("source")), raw) if raw is not None else None
        if lens == "raw":
            if conv is None:
                out["messages"] = None
                out["missing"] = "the raw source file is not on this machine: run `make download`"
            else:
                out["messages"] = views.raw_messages(conv, drop["rule"] if drop else None)
                out["termination"] = conv.termination
                out["reward"] = conv.reward
            return out
        if kept:
            the_row = kept
        elif conv is not None:
            from tau_telecom_sft.build.convert import to_row

            the_row = to_row(conv, v)
        else:
            out["messages"] = None
            out["missing"] = "the raw source file is not on this machine: run `make download`"
            return out
        if lens == "chat":
            out["messages"] = views.chat_messages(the_row)
            out["n_tools"] = len(the_row["tools"])
            return out
        key = (v, sim_id)
        with tokens_lock:
            cached = tokens_cache.get(key)
        if cached is None:
            try:
                cached = views.token_segments(the_row)
            except OSError as e:  # the tokenizer is not in the Hugging Face cache, and no network
                raise HTTPException(503, f"the student's tokenizer is not available: {e}") from e
            with tokens_lock:
                tokens_cache[key] = cached
        out["tokens"] = cached
        return out

    @app.get("/api/data/{v}/review")
    def review_status(v: str) -> dict[str, Any]:
        stats_or_404(v)
        d = store.version_dir(v)
        assert d is not None
        st = review.status(d)
        index = store.kept_index(v)
        drops = {x["sim_id"]: x for x in store.drops(v) if x.get("sim_id")}

        def item(sim_id: str, status: str) -> dict[str, Any]:
            meta = index[sim_id][1] if sim_id in index else drops.get(sim_id, {})
            dr = drops.get(sim_id)
            return {
                "sim_id": sim_id,
                "status": status,
                "task_id": meta.get("task_id"),
                "source": meta.get("source"),
                "trial": meta.get("trial"),
                "rule": dr.get("rule") if dr else None,
                "reason": dr.get("reason") if dr else None,
                "latest": st["latest"].get(sim_id),
            }

        return {
            "version": v,
            "seed": st["sample"].get("seed"),
            "drawn": st["sample"].get("drawn"),
            "items": [item(s, "kept") for s in st["sample"]["kept"]]
            + [item(s, "dropped") for s in st["sample"]["dropped"]],
            "reviewed": st["reviewed"],
            "total": st["total"],
            "signed_off": st["signed_off"],
            "verdicts": list(review.VERDICTS),
            "log": review.log(d),
        }

    @app.post("/api/data/{v}/review")
    def review_append(v: str, body: ReviewBody, request: Request) -> dict[str, Any]:
        host = request.client.host if request.client else ""
        if host not in allowed_hosts:
            raise HTTPException(403, "reviews are written only from this machine")
        stats_or_404(v)
        d = store.version_dir(v)
        assert d is not None
        try:
            rec = review.append(d, body.model_dump())
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"saved": rec, "status": review_status(v)}

    # ---- training runs -----------------------------------------------------------------

    @app.get("/api/train")
    def train_runs() -> dict[str, Any]:
        evals = store.runs("eval")
        return {"runs": [train_summary(t, evals) for t in store.runs("train")]}

    @app.get("/api/train/{run_id}")
    def train_run(run_id: str) -> dict[str, Any]:
        doc = run_or_404(run_id, "train")
        evals = store.runs("eval")
        return {
            "run": doc,
            "summary": train_summary(doc, evals),
            "config_flat": flat_config(doc.get("config") or {}),
            "metrics": store.metrics(run_id),
            "mask": store.mask(run_id),
        }

    # ---- scored models -----------------------------------------------------------------

    def served_summary(run_id: str) -> dict[str, Any]:
        rows = store.served(run_id)
        return {
            "requests": len(rows),
            "system_prompt_sha256": sorted({str(r.get("system_prompt_sha256")) for r in rows}),
            "tools_sha256": sorted({str(r.get("tools_sha256")) for r in rows}),
            "with_tool_calls": sum(1 for r in rows if r.get("tool_calls")),
            "text_tool_calls": sum(1 for r in rows if r.get("text_tool_call")),
            "latency_s_median": _median(
                float(r["latency_s"]) for r in rows if r.get("latency_s") is not None
            ),
            "prompt_tokens_max": max(
                (int(r.get("prompt_tokens") or 0) for r in rows), default=None
            ),
        }

    @app.get("/api/eval")
    def eval_runs() -> dict[str, Any]:
        return {"runs": [eval_summary(e) for e in store.runs("eval")]}

    @app.get("/api/eval/{run_id}")
    def eval_run(run_id: str, vs: str | None = None) -> dict[str, Any]:
        doc = run_or_404(run_id, "eval")
        mine = scores_of(doc)
        trials = max((len(v) for v in mine.values()), default=0)
        base_id = vs or (doc.get("vs") or {}).get("run_id")
        cmp = None
        if base_id:
            base = store.run(base_id)
            if base is None or base.get("kind") != "eval":
                raise HTTPException(404, f"no eval run {base_id} to compare with")
            cmp = {"vs": eval_summary(base), **views.compare(mine, scores_of(base))}
        results = store.loop_results(run_id)
        return {
            "run": doc,
            "summary": eval_summary(doc),
            "grid": [{"task_id": t, "trials": v} for t, v in sorted(mine.items())],
            "trials": trials,
            "pass_hat": {str(k): views.pass_hat(mine, k) for k in range(1, trials + 1)},
            "results": results,
            "served": served_summary(run_id),
            "loop_run": store.loop_run(run_id),
            "compare": cmp,
        }

    @app.get("/api/eval/{run_id}/{task_id}/{trial}")
    def eval_conversation(run_id: str, task_id: str, trial: str) -> dict[str, Any]:
        run_or_404(run_id, "eval")
        try:
            n = int(trial.removeprefix("t"))
        except ValueError as e:
            raise HTTPException(422, "trial is t1, t2, …") from e
        sim = store.loop_sim(run_id, task_id, n)
        if sim is None:
            raise HTTPException(404, f"no conversation {task_id} t{n} in {run_id}")
        conv = views.conversation(f"eval {run_id}", sim)
        result = next(
            (
                r
                for r in store.loop_results(run_id)
                if r.get("task_id") == task_id and int(r.get("trial") or 1) == n
            ),
            None,
        )
        return {
            "run_id": run_id,
            "task_id": task_id,
            "trial": n,
            "reward": conv.reward,
            "termination": conv.termination,
            "result": result,
            "messages": views.raw_messages(conv),
        }

    # ---- method ------------------------------------------------------------------------

    @app.get("/api/method")
    def method() -> dict[str, Any]:
        trains = store.runs("train")
        latest = next(
            (t for t in trains if t.get("status") == "finished"), trains[0] if trains else None
        )
        datasets = store.versions()
        st = store.stats(datasets[-1]) if datasets else None
        return {
            "train_run": latest,
            "metrics": store.metrics(latest["run_id"]) if latest else [],
            "mask": store.mask(latest["run_id"]) if latest else None,
            "dataset": {
                "version": datasets[-1],
                "n_kept": st.get("n_kept"),
                "tokens": st.get("tokens"),
                "max_tokens": st.get("max_tokens"),
                "token_counter": st.get("token_counter"),
            }
            if st and datasets
            else None,
        }

    # ---- the built frontend ------------------------------------------------------------

    dist = store.paths.dist
    if (dist / "index.html").exists():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, f"no route /{path}")
            f = dist / path
            if path and f.is_file() and f.resolve().is_relative_to(dist.resolve()):
                return FileResponse(f)
            return FileResponse(dist / "index.html")

    return app
