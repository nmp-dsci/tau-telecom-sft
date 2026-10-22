"""Every file the viewer reads, and nothing else (plan §10: committed files are the record).

Datasets come from `data/sft/<v>/` (stats.json, card.md, drops.jsonl, review_sample.json,
review.jsonl, and the Mac's git-ignored train.jsonl). Raw conversations come from the source
files the card names, found under `raw/board/` (or the fixtures). Runs come from
`runs/<id>/` in the shape `docs/run_folders.md` fixes. No number here is read from MLflow.

Two caches keep the pages quick and change nothing on record: a byte-offset index of each
train.jsonl, and each raw source split into one small file per conversation under
`.cache/viewer/` (git-ignored), written the first time a raw conversation is opened.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau_telecom_sft import config

RUN_KINDS = ("train", "eval")
MILESTONE = re.compile(r"^- \[(?P<done>[ x])\] \*\*(?P<id>M\d+) · (?P<title>.+?)\*\*(?P<rest>.*)$")


@dataclass(frozen=True)
class Paths:
    sft: Path = config.SFT_DIR
    runs: Path = config.ROOT / "runs"
    raw_dirs: tuple[Path, ...] = (config.RAW_DIR / "board", config.FIXTURES_DIR)
    checklist: Path = config.ROOT / "ai_specs" / "s00_project_plan.md"
    cache: Path = config.ROOT / ".cache" / "viewer"
    dist: Path = config.ROOT / "frontend" / "dist"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe(name: str) -> bool:
    """A path segment from a URL: no separators, no parent references."""
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..")


class Store:
    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or Paths()
        self._lock = threading.Lock()
        self._index: dict[tuple[str, float], dict[str, tuple[int, dict[str, Any]]]] = {}
        self._sims: dict[tuple[str, str], dict[str, Any]] = {}

    # ---- datasets ----------------------------------------------------------------------

    def versions(self) -> list[str]:
        if not self.paths.sft.is_dir():
            return []
        return sorted(
            d.name for d in self.paths.sft.iterdir() if d.is_dir() and (d / "stats.json").exists()
        )

    def version_dir(self, v: str) -> Path | None:
        d = self.paths.sft / v
        return d if _safe(v) and (d / "stats.json").exists() else None

    def stats(self, v: str) -> dict[str, Any]:
        d = self.version_dir(v)
        if d is None:
            raise FileNotFoundError(v)
        doc: dict[str, Any] = read_json(d / "stats.json")
        return doc

    def card(self, v: str) -> str:
        d = self.version_dir(v)
        return (d / "card.md").read_text() if d and (d / "card.md").exists() else ""

    def has_rows(self, v: str) -> bool:
        d = self.version_dir(v)
        return bool(d and (d / "train.jsonl").exists())

    def kept_index(self, v: str) -> dict[str, tuple[int, dict[str, Any]]]:
        """sim_id → (byte offset, meta) for every kept row; empty when train.jsonl is absent."""
        d = self.version_dir(v)
        if d is None or not (d / "train.jsonl").exists():
            return {}
        path = d / "train.jsonl"
        key = (str(path), path.stat().st_mtime)
        with self._lock:
            if key not in self._index:
                out: dict[str, tuple[int, dict[str, Any]]] = {}
                with path.open("rb") as f:
                    offset = 0
                    for line in f:
                        if line.strip():
                            row = json.loads(line)
                            meta = dict(row.get("meta") or {})
                            meta["n_messages"] = len(row.get("messages") or [])
                            if meta.get("sim_id"):
                                out[str(meta["sim_id"])] = (offset, meta)
                        offset += len(line)
                self._index[key] = out
            return self._index[key]

    def kept_row(self, v: str, sim_id: str) -> dict[str, Any] | None:
        hit = self.kept_index(v).get(sim_id)
        d = self.version_dir(v)
        if hit is None or d is None:
            return None
        with (d / "train.jsonl").open("rb") as f:
            f.seek(hit[0])
            row: dict[str, Any] = json.loads(f.readline())
        return row

    def drops(self, v: str) -> list[dict[str, Any]]:
        d = self.version_dir(v)
        return read_jsonl(d / "drops.jsonl") if d else []

    # ---- raw conversations -------------------------------------------------------------

    def source_file(self, v: str, source: str) -> tuple[Path, str] | None:
        """The raw file a row's `source` names ("S1 board/glm-5"), and its sha256, when here."""
        for s in self.stats(v).get("sources", []):
            if f"{s['id']} {s['label']}" != source:
                continue
            for base in self.paths.raw_dirs:
                p = base / s["file"]
                if p.exists():
                    return p, str(s["sha256"])
        return None

    def raw_sim(self, v: str, source: str, sim_id: str) -> dict[str, Any] | None:
        """One simulation's JSON from its source file, through the per-conversation cache."""
        found = self.source_file(v, source)
        if found is None or not _safe(sim_id):
            return None
        path, sha = found
        key = (sha, sim_id)
        with self._lock:
            if key in self._sims:
                return self._sims[key]
            folder = self.paths.cache / sha[:16]
            one = folder / f"{sim_id}.json"
            if not (folder / ".done").exists():
                folder.mkdir(parents=True, exist_ok=True)
                for each in read_json(path).get("simulations", []):
                    (folder / f"{each['id']}.json").write_text(json.dumps(each))
                (folder / ".done").write_text(path.name + "\n")
            if not one.exists():
                return None
            sim: dict[str, Any] = dict(read_json(one))
            if len(self._sims) > 256:
                self._sims.clear()
            self._sims[key] = sim
            return sim

    # ---- runs --------------------------------------------------------------------------

    def runs(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Every run folder's run.json, newest first. A missing runs/ is no runs yet."""
        if not self.paths.runs.is_dir():
            return []
        out = []
        for d in self.paths.runs.iterdir():
            f = d / "run.json"
            if d.is_dir() and f.exists():
                try:
                    doc = read_json(f)
                except json.JSONDecodeError:
                    continue
                doc.setdefault("run_id", d.name)
                if doc.get("kind") in RUN_KINDS and (kind is None or doc.get("kind") == kind):
                    out.append(doc)
        return sorted(out, key=lambda r: (str(r.get("created") or ""), r["run_id"]), reverse=True)

    def run_dir(self, run_id: str) -> Path | None:
        d = self.paths.runs / run_id
        return d if _safe(run_id) and (d / "run.json").exists() else None

    def run(self, run_id: str) -> dict[str, Any] | None:
        d = self.run_dir(run_id)
        if d is None:
            return None
        doc: dict[str, Any] = read_json(d / "run.json")
        doc.setdefault("run_id", run_id)
        return doc

    def run_file(self, run_id: str, name: str) -> Path | None:
        d = self.run_dir(run_id)
        return d / name if d and (d / name).exists() else None

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        p = self.run_file(run_id, "metrics.jsonl")
        return read_jsonl(p) if p else []

    def mask(self, run_id: str) -> dict[str, Any] | None:
        p = self.run_file(run_id, "mask.json")
        doc: dict[str, Any] | None = read_json(p) if p else None
        return doc

    def served(self, run_id: str) -> list[dict[str, Any]]:
        p = self.run_file(run_id, "served.jsonl")
        return read_jsonl(p) if p else []

    def loop_results(self, run_id: str) -> list[dict[str, Any]]:
        p = self.run_file(run_id, "tau2_loop/results.jsonl")
        return read_jsonl(p) if p else []

    def loop_run(self, run_id: str) -> dict[str, Any] | None:
        p = self.run_file(run_id, "tau2_loop/run.json")
        doc: dict[str, Any] | None = read_json(p) if p else None
        return doc

    def loop_sim(self, run_id: str, task_id: str, trial: int) -> dict[str, Any] | None:
        """One scored conversation from the run's copy of tau2's results (trial is 1-based)."""
        p = self.run_file(run_id, "tau2_loop/tau2_results.json")
        if p is None:
            return None
        for sim in read_json(p).get("simulations", []):
            if sim.get("task_id") == task_id and int(sim.get("trial") or 0) + 1 == trial:
                found: dict[str, Any] = sim
                return found
        return None

    # ---- milestones --------------------------------------------------------------------

    def milestones(self) -> list[dict[str, Any]]:
        """The checklist in ai_specs/: one entry per milestone, ticked or not."""
        p = self.paths.checklist
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            m = MILESTONE.match(line)
            if m:
                out.append(
                    {
                        "id": m["id"],
                        "title": m["title"].strip(),
                        "done": m["done"] == "x",
                        "detail": m["rest"].strip(),
                    }
                )
        return out
