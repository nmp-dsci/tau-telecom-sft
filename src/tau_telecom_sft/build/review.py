"""M1's sign-off: a person reads 10 kept and 10 dropped conversations before any training.

The sample is drawn once per dataset version with a fixed seed and committed as
`review_sample.json`. Dropped rows are drawn one rule at a time, so every rule that dropped a
training-task conversation is read at least once. A test task's drop is never drawn: it is
WITHHELD in drops.jsonl. The viewer's one write route appends to `review.jsonl`
(append-only, committed; D10): a verdict per row, then one sign-off.

A person may also sign off without reading every row, on other evidence. That sign-off is
written from here, never from the viewer, with `unread_ok`, and it records how many sampled
rows had no verdict, so the log never claims a read that did not happen.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEED = 300
N_EACH = 10
VERDICTS = ("good", "bad", "note")
SAMPLE = "review_sample.json"
LOG = "review.jsonl"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw(out_dir: Path, seed: int = SEED, n: int = N_EACH) -> dict[str, Any]:
    """Draw the sample once; an existing sample is returned unchanged."""
    path = out_dir / SAMPLE
    if path.exists():
        existing: dict[str, Any] = json.loads(path.read_text())
        return existing
    rng = random.Random(seed)
    kept = sorted(r["meta"]["sim_id"] for r in _rows(out_dir / "train.jsonl"))
    by_rule: dict[str, list[str]] = defaultdict(list)
    for d in _rows(out_dir / "drops.jsonl"):
        if d.get("sim_id"):
            by_rule[d["rule"]].append(d["sim_id"])
    for ids in by_rule.values():
        ids.sort()
        rng.shuffle(ids)
    dropped: list[str] = []
    while len(dropped) < n and any(by_rule.values()):
        for rule in sorted(by_rule, key=lambda r: int(r[1:])):
            if by_rule[rule] and len(dropped) < n:
                dropped.append(by_rule[rule].pop())
    sample = {
        "seed": seed,
        "drawn": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kept": rng.sample(kept, min(n, len(kept))),
        "dropped": dropped,
    }
    path.write_text(json.dumps(sample, indent=2) + "\n")
    return sample


def log(out_dir: Path) -> list[dict[str, Any]]:
    return _rows(out_dir / LOG)


def append(out_dir: Path, record: dict[str, Any], unread_ok: bool = False) -> dict[str, Any]:
    """One verdict on a sampled row, or the sign-off once every row has one (or, with
    `unread_ok`, a sign-off that records how many rows were not read)."""
    sample = draw(out_dir)
    sampled = set(sample["kept"]) | set(sample["dropped"])
    kind = record.get("kind", "verdict")
    missing: set[str] = set()
    if kind == "verdict":
        if record.get("sim_id") not in sampled:
            raise ValueError("only a sampled row can be reviewed")
        if record.get("verdict") not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
    elif kind == "sign-off":
        missing = sampled - {r["sim_id"] for r in log(out_dir) if r.get("kind") == "verdict"}
        if missing and not unread_ok:
            raise ValueError(f"{len(missing)} sampled row(s) have no verdict yet")
        if not str(record.get("note") or "").strip() and missing:
            raise ValueError("a sign-off without every verdict needs a note saying why")
    else:
        raise ValueError("kind must be verdict or sign-off")
    rec = {
        "kind": kind,
        "sim_id": record.get("sim_id"),
        "verdict": record.get("verdict"),
        "note": str(record.get("note") or "")[:2000],
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if kind == "sign-off":
        rec["unread"] = len(missing)
    with (out_dir / LOG).open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def status(out_dir: Path) -> dict[str, Any]:
    """Read-only: the sample as committed, each row's latest verdict, and the sign-off."""
    path = out_dir / SAMPLE
    sample = json.loads(path.read_text()) if path.exists() else {"kept": [], "dropped": []}
    latest: dict[str, dict[str, Any]] = {}
    signed: dict[str, Any] | None = None
    for r in log(out_dir):
        if r["kind"] == "verdict":
            latest[r["sim_id"]] = r
        elif r["kind"] == "sign-off":
            signed = r
    sampled = list(sample["kept"]) + list(sample["dropped"])
    return {
        "sample": sample,
        "latest": latest,
        "reviewed": sum(1 for s in sampled if s in latest),
        "total": len(sampled),
        "signed_off": signed,
    }
