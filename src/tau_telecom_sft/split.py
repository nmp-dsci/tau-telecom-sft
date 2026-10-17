"""tau2-loop's telecom split v3, copied once and pinned by its sha256.

The file is a byte-for-byte copy of tau2-loop's `data/splits/telecom.json` (version 3: 74
train, 40 test, seed 300, stratified by symptom and persona). `data/splits/manifest.json`
records its sha256 and where it came from, so the student is trained and scored on exactly
the tasks the Claude v2 agent played.

A telecom task id spells out its makeup: `[symptom]issue_a|issue_b|…[PERSONA:Easy]`. Two
tasks with the same set of root issues are the same problem for a different customer, so
the leak rule compares root-issue sets as well as ids (§3).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau_telecom_sft import config

TASK_ID = re.compile(r"^\[(?P<symptom>\w+)\](?P<issues>[\w|]*)\[PERSONA:(?P<persona>\w+)\]$")


@dataclass(frozen=True)
class TaskId:
    raw: str
    symptom: str
    issues: frozenset[str]
    persona: str


def parse_task_id(task_id: str) -> TaskId:
    m = TASK_ID.match(task_id)
    if not m:
        raise ValueError(f"not a telecom task id: {task_id!r}")
    issues = frozenset(i for i in m["issues"].split("|") if i)
    return TaskId(task_id, m["symptom"], issues, m["persona"])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Split:
    version: int
    train: tuple[str, ...]
    test: tuple[str, ...]
    sha256: str

    @functools.cached_property
    def train_set(self) -> frozenset[str]:
        return frozenset(self.train)

    @functools.cached_property
    def test_set(self) -> frozenset[str]:
        return frozenset(self.test)

    @functools.cached_property
    def test_issue_sets(self) -> frozenset[frozenset[str]]:
        """Every test task's exact set of root issues, under any persona or symptom."""
        return frozenset(parse_task_id(t).issues for t in self.test)

    def is_test_twin(self, task_id: str) -> bool:
        """True if the id is a test task, or shares a test task's exact set of root issues."""
        if task_id in self.test_set:
            return True
        return parse_task_id(task_id).issues in self.test_issue_sets


def load(path: Path = config.SPLIT_PATH) -> Split:
    doc: dict[str, Any] = json.loads(path.read_text())
    return Split(
        version=int(doc["version"]),
        train=tuple(doc["train"]),
        test=tuple(doc["test"]),
        sha256=sha256_file(path),
    )


@functools.cache
def current() -> Split:
    return load()


def manifest() -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(config.SPLIT_MANIFEST.read_text())
    return doc


@dataclass(frozen=True)
class SplitCheck:
    sha256: str
    recorded: str
    sibling: str | None  # tau2-loop's copy, when the sibling is checked out

    @property
    def ok(self) -> bool:
        return self.sha256 == self.recorded and self.sibling in (None, self.sha256)


def check() -> SplitCheck:
    """Our copy's sha256 against the manifest, and against tau2-loop's file when present."""
    sibling = sha256_file(config.TAU2_LOOP_SPLIT) if config.TAU2_LOOP_SPLIT.exists() else None
    return SplitCheck(sha256_file(config.SPLIT_PATH), str(manifest()["sha256"]), sibling)


def problems(s: Split) -> list[str]:
    """What would make the split unusable: overlap, twins across sides, or a wrong shape."""
    out: list[str] = []
    if s.version != config.SPLIT_VERSION:
        out.append(f"split version {s.version}, expected {config.SPLIT_VERSION}")
    if len(s.train) != 74 or len(s.test) != 40:
        out.append(f"{len(s.train)}/{len(s.test)} tasks, expected 74/40")
    if both := s.train_set & s.test_set:
        out.append(f"{len(both)} tasks on both sides")
    twins = [t for t in s.train if parse_task_id(t).issues in s.test_issue_sets]
    if twins:
        out.append(f"{len(twins)} train tasks share a test task's root issues: {twins[:3]}")
    return out
