"""The leak check: no test task, and no test task's twin, in anything the trainer reads.

A twin is a task with a test task's exact set of root issues under any persona or symptom:
the same problem in different words (§3, F8). The scan reads every file under the trainer's
inputs (`config.TRAINER_INPUTS`: data/sft/ and configs/) and finds every telecom task id in
it, wherever it appears: a meta field, a card table, a config comment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tau_telecom_sft import config, split

# Any telecom task id in free text: `[symptom]issue|issue…[PERSONA:…]`
ANY_TASK_ID = re.compile(r"\[\w+\][\w|]*\[PERSONA:\w+\]")


@dataclass(frozen=True)
class Leak:
    path: Path
    line: int
    task_id: str
    reason: str  # "test task" | "test twin"

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}: {self.task_id}"


def scan_text(text: str, s: split.Split, path: Path = Path("<text>")) -> Iterator[Leak]:
    for n, line in enumerate(text.splitlines(), start=1):
        for m in ANY_TASK_ID.finditer(line):
            tid = m.group(0)
            if tid in s.test_set:
                yield Leak(path, n, tid, "test task")
            elif split.parse_task_id(tid).issues in s.test_issue_sets:
                yield Leak(path, n, tid, "test twin")


def _files(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(p for p in root.rglob("*") if p.is_file())


def scan(roots: Iterable[Path] = config.TRAINER_INPUTS, s: split.Split | None = None) -> list[Leak]:
    s = s or split.current()
    leaks: list[Leak] = []
    for path in _files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # weights and other binaries carry no task ids
        leaks.extend(scan_text(text, s, path))
    return leaks


def scanned_files(roots: Iterable[Path] = config.TRAINER_INPUTS) -> int:
    return sum(1 for _ in _files(roots))
