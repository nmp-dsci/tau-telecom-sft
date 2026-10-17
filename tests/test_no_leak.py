"""AGENTS.md rule 1: no test task id, and no test task's exact set of root issues, in anything
the trainer reads. The first test scans the real trainer inputs; the rest prove it bites."""

from __future__ import annotations

from pathlib import Path

from tau_telecom_sft import config, leak, split
from tau_telecom_sft.build.dataset import WITHHELD, Built


def _twin_of(test_id: str, s: split.Split) -> str:
    """The same root issues as a test task, under a different persona."""
    t = split.parse_task_id(test_id)
    persona = next(p for p in ("Easy", "Hard", "None") if p != t.persona)
    twin = f"[{t.symptom}]{'|'.join(sorted(t.issues))}[PERSONA:{persona}]"
    assert twin not in s.test_set
    return twin


def test_trainer_inputs_hold_no_test_task_or_twin() -> None:
    assert leak.scan(config.TRAINER_INPUTS) == []


def test_a_test_id_is_caught(tmp_path: Path) -> None:
    s = split.current()
    (tmp_path / "train.jsonl").write_text(f'{{"meta": {{"task_id": "{s.test[0]}"}}}}\n')
    leaks = leak.scan([tmp_path])
    assert [(lk.reason, lk.task_id, lk.line) for lk in leaks] == [("test task", s.test[0], 1)]


def test_a_test_twin_under_another_persona_is_caught(tmp_path: Path) -> None:
    s = split.current()
    twin = _twin_of(s.test[0], s)
    (tmp_path / "sft_a.yaml").write_text(f"# tasks\n#   - {twin}\n")
    leaks = leak.scan([tmp_path])
    assert [(lk.reason, lk.task_id, lk.line) for lk in leaks] == [("test twin", twin, 2)]


def test_training_ids_pass(tmp_path: Path) -> None:
    s = split.current()
    (tmp_path / "card.md").write_text("\n".join(f"| `{t}` | 4 |" for t in s.train))
    assert leak.scan([tmp_path]) == []


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "adapter.safetensors").write_bytes(b"\xff\xfe\x00\x81" * 64)
    assert leak.scan([tmp_path]) == []


def test_a_fixture_build_names_no_test_task(fixture_build: Built) -> None:
    """The fixture holds a pass on a test task; the build drops it and never writes its id."""
    assert leak.scan([fixture_build.out_dir]) == []
    drops = (fixture_build.out_dir / "drops.jsonl").read_text()
    assert WITHHELD in drops
