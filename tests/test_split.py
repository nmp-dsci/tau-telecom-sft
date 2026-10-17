"""The split is tau2-loop's split v3, byte for byte, and it is leak-free on its own."""

from __future__ import annotations

from collections import Counter

import pytest

from tau_telecom_sft import config, split


def test_split_matches_its_recorded_sha256() -> None:
    c = split.check()
    assert c.sha256 == c.recorded == split.manifest()["sha256"]


@pytest.mark.skipif(not config.TAU2_LOOP_SPLIT.exists(), reason="tau2-loop is not checked out")
def test_split_matches_tau2_loops_split_v3() -> None:
    c = split.check()
    assert c.sibling == c.sha256


def test_split_is_v3_with_74_train_and_40_test() -> None:
    s = split.current()
    assert (s.version, len(s.train), len(s.test)) == (3, 74, 40)
    assert split.problems(s) == []


def test_symptom_counts_match_the_plan() -> None:
    s = split.current()
    train = Counter(split.parse_task_id(t).symptom for t in s.train)
    test = Counter(split.parse_task_id(t).symptom for t in s.test)
    assert train == {"mms_issue": 32, "mobile_data_issue": 22, "service_issue": 20}
    assert test == {"mms_issue": 17, "mobile_data_issue": 14, "service_issue": 9}


def test_every_test_root_issue_is_taught_in_train() -> None:
    """§1: test tasks recombine known faults; none is new."""
    s = split.current()
    taught = set().union(*(split.parse_task_id(t).issues for t in s.train))
    tested = set().union(*(split.parse_task_id(t).issues for t in s.test))
    assert tested <= taught
    assert len(taught) == 19


def test_parse_task_id() -> None:
    t = split.parse_task_id("[mms_issue]airplane_mode_on|bad_wifi_calling[PERSONA:Hard]")
    assert (t.symptom, t.issues, t.persona) == (
        "mms_issue",
        frozenset({"airplane_mode_on", "bad_wifi_calling"}),
        "Hard",
    )
    with pytest.raises(ValueError):
        split.parse_task_id("airline_task_7")
