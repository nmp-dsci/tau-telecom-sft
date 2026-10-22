"""The scorer's arithmetic and the prompt-bytes check, offline."""

from __future__ import annotations

from tau_telecom_sft.score.pull import compare, pass_hat, sign_test
from tau_telecom_sft.serve.proxy import record, summarise


def test_pass_hat_is_taus_estimator() -> None:
    by_task = {"a": [1, 1, 1, 1], "b": [1, 0, 1, 0], "c": [0, 0, 0, 0]}
    ph = pass_hat(by_task)
    assert ph["1"] == round((1 + 0.5 + 0) / 3, 4)
    assert ph["2"] == round((1 + 1 / 6 + 0) / 3, 4)  # C(2,2)/C(4,2) = 1/6
    assert ph["4"] == round(1 / 3, 4)


def test_the_sign_test_is_one_sided_on_changed_tasks() -> None:
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 0) == 0.0312  # 1 / 32
    assert sign_test(3, 3) == 0.6562
    c = compare({"a": [1], "b": [0], "c": [1]}, {"a": [0], "b": [1], "c": [1]})
    assert (c["fixed"], c["broke"], c["same"], c["p"]) == (1, 1, 1, 0.75)


def test_the_prompt_bytes_check_needs_one_prompt_and_one_tool_set() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    reply = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"content": "", "tool_calls": [{"function": {"name": "f"}}]},
            }
        ]
    }
    line = record(body, reply, 0.1)
    assert line["tool_calls"] == 1 and not line["text_tool_call"]
    assert "hi" not in str(line) and line["system_prompt_sha256"] != "S"  # hashes, never text
    expected = {
        "system_prompt_sha256": line["system_prompt_sha256"],
        "tools_sha256": line["tools_sha256"],
    }
    assert summarise([line], expected)["prompt_bytes"]["ok"]
    other = record({**body, "messages": [{"role": "system", "content": "T"}]}, reply, 0.1)
    assert not summarise([line, other], expected)["prompt_bytes"]["ok"]
    text = record(body, {"choices": [{"message": {"content": "<tool_call>{}"}}]}, 0.1)
    assert text["text_tool_call"]
