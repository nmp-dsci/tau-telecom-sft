"""The fixture build end to end: agent view, F1 to F10, the row format and the card."""

from __future__ import annotations

import json
from dataclasses import replace

from tau_telecom_sft import split
from tau_telecom_sft.build import filters
from tau_telecom_sft.build.agent_view import agent_view
from tau_telecom_sft.build.convert import (
    agent_tool_names,
    agent_tools,
    phone_tool_names,
    prompt_fingerprint,
    system_prompt,
    to_row,
)
from tau_telecom_sft.build.dataset import Built
from tau_telecom_sft.sources.results import Conversation


def test_telecom_has_13_agent_tools_and_30_phone_tools() -> None:
    assert len(agent_tool_names()) == 13
    assert len(phone_tool_names()) == 30
    assert not agent_tool_names() & phone_tool_names()


def test_agent_view_drops_the_customers_phone(fixture_convs: list[Conversation]) -> None:
    conv = fixture_convs[0]
    phone_results = [m for m in conv.messages if m.role == "tool" and m.requestor == "user"]
    assert phone_results, "the fixture should exercise the customer's phone tools"
    view = agent_view(conv.messages)
    assert all(not (m.role == "tool" and m.requestor == "user") for m in view)
    assert all(not (m.role == "user" and m.tool_calls) for m in view)
    assert any(m.role == "tool" and m.requestor == "assistant" for m in view)


def test_every_rule_drops_what_the_fixture_planted(fixture_build: Built) -> None:
    st = fixture_build.stats
    dropped = {f["rule"]: f["dropped"] for f in st["filters"]}
    assert dropped == {
        "F1": 1, "F2": 1, "F3": 1, "F4": 1, "F5": 1,
        "F6": 1, "F7": 1, "F8": 0, "F9": 1, "F10": 1,
    }  # fmt: skip
    assert (st["n_loaded"], st["n_kept"]) == (17, 8)
    assert sum(dropped.values()) + st["n_kept"] == st["n_loaded"]


def test_f8_drops_a_pool_twin_when_the_pool_is_on(fixture_convs: list[Conversation]) -> None:
    """D6: generated-pool tasks may enter, but never a test task's twin."""
    s = split.current()
    t = split.parse_task_id(s.test[0])
    twin = f"[{t.symptom}]{'|'.join(sorted(t.issues))}[PERSONA:{'Hard' if t.persona != 'Hard' else 'Easy'}]"
    conv = fixture_convs[0]
    pool_conv = replace(conv, sim=conv.sim.model_copy(update={"task_id": twin}))
    ctx = filters.Context.build(
        [pool_conv], split=s, agent_tools=agent_tool_names(), count_tokens=lambda c: 0,
        max_tokens=1, allow_pool=True,
    )  # fmt: skip
    assert filters.f2_training_task(pool_conv, ctx) is None
    assert filters.f8_no_test_twin(pool_conv, ctx) == "shares a test task's root issues"


def test_f3_drops_an_unexpected_transfer(fixture_convs: list[Conversation]) -> None:
    from tau2.data_model.message import AssistantMessage, ToolCall

    conv = fixture_convs[0]
    transfer = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="t1", name="transfer_to_human_agents", arguments={"summary": "x"})],
    )
    moved = replace(conv, sim=conv.sim.model_copy(update={"messages": [*conv.messages, transfer]}))
    ctx = filters.Context.build(
        [moved], split=split.current(), agent_tools=agent_tool_names(), count_tokens=lambda c: 0,
        max_tokens=1,
    )  # fmt: skip
    assert filters.f3_ended_normally(moved, ctx) == (
        "transferred to a human when the task does not expect it"
    )


def test_text_tool_call_pattern() -> None:
    hit = filters.TEXT_TOOL_CALL.search
    assert hit('<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>')
    assert hit('{"name": "get_data_usage", "arguments": {"line_id": "L1"}}')
    assert not hit("Please check the name on your account and your data usage.")


def test_a_row_is_the_agents_view_in_chat_format(fixture_convs: list[Conversation]) -> None:
    row = to_row(fixture_convs[0], "smoke")
    msgs = row["messages"]
    assert msgs[0] == {"role": "system", "content": system_prompt()}
    assert msgs[1] == {"role": "assistant", "content": "Hi! How can I help you today?"}
    assert {m["role"] for m in msgs} == {"system", "assistant", "user", "tool"}
    calls = [tc for m in msgs for tc in m.get("tool_calls", [])]
    assert all(isinstance(tc["function"]["arguments"], str) for tc in calls)
    assert {tc["function"]["name"] for tc in calls} <= agent_tool_names()
    assert all(m["content"] for m in msgs if m["role"] == "user")  # no empty phone turns
    assert row["tools"] == list(agent_tools())
    assert set(row["meta"]) == {
        "task_id",
        "symptom",
        "source",
        "trial",
        "reward",
        "agent_turns",
        "dataset",
    }


def test_system_prompt_is_taus_stock_prompt() -> None:
    p = system_prompt()
    assert p.startswith("<instructions>\nYou are a customer service agent")
    assert "<main_policy>" in p and "</policy>" in p
    assert set(prompt_fingerprint()) == {"system_prompt_sha256", "tools_sha256", "tau2_pin"}


def test_the_card_adds_up(fixture_build: Built) -> None:
    card = (fixture_build.out_dir / "card.md").read_text()
    st = fixture_build.stats
    assert f"**{st['n_kept']} kept**" in card
    assert st["train_sha256"] in card
    rows = (fixture_build.out_dir / "train.jsonl").read_text().splitlines()
    assert len(rows) == st["n_kept"]
    assert all(json.loads(r)["meta"]["reward"] == 1.0 for r in rows)
