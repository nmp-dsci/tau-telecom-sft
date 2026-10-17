"""The agent's view of a conversation, by tau2's own rule.

A telecom conversation has two tool users: the agent and the customer's phone. tau2 shows the
agent only its own calls and results and the customer's text replies
(`is_valid_agent_history_message` in tau2's base_agent.py). The converter applies that same
function, so the customer's phone calls and their results never reach a training row (§6).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tau_telecom_sft import config


def _flatten(messages: Iterable[Any]) -> list[Any]:
    """A MultiToolMessage holds several tool results; the agent sees each one."""
    config.quiet_tau2()
    from tau2.data_model.message import MultiToolMessage

    out: list[Any] = []
    for m in messages:
        if isinstance(m, MultiToolMessage):
            out.extend(m.tool_messages)
        else:
            out.append(m)
    return out


def agent_view(messages: Iterable[Any]) -> list[Any]:
    from tau2.agent.base_agent import is_valid_agent_history_message

    return [m for m in _flatten(messages) if is_valid_agent_history_message(m)]


def agent_tool_calls(messages: Iterable[Any]) -> list[Any]:
    """Every tool call the agent made, in order."""
    return [
        tc
        for m in agent_view(messages)
        if getattr(m, "role", None) == "assistant"
        for tc in (m.tool_calls or [])
    ]


def agent_turns(messages: Iterable[Any]) -> int:
    return sum(1 for m in agent_view(messages) if getattr(m, "role", None) == "assistant")
