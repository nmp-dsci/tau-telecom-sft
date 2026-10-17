"""The agent's view of a conversation as one chat-and-tools row (§6).

One conversation is one row: the stock system prompt, every turn the agent saw, and the 13
agent tool schemas in `tools` so the student's own chat template places them. The loss goes on
every assistant turn (TRL's assistant_only_loss); customer turns and tool results are context.
`meta` is provenance for the card and the logs, never model input.

The system prompt is tau2's stock `SYSTEM_PROMPT` with the telecom policy (the main policy
plus the tech-support manual) at the pin, built the way tau2's own runner builds it. The
student is scored with exactly these bytes, so the card records their sha256.
"""

from __future__ import annotations

import functools
import hashlib
import json
from typing import Any

from tau_telecom_sft import config
from tau_telecom_sft.build.agent_view import agent_turns, agent_view
from tau_telecom_sft.sources.results import Conversation
from tau_telecom_sft.split import parse_task_id


@functools.cache
def _environment() -> Any:
    config.quiet_tau2()
    from tau2.domains.telecom.environment import get_environment

    return get_environment()


@functools.cache
def system_prompt() -> str:
    policy = _environment().get_policy()  # first, so tau2 is quietened before it is imported
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    prompt: str = SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=policy)
    return prompt


@functools.cache
def agent_tools() -> tuple[dict[str, Any], ...]:
    return tuple(t.openai_schema for t in _environment().get_tools())


@functools.cache
def agent_tool_names() -> frozenset[str]:
    return frozenset(t.name for t in _environment().get_tools())


@functools.cache
def phone_tool_names() -> frozenset[str]:
    """The customer's phone tools: the agent can only ask for these in words."""
    return frozenset(t.name for t in _environment().get_user_tools())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_fingerprint() -> dict[str, str]:
    """The bytes the student trains on, and must be served with (the prompt-bytes tests, M2, M3)."""
    return {
        "system_prompt_sha256": _sha256(system_prompt()),
        "tools_sha256": _sha256(json.dumps(agent_tools(), sort_keys=True)),
        "tau2_pin": config.TAU2_PIN,
    }


def _assistant(m: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
    if m.tool_calls:
        row["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in m.tool_calls
        ]
    return row


def to_messages(conv: Conversation) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt()}]
    for m in agent_view(conv.messages):
        if m.role == "assistant":
            out.append(_assistant(m))
        elif m.role == "user":
            out.append({"role": "user", "content": m.content or ""})
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.id, "content": m.content or ""})
    return out


def to_row(conv: Conversation, dataset: str) -> dict[str, Any]:
    return {
        "messages": to_messages(conv),
        "tools": list(agent_tools()),
        "meta": {
            "task_id": conv.task_id,
            "symptom": parse_task_id(conv.task_id).symptom,
            "source": conv.source,
            "trial": conv.trial,
            "reward": conv.reward,
            "agent_turns": agent_turns(conv.messages),
            "dataset": dataset,
        },
    }


def estimate_tokens(row: dict[str, Any]) -> int:
    """Characters / 4: a stand-in for the smoke build only. M1 counts with the student's
    tokenizer and chat template, and the card says which counter a build used."""
    return len(json.dumps({"messages": row["messages"], "tools": row["tools"]})) // 4
