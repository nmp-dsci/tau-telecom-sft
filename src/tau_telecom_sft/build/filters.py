"""F1 to F10 (§5): the rules a conversation must pass, each a pure function.

A rule returns None to keep a conversation, or the reason it drops it. They run in order and
each drop is recorded against the first rule it failed, so the card prints one row per rule:
how many came in, how many were dropped, and why. A pass alone is not enough, which was the
talk's main lesson: F3 to F10 drop passes that taught something wrong.

Two rules need the whole pool, not one conversation: F6 compares a conversation's agent turns
with the median of its task's passes, and F9 keeps at most three conversations per task that
make the same agent tool calls. `Context.build` computes F6's medians once, before any rule
runs; F9 keeps the first three in `Conversation.key` order.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tau_telecom_sft import split as split_mod
from tau_telecom_sft.build.agent_view import agent_tool_calls, agent_turns, agent_view
from tau_telecom_sft.sources.results import Conversation

TRANSFER_TOOL = "transfer_to_human_agents"
NORMAL_ENDINGS = frozenset({"user_stop", "agent_stop"})

# A tool call written into a reply instead of made (F7): a call tag, or a JSON call object.
TEXT_TOOL_CALL = re.compile(
    r"</?tool_call>|<function[=_ ]|\"tool_calls\"\s*:|\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"(arguments|parameters)\"\s*:"
)


@dataclass(frozen=True)
class Rule:
    id: str
    keep_if: str
    why: str


RULES: tuple[Rule, ...] = (
    Rule("F1", "tau2 scored it a pass (reward 1)", "failed conversations teach wrong diagnoses"),
    Rule("F2", "its task is one of split v3's 74 training tasks", "a test task voids the result"),
    Rule(
        "F3",
        "it ended normally: the customer stopped, or the agent transferred when the task expects it",
        "a conversation cut off by a limit shows no finished job",
    ),
    Rule(
        "F4",
        "every agent tool call names one of the 13 agent tools",
        "invented tools teach inventing",
    ),
    Rule("F5", "no agent tool result is an error", "a recovered bad call teaches the bad call too"),
    Rule(
        "F6",
        "its agent turns are at most 1.5 x the median of its task's passes",
        "wandering teaches wandering",
    ),
    Rule("F7", "no reply holds a tool call written as text", "a written-out call runs nothing"),
    Rule(
        "F8",
        "not a test twin: no test task's exact set of root issues",
        "a twin leaks test into training",
    ),
    Rule(
        "F9",
        "at most 3 conversations per task with the same agent tool calls",
        "copies teach memorising",
    ),
    Rule("F10", "it fits the length limit once rendered", "a cut row loses its ending"),
)


@dataclass(frozen=True)
class Context:
    split: split_mod.Split
    agent_tools: frozenset[str]
    median_turns: Mapping[str, float]
    count_tokens: Callable[[Conversation], int]
    max_tokens: int
    allow_pool: bool = False  # D6: generated-pool tasks may enter, minus any test twin (F8)
    max_turn_ratio: float = 1.5
    max_same_path: int = 3

    @classmethod
    def build(
        cls,
        convs: Iterable[Conversation],
        *,
        split: split_mod.Split,
        agent_tools: frozenset[str],
        count_tokens: Callable[[Conversation], int],
        max_tokens: int,
        allow_pool: bool = False,
    ) -> Context:
        by_task: dict[str, list[int]] = {}
        for c in convs:
            if c.reward == 1.0:
                by_task.setdefault(c.task_id, []).append(agent_turns(c.messages))
        medians = {t: float(statistics.median(v)) for t, v in by_task.items()}
        return cls(split, agent_tools, medians, count_tokens, max_tokens, allow_pool)


def f1_pass(c: Conversation, ctx: Context) -> str | None:
    return None if c.reward == 1.0 else f"reward {c.reward}"


def f2_training_task(c: Conversation, ctx: Context) -> str | None:
    if c.task_id in ctx.split.train_set:
        return None
    if c.task_id in ctx.split.test_set:
        return "a test task"
    return None if ctx.allow_pool else "not a split v3 training task"


def _expects_transfer(c: Conversation) -> bool:
    criteria = getattr(c.task, "evaluation_criteria", None)
    actions = getattr(criteria, "actions", None) or []
    return any(a.name == TRANSFER_TOOL for a in actions)


def f3_ended_normally(c: Conversation, ctx: Context) -> str | None:
    if c.termination not in NORMAL_ENDINGS:
        return f"ended by {c.termination}"
    transferred = any(tc.name == TRANSFER_TOOL for tc in agent_tool_calls(c.messages))
    if transferred and not _expects_transfer(c):
        return "transferred to a human when the task does not expect it"
    return None


def f4_known_tools(c: Conversation, ctx: Context) -> str | None:
    unknown = sorted({tc.name for tc in agent_tool_calls(c.messages)} - ctx.agent_tools)
    return f"called {', '.join(unknown)}" if unknown else None


def f5_no_tool_errors(c: Conversation, ctx: Context) -> str | None:
    errors = [m for m in agent_view(c.messages) if m.role == "tool" and m.error]
    return f"{len(errors)} agent tool error(s)" if errors else None


def f6_not_long(c: Conversation, ctx: Context) -> str | None:
    median = ctx.median_turns.get(c.task_id)
    turns = agent_turns(c.messages)
    if median is not None and turns > ctx.max_turn_ratio * median:
        return f"{turns} agent turns, over {ctx.max_turn_ratio} x the task median {median:g}"
    return None


def f7_no_text_tool_calls(c: Conversation, ctx: Context) -> str | None:
    for m in agent_view(c.messages):
        if m.role == "assistant" and m.content and TEXT_TOOL_CALL.search(m.content):
            return "a reply holds a tool call written as text"
    return None


def f8_no_test_twin(c: Conversation, ctx: Context) -> str | None:
    return "shares a test task's root issues" if ctx.split.is_test_twin(c.task_id) else None


def f10_fits(c: Conversation, ctx: Context) -> str | None:
    n = ctx.count_tokens(c)
    return f"{n} tokens, over {ctx.max_tokens}" if n > ctx.max_tokens else None


def tool_path(c: Conversation) -> tuple[tuple[str, str], ...]:
    """F9's signature: the agent's tool calls and arguments, in order."""
    return tuple(
        (tc.name, json.dumps(tc.arguments, sort_keys=True)) for tc in agent_tool_calls(c.messages)
    )


def flat_messages(c: Conversation) -> list[Any]:
    """Every message in order, each MultiToolMessage split into its results: the raw lens's rows."""
    from tau_telecom_sft.build.agent_view import flatten

    return flatten(c.messages)


def offending(c: Conversation, rule: str, agent_tools: frozenset[str]) -> list[int]:
    """The messages (indexes into `flat_messages`) that broke a per-turn rule, for the viewer
    to mark: F4's unknown tool calls, F5's tool errors, F7's written-out calls. Whole-
    conversation rules (F1–F3, F6, F8–F10) mark nothing."""
    from tau2.agent.base_agent import is_valid_agent_history_message

    def broke(m: Any) -> bool:
        if rule == "F4" and m.role == "assistant":
            return any(tc.name not in agent_tools for tc in (m.tool_calls or []))
        if rule == "F5" and m.role == "tool":
            return bool(m.error)
        if rule == "F7" and m.role == "assistant" and m.content:
            return TEXT_TOOL_CALL.search(m.content) is not None
        return False

    out: list[int] = []
    for i, m in enumerate(flat_messages(c)):
        if is_valid_agent_history_message(m) and broke(m):
            out.append(i)
    return out


PER_CONVERSATION: dict[str, Callable[[Conversation, Context], str | None]] = {
    "F1": f1_pass,
    "F2": f2_training_task,
    "F3": f3_ended_normally,
    "F4": f4_known_tools,
    "F5": f5_no_tool_errors,
    "F6": f6_not_long,
    "F7": f7_no_text_tool_calls,
    "F8": f8_no_test_twin,
    "F10": f10_fits,
}


@dataclass(frozen=True)
class Drop:
    conv: Conversation
    rule: str
    reason: str


@dataclass(frozen=True)
class RuleCount:
    rule: Rule
    n_in: int
    n_dropped: int


@dataclass
class Outcome:
    kept: list[Conversation] = field(default_factory=list)
    drops: list[Drop] = field(default_factory=list)
    table: list[RuleCount] = field(default_factory=list)


def apply(convs: Iterable[Conversation], ctx: Context) -> Outcome:
    """Run F1 to F10 in order; each drop is charged to the first rule it failed."""
    survivors = sorted(convs, key=lambda c: c.key)
    out = Outcome()
    for rule in RULES:
        n_in = len(survivors)
        nxt: list[Conversation] = []
        if rule.id == "F9":
            seen: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
            for c in survivors:
                sig = (c.task_id, tool_path(c))
                seen[sig] += 1
                if seen[sig] > ctx.max_same_path:
                    out.drops.append(Drop(c, "F9", f"copy {seen[sig]} of one tool path"))
                else:
                    nxt.append(c)
        else:
            fn = PER_CONVERSATION[rule.id]
            for c in survivors:
                reason = fn(c, ctx)
                if reason is None:
                    nxt.append(c)
                else:
                    out.drops.append(Drop(c, rule.id, reason))
        out.table.append(RuleCount(rule, n_in, n_in - len(nxt)))
        survivors = nxt
    out.kept = survivors
    return out
