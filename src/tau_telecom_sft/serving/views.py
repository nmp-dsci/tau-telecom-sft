"""What each lens shows: a conversation as the agent saw it, as raw tau2 wrote it, or as the
exact text TRL trains on with the trained tokens marked; and the numbers a scored run needs.

The chat lens is the row itself (kept) or the converter's output rebuilt from raw (dropped).
The raw lens is every message, each MultiToolMessage split, with whether the agent saw it
(tau2's own `is_valid_agent_history_message`) and whether it broke the row's rule
(`filters.offending`). The tokens lens renders the row with TRL's training copy of the
student's chat template and returns the text cut where the assistant mask changes: the same
template and mask `train/sft.py` trains with.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable
from typing import Any

from tau_telecom_sft import config
from tau_telecom_sft.build.tokens import STUDENT_TOKENIZER


@functools.cache
def _tau2() -> Any:
    config.quiet_tau2()
    from tau2.agent.base_agent import is_valid_agent_history_message
    from tau2.data_model.simulation import SimulationRun

    return SimulationRun, is_valid_agent_history_message


def conversation(source: str, sim: dict[str, Any]) -> Any:
    """A raw simulation as the build's `Conversation` (no task: the lenses never need one)."""
    from tau_telecom_sft.sources.results import Conversation

    simulation_run, _ = _tau2()
    return Conversation(source, simulation_run.model_validate(sim), None)


def _calls(m: Any) -> list[dict[str, Any]]:
    return [
        {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in (m.tool_calls or [])
    ]


def raw_messages(conv: Any, rule: str | None = None) -> list[dict[str, Any]]:
    from tau_telecom_sft.build import filters
    from tau_telecom_sft.build.convert import agent_tool_names

    _, valid = _tau2()
    marked = set(filters.offending(conv, rule, agent_tool_names())) if rule else set()
    out = []
    for i, m in enumerate(filters.flat_messages(conv)):
        out.append(
            {
                "i": i,
                "role": m.role,
                "content": getattr(m, "content", None),
                "tool_calls": _calls(m) if getattr(m, "tool_calls", None) else [],
                "requestor": getattr(m, "requestor", None),
                "error": bool(getattr(m, "error", False)),
                "tool_call_id": m.id if m.role == "tool" else None,
                "in_agent_view": bool(valid(m)),
                "offending": i in marked,
            }
        )
    return out


def chat_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The row's messages for display: tool-call arguments stay the JSON strings trained on."""
    return [
        {
            "role": m["role"],
            "content": m.get("content") or "",
            "tool_calls": [
                {
                    "id": tc.get("id"),
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for tc in (m.get("tool_calls") or [])
            ],
            "tool_call_id": m.get("tool_call_id"),
        }
        for m in row["messages"]
    ]


@functools.cache
def _training_template() -> tuple[Any, str]:
    from trl.chat_template_utils import get_training_chat_template

    from tau_telecom_sft.build import tokens

    tok = tokens.tokenizer(STUDENT_TOKENIZER)
    tmpl = get_training_chat_template(tok) or tok.chat_template
    return tok, str(tmpl)


def token_segments(row: dict[str, Any]) -> dict[str, Any]:
    """The rendered row cut where the loss mask changes: [{text, trained}], plus the counts."""
    tok, tmpl = _training_template()
    enc = tok.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        chat_template=tmpl,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids: list[int] = list(enc["input_ids"])
    mask: list[int] = list(enc["assistant_masks"])
    segments: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or mask[i] != mask[start]:
            segments.append({"text": tok.decode(ids[start:i]), "trained": bool(mask[start])})
            start = i
    return {
        "tokenizer": STUDENT_TOKENIZER,
        "template": "TRL's training copy of the student's chat template",
        "n_tokens": len(ids),
        "n_trained": int(sum(mask)),
        "segments": segments,
    }


# ---- scores ------------------------------------------------------------------------------


def pass_hat(by_task: dict[str, list[int]], k: int) -> float | None:
    """tau2's pass^k: per task, C(c, k) / C(n, k) over its n trials with c passes, averaged."""
    vals = []
    for trials in by_task.values():
        n, c = len(trials), sum(trials)
        if n < k:
            return None
        vals.append(math.comb(c, k) / math.comb(n, k))
    return float(sum(vals) / len(vals)) if vals else None


def sign_test(fixed: int, broke: int) -> float | None:
    """One-sided exact sign test: P(X >= fixed) for X ~ Bin(fixed + broke, 1/2)."""
    n = fixed + broke
    if n == 0:
        return None
    return float(sum(math.comb(n, i) for i in range(fixed, n + 1)) / 2**n)


def by_task(rows: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    """tau2-loop's results.jsonl rows → task → [pass per trial, in trial order]."""
    out: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        out.setdefault(str(r["task_id"]), []).append(
            (int(r.get("trial") or 1), 1 if r.get("correct") else 0)
        )
    return {t: [p for _, p in sorted(v)] for t, v in out.items()}


def compare(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, Any]:
    """Run a against base b, paired by task on the tasks both played: a task's score is its pass
    fraction; fixed = a higher, broke = a lower. The same gate tau2-loop uses."""
    shared = sorted(set(a) & set(b))
    rows = []
    fixed = broke = 0
    for t in shared:
        sa, sb = sum(a[t]) / len(a[t]), sum(b[t]) / len(b[t])
        change = "fixed" if sa > sb else "broke" if sa < sb else "same"
        fixed += change == "fixed"
        broke += change == "broke"
        rows.append({"task_id": t, "a": a[t], "b": b[t], "change": change})
    return {
        "n_tasks": len(shared),
        "fixed": fixed,
        "broke": broke,
        "same": len(shared) - fixed - broke,
        "p": sign_test(fixed, broke),
        "tasks": rows,
    }
