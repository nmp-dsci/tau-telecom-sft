"""One loader, three kinds of file: the board's S1 and S2 and a tau2-loop telecom run (§4).

M1 loads each with tau2's own `Results` model at the pin and checks they have the same shape:
the same top-level parts, the same message roles, tool results marked by who asked for them,
and structured tool calls. The tau2-loop run is Claude's (S5): it is read here for its shape,
and never trained on (AGENTS.md rule 2). Its file's conversations are counted, nothing more.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tau_telecom_sft import config
from tau_telecom_sft.build.agent_view import agent_view, flatten
from tau_telecom_sft.sources.results import conversations

# tau2-loop's v2 agent on split v3's 74 training tasks (16 Oct): a train run, so the check reads
# no test conversation either.
TAU2_LOOP_RUN = "20261016T124631Z_telecom_v2_train"
CORE_SIM_KEYS = ("task_id", "trial", "messages", "reward_info", "termination_reason")
CORE_PARTS = ("info", "tasks", "simulations")


def tau2_loop_results(run: str = TAU2_LOOP_RUN) -> Path:
    return config.TAU2_LOOP_ROOT / "runs" / run / "tau2_results.json"


def profile(path: Path, source: str) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    convs = conversations(path, source)
    roles: Counter[str] = Counter()
    requestors: Counter[str] = Counter()
    calls = structured = 0
    seen = removed = 0
    for c in convs:
        flat = flatten(c.messages)
        view = agent_view(c.messages)
        seen += len(view)
        removed += len(flat) - len(view)
        for m in flat:
            roles[m.role] += 1
            if m.role == "tool":
                requestors[str(m.requestor)] += 1
            for tc in getattr(m, "tool_calls", None) or []:
                calls += 1
                structured += int(isinstance(tc.arguments, dict) and bool(tc.name) and bool(tc.id))
    sim_keys = sorted({k for s in raw["simulations"] for k in s})
    return {
        "source": source,
        "file": path.name,
        "top_level": sorted(raw),
        "conversations": len(convs),
        "core_sim_keys_present": all(k in sim_keys for k in CORE_SIM_KEYS),
        "roles": dict(sorted(roles.items())),
        "tool_results_by_requestor": dict(sorted(requestors.items())),
        "tool_calls": calls,
        "tool_calls_structured": structured,
        "agent_view_messages": seen,
        "agent_view_removed": removed,
    }


def differences(profiles: list[dict[str, Any]]) -> list[str]:
    """Parts one file has and another lacks: fields tau2 added after February (§4), which the
    loader does not need. Reported, not failed."""
    common = set.intersection(*(set(p["top_level"]) for p in profiles))
    return [
        f"{p['source']} also has {', '.join(sorted(set(p['top_level']) - common))}"
        for p in profiles
        if set(p["top_level"]) - common
    ]


def problems(profiles: list[dict[str, Any]]) -> list[str]:
    """Empty when every file has the same shape, so one loader and one agent view serve all."""
    out: list[str] = []
    first = profiles[0]
    for p in profiles:
        name = p["source"]
        if not set(CORE_PARTS) <= set(p["top_level"]):
            out.append(f"{name}: lacks one of the parts {CORE_PARTS}")
        if set(p["roles"]) != set(first["roles"]):
            out.append(f"{name}: roles {sorted(p['roles'])} != {sorted(first['roles'])}")
        if set(p["tool_results_by_requestor"]) != {"assistant", "user"}:
            out.append(f"{name}: tool results not split by requestor assistant/user")
        if not p["core_sim_keys_present"]:
            out.append(f"{name}: a conversation lacks one of {CORE_SIM_KEYS}")
        if p["tool_calls"] != p["tool_calls_structured"]:
            out.append(f"{name}: {p['tool_calls'] - p['tool_calls_structured']} unstructured calls")
        if p["agent_view_removed"] == 0:
            out.append(f"{name}: the agent view removed nothing (no phone calls found)")
    return out
