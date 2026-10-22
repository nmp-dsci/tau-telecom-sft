"""The dataset card: every build's sources, filter table, counts and lengths, as markdown.

`data/sft/<version>/card.md` is committed; it is rendered from the build's stats.json and
never edited by hand. A version never changes: a new rule means a new version (§6).
"""

from __future__ import annotations

from typing import Any


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _length_section(st: dict[str, Any]) -> list[str]:
    hist = st.get("tokens_hist") or []
    over = st.get("over_limit") or {}
    if not hist:
        return []
    top = max(h["n"] for h in hist)
    out = [
        "### Histogram (1,024-token bins)",
        "",
        "```",
        *(
            f"{h['from']:>6,} {'#' * max(1 if h['n'] else 0, round(40 * h['n'] / top))} {h['n']}"
            for h in hist
        ),
        "```",
        "",
    ]
    if over:
        out += [
            f"### What a limit would cut, of the {st.get('reached_f10', 0)} rows that reached F10",
            "",
            *_table(["limit", "rows over it"], [[f"{int(k):,}", v] for k, v in over.items()]),
            "",
        ]
    return out


CHECK_ROWS = (
    ("agent_llm", "agent model, as tau2 recorded it"),
    ("agent_llm_args", "agent sampling, as tau2 recorded it"),
    ("customer_llm", "customer model"),
    ("customer_llm_args", "customer settings"),
    ("tau2_commit", "tau2 commit the board ran"),
    ("num_trials", "trials"),
    ("phone_tool_results", "customer phone-tool results in the file (removed from every row)"),
    ("agent_tool_results", "agent tool results"),
    ("assistant_turns_with_reasoning", "assistant turns with reasoning in raw_data (dropped)"),
    (
        "assistant_turns_with_text_and_calls",
        "assistant turns with both text and tool calls (kept as is)",
    ),
    ("system_prompt_stored", "system prompt stored in the file"),
    ("policy_matches_pin", "the file's policy is the pin's, byte for byte"),
    ("licence", "licence"),
)


def _checks_section(st: dict[str, Any]) -> list[str]:
    checks = st.get("source_checks") or []
    if not checks:
        return []
    return [
        "## Source checks (§4: five things M1 checks after downloading)",
        "",
        *_table(
            ["", *(c["source"] + " " + c["model"] for c in checks)],
            [[label, *(f"`{c.get(k)}`" for c in checks)] for k, label in CHECK_ROWS],
        ),
        "",
    ]


def render(st: dict[str, Any]) -> str:
    tok = st["tokens"]
    lines: list[str] = [
        f"# Dataset {st['dataset']}: tau-telecom-sft",
        "",
        f"Built {st['built']} at code `{st['code_sha']}` from {st['n_loaded']} conversations in "
        f"{len(st['sources'])} source(s): **{st['n_kept']} kept**, {st['n_dropped']} dropped. "
        f"{st['training_tasks_covered']} of the {st['training_tasks_total']} training tasks have "
        "at least one row.",
        "",
        *_table(
            ["", ""],
            [
                ["train.jsonl sha256", f"`{st['train_sha256']}`"],
                ["split", f"v{st['split_version']}, sha256 `{st['split_sha256']}`"],
                ["tau2 pin", f"`{st['tau2_pin']}`"],
                ["system prompt sha256", f"`{st['system_prompt_sha256']}`"],
                ["tools sha256", f"`{st['tools_sha256']}`"],
                ["token counter", f"{st['token_counter']}; limit {st['max_tokens']:,}"],
            ],
        ),
        "",
        "## Sources",
        "",
        *_table(
            ["id", "source", "file", "bytes", "sha256", "conversations"],
            [
                [
                    s["id"],
                    s["label"],
                    s["file"],
                    f"{s['bytes']:,}",
                    f"`{s['sha256'][:16]}…`",
                    s["conversations"],
                ]
                for s in st["sources"]
            ],
        ),
        "",
        "## Filters, in order",
        "",
        "Each drop is charged to the first rule it failed.",
        "",
        *_table(
            ["rule", "keep a conversation only if", "in", "dropped", "kept"],
            [
                [f["rule"], f["keep_if"], f["in"], f["dropped"], f["in"] - f["dropped"]]
                for f in st["filters"]
            ],
        ),
        "",
        "### Why they were dropped",
        "",
        *_table(
            ["rule: reason", "conversations"],
            [[k, v] for k, v in sorted(st["drop_reasons"].items())],
        ),
        "",
        "## What was kept",
        "",
        "### By symptom",
        "",
        *_table(["symptom", "rows"], [[k, v] for k, v in st["kept_by_symptom"].items()]),
        "",
        "### By root issue",
        "",
        *_table(["root issue", "rows"], [[k, v] for k, v in st["kept_by_root_issue"].items()]),
        "",
        "### By source",
        "",
        *_table(["source", "rows"], [[k, v] for k, v in st["kept_by_source"].items()]),
        "",
        "### By task",
        "",
        *_table(["task", "rows"], [[f"`{k}`", v] for k, v in st["kept_by_task"].items()]),
        "",
        "## Length of the kept rows (tokens)",
        "",
        f"Counted with the {st['token_counter']}. The system prompt and the 13 tools alone take "
        f"{tok.get('prompt_and_tools', 0):,} tokens. Kept rows have a median of "
        f"{st.get('agent_turns', {}).get('median', 0)} agent turns (max "
        f"{st.get('agent_turns', {}).get('max', 0)}).",
        "",
        *_table(
            ["min", "median", "p90", "p99", "max"],
            [[tok["min"], tok["median"], tok["p90"], tok.get("p99", ""), tok["max"]]],
        ),
        "",
        *_length_section(st),
        *_checks_section(st),
        "## Agent text",
        "",
        "Each agent turn's text is kept without the whitespace around it (from v1.1). The board's "
        "Qwen3.5 replies begin with two newlines, which after the reply header make one token "
        "with the header's own newline, so the trainer would learn the header as part of the "
        "reply; served, that newline is a token of its own. v1 kept them, and M4's mask check "
        "refused it.",
        "",
        "## Licence",
        "",
        "tau2-bench is MIT. The published board conversations carry no separate licence. Customer "
        "turns are another model's output and are context only, with no loss on them. No Claude "
        "output is in this dataset.",
        "",
    ]
    return "\n".join(lines)
