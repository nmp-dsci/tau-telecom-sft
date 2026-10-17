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
        *_table(
            ["min", "median", "p90", "max"], [[tok["min"], tok["median"], tok["p90"], tok["max"]]]
        ),
        "",
        "## Licence",
        "",
        "tau2-bench is MIT. The published board conversations carry no separate licence. Customer "
        "turns are another model's output and are context only, with no loss on them. No Claude "
        "output is in this dataset.",
        "",
    ]
    return "\n".join(lines)
