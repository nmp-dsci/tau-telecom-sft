"""One dataset build: sources -> agent view -> F1..F10 -> chat-and-tools JSONL, card and counts.

A build writes four files into its folder, normally `data/sft/<version>/`:
  train.jsonl   one row per kept conversation (git-ignored; S3 under its sha256 from M1)
  card.md       the dataset card (committed): sources, the filter table, counts, lengths
  stats.json    the same numbers for MLflow and M7's trigger
  drops.jsonl   every dropped conversation with its rule and reason, and its simulation id so
                the viewer can open it from raw/ (none for a test task or twin)

Every file lands under the trainer's inputs, so none may name a test task: a dropped test
task or twin is written as WITHHELD, and the leak check scans the folder.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau_telecom_sft import config, split
from tau_telecom_sft.build import card, filters
from tau_telecom_sft.build.agent_view import agent_turns
from tau_telecom_sft.build.convert import (
    agent_tool_names,
    agent_tools,
    estimate_tokens,
    prompt_fingerprint,
    system_prompt,
    to_row,
)
from tau_telecom_sft.sources.results import Conversation, conversations

WITHHELD = "WITHHELD"


@dataclass(frozen=True)
class Source:
    id: str  # S1, S2, … or "fixture"
    label: str  # e.g. "board/glm-5"
    path: Path

    @property
    def name(self) -> str:
        return f"{self.id} {self.label}"


@dataclass(frozen=True)
class Built:
    out_dir: Path
    stats: dict[str, Any]

    @property
    def n_kept(self) -> int:
        return int(self.stats["n_kept"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


HIST_BIN = 1024  # tokens per bar in the card's and the viewer's length histogram


def _hist(xs: Sequence[int]) -> list[dict[str, int]]:
    counts = Counter(x // HIST_BIN for x in xs)
    return (
        [{"from": b * HIST_BIN, "n": counts[b]} for b in range(min(counts), max(counts) + 1)]
        if counts
        else []
    )


def _pctl(xs: Sequence[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def build(
    sources: Sequence[Source],
    out_dir: Path,
    version: str,
    *,
    count_tokens: Callable[[dict[str, Any]], int] = estimate_tokens,
    token_counter: str = "estimate: characters / 4 (smoke only)",
    max_tokens: int = config.MAX_TOKENS,
    allow_pool: bool = False,
    the_split: split.Split | None = None,
    source_checks: Sequence[dict[str, Any]] = (),
) -> Built:
    s = the_split or split.current()
    convs: list[Conversation] = []
    source_rows: list[dict[str, Any]] = []
    for src in sources:
        loaded = conversations(src.path, src.name)
        convs.extend(loaded)
        source_rows.append(
            {
                "id": src.id,
                "label": src.label,
                "file": src.path.name,
                "bytes": src.path.stat().st_size,
                "sha256": _sha256(src.path),
                "conversations": len(loaded),
            }
        )

    rows: dict[str, dict[str, Any]] = {}

    def row_of(c: Conversation) -> dict[str, Any]:
        if c.sim.id not in rows:
            rows[c.sim.id] = to_row(c, version)
        return rows[c.sim.id]

    lengths: dict[str, int] = {}

    def tokens_of(c: Conversation) -> int:
        if c.sim.id not in lengths:
            lengths[c.sim.id] = count_tokens(row_of(c))
        return lengths[c.sim.id]

    ctx = filters.Context.build(
        convs,
        split=s,
        agent_tools=agent_tool_names(),
        count_tokens=tokens_of,
        max_tokens=max_tokens,
        allow_pool=allow_pool,
    )
    outcome = filters.apply(convs, ctx)

    out_dir.mkdir(parents=True, exist_ok=True)
    train = out_dir / "train.jsonl"
    with train.open("w") as f:
        for c in outcome.kept:
            f.write(json.dumps(row_of(c), ensure_ascii=False) + "\n")

    def safe_id(task_id: str) -> str:
        return WITHHELD if s.is_test_twin(task_id) else task_id

    with (out_dir / "drops.jsonl").open("w") as f:
        for d in outcome.drops:
            withheld = safe_id(d.conv.task_id) == WITHHELD
            rec = {
                "rule": d.rule,
                "reason": d.reason,
                "task_id": safe_id(d.conv.task_id),
                "source": d.conv.source,
                "trial": d.conv.trial,
                "sim_id": None if withheld else d.conv.sim.id,
                "agent_turns": agent_turns(d.conv.messages),
            }
            f.write(json.dumps(rec) + "\n")

    kept_lengths = [tokens_of(c) for c in outcome.kept]
    kept_turns = [agent_turns(c.messages) for c in outcome.kept]
    # every conversation that reached F10, so the card shows what the limit cut and what it spared
    f10 = {d.conv.sim.id for d in outcome.drops if d.rule == "F10"}
    at_f10 = kept_lengths + [lengths[i] for i in f10 if i in lengths]
    prompt_only = {
        "messages": [{"role": "system", "content": system_prompt()}],
        "tools": list(agent_tools()),
    }
    kept_ids = [split.parse_task_id(c.task_id) for c in outcome.kept]
    stats: dict[str, Any] = {
        "dataset": version,
        "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": config.settings().code_sha,
        "split_version": s.version,
        "split_sha256": s.sha256,
        **prompt_fingerprint(),
        "sources": source_rows,
        "n_loaded": len(convs),
        "n_kept": len(outcome.kept),
        "n_dropped": len(outcome.drops),
        "filters": [
            {
                "rule": rc.rule.id,
                "keep_if": rc.rule.keep_if,
                "in": rc.n_in,
                "dropped": rc.n_dropped,
            }
            for rc in outcome.table
        ],
        "drop_reasons": dict(Counter(f"{d.rule}: {d.reason}" for d in outcome.drops)),
        "kept_by_task": dict(sorted(Counter(c.task_id for c in outcome.kept).items())),
        "kept_by_symptom": dict(sorted(Counter(t.symptom for t in kept_ids).items())),
        "kept_by_root_issue": dict(sorted(Counter(i for t in kept_ids for i in t.issues).items())),
        "kept_by_source": dict(sorted(Counter(c.source for c in outcome.kept).items())),
        "training_tasks_covered": len({c.task_id for c in outcome.kept}),
        "training_tasks_total": len(s.train),
        "token_counter": token_counter,
        "max_tokens": max_tokens,
        "tokens": {
            "min": min(kept_lengths, default=0),
            "median": int(statistics.median(kept_lengths)) if kept_lengths else 0,
            "p90": _pctl(kept_lengths, 0.9),
            "p99": _pctl(kept_lengths, 0.99),
            "max": max(kept_lengths, default=0),
            "prompt_and_tools": count_tokens(prompt_only),
        },
        "tokens_hist": _hist(kept_lengths),
        "reached_f10": len(at_f10),
        "over_limit": {
            str(lim): sum(1 for n in at_f10 if n > lim)
            for lim in (8192, 12288, 16384, 20480, 24576)
        },
        "agent_turns": {
            "median": int(statistics.median(kept_turns)) if kept_turns else 0,
            "max": max(kept_turns, default=0),
        },
        "source_checks": list(source_checks),
        "train_sha256": _sha256(train),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out_dir / "card.md").write_text(card.render(stats))
    return Built(out_dir, stats)
