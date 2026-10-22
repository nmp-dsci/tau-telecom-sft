"""The board's telecom conversations, S1 and S2 (§4): taubench.com's public S3 bucket.

Sierra ran GLM-5 and Qwen3.5-397B on every telecom task, 4 trials each, in tau2's standard
tool-calling loop with the stock prompt, and published the results files. They are tau2's own
results JSON, so `sources.results` loads them unchanged. `raw/` is immutable: a download that
would replace a file with different bytes is refused, and each file's size and sha256 are
pinned here from the first download (17 Oct 2026).

`checks()` is M1's five checks from §4 ("Before you rely on the board files"), read from the
file itself so the card can say what each one found.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau_telecom_sft import config

BUCKET_URL = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions"
BOARD_DIR = config.RAW_DIR / "board"


@dataclass(frozen=True)
class BoardFile:
    id: str  # S1, S2
    label: str  # board/glm-5
    model: str
    submission: str  # the folder under web/leaderboard/public/submissions at the pin
    file: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{BUCKET_URL}/{self.submission}/trajectories/{self.file}"

    @property
    def path(self) -> Path:
        return BOARD_DIR / self.file


S1 = BoardFile(
    "S1",
    "board/glm-5",
    "GLM-5",
    "glm-5-think_sierra_2026-03-02",
    "glm-5_enabled_telecom_gpt-5.2_4trials.json",
    56_311_944,
    "18e41808e95cb4fbc4ad8cfd7b8fa358f2ed45c69abadd0c5d89ed3143edae0b",
)
S2 = BoardFile(
    "S2",
    "board/qwen3.5-397b",
    "Qwen3.5-397B-A17B",
    "qwen3.5-397b-a17b-think_sierra_2026-03-02",
    "qwen3.5-397b-a17b_enabled_telecom_gpt-5.2_4trials.json",
    71_968_908,
    "4038e532bef560a3aa47591b5a6c644ac0fb68a969fcc678ed2a52a9e7f3ed2d",
)
BOARD: tuple[BoardFile, ...] = (S1, S2)


class RawChangedError(RuntimeError):
    """A file under raw/ differs from its pinned bytes; raw/ never changes."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(bf: BoardFile) -> bool:
    """True when the file is present with its pinned size and sha256; raises when it differs."""
    if not bf.path.exists():
        return False
    if bf.path.stat().st_size != bf.bytes or _sha256(bf.path) != bf.sha256:
        raise RawChangedError(f"{bf.path} is not the pinned {bf.id} ({bf.sha256[:12]}…)")
    return True


def download(bf: BoardFile) -> Path:
    """Fetch one board file into raw/board/, once. A present, verified file is left alone."""
    if verify(bf):
        return bf.path
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    part = bf.path.with_suffix(".part")
    urllib.request.urlretrieve(bf.url, part)  # noqa: S310 - the board's public bucket
    if part.stat().st_size != bf.bytes or _sha256(part) != bf.sha256:
        part.unlink()
        raise RawChangedError(f"{bf.url} no longer has the pinned bytes ({bf.sha256[:12]}…)")
    part.rename(bf.path)
    return bf.path


def checks(bf: BoardFile) -> dict[str, Any]:
    """§4's five checks, read from the file: what the agent never saw, what thinking was kept,
    the prompt, the customer, the licence."""
    from tau_telecom_sft.build.convert import system_prompt

    data = json.loads(bf.path.read_text())
    info = data["info"]
    sims = data["simulations"]
    user_tool_results = agent_tool_results = 0
    reasoning_turns = assistant_turns = both = 0
    for s in sims:
        for m in s["messages"]:
            if m["role"] == "tool":
                if m.get("requestor") == "user":
                    user_tool_results += 1
                else:
                    agent_tool_results += 1
            if m["role"] != "assistant":
                continue
            assistant_turns += 1
            if m.get("content") and m.get("tool_calls"):
                both += 1
            raw = m.get("raw_data") or {}
            msg = ((raw.get("choices") or [{}])[0] or {}).get("message") or {}
            if msg.get("reasoning_content") or msg.get("reasoning"):
                reasoning_turns += 1
    board_policy = (info.get("environment_info") or {}).get("policy") or ""
    return {
        "source": bf.id,
        "model": bf.model,
        "agent_llm": (info.get("agent_info") or {}).get("llm"),
        "agent_llm_args": {
            k: v
            for k, v in ((info.get("agent_info") or {}).get("llm_args") or {}).items()
            if k != "api_base"
        },
        "customer_llm": (info.get("user_info") or {}).get("llm"),
        "customer_llm_args": (info.get("user_info") or {}).get("llm_args"),
        "tau2_commit": info.get("git_commit"),
        "num_trials": info.get("num_trials"),
        "seed": info.get("seed"),
        "phone_tool_results": user_tool_results,
        "agent_tool_results": agent_tool_results,
        "assistant_turns": assistant_turns,
        "assistant_turns_with_reasoning": reasoning_turns,
        "assistant_turns_with_text_and_calls": both,
        "system_prompt_stored": any(m["role"] == "system" for s in sims for m in s["messages"]),
        "policy_matches_pin": board_policy in system_prompt(),
        "licence": "none found beside the files; tau2-bench itself is MIT",
    }
