"""F10's counter from M1 on: the student's own tokenizer and chat template (§5, §7).

A row's length is the number of tokens TRL trains on: the row's messages and its 13 tools,
rendered by the student's chat template. Qwen3-4B-Instruct-2507 is step 1's student; the
30B and the Mac's 0.6B share its vocabulary, so one count serves all three. The tokenizer
comes from the Hugging Face cache (downloaded once; no key needed, the model is public).
"""

from __future__ import annotations

import functools
from typing import Any

STUDENT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


@functools.cache
def tokenizer(name: str = STUDENT_TOKENIZER) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def render(row: dict[str, Any], name: str = STUDENT_TOKENIZER) -> str:
    """The exact text the trainer sees for a row, before the mask."""
    text: str = tokenizer(name).apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False
    )
    return text


def count(row: dict[str, Any], name: str = STUDENT_TOKENIZER) -> int:
    enc = tokenizer(name).apply_chat_template(row["messages"], tools=row["tools"], tokenize=True)
    return len(enc["input_ids"])


def label(name: str = STUDENT_TOKENIZER) -> str:
    return f"{name} tokenizer and chat template"
