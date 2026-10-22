"""The student served on Modal (M3, M5): where it is and what it serves.

The Mac's MLX serving writes `serve.json` beside the converted weights (serve/mlx.py). On Modal
the same facts come from cloud/modal_app.py, plus the chat template's sha256, read from the
tokenizer at the served revision: vLLM applies the model's own template, the one the dataset
was rendered with. `describe()` is that record for one served name; `url()` asks Modal where
the deployed Student answers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tau_telecom_sft import config
from tau_telecom_sft.cloud import modal_app

RUNS_DIR = config.ROOT / "runs"


def served_names() -> dict[str, str | None]:
    """Every name the Student answers to, with the train run behind it (None: the base)."""
    return {modal_app.BASE_NAME: None, **modal_app.ADAPTERS}


def describe(name: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    names = served_names()
    if name not in names:
        raise ValueError(f"the Student does not serve {name!r}: {sorted(names)}")
    train_run = names[name]
    adapter_sha = None
    if train_run:
        run = json.loads((RUNS_DIR / train_run / "run.json").read_text())
        adapter_sha = run["adapter"]["sha256"]
    tok = AutoTokenizer.from_pretrained(modal_app.BASE, revision=modal_app.BASE_REVISION)
    return {
        "model": name,
        "base_model": modal_app.BASE,
        "revision": modal_app.BASE_REVISION,
        "train_run": train_run,
        "adapter_sha256": adapter_sha,
        "tokenizer": modal_app.BASE,
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        "dtype": "bfloat16",
        "server": f"vllm {modal_app.VLLM} on Modal {modal_app.GPU}",
        "max_model_len": modal_app.MAX_MODEL_LEN,
    }


def url() -> str:
    """The deployed Student's URL (needs `modal setup` and `make modal-serve` first)."""
    import modal

    server = modal.Server.from_name(modal_app.APP, modal_app.SERVER)
    return str(server.get_url()).rstrip("/")
