"""M2's serving on the Mac (plan §8, option B): merge, convert to MLX in bf16, serve.

`prepare("base")` converts the untrained Qwen3-0.6B; `prepare(<train run id>)` merges that
run's LoRA adapter into the base first. Either way the served folder carries the student's
tokenizer and chat template (Qwen3-4B-Instruct-2507's), the same text format the run trained
on, and a `serve.json` beside it records what it is. `mlx_lm.server` then serves it with its
tool-call parser on; `serve/proxy.py` sits in front of it on :8020 and records every request.
bf16 throughout: never quantize one row and not another (§8).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from tau_telecom_sft import config

OUTPUTS = config.ROOT / "outputs" / "mlx"
RUNS_DIR = config.ROOT / "runs"
UPSTREAM_PORT = 8120  # mlx_lm.server; the recording proxy takes :8020, where tau2-loop calls
DTYPE = "bfloat16"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def served_dir(model: str) -> Path:
    return OUTPUTS / "qwen3-0.6b-base" if model == "base" else RUNS_DIR / model / "mlx"


def _use_student_template(mlx_dir: Path, tokenizer_name: str) -> str:
    """Replace the converted folder's tokenizer with the student's: same vocabulary, the
    student's chat template. Returns the template's sha256."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    for f in mlx_dir.glob("*"):
        if f.name.startswith(("tokenizer", "chat_template", "special_tokens", "vocab", "merges")):
            f.unlink()
    tok.save_pretrained(str(mlx_dir))
    return _sha256_text(tok.chat_template)


def prepare(model: str, *, force: bool = False) -> Path:
    """The MLX folder for `base` or a train run, built once."""
    from mlx_lm import convert

    out = served_dir(model)
    if (out / "serve.json").exists() and not force:
        return out
    if out.exists():
        shutil.rmtree(out)
    if model == "base":
        # the smoke config's base and tokenizer: the untrained row is the same model, untouched
        smoke = yaml.safe_load((config.CONFIGS_DIR / "smoke.yaml").read_text())
        base_model, tokenizer_name = smoke["base_model"], smoke["tokenizer"]
        adapter_sha, source = None, base_model
        convert(base_model, mlx_path=str(out), dtype=DTYPE)
    else:
        run = json.loads((RUNS_DIR / model / "run.json").read_text())
        if run.get("status") != "finished":
            raise ValueError(f"{model} has not finished training")
        base_model, tokenizer_name = run["base_model"], run["tokenizer"]
        adapter_sha = run["adapter"]["sha256"]
        merged = RUNS_DIR / model / "merged"
        _merge(base_model, RUNS_DIR / model / "adapter", merged)
        source = str(merged.relative_to(config.ROOT))
        convert(str(merged), mlx_path=str(out), dtype=DTYPE)
    template_sha = _use_student_template(out, tokenizer_name)
    record: dict[str, Any] = {
        "model": model,
        "base_model": base_model,
        "train_run": None if model == "base" else model,
        "adapter_sha256": adapter_sha,
        "converted_from": source,
        "tokenizer": tokenizer_name,
        "chat_template_sha256": template_sha,
        "dtype": DTYPE,
        "server": "mlx_lm.server",
    }
    (out / "serve.json").write_text(json.dumps(record, indent=2) + "\n")
    return out


def _merge(base_model: str, adapter: Path, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)


def server_command(model: str, port: int = UPSTREAM_PORT) -> list[str]:
    """mlx_lm.server on 127.0.0.1, with room for a long agent turn; sampling comes per request
    from the student's tau2-loop version."""
    import sys

    return [
        sys.executable,
        "-m",
        "tau_telecom_sft.serve.mlx_server",
        "--model",
        str(served_dir(model)),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-tokens",
        "2048",
        # a few conversations' KV caches, not ten: the Mac also trains (serve/mlx_server.py)
        "--prompt-cache-size",
        "4",
        "--prompt-cache-bytes",
        "4GB",
        "--log-level",
        "WARNING",
    ]
