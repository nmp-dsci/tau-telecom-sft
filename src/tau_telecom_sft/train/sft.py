"""The trainer: LoRA SFT in TRL on a dataset version, with the loss on the agent's turns only.

One script for the Mac's proof of concept (M2, `configs/smoke.yaml`, MPS) and the GPU jobs
(M4 and M6, `configs/sft_a.yaml`, CUDA on Modal). A config names the base model and its
revision, the tokenizer, the dataset version, which rows to take, the LoRA adapter and TRL's
SFTConfig, under their real names (plan §7's table).

Before the model loads, the version's train.jsonl is checked against its card's sha256, so a
copy on another machine (Modal's data Volume) is the data that was signed off. Before the
first step every row the trainer tokenized is read back and checked (`train.mask`): one trained
span per agent turn and nothing else. A failed check on any row stops the job before it costs
anything. Every run writes its folder under `runs/` (docs/run_folders.md): run.json,
metrics.jsonl at every logged step, mask.json, the checkpoints and the adapter. The same
callback logs each step to the central MLflow when it can be reached and carries on when it
can't, so the folder, not the connection, is the record (`make sync-runs` fills MLflow later).

A job given a run id whose folder already holds a checkpoint resumes from the newest complete
one: Modal restarts a preempted job on the same input (s02's M-1), so the same run id.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tau_telecom_sft import config, track
from tau_telecom_sft.build import tokens
from tau_telecom_sft.train import attention
from tau_telecom_sft.train import mask as mask_mod

RUNS_DIR = config.ROOT / "runs"


def load_config(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    for key in ("label", "purpose", "base_model", "tokenizer", "dataset", "lora", "sft"):
        if key not in cfg:
            raise ValueError(f"{path}: missing {key!r}")
    if cfg["purpose"] not in ("smoke", "result"):
        raise ValueError(f"{path}: purpose must be smoke or result")
    return cfg


def run_label(cfg: dict[str, Any], max_steps: int | None) -> str:
    """A measured step is labelled as one (`sft-a-4b-step1`), so it is never read as the run."""
    return f"{cfg['label']}-step{max_steps}" if max_steps is not None else str(cfg["label"])


def new_run_id(kind: str, label: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{kind}_{label}"


def check_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """The version's stats.json, once its train.jsonl is the file the card names; anything else
    is refused before the model loads."""
    folder = config.SFT_DIR / cfg["dataset"]
    stats: dict[str, Any] = json.loads((folder / "stats.json").read_text())
    sha = hashlib.sha256((folder / "train.jsonl").read_bytes()).hexdigest()
    if sha != stats["train_sha256"]:
        raise RuntimeError(
            f"{folder}/train.jsonl has sha256 {sha[:8]}…, not the card's "
            f"{stats['train_sha256'][:8]}…; re-upload it (make modal-data)"
        )
    return stats


def select_rows(
    cfg: dict[str, Any], longest: int | None = None
) -> tuple[list[dict[str, Any]], str]:
    """The config's rows from the version's train.jsonl: those that fit `rows.max_tokens`
    whole, then `rows.limit` of them drawn with `rows.seed`, or, for a measured step, the
    `longest` rows, which set the job's peak memory. Returns the rows and a note."""
    path = config.SFT_DIR / cfg["dataset"] / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    spec = cfg.get("rows") or {}
    limit = spec.get("max_tokens")
    fit = [r for r in rows if limit is None or tokens.count(r, cfg["tokenizer"]) <= limit]
    note = f"{len(fit)} of {len(rows)} rows fit {limit:,} tokens" if limit else f"all {len(rows)}"
    if longest:
        fit = sorted(fit, key=lambda r: (-tokens.count(r, cfg["tokenizer"]), r["meta"]["sim_id"]))
        fit = fit[:longest]
        note += f"; the {longest} longest, for a measured step"
    elif spec.get("limit") and spec["limit"] < len(fit):
        fit = sorted(fit, key=lambda r: r["meta"]["sim_id"])
        fit = random.Random(spec.get("seed", 300)).sample(fit, spec["limit"])
        note += f"; {spec['limit']} drawn with seed {spec.get('seed', 300)}"
    return fit, note


def newest_checkpoint(run_dir: Path) -> Path | None:
    """The newest checkpoint the job finished writing. Trainer writes trainer_state.json last,
    so a checkpoint cut short by a preemption has none and is passed over."""
    done = [
        p
        for p in (run_dir / "checkpoints").glob("checkpoint-*")
        if (p / "trainer_state.json").exists()
    ]
    return max(done, key=lambda p: int(p.name.rsplit("-", 1)[1]), default=None)


def _sha256_dir(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        data = f.read_bytes()
        h.update(f.relative_to(path).as_posix().encode())
        h.update(data)
        size += len(data)
    return h.hexdigest(), size


def _instance(device: str) -> str:
    box = config.modal_container()
    if box is not None:
        import torch

        return f"Modal {torch.cuda.get_device_name(0)} ({box['region']})"
    if device == "mps":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            mem = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                ).stdout.strip()
            )
            return f"Mac {chip}, {mem // 2**30} GB"
        except (OSError, ValueError):
            return "Mac"
    return platform.node()


def _stack() -> dict[str, str]:
    """The training stack's versions as the job found them: on Modal, the image's."""
    from importlib.metadata import version

    return {p: version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate")}


def _flat(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{k}": v for k, v in d.items()}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def train(
    config_path: Path,
    *,
    runs_dir: Path = RUNS_DIR,
    run_id: str | None = None,
    max_steps: int | None = None,
    longest: int | None = None,
    use_mlflow: bool = True,
    console: Any = None,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from tau_telecom_sft.train.callbacks import RunFolder

    say = console.print if console is not None else print
    cfg = load_config(config_path)
    stats = check_data(cfg)
    rows, rows_note = select_rows(cfg, longest)
    label = run_label(cfg, max_steps)
    # a caller's run id (Modal's) may name a folder an earlier attempt of the same job began
    given = run_id is not None
    run_id = run_id or new_run_id("train", label)
    run_dir = runs_dir / run_id
    resume = newest_checkpoint(run_dir)
    run_dir.mkdir(parents=True, exist_ok=given)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    sft_kwargs = dict(cfg["sft"])
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps
    box = config.modal_container()
    if resume is not None:
        meta: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
        meta["status"] = "running"
        meta["resumed_from"] = [*meta.get("resumed_from", []), resume.name]
        say(f"run {run_id}: resuming from {resume.name}")
    else:
        meta = {
            "run_id": run_id,
            "kind": "train",
            "label": label,
            # a measured step is never a result, whatever its config says
            "purpose": "smoke" if max_steps is not None else cfg["purpose"],
            "status": "running",
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_sha": config.settings().code_sha,
            "image_tag": box["image_id"] if box else None,
            "config_file": str(config_path.resolve().relative_to(config.ROOT))
            if config_path.resolve().is_relative_to(config.ROOT)
            else str(config_path),
            "config": {**cfg, "sft": sft_kwargs},
            "base_model": cfg["base_model"],
            "revision": cfg.get("revision"),
            "tokenizer": cfg["tokenizer"],
            "dataset": cfg["dataset"],
            "train_sha256": stats["train_sha256"],
            "n_rows": len(rows),
            "rows_selected": rows_note,
            "row_sim_ids": [r["meta"]["sim_id"] for r in rows],
            "device": device,
            "instance": _instance(device),
            "stack": _stack(),
            "billing": "modal" if box else "none",
            "cost_usd": 0.0 if device != "cuda" else None,
            "resumed_from": [],
        }
    write_json(run_dir / "run.json", meta)
    say(f"run {run_id}: {len(rows)} rows ({rows_note}) on {meta['instance']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], revision=cfg.get("revision"))
    dtype = torch.bfloat16 if sft_kwargs.get("bf16") else torch.float32
    model_init = dict(cfg.get("model_init") or {})
    if model_init.get("attn_implementation") == attention.NAME:
        attention.register()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], revision=cfg.get("revision"), dtype=dtype, **model_init
    )
    dataset = Dataset.from_list(
        [{"messages": r["messages"], "tools": json.dumps(r["tools"])} for r in rows]
    )
    args = SFTConfig(output_dir=str(run_dir / "checkpoints"), report_to="none", **sft_kwargs)
    # steps past the checkpoint are trained again, so their old lines go (all of them when an
    # earlier attempt died before its first checkpoint)
    step = int(resume.name.rsplit("-", 1)[1]) if resume is not None else 0
    folder = RunFolder(run_dir, device, resume_step=step)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
        callbacks=[folder],
    )

    # the trainer's own tokenized rows, read back: their labels are what the loss sees (-100 is
    # ignored), so the mask check runs on exactly that, on every row, before any step (one row
    # was not enough: M4's first measured step failed on the 273 rows M2's first row hid)
    if len(trainer.train_dataset) != len(rows):
        raise RuntimeError(f"TRL kept {len(trainer.train_dataset)} of {len(rows)} rows")
    failed: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        ex = trainer.train_dataset[i]
        trained = [int(label != -100) for label in ex["labels"]]
        segs = mask_mod.segments(tokenizer, ex["input_ids"], trained)
        chk = mask_mod.check(row, segs)
        if i == 0:
            first = {"ex": ex, "trained": trained, "segs": segs, "chk": chk}
        if not chk["ok"]:
            failed.append({"sim_id": row["meta"]["sim_id"], "problems": chk["problems"][:3]})
    ex, trained, segs, chk = first["ex"], first["trained"], first["segs"], first["chk"]
    write_json(
        run_dir / "mask.json",
        {
            "row_sim_id": rows[0]["meta"]["sim_id"],
            "task_id": rows[0]["meta"]["task_id"],
            "n_tokens": len(ex["input_ids"]),
            "n_trained": sum(trained),
            "chat_template": "trl qwen3_instruct_2507_training.jinja (TRL's training copy)",
            "segments": segs,
            "check": chk,
            "rows_checked": len(rows),
            "rows_failed": failed,
        },
    )
    meta["mask_check"] = {
        "row_sim_id": rows[0]["meta"]["sim_id"],
        **chk,
        "ok": not failed,
        "rows_checked": len(rows),
        "rows_failed": len(failed),
    }
    print_mask(segs, chk, say)
    say(f"every row's mask: {len(rows) - len(failed)} of {len(rows)} ok")
    if failed:
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        raise RuntimeError(
            f"loss-mask check failed on {len(failed)} of {len(rows)} rows, the first "
            f"{failed[0]['sim_id']}: {failed[0]['problems']}"
        )

    mlflow_run = track.start_train(meta) if use_mlflow else None
    folder.mlflow_run_id = mlflow_run
    meta["mlflow_run_id"] = mlflow_run
    meta["mlflow_experiment"] = config.EXPERIMENT_TRAIN if mlflow_run else None
    write_json(run_dir / "run.json", meta)

    t0 = time.time()
    try:
        out = trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    except BaseException:
        meta["status"] = "failed"
        write_json(run_dir / "run.json", meta)
        track.end_train(mlflow_run, run_dir, status="FAILED")
        raise
    adapter = run_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    sha, size = _sha256_dir(adapter)
    meta.update(
        {
            "status": "finished",
            "finished": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": round(time.time() - t0, 1),
            "steps": int(out.global_step),
            "epochs": float(trainer.state.epoch or 0),
            "final_loss": folder.last_loss,
            "train_loss": float(out.training_loss),
            "peak_mem_gb": folder.peak_mem_gb,
            "tokens_trained": folder.tokens,
            "adapter": {
                "path": str(adapter.relative_to(config.ROOT)),
                "sha256": sha,
                "bytes": size,
            },
        }
    )
    write_json(run_dir / "run.json", meta)
    track.end_train(mlflow_run, run_dir, status="FINISHED", summary=meta)
    say(f"finished {run_id}: {meta['steps']} steps, final loss {meta['final_loss']}")
    return run_dir


def print_mask(segs: list[dict[str, Any]], chk: dict[str, Any], say: Any) -> None:
    """One row's loss mask: trained text in green, context dimmed and shortened."""
    from rich.markup import escape

    for s in segs:
        text = s["text"]
        if s["trained"]:
            say(f"[green]{escape(text)}[/green]", end="")
        else:
            short = (
                text
                if len(text) < 400
                else f"{text[:160]} … [{s['tokens']:,} tokens] … {text[-160:]}"
            )
            say(f"[dim]{escape(short)}[/dim]", end="")
    say("")
    mark = "[green]ok[/green]" if chk["ok"] else "[red]FAILED[/red]"
    say(
        f"loss mask {mark}: {chk['trained_turns']}/{chk['assistant_turns']} agent turns trained, "
        f"{chk['trained_tokens']:,} of {chk['total_tokens']:,} tokens carry loss"
    )
    for p in chk["problems"]:
        say(f"  [red]{escape(p)}[/red]")
