"""The trainer's pieces that need no model download: block-wise attention, the mask check, and
M4's data check, row choice, resume and checkpoint spacing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

torch = pytest.importorskip("torch")

from tau_telecom_sft import config  # noqa: E402
from tau_telecom_sft.build import tokens  # noqa: E402
from tau_telecom_sft.train import attention, sft  # noqa: E402
from tau_telecom_sft.train.mask import check, segments  # noqa: E402


def test_chunked_attention_matches_sdpa_with_gradients() -> None:
    from torch.nn import functional

    torch.manual_seed(0)
    old = attention.BLOCK
    attention.BLOCK = 64
    try:
        q = torch.randn(1, 4, 200, 16, requires_grad=True)
        k = torch.randn(1, 2, 200, 16, requires_grad=True)
        v = torch.randn(1, 2, 200, 16, requires_grad=True)
        out, _ = attention.chunked_sdpa(None, q, k, v, None, scaling=0.25)
        ref = functional.scaled_dot_product_attention(
            q, attention._repeat_kv(k, 2), attention._repeat_kv(v, 2), is_causal=True, scale=0.25
        ).transpose(1, 2)
        assert torch.allclose(out, ref, atol=1e-5)
        g1 = torch.autograd.grad(out.square().sum(), (q, k, v))
        g2 = torch.autograd.grad(ref.square().sum(), (q, k, v))
        assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(g1, g2, strict=True))
    finally:
        attention.BLOCK = old


class _Tok:
    """A tokenizer whose ids are characters, so segments decode to the exact text."""

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


def _render(row: dict, trained_turns: set[int]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []

    def put(text: str, m: int) -> None:
        ids.extend(ord(c) for c in text)
        mask.extend([m] * len(text))

    k = 0
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            put("<|im_start|>assistant\n", 0)
            body = msg["content"]
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                body += f'\n<tool_call>\n{{"name": "{fn["name"]}", "arguments": {fn["arguments"]}}}\n</tool_call>'
            put(body + "<|im_end|>\n", int(k in trained_turns))
            k += 1
        else:
            put(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n", 0)
    return ids, mask


ROW = {
    "messages": [
        {"role": "system", "content": "You are a telecom agent. Follow the policy below."},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "My mobile data stopped working while I am abroad."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_customer_by_phone",
                        "arguments": '{"phone_number": "555-123-2002"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"customer_id": "C1001", "line": "L1"}'},
        {"role": "assistant", "content": "Roaming is off on your line. Shall I turn it on?"},
    ]
}


def test_mask_check_passes_when_exactly_the_agent_turns_are_trained() -> None:
    ids, mask = _render(ROW, {0, 1, 2})
    chk = check(ROW, segments(_Tok(), ids, mask))
    assert chk["ok"], chk["problems"]
    assert chk["trained_turns"] == chk["assistant_turns"] == 3


def test_mask_check_fails_on_a_missing_turn_or_trained_context() -> None:
    ids, mask = _render(ROW, {0, 2})
    assert not check(ROW, segments(_Tok(), ids, mask))["ok"]
    ids, mask = _render(ROW, {0, 1, 2})
    assert not check(ROW, segments(_Tok(), ids, [1] * len(mask)))["ok"]


# M4: the job's pieces that decide what it trains and where it resumes, with no model download


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lengths: list[int]) -> Path:
    """A version v1 under tmp_path whose rows count as `lengths` tokens, and a config for it."""
    folder = tmp_path / "sft" / "v1"
    folder.mkdir(parents=True)
    rows = [
        {
            "messages": ROW["messages"],
            "tools": [],
            "meta": {"sim_id": f"s{i}", "task_id": "t"},
            "n": n,
        }
        for i, n in enumerate(lengths)
    ]
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (folder / "train.jsonl").write_bytes(data)
    stats = {"train_sha256": hashlib.sha256(data).hexdigest(), "n_kept": len(rows)}
    (folder / "stats.json").write_text(json.dumps(stats))
    monkeypatch.setattr(config, "SFT_DIR", tmp_path / "sft")
    monkeypatch.setattr(tokens, "count", lambda row, name: row["n"])
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    cfg = yaml.safe_load((config.CONFIGS_DIR / "sft_a.yaml").read_text())
    path = tmp_path / "sft_a.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class _LoadedError(Exception):
    """Raised where the tokenizer would load: everything before it has run."""


def _stop_at_load(*args: Any, **kwargs: Any) -> Any:
    raise _LoadedError


def test_a_changed_train_jsonl_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = _dataset(tmp_path, monkeypatch, [10, 20])
    path = tmp_path / "sft" / "v1" / "train.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"n": 10', b'"n": 11'))
    with pytest.raises(RuntimeError, match="re-upload"):
        sft.train(cfg_path, runs_dir=tmp_path / "runs", use_mlflow=False)
    assert not (tmp_path / "runs").exists()  # stopped before the folder, let alone the model


def test_the_measured_step_takes_the_longest_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    from tau_telecom_sft.cloud.modal_app import BASE_REVISION

    cfg_path = _dataset(tmp_path, monkeypatch, [10, 30, 20, 30, 5])
    cfg = sft.load_config(cfg_path)
    rows, note = sft.select_rows(cfg, longest=3)
    assert [r["meta"]["sim_id"] for r in rows] == ["s1", "s3", "s2"]  # ties by sim_id
    assert "3 longest" in note
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _stop_at_load)
    with pytest.raises(_LoadedError):
        sft.train(cfg_path, runs_dir=tmp_path / "runs", max_steps=1, longest=3, use_mlflow=False)
    (run_dir,) = (tmp_path / "runs").iterdir()
    meta = json.loads((run_dir / "run.json").read_text())
    assert run_dir.name.endswith("_train_sft-a-4b-step1") and meta["label"] == "sft-a-4b-step1"
    assert meta["purpose"] == "smoke"  # sft_a.yaml says result; a measured step never is
    assert meta["config"]["sft"]["max_steps"] == 1
    assert meta["row_sim_ids"] == ["s1", "s3", "s2"]
    assert meta["revision"] == BASE_REVISION  # the snapshot the Student serves


def test_resume_takes_the_newest_complete_checkpoint(tmp_path: Path) -> None:
    assert sft.newest_checkpoint(tmp_path) is None
    for step, done in ((9, True), (11, True), (22, True), (33, False)):
        ckpt = tmp_path / "checkpoints" / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        if done:
            (ckpt / "trainer_state.json").write_text("{}")
    # 33 was cut short by a preemption; 22 beats 9 as a number, not as text
    assert sft.newest_checkpoint(tmp_path) == tmp_path / "checkpoints" / "checkpoint-22"


def test_a_restart_drops_the_steps_it_will_redo(tmp_path: Path) -> None:
    from tau_telecom_sft.train.callbacks import RunFolder

    path = tmp_path / "metrics.jsonl"
    lines = "".join(json.dumps({"step": s, "loss": 1.0}) + "\n" for s in range(1, 30))
    path.write_text(lines)
    RunFolder(tmp_path, "cpu", resume_step=22)
    assert [json.loads(x)["step"] for x in path.read_text().splitlines()] == list(range(1, 23))
    RunFolder(tmp_path, "cpu")  # no checkpoint yet: every step is trained again
    assert path.read_text() == ""


def test_checkpoints_land_on_each_epochs_end() -> None:
    cfg = sft.load_config(config.CONFIGS_DIR / "sft_a.yaml")
    stats = json.loads((config.SFT_DIR / "v1" / "stats.json").read_text())
    s = cfg["sft"]
    per_step = s["per_device_train_batch_size"] * s["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(stats["n_kept"] / per_step)
    assert (stats["n_kept"], steps_per_epoch) == (518, 33)
    assert steps_per_epoch % s["save_steps"] == 0 and s["save_strategy"] == "steps"
    assert s["num_train_epochs"] == 3  # D1: fixed before the run
    assert stats["tokens"]["max"] <= min(s["max_length"], cfg["rows"]["max_tokens"])  # all fit
    assert "use_liger_kernel" not in s
