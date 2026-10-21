"""The Modal path (s02): the proxy's token and retries, the vLLM command and the served names
(M3), and the trainer's image, its run's record and the sync home (M4). Offline: nothing here
reaches Modal."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest
from fastapi.testclient import TestClient

from tau_telecom_sft import config, track
from tau_telecom_sft.cloud import modal_app, sync
from tau_telecom_sft.serve import proxy, remote
from tau_telecom_sft.train import sft

REPLY = {"choices": [{"message": {"content": "hi", "tool_calls": []}, "finish_reason": "stop"}]}


def _body() -> dict[str, Any]:
    return {"model": "student-qwen3-4b", "messages": [{"role": "system", "content": "x"}]}


def test_the_proxy_sends_the_token_and_waits_out_a_cold_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503) if len(seen) < 3 else httpx.Response(200, json=REPLY)

    async def no_wait(_: float) -> None:
        return None

    monkeypatch.setattr(proxy.asyncio, "sleep", no_wait)
    log = tmp_path / "served.jsonl"
    app = proxy.make_app(
        "https://student.example",
        log,
        headers={"Modal-Key": "wk-1", "Modal-Secret": "ws-1"},
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json=_body())
    assert r.status_code == 200 and r.json() == REPLY
    assert len(seen) == 3  # two 503s, then the answer
    assert all(q.headers["Modal-Key"] == "wk-1" for q in seen)
    line = json.loads(log.read_text())
    assert line["retries"] == 2 and line["error"] is None
    assert "wk-1" not in log.read_text() and "ws-1" not in log.read_text()  # never logged


def test_the_proxy_gives_up_after_its_window(tmp_path: Path) -> None:
    app = proxy.make_app(
        "https://student.example",
        tmp_path / "served.jsonl",
        retry_for_s=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    )
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=_body()).status_code == 503
    line = json.loads((tmp_path / "served.jsonl").read_text())
    assert line["retries"] == 0 and line["error"] == "upstream failed"


def test_the_proxy_logs_a_200_that_is_not_json_as_a_failure(tmp_path: Path) -> None:
    app = proxy.make_app(
        "https://student.example",
        tmp_path / "served.jsonl",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'{"choices": [')),
    )
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=_body()).status_code == 502
    line = json.loads((tmp_path / "served.jsonl").read_text())
    assert line["error"] == "upstream failed" and line["n_messages"] == 1


def test_the_vllm_command_is_the_plans() -> None:
    cmd = modal_app.vllm_command()
    for flag in ("--enable-prefix-caching", "--enable-auto-tool-choice"):
        assert flag in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"
    assert cmd[cmd.index("--served-model-name") + 1] == modal_app.BASE_NAME
    assert cmd[cmd.index("--revision") + 1] == modal_app.BASE_REVISION
    assert "--enable-lora" not in cmd  # R0: the untrained base only


def test_an_adapter_is_served_under_its_own_name() -> None:
    cmd = modal_app.vllm_command({"student-qwen3-4b-sft-a": "20261020T000000Z_train_sft-a-4b"})
    assert cmd[cmd.index("--max-lora-rank") + 1] == "16"
    assert "student-qwen3-4b-sft-a=/runs/20261020T000000Z_train_sft-a-4b/adapter" in cmd


def test_the_student_refuses_a_name_it_does_not_serve() -> None:
    assert remote.served_names() == {modal_app.BASE_NAME: None, **modal_app.ADAPTERS}
    with pytest.raises(ValueError, match="does not serve"):
        remote.describe("student-qwen3-0.6b")


def test_the_proxy_token_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-2")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-2")
    assert config.modal_proxy_headers() == {"Modal-Key": "wk-2", "Modal-Secret": "ws-2"}
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET")
    assert config.modal_proxy_headers() == {}


# M4: the trainer on Modal and the sync home. A fake Volume and a fake MLflow stand in.


def test_the_trainer_image_is_uv_locks() -> None:
    lock = tomllib.loads((config.ROOT / "uv.lock").read_text())
    locked = {p["name"]: p["version"] for p in lock["package"]}
    for pin in modal_app.TRAIN_PINS:
        name, version = pin.split("==")
        assert locked[name] == version, pin


def test_the_trainer_imports_no_tau2() -> None:
    # the image has neither (TRAIN_PINS), and a tau2 import would bring its test tasks along
    code = (
        "import sys, tau_telecom_sft.train.sft\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'tau2', 'tau2_loop', 'mlflow'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_a_job_on_modal_records_where_it_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    assert config.modal_container() is None  # the Mac
    monkeypatch.setenv("MODAL_TASK_ID", "ta-1")
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-1")
    monkeypatch.setenv("MODAL_REGION", "us-east")
    assert config.modal_container() == {"task_id": "ta-1", "image_id": "im-1", "region": "us-east"}


def test_a_modal_job_refuses_uncommitted_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "a")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert len(config.clean_code_sha()) >= 7
    (tmp_path / "notes.txt").write_text("not shipped\n")  # outside src/ and configs/: fine
    config.clean_code_sha()
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="commit src/ and configs/ first"):
        config.clean_code_sha()


class _Entry:
    def __init__(self, path: str) -> None:
        self.path = path
        self.type = SimpleNamespace(name="FILE")


class _Volume:
    """Modal's Volume as sync uses it: paths from its root, files read in chunks."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def listdir(self, path: str, recursive: bool = False) -> list[_Entry]:
        prefix = path.strip("/") + "/" if path.strip("/") else ""
        return [_Entry(p) for p in self.files if p.startswith(prefix)]

    def read_file(self, path: str) -> Iterator[bytes]:
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        return iter([data[:3], data[3:]])


RUN = "20261020T120000Z_train_sft-a-4b"


def _finished_run(tmp_path: Path) -> dict[str, bytes]:
    adapter = {"adapter_config.json": b'{"r": 16}', "adapter_model.safetensors": b"\x00weights"}
    src = tmp_path / "src_adapter"
    src.mkdir()
    for name, data in adapter.items():
        (src / name).write_bytes(data)
    sha, size = sft._sha256_dir(src)
    meta = {"run_id": RUN, "status": "finished", "adapter": {"sha256": sha, "bytes": size}}
    files = {f"{RUN}/adapter/{n}": d for n, d in adapter.items()}
    files[f"{RUN}/run.json"] = json.dumps(meta).encode()
    files[f"{RUN}/metrics.jsonl"] = b'{"step": 1, "loss": 1.2}\n'
    files[f"{RUN}/checkpoints/checkpoint-33/trainer_state.json"] = b"{}"
    return files


def test_sync_copies_the_folder_and_checks_the_adapter(tmp_path: Path) -> None:
    files = _finished_run(tmp_path)
    home = tmp_path / "runs"
    (home / RUN).mkdir(parents=True)
    (home / RUN / "run.json").write_text(json.dumps({"mlflow_run_id": "m-1", "status": "running"}))
    meta = sync.pull(_Volume(files), RUN, home)
    assert meta["status"] == "finished" and meta["mlflow_run_id"] == "m-1"  # the Mac's link kept
    assert (home / RUN / "metrics.jsonl").exists() and not (home / RUN / "mask.json").exists()
    assert sorted(p.name for p in (home / RUN / "adapter").iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert not (home / RUN / "checkpoints").exists()  # they stay on the Volume
    files[f"{RUN}/adapter/adapter_model.safetensors"] = b"\x01weights"
    with pytest.raises(RuntimeError, match="not the job's"):
        sync.pull(_Volume(files), RUN, tmp_path / "elsewhere")


def test_sync_replaces_an_adapter_left_half_copied(tmp_path: Path) -> None:
    files = _finished_run(tmp_path)
    home = tmp_path / "runs"
    (home / RUN / "adapter").mkdir(parents=True)
    (home / RUN / "adapter" / "adapter_config.json").write_bytes(b'{"r"')  # a sync cut short
    sync.pull(_Volume(files), RUN, home)
    adapter = home / RUN / "adapter"
    assert (adapter / "adapter_config.json").read_bytes() == b'{"r": 16}'
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"\x00weights"


class _Client:
    logged: ClassVar[list[Any]] = []

    def get_metric_history(self, run_id: str, key: str) -> list[Any]:
        return [m for m in self.logged if m.key == key]

    def log_batch(self, run_id: str, metrics: list[Any]) -> None:
        self.logged.extend(metrics)


def test_sync_logs_each_step_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    _Client.logged = []
    ended: list[str] = []
    monkeypatch.setattr(mlflow, "MlflowClient", _Client)
    monkeypatch.setattr(mlflow, "active_run", lambda: object())
    monkeypatch.setattr(track, "start_train", lambda meta, system_metrics=True: "m-2")
    monkeypatch.setattr(
        track, "end_train", lambda rid, run_dir, status, summary=None: ended.append(status)
    )

    def steps(*numbers: int) -> None:
        with (tmp_path / "metrics.jsonl").open("a") as f:
            for n in numbers:
                f.write(json.dumps({"step": n, "loss": 1.0 / n, "tokens_per_s": 1200.0}) + "\n")

    meta: dict[str, Any] = {"run_id": RUN, "status": "running"}
    steps(1, 2, 3)
    sync.log(meta, tmp_path)
    assert json.loads((tmp_path / "run.json").read_text())["mlflow_run_id"] == "m-2"
    steps(4, 5)
    sync.log(meta, tmp_path)
    assert not ended  # still running: the run stays open
    meta["status"] = "finished"
    sync.log(meta, tmp_path)
    losses = [m.step for m in _Client.logged if m.key == "loss"]
    assert losses == [1, 2, 3, 4, 5]  # each step once, across three syncs
    assert len(_Client.logged) == 10 and ended == ["FINISHED"]
