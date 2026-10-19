"""The Modal path (s02, M3): the proxy's token and retries, the vLLM command and the served names.
Offline: nothing here reaches Modal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tau_telecom_sft import config
from tau_telecom_sft.cloud import modal_app
from tau_telecom_sft.serve import proxy, remote

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
