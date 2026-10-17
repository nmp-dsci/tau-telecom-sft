"""A recording proxy in front of the student's server: what was served is what was trained on.

tau2-loop calls this on :8020 as it would call the model. Each chat completion is forwarded
unchanged to the server behind it (mlx_lm.server on :8120 on the Mac; on EC2, the port forward
to vLLM), and one line is appended to the log: the sha256 of the system prompt and of the tools
the scorer sent, how many messages, and whether the reply's tool calls came back structured
(`tool_calls`) or written into the text. The prompt-bytes check (M2, M3) compares those hashes
with the dataset card's. Nothing in a line is conversation text.

mlx_lm.server treats the request's `model` as a path to load unless it is `default_model`, so
on the Mac the proxy asks for that (`rewrite_model`) and logs the name tau2-loop asked for.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

# at module level: FastAPI resolves the handlers' (string) annotations against these globals
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

TEXT_TOOL_CALL = re.compile(r"</?tool_call>|\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def record(body: dict[str, Any], reply: dict[str, Any] | None, latency: float) -> dict[str, Any]:
    """The log line for one request: hashes and counts, no conversation text."""
    msgs = body.get("messages") or []
    system = msgs[0].get("content") if msgs and msgs[0].get("role") == "system" else None
    choice = ((reply or {}).get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    usage = (reply or {}).get("usage") or {}
    return {
        "t": round(time.time(), 3),
        "model": body.get("model"),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest() if system else None,
        "tools_sha256": sha256_json(body["tools"]) if body.get("tools") else None,
        "n_tools": len(body.get("tools") or []),
        "n_messages": len(msgs),
        "sampling": {k: body.get(k) for k in ("temperature", "top_p", "top_k") if k in body},
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": len(calls),
        "tool_call_names": [c.get("function", {}).get("name") for c in calls],
        "text_tool_call": bool(TEXT_TOOL_CALL.search(message.get("content") or "")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(latency, 2),
        "error": None if reply is not None else "upstream failed",
    }


def make_app(upstream: str, log: Path, rewrite_model: str | None = None) -> FastAPI:
    import httpx

    app = FastAPI(title="tau-telecom-sft recording proxy")
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(1800.0))
    log.parent.mkdir(parents=True, exist_ok=True)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.json()
        sent = dict(body)
        if rewrite_model:
            sent["model"] = rewrite_model
        t0 = time.time()
        reply: dict[str, Any] | None = None
        try:
            r = await client.post("/v1/chat/completions", json=sent)
            status = r.status_code
            content = r.content
            if status == 200:
                reply = r.json()
        except httpx.HTTPError as e:
            status, content = 502, json.dumps({"error": str(e)}).encode()
        with log.open("a") as f:
            f.write(json.dumps(record(body, reply, time.time() - t0)) + "\n")
        return Response(content=content, status_code=status, media_type="application/json")

    @app.get("/v1/models")
    async def models() -> Response:
        r = await client.get("/v1/models")
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True, "upstream": upstream, "log": str(log)})

    return app


def summarise(lines: list[dict[str, Any]], expected: dict[str, str]) -> dict[str, Any]:
    """The prompt-bytes and tool-call checks over a run's requests."""
    sys_shas = sorted({x["system_prompt_sha256"] for x in lines if x.get("system_prompt_sha256")})
    tool_shas = sorted({x["tools_sha256"] for x in lines if x.get("tools_sha256")})
    with_calls = sum(1 for x in lines if x.get("tool_calls"))
    text_calls = sum(1 for x in lines if x.get("text_tool_call"))
    return {
        "prompt_bytes": {
            "ok": bool(lines)
            and sys_shas == [expected["system_prompt_sha256"]]
            and tool_shas == [expected["tools_sha256"]],
            "requests": len(lines),
            "system_prompt_sha256": sys_shas,
            "tools_sha256": tool_shas,
            "expected_system_prompt_sha256": expected["system_prompt_sha256"],
            "expected_tools_sha256": expected["tools_sha256"],
        },
        "tool_calls": {
            "ok": with_calls > 0,
            "requests_with_tool_calls": with_calls,
            "text_tool_calls": text_calls,
            "errors": sum(1 for x in lines if x.get("error")),
        },
    }
