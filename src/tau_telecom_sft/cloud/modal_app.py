"""The student on Modal (s02, M3 and M5): vLLM on one L4, in place of EC2's `docker run`.

`Student` is vLLM's own OpenAI-compatible server for Qwen3-4B-Instruct-2507 in bf16, with the
plan's flags (§8): a 32K context, prefix caching and the hermes tool-call parser. It serves the
untrained base as `student-qwen3-4b` (R0) and each adapter in ADAPTERS under its own name (M5:
`student-qwen3-4b-sft-a`), and tau2-loop's student versions ask for those names, so swapping the
base for the tuned model is a name. Modal's proxy requires a token, its default for a Server;
serve/proxy.py on the Mac adds it to each request it forwards.

Only the agent runs here. tau2-loop on the Mac plays the customer, runs every tool call and
grades. Modal may preempt the container; the proxy retries until a new one is up.

    uv run modal deploy -m tau_telecom_sft.cloud.modal_app    # make modal-serve
    uv run modal app stop tau-telecom-sft                       # make modal-stop
"""

from __future__ import annotations

import subprocess

import modal

APP = "tau-telecom-sft"
SERVER = "Student"
BASE = "Qwen/Qwen3-4B-Instruct-2507"
# the snapshot the Mac's tokenizer, chat template and dataset were built with
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE_NAME = "student-qwen3-4b"
GPU = "L4"
GPU_USD_PER_S = 0.000222  # Modal's L4 list price, 19 Oct: the eval run's cost estimate
VLLM = "0.21.0"  # the pin in Modal's vLLM example, 19 Oct
PORT = 8000
MAX_MODEL_LEN = 32768
MINUTES = 60
# M5: {"student-qwen3-4b-sft-a": "<M4's train run id>"}; one --lora-modules entry each, read from
# the runs Volume, where M4's job writes its run folder
ADAPTERS: dict[str, str] = {}
LORA_RANK = 16

hf_cache = modal.Volume.from_name("tau-telecom-sft-hf", create_if_missing=True)
vllm_cache = modal.Volume.from_name("tau-telecom-sft-vllm", create_if_missing=True)
runs = modal.Volume.from_name("tau-telecom-sft-runs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "10"})
)

app = modal.App(APP)


def vllm_command(adapters: dict[str, str] | None = None) -> list[str]:
    """`vllm serve` with the plan's flags; the adapters are {served name: train run id}."""
    cmd = [
        "vllm",
        "serve",
        BASE,
        "--revision",
        BASE_REVISION,
        "--served-model-name",
        BASE_NAME,
        "--host",
        "0.0.0.0",  # noqa: S104 - inside the container; Modal's proxy is the only way in
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    if adapters:
        cmd += ["--enable-lora", "--max-lora-rank", str(LORA_RANK), "--lora-modules"]
        cmd += [f"{name}=/runs/{run}/adapter" for name, run in adapters.items()]
    return cmd


# no target_concurrency: one container (Modal's advice for a singleton), so one prefix cache
# serves every conversation
@app.server(
    image=image,
    gpu=GPU,
    port=PORT,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/runs": runs,
    },
    scaledown_window=15 * MINUTES,
    startup_timeout=20 * MINUTES,
)
class Student:
    @modal.enter()
    def start(self) -> None:
        self.process = subprocess.Popen(vllm_command(ADAPTERS))

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
