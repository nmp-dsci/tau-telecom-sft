"""The student on Modal (s02): vLLM on one L4 in place of EC2's `docker run` (M3, M5), and the
trainer on one L4 in place of EC2's training container (M4).

`Student` is vLLM's own OpenAI-compatible server for Qwen3-4B-Instruct-2507 in bf16, with the
plan's flags (§8): a 32K context, prefix caching and the hermes tool-call parser. It serves the
untrained base as `student-qwen3-4b` (R0) and each adapter in ADAPTERS under its own name (M5:
`student-qwen3-4b-sft-a`), and tau2-loop's student versions ask for those names, so swapping the
base for the tuned model is a name. Modal's proxy requires a token, its default for a Server;
serve/proxy.py on the Mac adds it to each request it forwards.

Only the agent runs here. tau2-loop on the Mac plays the customer, runs every tool call and
grades. Modal may preempt the container; the proxy retries until a new one is up.

`train` is train/sft.py on one L4, with uv.lock's versions of the training stack and only
src/ and configs/ from the repo: no tau2, no raw/, no test task (rule 1). It reads dataset v1
from the data Volume and the weights from the hf Volume the Student filled at M3, and writes
its run folder (checkpoints, adapter) to the runs Volume, where the Student finds the adapter
at M5 and `tausft sync-runs` copies the folder home. Modal can preempt it too, and re-runs an
input cut short by a preemption or an OOM on its own with the same arguments, so the same run
id, and the trainer resumes from its newest checkpoint. No `retries`: they would add re-runs of
the job's own exceptions (the data check, the mask check, a bug), and those fail again.
Nothing here reaches MLflow; sync-runs logs the folder from the Mac.

    uv run modal deploy -m tau_telecom_sft.cloud.modal_app                    # make modal-serve
    uv run modal run --detach -m tau_telecom_sft.cloud.modal_app::train_job   # make modal-train
    uv run modal app stop tau-telecom-sft                                     # make modal-stop
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

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

# M4's trainer: uv.lock's versions (tests/test_modal.py holds them to it). Liger is not among
# them: TRL 1.16 scores only the trained positions, so 24K tokens fit an L4 without it.
TRAIN_PINS = (
    "torch==2.14.1",
    "transformers==5.20.1",
    "trl==1.16.0",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "datasets==5.1.0",
    "pyyaml==6.0.3",
    "rich==15.0.0",
    "python-dotenv==1.2.4",
    "jinja2==3.1.6",
)
TRAIN_CPU = 4
TRAIN_MEMORY_GIB = 16
# list prices, 19 Oct (s02): the GPU, then per core and per GiB, each per second
TRAIN_USD_PER_S = GPU_USD_PER_S + TRAIN_CPU * 0.0000131 + TRAIN_MEMORY_GIB * 0.00000222
TRAIN_HOURS = 8  # per attempt; the estimate is 3 to 4 hours, and a re-run starts a new clock
REPO = "/repo"  # config.ROOT inside the trainer
LOCAL_ROOT = Path(__file__).resolve().parents[3]

hf_cache = modal.Volume.from_name("tau-telecom-sft-hf", create_if_missing=True)
vllm_cache = modal.Volume.from_name("tau-telecom-sft-vllm", create_if_missing=True)
RUNS_VOLUME = "tau-telecom-sft-runs"
runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
data = modal.Volume.from_name("tau-telecom-sft-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "10"})
)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*TRAIN_PINS)
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONPATH": f"{REPO}/src", "TAUSFT_ROOT": REPO})
    .add_local_dir(
        LOCAL_ROOT / "src" / "tau_telecom_sft",
        f"{REPO}/src/tau_telecom_sft",
        ignore=["**/__pycache__"],
    )
    .add_local_dir(LOCAL_ROOT / "configs", f"{REPO}/configs")
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


@app.function(
    image=train_image,
    gpu=GPU,
    cpu=TRAIN_CPU,
    memory=TRAIN_MEMORY_GIB * 1024,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{REPO}/data/sft": data,
        f"{REPO}/runs": runs,
    },
    timeout=TRAIN_HOURS * 60 * MINUTES,
)
def train(
    config_file: str,
    run_id: str,
    code_sha: str,
    max_steps: int | None = None,
    longest: int | None = None,
) -> dict[str, Any]:
    """train/sft.py on one L4, into the runs Volume. Modal's re-run after a preemption gets the
    same arguments, so the same folder, and resumes from its newest checkpoint."""
    import os
    import time

    t0 = time.time()
    os.environ["GIT_SHA"] = code_sha  # there is no .git in here: the caller's commit
    from tau_telecom_sft.train.sft import train as run_train
    from tau_telecom_sft.train.sft import write_json

    run_dir = run_train(
        Path(REPO) / config_file,
        run_id=run_id,
        max_steps=max_steps,
        longest=longest,
        use_mlflow=False,  # MLflow listens on the Mac; sync-runs logs the folder
    )
    meta: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
    meta["cost_usd"] = round((time.time() - t0) * TRAIN_USD_PER_S, 2)
    meta["cost_basis"] = (
        f"Modal list price × this attempt's wall time: {GPU} + {TRAIN_CPU} cores + "
        f"{TRAIN_MEMORY_GIB} GiB; an attempt cut short by a preemption is not counted"
    )
    write_json(run_dir / "run.json", meta)
    return meta


@app.local_entrypoint()
def train_job(config: str = "configs/sft_a.yaml", max_steps: int = 0, longest: int = 0) -> None:
    """`make modal-train`: the run id is minted here, so every re-run of the job writes one folder;
    with `modal run --detach` the job carries on if the Mac sleeps."""
    from tau_telecom_sft import config as cfg
    from tau_telecom_sft.train.sft import load_config, new_run_id, run_label

    code_sha = cfg.clean_code_sha()
    steps = max_steps or None
    run_id = new_run_id("train", run_label(load_config(Path(config)), steps))
    call = train.spawn(config, run_id, code_sha, steps, longest or None)
    # the run's own app id: `make modal-stop` names the deployed Student, not this job
    print(f"{run_id} at {code_sha}: call {call.object_id}")
    print(f"logs: uv run modal app logs -f {app.app_id} · stop: uv run modal app stop {app.app_id}")
    print(f"home: make sync-runs RUN={run_id} WATCH=60")
    meta = call.get()  # .spawn().get(), not .remote(): a .remote() call expires after 24 h
    print(
        f"{run_id}: {meta['status']}, {meta.get('steps')} steps, final loss "
        f"{meta.get('final_loss')}, about ${meta['cost_usd']}"
    )
