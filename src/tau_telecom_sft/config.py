"""Paths, pins and settings. The only module that reads os.environ.

Keys (OPENROUTER_API_KEY, HF_TOKEN) come from ~/.env on the Mac and from the cloud's secret
store when running there; nothing here reads them until a milestone needs them.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "tau-telecom-sft"
DOMAIN = "telecom"

# tau2-bench, vendored as a submodule at the same pin as tau2-loop. Never edited.
TAU2_ROOT = ROOT / "vendor" / "tau2-bench"
TAU2_PIN = "2174a603f6d014ef94473ffa95957f6ce27100db"
TAU2_DATA_DIR = TAU2_ROOT / "data"
os.environ.setdefault("TAU2_DATA_DIR", str(TAU2_DATA_DIR))

DATA_DIR = ROOT / "data"
SPLIT_PATH = DATA_DIR / "splits" / "telecom.json"
SPLIT_MANIFEST = DATA_DIR / "splits" / "manifest.json"
SFT_DIR = DATA_DIR / "sft"
CONFIGS_DIR = ROOT / "configs"
RAW_DIR = ROOT / "raw"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

# The sibling the split is copied from, and scored in (D5). Absent in CI.
TAU2_LOOP_ROOT = ROOT.parent / "tau2-loop"
TAU2_LOOP_SPLIT = TAU2_LOOP_ROOT / "data" / "splits" / "telecom.json"
SPLIT_VERSION = 3

# Everything the trainer reads. The leak check scans these (AGENTS.md rule 1).
TRAINER_INPUTS = (SFT_DIR, CONFIGS_DIR)

# Central MLflow (nmp-central-ai). Experiment names only: ids come from `make mlflow-init`.
EXPERIMENT_DATA = f"{PROJECT}/data"
EXPERIMENT_TRAIN = f"{PROJECT}/train"
EXPERIMENT_EVAL = f"{PROJECT}/eval"
REGISTERED_MODEL = f"{PROJECT}-student"

# F10's starting limit (§5): M1 sets the real one from the student's tokenizer.
MAX_TOKENS = 24_576


def quiet_tau2() -> None:
    """tau2 logs at DEBUG through loguru by default; keep only warnings on our console."""
    try:
        from loguru import logger

        logger.remove()
        logger.add(lambda m: print(m, end=""), level="WARNING")
    except Exception:  # noqa: BLE001 - logging is a convenience
        return


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str
    env: str
    code_sha: str


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


@functools.cache
def settings() -> Settings:
    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        env=os.environ.get("TAUSFT_ENV", "local"),
        code_sha=os.environ.get("GIT_SHA") or _git_sha(),
    )


def modal_proxy_headers() -> dict[str, str]:
    """The proxy token a request needs to reach the Student on Modal (s02), from ~/.env; never
    logged. Empty when unset, and Modal's proxy then answers 401."""
    settings()  # loads ~/.env and .env
    key = os.environ.get("MODAL_PROXY_TOKEN_ID")
    secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    return {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}
