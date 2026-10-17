"""Write frontend/src/test/api.json: the real API's answers on the offline fixtures.

The viewer's vitest suite (routes.test.tsx) serves these to every page, so the frontend is
tested against exactly the shapes serving/app.py returns, not a hand-copied imitation. It reads
the fixture build (board_mini.json) and tests/fixtures/runs/ (make_run_fixtures.py).

Run: uv run python tests/fixtures/make_api_fixtures.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tau_telecom_sft import config
from tau_telecom_sft.build import review
from tau_telecom_sft.build.dataset import Source, build
from tau_telecom_sft.serving.app import create_app
from tau_telecom_sft.serving.store import Paths

OUT = config.ROOT / "frontend" / "src" / "test" / "api.json"
V = "smoke"
TRAIN = "20261017T000000Z_train_fixture-smoke"
BASE = "20261017T010000Z_eval_fixture-base"
TUNED = "20261017T020000Z_eval_fixture-tuned"


def _trim(doc: Any) -> Any:
    """Long texts (the 30K-character system prompt) cut, so the fixture stays small."""
    if isinstance(doc, dict):
        return {k: _trim(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_trim(v) for v in doc]
    if isinstance(doc, str) and len(doc) > 2500:
        return doc[:2500] + " …"
    return doc


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "sft" / V
        build(
            [
                Source(
                    "fixture", "tests/fixtures/board_mini", config.FIXTURES_DIR / "board_mini.json"
                )
            ],
            out,
            V,
        )
        review.draw(out, n=3)
        paths = Paths(
            sft=root / "sft",
            runs=config.FIXTURES_DIR / "runs",
            raw_dirs=(config.FIXTURES_DIR,),
            checklist=config.ROOT / "ai_specs" / "s00_project_plan.md",
            cache=root / "cache",
            dist=root / "no-dist",
        )
        c = TestClient(create_app(paths, mlflow=False))
        rows = c.get(f"/api/data/{V}/rows").json()["items"]
        kept = next(r["sim_id"] for r in rows if r["status"] == "kept")
        f4 = next(r["sim_id"] for r in rows if r["rule"] == "F4")
        task = c.get(f"/api/eval/{TUNED}").json()["grid"][0]["task_id"]
        urls = [
            "/healthz",
            "/api/overview",
            "/api/data",
            f"/api/data/{V}",
            f"/api/data/{V}/coverage",
            f"/api/data/{V}/rows",
            f"/api/data/{V}/rows?status=dropped&rule=F4",
            f"/api/data/{V}/rows/{kept}?lens=chat",
            f"/api/data/{V}/rows/{kept}?lens=raw",
            f"/api/data/{V}/rows/{kept}?lens=tokens",
            f"/api/data/{V}/rows/{f4}?lens=raw",
            f"/api/data/{V}/review",
            "/api/train",
            f"/api/train/{TRAIN}",
            "/api/eval",
            f"/api/eval/{TUNED}",
            f"/api/eval/{BASE}",
            f"/api/eval/{TUNED}/{task}/t1",
            "/api/method",
        ]
        answers: dict[str, Any] = {}
        for u in urls:
            r = c.get(u)
            assert r.status_code == 200, (u, r.status_code, r.text[:200])
            answers[u] = _trim(r.json())
        doc = {
            "ids": {
                "version": V,
                "kept": kept,
                "f4": f4,
                "train": TRAIN,
                "base": BASE,
                "tuned": TUNED,
                "task": task,
            },
            "answers": answers,
        }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(
        f"wrote {OUT.relative_to(config.ROOT)}: {len(urls)} answers, {OUT.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
