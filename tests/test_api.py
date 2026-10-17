"""The viewer's API, offline: the fixture build and tests/fixtures/runs/ (make_run_fixtures.py).

Every GET reads files; no response ever names a test task; the one write route appends to
review.jsonl and only from this machine; a missing runs/ is an ordinary empty state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tau_telecom_sft import config, split
from tau_telecom_sft.build import review
from tau_telecom_sft.build.dataset import Source, build
from tau_telecom_sft.serving.app import create_app
from tau_telecom_sft.serving.store import Paths

FIXTURE = config.FIXTURES_DIR / "board_mini.json"
RUNS = config.FIXTURES_DIR / "runs"
TRAIN = "20261017T000000Z_train_fixture-smoke"
BASE = "20261017T010000Z_eval_fixture-base"
TUNED = "20261017T020000Z_eval_fixture-tuned"
V = "smoke"


@pytest.fixture(scope="module")
def paths(tmp_path_factory: pytest.TempPathFactory) -> Paths:
    root: Path = tmp_path_factory.mktemp("viewer")
    out = root / "sft" / V
    build([Source("fixture", "tests/fixtures/board_mini", FIXTURE)], out, V)
    review.draw(out, n=3)
    return Paths(
        sft=root / "sft",
        runs=RUNS,
        raw_dirs=(config.FIXTURES_DIR,),
        checklist=config.ROOT / "ai_specs" / "s00_project_plan.md",
        cache=root / "cache",
        dist=root / "no-dist",
    )


@pytest.fixture(scope="module")
def client(paths: Paths) -> Iterator[TestClient]:
    with TestClient(create_app(paths, allowed_hosts=frozenset({"testclient"}), mlflow=False)) as c:
        yield c


def _no_test_task(text: str) -> None:
    named = [t for t in split.current().test if t in text]
    assert not named, f"a response names test task(s): {named[:2]}"


def get(c: TestClient, url: str, status: int = 200) -> Any:
    r = c.get(url)
    assert r.status_code == status, (url, r.status_code, r.text[:300])
    _no_test_task(r.text)
    return r.json()


def test_health_and_overview(client: TestClient) -> None:
    assert get(client, "/healthz")["mlflow"] == "down"  # mlflow=False: never contacted
    o = get(client, "/api/overview")
    assert [m["id"] for m in o["milestones"]][:3] == ["M0", "M1", "M2"]
    assert o["datasets"][0]["version"] == V
    assert [r["row"] for r in o["results"]] == ["R0", "R1", "R2", "R3", "R4", "R5"]
    assert all(r["run"] is None for r in o["results"])  # smoke runs never fill a results row
    assert {e["run_id"] for e in o["smoke"]} == {BASE, TUNED}
    (chain,) = o["chains"]
    assert chain["train_run"] == TRAIN and chain["adapter_sha256"] == "a" * 64
    assert [e["run_id"] for e in chain["evals"]] == [TUNED]
    assert [e["run_id"] for e in chain["bases"]] == [BASE]


def test_dataset_and_coverage(client: TestClient) -> None:
    d = get(client, f"/api/data/{V}")
    st = d["stats"]
    assert st["n_kept"] == 8 and d["summary"]["rows_here"] is True
    assert d["withheld"] >= 1  # the fixture's test task: counted, never listed
    assert "## Filters, in order" in d["card"]
    cov = get(client, f"/api/data/{V}/coverage")
    assert sum(i["test_tasks"] for i in cov["issues"]) > 0
    assert all(
        set(i) == {"issue", "kept_rows", "train_tasks", "test_tasks", "thin"} for i in cov["issues"]
    )
    get(client, "/api/data/v9", 404)


def test_rows_list_filters(client: TestClient) -> None:
    allrows = get(client, f"/api/data/{V}/rows")
    assert allrows["counts"]["kept"] == 8
    assert allrows["counts"]["kept"] + allrows["counts"]["dropped"] == allrows["total"]
    kept = get(client, f"/api/data/{V}/rows?status=kept")
    assert kept["total"] == 8 and {r["status"] for r in kept["items"]} == {"kept"}
    f4 = get(client, f"/api/data/{V}/rows?status=dropped&rule=F4")
    assert f4["total"] == 1 and f4["items"][0]["rule"] == "F4"
    page = get(client, f"/api/data/{V}/rows?limit=2&offset=2")
    assert len(page["items"]) == 2


def test_row_lenses(client: TestClient) -> None:
    kept = get(client, f"/api/data/{V}/rows?status=kept")["items"][0]["sim_id"]
    chat = get(client, f"/api/data/{V}/rows/{kept}")
    assert chat["status"] == "kept" and chat["lens"] == "chat"
    assert chat["messages"][0]["role"] == "system" and chat["n_tools"] == 13
    raw = get(client, f"/api/data/{V}/rows/{kept}?lens=raw")
    hidden = [m for m in raw["messages"] if not m["in_agent_view"]]
    assert hidden, "the customer's phone calls are in the raw lens, greyed"
    assert all(not m["offending"] for m in raw["messages"])
    for rule in ("F4", "F5", "F7"):
        sim = get(client, f"/api/data/{V}/rows?status=dropped&rule={rule}")["items"][0]["sim_id"]
        r = get(client, f"/api/data/{V}/rows/{sim}?lens=raw")
        assert r["drop"]["rule"] == rule
        assert [m for m in r["messages"] if m["offending"]], f"{rule}'s turn is marked"
        c = get(client, f"/api/data/{V}/rows/{sim}")
        assert c["status"] == "dropped" and c["messages"]  # rebuilt from raw by the converter
    get(client, f"/api/data/{V}/rows/no-such-row", 404)
    get(client, f"/api/data/{V}/rows/{kept}?lens=bogus", 422)


def _tokenizer_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from tau_telecom_sft.build.tokens import STUDENT_TOKENIZER

        return isinstance(try_to_load_from_cache(STUDENT_TOKENIZER, "tokenizer.json"), str)
    except Exception:  # noqa: BLE001 - no cache, no test
        return False


@pytest.mark.skipif(
    not _tokenizer_cached(), reason="the student's tokenizer is not in the HF cache"
)
def test_tokens_lens_marks_only_agent_turns(client: TestClient) -> None:
    kept = get(client, f"/api/data/{V}/rows?status=kept")["items"][0]["sim_id"]
    t = get(client, f"/api/data/{V}/rows/{kept}?lens=tokens")["tokens"]
    trained = [s["text"] for s in t["segments"] if s["trained"]]
    assert 0 < t["n_trained"] < t["n_tokens"]
    assert trained[0].startswith("Hi! How can I help you today?")
    assert all(s.rstrip("\n").endswith("<|im_end|>") for s in trained)
    assert not any("<|im_start|>user" in s for s in trained)


def test_review_writes_only_locally(client: TestClient, paths: Paths) -> None:
    st = get(client, f"/api/data/{V}/review")
    assert st["total"] == len(st["items"]) > 0 and st["signed_off"] is None
    first = st["items"][0]["sim_id"]
    bad = client.post(f"/api/data/{V}/review", json={"kind": "sign-off"})
    assert bad.status_code == 422  # not every row has a verdict
    ok = client.post(
        f"/api/data/{V}/review", json={"sim_id": first, "verdict": "good", "note": "clean"}
    )
    assert ok.status_code == 200 and ok.json()["saved"]["verdict"] == "good"
    assert (
        client.post(f"/api/data/{V}/review", json={"sim_id": "nope", "verdict": "good"}).status_code
        == 422
    )
    lines = (paths.sft / V / "review.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["sim_id"] == first
    with TestClient(create_app(paths, mlflow=False)) as remote:  # default: only 127.0.0.1 may write
        assert (
            remote.post(
                f"/api/data/{V}/review", json={"sim_id": first, "verdict": "bad"}
            ).status_code
            == 403
        )


def test_train_runs(client: TestClient) -> None:
    (row,) = get(client, "/api/train")["runs"]
    assert row["run_id"] == TRAIN and row["mask_ok"] is True
    assert row["settings"]["lora_r"] == 16 and row["settings"]["learning_rate"] == 1e-4
    assert [e["run_id"] for e in row["scored"]] == [TUNED]
    one = get(client, f"/api/train/{TRAIN}")
    assert [m["step"] for m in one["metrics"]] == [1, 2, 3, 4, 5, 6]
    assert one["mask"]["check"]["ok"] and one["config_flat"]["sft.max_length"] == 12288
    get(client, "/api/train/nope", 404)
    get(client, f"/api/train/{BASE}", 404)  # an eval run is not a training run


def test_eval_runs_and_the_gate(client: TestClient) -> None:
    runs = {e["run_id"]: e for e in get(client, "/api/eval")["runs"]}
    assert runs[TUNED]["passed"] == 2 and runs[BASE]["passed"] == 1
    e = get(client, f"/api/eval/{TUNED}")
    assert e["trials"] == 1 and len(e["grid"]) == 3
    assert e["pass_hat"]["1"] == pytest.approx(2 / 3)
    cmp = e["compare"]  # run.json's vs: its base, by default
    assert cmp["vs"]["run_id"] == BASE and (cmp["fixed"], cmp["broke"], cmp["same"]) == (1, 0, 2)
    assert cmp["p"] == pytest.approx(0.5)
    assert e["served"]["requests"] == 4 and e["served"]["system_prompt_sha256"] == ["b" * 64]
    rev = get(client, f"/api/eval/{BASE}?vs={TUNED}")["compare"]
    assert (rev["fixed"], rev["broke"]) == (0, 1)
    task = e["grid"][0]["task_id"]
    conv = get(client, f"/api/eval/{TUNED}/{task}/t1")
    assert conv["trial"] == 1 and conv["messages"] and conv["result"]["task_id"] == task
    get(client, f"/api/eval/{TUNED}/{task}/t9", 404)
    get(client, f"/api/eval/{TUNED}?vs=nope", 404)


def test_method_reads_the_latest_finished_job(client: TestClient) -> None:
    m = get(client, "/api/method")
    assert m["train_run"]["run_id"] == TRAIN and len(m["metrics"]) == 6 and m["mask"]
    assert m["dataset"]["version"] == V


def test_no_runs_is_an_empty_state(tmp_path: Path) -> None:
    empty = Paths(
        sft=tmp_path / "sft",
        runs=tmp_path / "runs",
        checklist=tmp_path / "none.md",
        cache=tmp_path / "c",
        dist=tmp_path / "d",
    )
    with TestClient(create_app(empty, mlflow=False)) as c:
        o = get(c, "/api/overview")
        assert (
            o["datasets"] == []
            and o["train_runs"] == []
            and o["chains"] == []
            and o["milestones"] == []
        )
        assert get(c, "/api/train")["runs"] == [] and get(c, "/api/eval")["runs"] == []
        assert get(c, "/api/method")["train_run"] is None
