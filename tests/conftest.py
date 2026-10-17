from __future__ import annotations

from pathlib import Path

import pytest

from tau_telecom_sft import config
from tau_telecom_sft.build.dataset import Built, Source, build
from tau_telecom_sft.sources.results import Conversation, conversations

FIXTURE = config.FIXTURES_DIR / "board_mini.json"


@pytest.fixture(scope="session")
def fixture_convs() -> list[Conversation]:
    return conversations(FIXTURE, "fixture tests/fixtures/board_mini")


@pytest.fixture(scope="session")
def fixture_build(tmp_path_factory: pytest.TempPathFactory) -> Built:
    out: Path = tmp_path_factory.mktemp("build")
    return build([Source("fixture", "tests/fixtures/board_mini", FIXTURE)], out, "smoke")
