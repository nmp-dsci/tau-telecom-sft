"""One loader for every conversation file: tau2's own `Results` model at the pin.

The board files (S1, S2), a tau2-loop run's `tau2_results.json` and the student's scored
runs are all tau2's results JSON, so they load the same way (§4). M1 adds the mapping for
any field the February board files lack; until then a file that fails validation fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau_telecom_sft import config


@dataclass(frozen=True)
class Conversation:
    """One simulation from one source file, with the task it played."""

    source: str  # e.g. "S1 board/glm-5", or "fixture" for the smoke build
    sim: Any  # tau2.data_model.simulation.SimulationRun
    task: Any | None  # tau2.data_model.tasks.Task

    @property
    def task_id(self) -> str:
        return str(self.sim.task_id)

    @property
    def trial(self) -> int | None:
        return None if self.sim.trial is None else int(self.sim.trial)

    @property
    def reward(self) -> float | None:
        info = self.sim.reward_info
        return None if info is None else float(info.reward)

    @property
    def termination(self) -> str:
        return str(getattr(self.sim.termination_reason, "value", self.sim.termination_reason))

    @property
    def messages(self) -> list[Any]:
        return list(self.sim.messages or [])

    @property
    def key(self) -> tuple[str, str, int, str]:
        """A stable order for every rule that keeps the first few (F9)."""
        return (
            self.source,
            self.task_id,
            self.trial if self.trial is not None else -1,
            self.sim.id,
        )


def load_results(path: Path) -> Any:
    config.quiet_tau2()
    from tau2.data_model.simulation import Results

    return Results.load(path)


def conversations(path: Path, source: str) -> list[Conversation]:
    results = load_results(path)
    tasks = {t.id: t for t in results.tasks}
    return [Conversation(source, sim, tasks.get(sim.task_id)) for sim in results.simulations]
