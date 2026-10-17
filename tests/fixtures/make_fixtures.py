"""Write tests/fixtures/board_mini.json: a tiny tau2 results file, handwritten, built with tau2's
own data models so it loads exactly as a board file does.

Every conversation is written by hand for the tests and the smoke build; none is a model's
output. Each one is either a clean pass or breaks exactly one rule (§5):

  task A (train): two clean passes; one failure                       -> kept 2, F1 1
  task T (test):  one pass                                            -> F2
  task B (train): three clean passes; a long one; one cut off by the
                  step limit; one calling a phone tool; one with a tool
                  error; one writing a tool call into its reply         -> kept 3, F6, F3, F4, F5, F7
  task C (train): four passes with one tool path; one too long to fit   -> kept 3, F9, F10

17 conversations: 8 kept, 9 dropped. F8 cannot fire while D6 is off (F2 drops any task outside
the 74 first), so tests/test_filters.py covers it with the generated pool switched on.

Run: uv run python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from tau_telecom_sft import config, split

config.quiet_tau2()

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage  # noqa: E402
from tau2.data_model.simulation import (  # noqa: E402
    AgentInfo,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.domains.telecom.environment import get_environment  # noqa: E402
from tau2.environment.environment import EnvironmentInfo  # noqa: E402
from tau2.runner.helpers import load_tasks  # noqa: E402

OUT = Path(__file__).with_name("board_mini.json")
_ids = itertools.count(1)


class Talk:
    """A conversation as tau2 saves it: the agent's turns, the customer's, and both tools."""

    def __init__(self) -> None:
        self.messages: list[Any] = [
            AssistantMessage(role="assistant", content="Hi! How can I help you today?")
        ]

    def agent(self, text: str) -> Talk:
        self.messages.append(AssistantMessage(role="assistant", content=text))
        return self

    def customer(self, text: str) -> Talk:
        self.messages.append(UserMessage(role="user", content=text))
        return self

    def call(self, name: str, result: str, *, error: bool = False, **args: Any) -> Talk:
        cid = f"call_{next(_ids)}"
        tc = ToolCall(id=cid, name=name, arguments=args, requestor="assistant")
        self.messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
        self.messages.append(
            ToolMessage(id=cid, role="tool", content=result, requestor="assistant", error=error)
        )
        return self

    def phone(self, name: str, result: str) -> Talk:
        """The customer's own phone tool: tau2 never shows it to the agent."""
        cid = f"call_{next(_ids)}"
        tc = ToolCall(id=cid, name=name, arguments={}, requestor="user")
        self.messages.append(UserMessage(role="user", content=None, tool_calls=[tc]))
        self.messages.append(ToolMessage(id=cid, role="tool", content=result, requestor="user"))
        return self


CUSTOMER = '{"customer_id": "C1001", "full_name": "John Smith", "line_ids": ["L1002"]}'
LINE = '{"line_id": "L1002", "status": "Active", "roaming_enabled": false}'


def opening(t: Talk) -> Talk:
    return (
        t.customer("My mobile data stopped working. I'm in France at the moment.")
        .agent("I can help with that. What's the phone number on the affected line?")
        .customer("It's 555-123-2002.")
        .call("get_customer_by_phone", CUSTOMER, phone_number="555-123-2002")
    )


def roaming_fix(t: Talk) -> Talk:
    return (
        t.call("get_details_by_id", LINE, id="L1002")
        .agent("Roaming is off on your line, which stops data abroad. Shall I turn it on?")
        .customer("Yes, please.")
        .call("enable_roaming", "Roaming enabled on L1002", customer_id="C1001", line_id="L1002")
        .agent("Done. Please turn on Data Roaming in your phone's settings, then run a speed test.")
        .phone("toggle_roaming", "Data Roaming: ON")
        .phone("run_speed_test", "Speed test: 200 Mbps, excellent")
        .customer("The speed test says excellent. Thanks! ###STOP###")
    )


def clean() -> Talk:
    return roaming_fix(opening(Talk()))


def airplane_first() -> Talk:
    """A different clean path: the agent checks airplane mode before roaming."""
    t = opening(Talk())
    t.agent("Is airplane mode on? Please check your status bar.")
    t.phone("check_status_bar", "Status Bar: Airplane Mode | 80%")
    t.customer("Yes, the airplane icon is showing. I've turned it off now.")
    return roaming_fix(t)


def long_one() -> Talk:
    t = opening(Talk())
    for _ in range(5):
        t.call("get_details_by_id", LINE, id="L1002")
        t.agent("Let me look at that once more.")
        t.customer("Okay.")
    return roaming_fix(t)


def phone_tool_by_agent() -> Talk:
    t = opening(Talk())
    t.call("toggle_airplane_mode", "Tool toggle_airplane_mode not found", error=False)
    return roaming_fix(t)


def tool_error() -> Talk:
    t = opening(Talk())
    t.call("get_details_by_id", "Error: id is required", error=True, id="")
    return roaming_fix(t)


def text_call() -> Talk:
    t = opening(Talk())
    t.agent(
        '<tool_call>\n{"name": "get_details_by_id", "arguments": {"id": "L1002"}}\n</tool_call>'
    )
    t.customer("Okay?")
    return roaming_fix(t)


def too_long() -> Talk:
    """Over F10's limit, on a tool path of its own so F9 does not catch it first."""
    t = Talk()
    t.customer("My mobile data stopped working. Here is everything on my screen: " + "x" * 80_000)
    t.agent("I can help with that. What's the phone number on the affected line?")
    t.customer("It's 555-123-2002.")
    t.call("get_customer_by_phone", CUSTOMER, phone_number="555-123-2002")
    t.call(
        "get_data_usage",
        '{"data_used_gb": 2.1, "data_limit_gb": 10}',
        customer_id="C1001",
        line_id="L1002",
    )
    return roaming_fix(t)


def sim(
    task_id: str, trial: int, talk: Talk, *, reward: float = 1.0, end: str = "user_stop"
) -> Any:
    return SimulationRun(
        id=f"fixture-{task_id[:24]}-{trial}-{next(_ids)}",
        task_id=task_id,
        start_time="2026-10-17T00:00:00",
        end_time="2026-10-17T00:05:00",
        duration=300.0,
        termination_reason=TerminationReason(end),
        reward_info=RewardInfo(reward=reward),
        messages=talk.messages,
        trial=trial,
        seed=300,
    )


def main() -> None:
    s = split.current()
    a, b, c = s.train[0], s.train[1], s.train[2]
    t = s.test[0]
    sims = [
        sim(a, 0, clean()),
        sim(a, 1, airplane_first()),
        sim(a, 2, clean(), reward=0.0),
        sim(t, 0, clean()),
        sim(b, 0, clean()),
        sim(b, 1, airplane_first()),
        sim(b, 2, clean()),
        sim(b, 3, long_one()),
        sim(b, 4, clean(), end="max_steps"),
        sim(b, 5, phone_tool_by_agent()),
        sim(b, 6, tool_error()),
        sim(b, 7, text_call()),
        *(sim(c, i, clean()) for i in range(4)),
        sim(c, 4, too_long()),
    ]
    env = get_environment()
    tasks = {task.id: task for task in load_tasks("telecom")}
    results = Results(
        timestamp="2026-10-17T00:00:00",
        info=Info(
            git_commit=config.TAU2_PIN,
            num_trials=8,
            max_steps=200,
            max_errors=10,
            user_info=UserInfo(implementation="user_simulator", llm="fixture/handwritten"),
            agent_info=AgentInfo(implementation="llm_agent", llm="fixture/handwritten"),
            environment_info=EnvironmentInfo(domain_name="telecom", policy=env.get_policy()),
            seed=300,
        ),
        tasks=[tasks[x] for x in (a, b, c, t)],
        simulations=sims,
    )
    OUT.write_text(json.dumps(json.loads(results.model_dump_json()), indent=1) + "\n")
    print(f"wrote {OUT} ({len(sims)} conversations)")


if __name__ == "__main__":
    main()
