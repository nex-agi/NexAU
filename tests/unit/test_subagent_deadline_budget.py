"""Public seam tests for sub-agent deadline and per-run budget contracts."""

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config.schema import SubAgentConfigEntry
from nexau.archs.main_sub.execution.subagent_manager import SubAgentBudgetExhaustedError
from nexau.archs.tool.builtin.agent_tool import call_sub_agent_async


def test_sub_agent_entry_round_trips_optional_controls() -> None:
    entry = SubAgentConfigEntry(
        name="worker",
        config_path="worker.yaml",
        timeout_seconds=2.5,
        max_calls_per_run=2,
    )
    assert entry.model_dump(exclude_none=True)["timeout_seconds"] == 2.5
    assert entry.model_dump(exclude_none=True)["max_calls_per_run"] == 2


def test_sub_agent_entry_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        SubAgentConfigEntry(name="worker", config_path="worker.yaml", timeout_seconds=0)


def test_sub_agent_entry_rejects_non_positive_call_budget() -> None:
    with pytest.raises(ValueError):
        SubAgentConfigEntry(name="worker", config_path="worker.yaml", max_calls_per_run=0)


class _Manager:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    async def call_sub_agent_async(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_deadline_result_is_structured_terminal() -> None:
    manager = _Manager({"status": "partial", "reason": "AGENT_DEADLINE_EXCEEDED", "terminal": True})
    state = SimpleNamespace(subagent_manager=manager)
    result = asyncio.run(call_sub_agent_async("worker", "work", agent_state=cast(AgentState, state)))
    assert result["status"] == "partial"
    assert result["reason"] == "AGENT_DEADLINE_EXCEEDED"
    assert result["terminal"] is True


def test_budget_exhaustion_does_not_start_a_second_call() -> None:
    manager = _Manager(SubAgentBudgetExhaustedError("worker"))
    state = SimpleNamespace(subagent_manager=manager)
    result = asyncio.run(call_sub_agent_async("worker", "work", agent_state=cast(AgentState, state)))
    assert result["reason"] == "DELEGATION_BUDGET_EXHAUSTED"
    assert result["terminal"] is True
    assert manager.calls == 1


def test_max_iterations_marker_is_normalized_to_partial() -> None:
    manager = _Manager("answer\n\n[Note: Maximum iteration limit reached]")
    state = SimpleNamespace(subagent_manager=manager)
    result = asyncio.run(call_sub_agent_async("worker", "work", agent_state=cast(AgentState, state)))
    assert result["reason"] == "MAX_ITERATIONS_REACHED"
    assert result["status"] == "partial"
    assert result["terminal"] is True
