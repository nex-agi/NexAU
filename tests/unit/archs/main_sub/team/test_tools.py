# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for claim_task and finish_team tools."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexau.archs.main_sub.team.tools.claim_task import claim_task
from nexau.archs.main_sub.team.tools.finish_team import finish_team
from nexau.archs.main_sub.team.types import (
    ClaimTaskResult,
    FinishTeamResult,
    TaskBlockedError,
    TaskInfo,
    TeammateInfo,
    ToolError,
)
from nexau.archs.session.task_lock_service import LockConflictError

# --- Helpers ---


def make_task_info(
    task_id: str = "T-001",
    title: str = "Test Task",
    status: str = "pending",
    assignee_agent_id: str | None = None,
    deliverable_path: str | None = ".nexau/tasks/T-001-test-task.md",
) -> TaskInfo:
    return TaskInfo(
        task_id=task_id,
        title=title,
        description="",
        status=status,
        priority=0,
        dependencies=[],
        assignee_agent_id=assignee_agent_id,
        result_summary=None,
        created_by="leader",
        is_blocked=False,
        deliverable_path=deliverable_path,
    )


def make_agent_state(
    agent_id: str = "agent1",
    is_leader: bool = False,
    task_board: MagicMock | None = None,
    team: MagicMock | None = None,
) -> MagicMock:
    state = MagicMock()
    state.agent_id = agent_id

    ts = MagicMock()
    ts.is_leader = is_leader
    ts.task_board = task_board or MagicMock()
    ts.team = team or MagicMock()
    state.team_state = ts

    return state


# --- claim_task tests ---


class TestClaimTask:
    def test_self_claim_success(self):
        async def run():
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[])
            task_board.claim_task = AsyncMock()
            task_board.get_task_info = AsyncMock(return_value=make_task_info())

            state = make_agent_state(agent_id="agent1", task_board=task_board)

            result = await claim_task(task_id="T-001", agent_state=state)

            assert isinstance(result, ClaimTaskResult)
            assert result.task_id == "T-001"
            assert result.assignee_agent_id == "agent1"
            assert result.status == "claimed"

        asyncio.run(run())

    def test_leader_assignment_success(self):
        async def run():
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[])
            task_board.claim_task = AsyncMock()
            task_board.get_task_info = AsyncMock(return_value=make_task_info())

            team = MagicMock()
            team.send_message_to_agent = MagicMock()

            state = make_agent_state(
                agent_id="leader",
                is_leader=True,
                task_board=task_board,
                team=team,
            )

            result = await claim_task(
                task_id="T-001",
                agent_state=state,
                assignee_agent_id="agent2",
            )

            assert isinstance(result, ClaimTaskResult)
            assert result.assignee_agent_id == "agent2"
            team.send_message_to_agent.assert_called_once()

        asyncio.run(run())

    def test_non_leader_cannot_assign_to_others(self):
        async def run():
            state = make_agent_state(agent_id="agent1", is_leader=False)

            result = await claim_task(
                task_id="T-001",
                agent_state=state,
                assignee_agent_id="agent2",
            )

            assert isinstance(result, ToolError)
            assert result.code == "permission_denied"

        asyncio.run(run())

    def test_busy_agent_cannot_claim_another_task(self):
        async def run():
            existing_task = make_task_info(
                task_id="T-001",
                status="in_progress",
                assignee_agent_id="agent1",
            )
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[existing_task])

            state = make_agent_state(agent_id="agent1", task_board=task_board)

            result = await claim_task(task_id="T-002", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "busy"

        asyncio.run(run())

    def test_lock_conflict_returns_conflict_error(self):
        async def run():
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[])
            task_board.claim_task = AsyncMock(side_effect=LockConflictError("conflict"))

            state = make_agent_state(agent_id="agent1", task_board=task_board)

            result = await claim_task(task_id="T-001", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "conflict"

        asyncio.run(run())

    def test_blocked_task_returns_blocked_error(self):
        async def run():
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[])
            task_board.claim_task = AsyncMock(side_effect=TaskBlockedError("blocked"))

            state = make_agent_state(agent_id="agent1", task_board=task_board)

            result = await claim_task(task_id="T-001", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "blocked"

        asyncio.run(run())

    def test_no_team_state_raises(self):
        async def run():
            state = MagicMock()
            state.team_state = None

            with pytest.raises(RuntimeError):
                await claim_task(task_id="T-001", agent_state=state)

        asyncio.run(run())


# --- finish_team tests ---


class TestFinishTeam:
    def test_leader_finishes_with_all_completed(self):
        async def run():
            completed_task = make_task_info(task_id="T-001", status="completed")
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[completed_task])

            team = MagicMock()
            team.get_teammate_info = MagicMock(return_value=[])

            state = make_agent_state(
                agent_id="leader",
                is_leader=True,
                task_board=task_board,
                team=team,
            )

            result = await finish_team(summary="all done", agent_state=state)

            assert isinstance(result, FinishTeamResult)
            assert result.summary == "all done"
            assert result.completed_tasks == 1
            assert result.total_tasks == 1

        asyncio.run(run())

    def test_non_leader_cannot_finish(self):
        async def run():
            state = make_agent_state(agent_id="agent1", is_leader=False)

            result = await finish_team(summary="done", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "permission_denied"

        asyncio.run(run())

    def test_finish_blocked_by_incomplete_tasks(self):
        async def run():
            pending_task = make_task_info(task_id="T-001", status="pending")
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[pending_task])

            team = MagicMock()
            team.get_teammate_info = MagicMock(return_value=[])

            state = make_agent_state(
                agent_id="leader",
                is_leader=True,
                task_board=task_board,
                team=team,
            )

            result = await finish_team(summary="done", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "invalid_state"
            assert "incomplete" in result.error

        asyncio.run(run())

    def test_finish_blocked_by_running_teammates(self):
        async def run():
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[])

            running_teammate = TeammateInfo(agent_id="agent1", role_name="researcher", status="running")
            team = MagicMock()
            team.get_teammate_info = MagicMock(return_value=[running_teammate])

            state = make_agent_state(
                agent_id="leader",
                is_leader=True,
                task_board=task_board,
                team=team,
            )

            result = await finish_team(summary="done", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "invalid_state"
            assert "running teammate" in result.error

        asyncio.run(run())

    def test_finish_blocked_by_both_incomplete_and_running(self):
        async def run():
            pending_task = make_task_info(task_id="T-001", status="in_progress")
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[pending_task])

            running_teammate = TeammateInfo(agent_id="agent1", role_name="coder", status="running")
            team = MagicMock()
            team.get_teammate_info = MagicMock(return_value=[running_teammate])

            state = make_agent_state(
                agent_id="leader",
                is_leader=True,
                task_board=task_board,
                team=team,
            )

            result = await finish_team(summary="done", agent_state=state)

            assert isinstance(result, ToolError)
            assert result.code == "invalid_state"
            # Both reasons should appear
            assert "incomplete" in result.error
            assert "running teammate" in result.error

        asyncio.run(run())

    def test_finish_with_no_tasks(self):
        async def run():
            task_board = MagicMock()
            task_board.list_tasks = AsyncMock(return_value=[])

            team = MagicMock()
            team.get_teammate_info = MagicMock(return_value=[])

            state = make_agent_state(
                agent_id="leader",
                is_leader=True,
                task_board=task_board,
                team=team,
            )

            result = await finish_team(summary="nothing to do", agent_state=state)

            assert isinstance(result, FinishTeamResult)
            assert result.completed_tasks == 0
            assert result.total_tasks == 0

        asyncio.run(run())

    def test_no_team_state_raises(self):
        async def run():
            state = MagicMock()
            state.team_state = None

            with pytest.raises(RuntimeError):
                await finish_team(summary="done", agent_state=state)

        asyncio.run(run())
