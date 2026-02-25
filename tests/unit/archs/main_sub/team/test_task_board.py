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

"""Unit tests for TaskBoard."""

import asyncio

import pytest

from nexau.archs.main_sub.team.task_board import TaskBoard, _slugify
from nexau.archs.main_sub.team.types import TaskBlockedError
from nexau.archs.session.models.team_task import TeamTaskModel
from nexau.archs.session.models.team_task_lock import TeamTaskLockModel
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.session.task_lock_service import LockConflictError, TaskLockService

# --- _slugify tests ---


def test_slugify_basic():
    assert _slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert _slugify("Task: Do Something!") == "task-do-something"


def test_slugify_truncate():
    long_title = "a" * 100
    result = _slugify(long_title, max_length=10)
    assert len(result) <= 10


def test_slugify_empty_fallback():
    assert _slugify("!!!") == "task"


def test_slugify_strips_hyphens():
    assert _slugify("-hello-") == "hello"


# --- TaskBoard fixtures ---


def make_board(engine: InMemoryDatabaseEngine) -> TaskBoard:
    lock_service = TaskLockService(engine=engine)
    return TaskBoard(
        engine=engine,
        task_lock_service=lock_service,
        user_id="u1",
        session_id="s1",
        team_id="t1",
    )


# --- TaskBoard tests ---


class TestTaskBoardCreate:
    def test_create_task_returns_task_info(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            info = await board.create_task(title="Write report", created_by="leader")

            assert info.task_id == "T-001"
            assert info.title == "Write report"
            assert info.status == "pending"
            assert info.deliverable_path == ".nexau/tasks/T-001-write-report.md"
            assert not info.is_blocked

        asyncio.run(run())

    def test_create_multiple_tasks_increments_id(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            t1 = await board.create_task(title="Task A")
            t2 = await board.create_task(title="Task B")

            assert t1.task_id == "T-001"
            assert t2.task_id == "T-002"

        asyncio.run(run())

    def test_create_task_with_dependencies(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            t1 = await board.create_task(title="Task A")
            t2 = await board.create_task(title="Task B", dependencies=[t1.task_id])

            assert t2.is_blocked  # T-001 is pending, so T-002 is blocked

        asyncio.run(run())

    def test_create_task_not_blocked_when_dep_completed(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            t1 = await board.create_task(title="Task A")
            await board.claim_task(task_id=t1.task_id, assignee_agent_id="agent1")
            await board.update_status(task_id=t1.task_id, status="completed")

            t2 = await board.create_task(title="Task B", dependencies=[t1.task_id])
            assert not t2.is_blocked

        asyncio.run(run())


class TestTaskBoardListAndGet:
    def test_list_tasks_empty(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            tasks = await board.list_tasks()
            assert tasks == []

        asyncio.run(run())

    def test_list_tasks_with_status_filter(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            t2 = await board.create_task(title="Task B")
            await board.claim_task(task_id=t2.task_id, assignee_agent_id="agent1")

            in_progress = await board.list_tasks(status="in_progress")
            assert len(in_progress) == 1
            assert in_progress[0].task_id == "T-002"

        asyncio.run(run())

    def test_get_task_info(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="My Task", description="desc", priority=1)
            info = await board.get_task_info("T-001")

            assert info.task_id == "T-001"
            assert info.description == "desc"
            assert info.priority == 1

        asyncio.run(run())

    def test_get_task_info_not_found(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            with pytest.raises(ValueError, match="not found"):
                await board.get_task_info("T-999")

        asyncio.run(run())


class TestTaskBoardClaimRelease:
    def test_claim_task(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            await board.claim_task(task_id="T-001", assignee_agent_id="agent1")

            info = await board.get_task_info("T-001")
            assert info.status == "in_progress"
            assert info.assignee_agent_id == "agent1"

        asyncio.run(run())

    def test_claim_already_assigned_raises(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            await board.claim_task(task_id="T-001", assignee_agent_id="agent1")

            with pytest.raises(LockConflictError):
                await board.claim_task(task_id="T-001", assignee_agent_id="agent2")

        asyncio.run(run())

    def test_claim_blocked_task_raises(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            await board.create_task(title="Task B", dependencies=["T-001"])

            with pytest.raises(TaskBlockedError):
                await board.claim_task(task_id="T-002", assignee_agent_id="agent1")

        asyncio.run(run())

    def test_release_task(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            await board.claim_task(task_id="T-001", assignee_agent_id="agent1")
            await board.release_task(task_id="T-001")

            info = await board.get_task_info("T-001")
            assert info.status == "pending"
            assert info.assignee_agent_id is None

        asyncio.run(run())


class TestTaskBoardUpdateStatus:
    def test_update_status_to_completed(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            await board.claim_task(task_id="T-001", assignee_agent_id="agent1")
            await board.update_status(task_id="T-001", status="completed", result_summary="done")

            info = await board.get_task_info("T-001")
            assert info.status == "completed"
            assert info.result_summary == "done"

        asyncio.run(run())

    def test_update_status_without_summary(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamTaskModel, TeamTaskLockModel])
            board = make_board(engine)

            await board.create_task(title="Task A")
            await board.claim_task(task_id="T-001", assignee_agent_id="agent1")
            await board.update_status(task_id="T-001", status="completed")

            info = await board.get_task_info("T-001")
            assert info.status == "completed"
            assert info.result_summary is None

        asyncio.run(run())
