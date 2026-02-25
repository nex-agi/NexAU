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

"""Unit tests for AgentTeam."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from nexau.archs.main_sub.team.agent_team import AgentTeam
from nexau.archs.main_sub.team.types import MaxTeammatesError
from nexau.archs.session.models.team import TeamModel
from nexau.archs.session.models.team_member import TeamMemberModel
from nexau.archs.session.models.team_message import TeamMessageModel
from nexau.archs.session.models.team_task import TeamTaskModel
from nexau.archs.session.models.team_task_lock import TeamTaskLockModel
from nexau.archs.session.orm import InMemoryDatabaseEngine

# --- Helpers ---


def make_agent_config(
    name: str = "worker",
    description: str = "A worker agent",
    system_prompt: str | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.name = name
    cfg.description = description
    cfg.system_prompt = system_prompt
    cfg.system_prompt_suffix = None
    cfg.tools = []
    cfg.middlewares = None
    cfg.llm_config = None
    cfg.sandbox_config = None
    cfg.stop_tools = None
    cfg.tracers = []
    cfg.resolved_tracer = None
    return cfg


def make_team(
    engine: InMemoryDatabaseEngine | None = None,
    max_teammates: int = 10,
) -> AgentTeam:
    if engine is None:
        engine = InMemoryDatabaseEngine()
    leader_config = make_agent_config(name="leader")
    candidate_config = make_agent_config(name="worker")
    session_manager = MagicMock()
    return AgentTeam(
        leader_config=leader_config,
        candidates={"worker": candidate_config},
        engine=engine,
        session_manager=session_manager,
        user_id="u1",
        session_id="s1",
        max_teammates=max_teammates,
    )


def make_mock_agent(name: str = "worker", is_idle: bool = True) -> MagicMock:
    agent = MagicMock()
    agent.config.name = name
    agent.executor.is_idle = is_idle
    agent.executor.force_stop = MagicMock()
    agent.enqueue_message = MagicMock()
    return agent


ALL_TEAM_MODELS = [
    TeamModel,
    TeamMemberModel,
    TeamTaskModel,
    TeamTaskLockModel,
    TeamMessageModel,
]


# --- TestAgentTeamProperties ---


class TestAgentTeamProperties:
    def test_properties_before_initialize_raise(self):
        team = make_team()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = team.task_board
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = team.message_bus

    def test_is_running_defaults_false(self):
        team = make_team()
        assert team.is_running is False

    def test_team_id_empty_before_initialize(self):
        team = make_team()
        assert team.team_id == ""

    def test_leader_agent_id_empty_before_initialize(self):
        team = make_team()
        assert team.leader_agent_id == ""

    def test_set_on_complete_stores_callback(self):
        team = make_team()
        cb = MagicMock()
        team.set_on_complete(cb)
        assert team._on_run_complete is cb


# --- TestAgentTeamInitialize ---


class TestAgentTeamInitialize:
    def test_initialize_creates_team_record(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)

            await team.initialize()

            assert team.team_id != ""
            assert team.leader_agent_id == "leader"
            assert team.task_board is not None
            assert team.message_bus is not None

        asyncio.run(run())

    def test_initialize_is_idempotent(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)

            await team.initialize()
            first_team_id = team.team_id

            # Second call should be a no-op
            await team.initialize()
            assert team.team_id == first_team_id

        asyncio.run(run())

    def test_initialize_restores_existing_team(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            # First team creates the record
            team1 = make_team(engine=engine)
            await team1.initialize()
            original_team_id = team1.team_id

            # Second team with same user/session should restore
            team2 = make_team(engine=engine)
            await team2.initialize()
            assert team2.team_id == original_team_id

        asyncio.run(run())

    def test_initialize_restores_role_counters(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            team = make_team(engine=engine)
            await team.initialize()

            # Manually insert a non-stopped member to simulate prior spawn
            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="idle",
            )
            await engine.create(member)

            # Re-initialize a fresh team with same engine/session
            team2 = make_team(engine=engine)
            await team2.initialize()
            assert team2._role_counters.get("worker", 0) == 1

        asyncio.run(run())

    def test_initialize_skips_stopped_members_in_counter(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="stopped",
            )
            await engine.create(member)

            team2 = make_team(engine=engine)
            await team2.initialize()
            assert team2._role_counters.get("worker", 0) == 0

        asyncio.run(run())


# --- TestAgentTeamGetTeammateInfo ---


class TestAgentTeamGetTeammateInfo:
    def test_empty_when_no_teammates(self):
        team = make_team()
        assert team.get_teammate_info() == []

    def test_idle_when_future_done(self):
        team = make_team()
        agent = make_mock_agent(name="worker", is_idle=False)
        future: Future[None] = Future()
        future.set_result(None)  # done
        team._teammate_agents["worker-1"] = agent
        team._teammate_futures["worker-1"] = future

        infos = team.get_teammate_info()
        assert len(infos) == 1
        assert infos[0].status == "idle"

    def test_idle_when_executor_is_idle(self):
        team = make_team()
        agent = make_mock_agent(name="worker", is_idle=True)
        future: Future[None] = Future()  # not done
        team._teammate_agents["worker-1"] = agent
        team._teammate_futures["worker-1"] = future

        infos = team.get_teammate_info()
        assert infos[0].status == "idle"

    def test_running_when_future_active_and_not_idle(self):
        team = make_team()
        agent = make_mock_agent(name="worker", is_idle=False)
        future: Future[None] = Future()  # not done
        team._teammate_agents["worker-1"] = agent
        team._teammate_futures["worker-1"] = future

        infos = team.get_teammate_info()
        assert infos[0].status == "running"

    def test_error_when_in_errored_agents(self):
        team = make_team()
        agent = make_mock_agent(name="worker", is_idle=False)
        future: Future[None] = Future()
        team._teammate_agents["worker-1"] = agent
        team._teammate_futures["worker-1"] = future
        team._errored_agents.add("worker-1")

        infos = team.get_teammate_info()
        assert infos[0].status == "error"

    def test_role_name_falls_back_to_agent_id(self):
        team = make_team()
        agent = make_mock_agent()
        agent.config.name = None  # no name set
        team._teammate_agents["worker-1"] = agent

        infos = team.get_teammate_info()
        assert infos[0].role_name == "worker-1"

    def test_multiple_teammates(self):
        team = make_team()
        for i in range(3):
            agent = make_mock_agent(name=f"role-{i}", is_idle=True)
            team._teammate_agents[f"role-{i}-1"] = agent

        infos = team.get_teammate_info()
        assert len(infos) == 3


# --- TestAgentTeamIsAllIdle ---


class TestAgentTeamIsAllIdle:
    def test_false_when_leader_not_set(self):
        team = make_team()
        assert team.is_all_idle() is False

    def test_false_when_leader_not_idle(self):
        team = make_team()
        leader = make_mock_agent(is_idle=False)
        team._leader_agent = leader
        assert team.is_all_idle() is False

    def test_true_when_leader_idle_no_teammates(self):
        team = make_team()
        leader = make_mock_agent(is_idle=True)
        team._leader_agent = leader
        assert team.is_all_idle() is True

    def test_false_when_teammate_not_idle(self):
        team = make_team()
        leader = make_mock_agent(is_idle=True)
        team._leader_agent = leader
        teammate = make_mock_agent(is_idle=False)
        team._teammate_agents["worker-1"] = teammate
        assert team.is_all_idle() is False

    def test_true_when_all_idle(self):
        team = make_team()
        leader = make_mock_agent(is_idle=True)
        team._leader_agent = leader
        for i in range(3):
            team._teammate_agents[f"worker-{i}"] = make_mock_agent(is_idle=True)
        assert team.is_all_idle() is True


# --- TestAgentTeamNotifyLeader ---


class TestAgentTeamNotifyLeader:
    def test_notify_leader_enqueues_to_leader(self):
        team = make_team()
        team._leader_agent_id = "leader"
        leader = make_mock_agent(name="leader")
        team._leader_agent = leader

        team.notify_leader(content="all done", from_agent_id="worker-1")

        leader.enqueue_message.assert_called_once()
        call_args = leader.enqueue_message.call_args[0][0]
        assert "worker-1" in call_args["content"]
        assert "all done" in call_args["content"]


# --- TestAgentTeamSendMessageToAgent ---


class TestAgentTeamSendMessageToAgent:
    def test_send_to_leader(self):
        team = make_team()
        team._leader_agent_id = "leader"
        leader = make_mock_agent(name="leader")
        team._leader_agent = leader

        team.send_message_to_agent(to_agent_id="leader", content="hello", from_agent_id="worker-1")

        leader.enqueue_message.assert_called_once()
        msg = leader.enqueue_message.call_args[0][0]
        assert msg["role"] == "user"
        assert "hello" in msg["content"]

    def test_send_to_teammate(self):
        team = make_team()
        team._leader_agent_id = "leader"
        teammate = make_mock_agent(name="worker")
        future: Future[None] = Future()
        team._teammate_agents["worker-1"] = teammate
        team._teammate_futures["worker-1"] = future

        team.send_message_to_agent(to_agent_id="worker-1", content="task", from_agent_id="leader")

        teammate.enqueue_message.assert_called_once()

    def test_send_to_unknown_agent_does_not_raise(self):
        team = make_team()
        team._leader_agent_id = "leader"
        # Should log warning but not raise
        team.send_message_to_agent(to_agent_id="ghost", content="hello", from_agent_id="leader")

    def test_send_to_leader_not_set_does_not_raise(self):
        team = make_team()
        team._leader_agent_id = "leader"
        team._leader_agent = None
        # Should log warning but not raise
        team.send_message_to_agent(to_agent_id="leader", content="hello", from_agent_id="worker-1")

    def test_send_restarts_exited_teammate_future(self):
        team = make_team()
        team._leader_agent_id = "leader"
        teammate = make_mock_agent(name="worker")
        done_future: Future[None] = Future()
        done_future.set_result(None)
        team._teammate_agents["worker-1"] = teammate
        team._teammate_futures["worker-1"] = done_future

        loop = MagicMock()
        new_future: Future[None] = Future()
        loop.call_soon_threadsafe = MagicMock()
        team._loop = loop

        with patch("asyncio.run_coroutine_threadsafe", return_value=new_future) as mock_rctf:
            team.send_message_to_agent(to_agent_id="worker-1", content="wake up", from_agent_id="leader")
            mock_rctf.assert_called_once()

        assert team._teammate_futures["worker-1"] is new_future


# --- TestAgentTeamEnqueueUserMessage ---


class TestAgentTeamEnqueueUserMessage:
    def test_enqueue_to_leader(self):
        team = make_team()
        team._leader_agent_id = "leader"
        leader = make_mock_agent(name="leader")
        team._leader_agent = leader

        team.enqueue_user_message(to_agent_id="leader", content="continue")

        leader.enqueue_message.assert_called_once()
        msg = leader.enqueue_message.call_args[0][0]
        assert msg["role"] == "user"
        assert msg["content"] == "continue"

    def test_enqueue_to_teammate(self):
        team = make_team()
        team._leader_agent_id = "leader"
        teammate = make_mock_agent(name="worker")
        future: Future[None] = Future()
        team._teammate_agents["worker-1"] = teammate
        team._teammate_futures["worker-1"] = future

        team.enqueue_user_message(to_agent_id="worker-1", content="stop")

        teammate.enqueue_message.assert_called_once()
        msg = teammate.enqueue_message.call_args[0][0]
        assert msg["role"] == "user"
        assert msg["content"] == "stop"

    def test_enqueue_to_unknown_does_not_raise(self):
        team = make_team()
        team._leader_agent_id = "leader"
        team.enqueue_user_message(to_agent_id="ghost", content="hello")

    def test_enqueue_restarts_exited_teammate(self):
        team = make_team()
        team._leader_agent_id = "leader"
        teammate = make_mock_agent(name="worker")
        done_future: Future[None] = Future()
        done_future.set_result(None)
        team._teammate_agents["worker-1"] = teammate
        team._teammate_futures["worker-1"] = done_future
        team._loop = MagicMock()

        new_future: Future[None] = Future()
        with patch("asyncio.run_coroutine_threadsafe", return_value=new_future) as mock_rctf:
            team.enqueue_user_message(to_agent_id="worker-1", content="resume")
            mock_rctf.assert_called_once()

        assert team._teammate_futures["worker-1"] is new_future


# --- TestAgentTeamRemoveTeammate ---


class TestAgentTeamRemoveTeammate:
    def test_remove_calls_force_stop(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            agent = make_mock_agent(name="worker")
            future: Future[None] = Future()
            future.set_result(None)
            team._teammate_agents["worker-1"] = agent
            team._teammate_futures["worker-1"] = future

            await team.remove_teammate("worker-1")

            agent.executor.force_stop.assert_called_once()
            assert "worker-1" not in team._teammate_agents

        asyncio.run(run())

    def test_remove_updates_db_status(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            # Insert a member record
            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="running",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            future: Future[None] = Future()
            future.set_result(None)
            team._teammate_agents["worker-1"] = agent
            team._teammate_futures["worker-1"] = future

            await team.remove_teammate("worker-1")

            from nexau.archs.session.orm import AndFilter, ComparisonFilter

            updated = await engine.find_first(
                TeamMemberModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("agent_id", "worker-1"),
                    ]
                ),
            )
            assert updated is not None
            assert updated.status == "stopped"

        asyncio.run(run())

    def test_remove_nonexistent_does_not_raise(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            # Should not raise even if agent doesn't exist
            await team.remove_teammate("ghost-1")

        asyncio.run(run())

    def test_remove_unregisters_from_watchdog(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            watchdog = MagicMock()
            team._watchdog = watchdog

            agent = make_mock_agent(name="worker")
            future: Future[None] = Future()
            future.set_result(None)
            team._teammate_agents["worker-1"] = agent
            team._teammate_futures["worker-1"] = future

            await team.remove_teammate("worker-1")

            watchdog.unregister.assert_called_once_with("worker-1")

        asyncio.run(run())


# --- TestAgentTeamSpawnTeammate ---


class TestAgentTeamSpawnTeammate:
    def test_spawn_raises_when_loop_not_set(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                with pytest.raises(RuntimeError, match="loop not set"):
                    await team.spawn_teammate("worker")

        asyncio.run(run())

    def test_spawn_raises_for_unknown_role(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            with pytest.raises(ValueError, match="Unknown role"):
                await team.spawn_teammate("nonexistent")

        asyncio.run(run())

    def test_spawn_raises_when_max_teammates_reached(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine, max_teammates=1)
            await team.initialize()
            team._loop = MagicMock()

            # Fill up to max
            team._teammate_agents["worker-1"] = make_mock_agent()

            with pytest.raises(MaxTeammatesError):
                await team.spawn_teammate("worker")

        asyncio.run(run())

    def test_spawn_increments_role_counter(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            new_future: Future[None] = Future()
            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                with patch("asyncio.run_coroutine_threadsafe", return_value=new_future):
                    agent_id = await team.spawn_teammate("worker")

            assert agent_id == "worker-1"
            assert team._role_counters["worker"] == 1

            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                with patch("asyncio.run_coroutine_threadsafe", return_value=Future()):
                    agent_id2 = await team.spawn_teammate("worker")

            assert agent_id2 == "worker-2"
            assert team._role_counters["worker"] == 2

        asyncio.run(run())

    def test_spawn_creates_member_record(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                with patch("asyncio.run_coroutine_threadsafe", return_value=Future()):
                    await team.spawn_teammate("worker")

            from nexau.archs.session.orm import AndFilter, ComparisonFilter

            member = await engine.find_first(
                TeamMemberModel,
                filters=AndFilter(filters=[ComparisonFilter.eq("agent_id", "worker-1")]),
            )
            assert member is not None
            assert member.role_name == "worker"
            assert member.status == "idle"

        asyncio.run(run())


# --- TestAgentTeamRestoreTeammates ---


class TestAgentTeamRestoreTeammates:
    def test_skips_stopped_members(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="stopped",
            )
            await engine.create(member)

            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                await team._restore_teammates()

            assert "worker-1" not in team._teammate_agents

        asyncio.run(run())

    def test_skips_already_in_memory_members(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="idle",
            )
            await engine.create(member)

            # Pre-populate in-memory
            existing_agent = make_mock_agent()
            team._teammate_agents["worker-1"] = existing_agent

            with patch("nexau.archs.main_sub.team.agent_team.Agent") as mock_agent_cls:
                await team._restore_teammates()
                mock_agent_cls.assert_not_called()

            # Still the original agent
            assert team._teammate_agents["worker-1"] is existing_agent

        asyncio.run(run())

    def test_skips_unknown_role(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="ghost-1",
                member_session_id="s1:ghost-1",
                role_name="ghost",  # not in candidates
                status="idle",
            )
            await engine.create(member)

            with patch("nexau.archs.main_sub.team.agent_team.Agent") as mock_agent_cls:
                await team._restore_teammates()
                mock_agent_cls.assert_not_called()

        asyncio.run(run())

    def test_raises_when_loop_not_set(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            # _loop is None by default

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="idle",
            )
            await engine.create(member)

            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                with pytest.raises(RuntimeError, match="loop not set"):
                    await team._restore_teammates()

        asyncio.run(run())

    def test_restores_valid_member(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="idle",
            )
            await engine.create(member)

            mock_agent = make_mock_agent()
            new_future: Future[None] = Future()
            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=mock_agent):
                with patch("asyncio.run_coroutine_threadsafe", return_value=new_future):
                    await team._restore_teammates()

            assert "worker-1" in team._teammate_agents
            assert team._teammate_futures["worker-1"] is new_future

        asyncio.run(run())

    def test_restores_multiple_members(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()
            team._loop = MagicMock()

            for i in range(3):
                member = TeamMemberModel(
                    user_id="u1",
                    session_id="s1",
                    team_id=team.team_id,
                    agent_id=f"worker-{i + 1}",
                    member_session_id=f"s1:worker-{i + 1}",
                    role_name="worker",
                    status="idle",
                )
                await engine.create(member)

            with patch("nexau.archs.main_sub.team.agent_team.Agent", return_value=make_mock_agent()):
                with patch("asyncio.run_coroutine_threadsafe", return_value=Future()):
                    await team._restore_teammates()

            assert len(team._teammate_agents) == 3

        asyncio.run(run())


# --- TestAgentTeamRunTeammateForever ---


class TestAgentTeamRunTeammateForever:
    def test_returns_early_when_agent_not_found(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            # Should return without error when agent_id not in _teammate_agents
            await team._run_teammate_forever("nonexistent")

        asyncio.run(run())

    def test_normal_exit_sets_status_idle(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="idle",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(return_value="done")
            team._teammate_agents["worker-1"] = agent

            await team._run_teammate_forever("worker-1")

            from nexau.archs.session.orm import AndFilter, ComparisonFilter

            updated = await engine.find_first(
                TeamMemberModel,
                filters=AndFilter(filters=[ComparisonFilter.eq("agent_id", "worker-1")]),
            )
            assert updated is not None
            assert updated.status == "idle"

        asyncio.run(run())

    def test_normal_exit_clears_errored_agents(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="error",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(return_value="done")
            team._teammate_agents["worker-1"] = agent
            team._errored_agents.add("worker-1")  # pre-mark as errored

            await team._run_teammate_forever("worker-1")

            assert "worker-1" not in team._errored_agents

        asyncio.run(run())

    def test_error_exit_adds_to_errored_agents(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="running",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(side_effect=RuntimeError("boom"))
            team._teammate_agents["worker-1"] = agent

            await team._run_teammate_forever("worker-1")

            assert "worker-1" in team._errored_agents

        asyncio.run(run())

    def test_error_exit_sets_status_error(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="running",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(side_effect=ValueError("fail"))
            team._teammate_agents["worker-1"] = agent

            await team._run_teammate_forever("worker-1")

            from nexau.archs.session.orm import AndFilter, ComparisonFilter

            updated = await engine.find_first(
                TeamMemberModel,
                filters=AndFilter(filters=[ComparisonFilter.eq("agent_id", "worker-1")]),
            )
            assert updated is not None
            assert updated.status == "error"

        asyncio.run(run())

    def test_error_exit_emits_sse_event(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="running",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(side_effect=RuntimeError("crash"))
            team._teammate_agents["worker-1"] = agent

            multiplexer = MagicMock()
            team._multiplexer = multiplexer

            await team._run_teammate_forever("worker-1")

            multiplexer.emit.assert_called_once()
            call_kwargs = multiplexer.emit.call_args[1]
            assert call_kwargs["agent_id"] == "worker-1"

        asyncio.run(run())

    def test_registers_and_unregisters_watchdog(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="idle",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(return_value="done")
            team._teammate_agents["worker-1"] = agent

            watchdog = MagicMock()
            team._watchdog = watchdog

            await team._run_teammate_forever("worker-1")

            watchdog.register.assert_called_once_with("worker-1")
            watchdog.unregister.assert_called_once_with("worker-1")

        asyncio.run(run())

    def test_unregisters_watchdog_on_error(self):
        async def run():
            from unittest.mock import AsyncMock

            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            member = TeamMemberModel(
                user_id="u1",
                session_id="s1",
                team_id=team.team_id,
                agent_id="worker-1",
                member_session_id="s1:worker-1",
                role_name="worker",
                status="running",
            )
            await engine.create(member)

            agent = make_mock_agent(name="worker")
            agent.run_async = AsyncMock(side_effect=RuntimeError("crash"))
            team._teammate_agents["worker-1"] = agent

            watchdog = MagicMock()
            team._watchdog = watchdog

            await team._run_teammate_forever("worker-1")

            watchdog.unregister.assert_called_once_with("worker-1")

        asyncio.run(run())


class TestAgentTeamRunStreaming:
    def test_run_streaming_yields_emitted_envelopes(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)

            from nexau.archs.llm.llm_aggregators.events import TextMessageContentEvent

            async def fake_run(message: str) -> str:
                # Emit one event via the multiplexer, then close it
                assert team._multiplexer is not None
                team._multiplexer.emit(
                    agent_id="leader",
                    event=TextMessageContentEvent(message_id="m1", delta="hello"),
                    role_name="leader",
                )
                team._multiplexer.close()
                team._multiplexer = None
                team._is_running = False
                return "done"

            envelopes = []
            with patch.object(team, "run", side_effect=fake_run):
                async for envelope in team.run_streaming("test message"):
                    envelopes.append(envelope)

            assert len(envelopes) == 1
            assert envelopes[0].agent_id == "leader"

        asyncio.run(run())

    def test_run_streaming_initializes_team(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)

            async def fake_run(message: str) -> str:
                assert team._multiplexer is not None
                team._multiplexer.close()
                team._multiplexer = None
                return "done"

            with patch.object(team, "run", side_effect=fake_run):
                async for _ in team.run_streaming("hello"):
                    pass

            # initialize() should have been called
            assert team.team_id != ""

        asyncio.run(run())

    def test_run_streaming_sets_multiplexer_before_run(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)

            captured_multiplexer: list[object] = []

            async def fake_run(message: str) -> str:
                captured_multiplexer.append(team._multiplexer)
                assert team._multiplexer is not None
                team._multiplexer.close()
                team._multiplexer = None
                return "done"

            with patch.object(team, "run", side_effect=fake_run):
                async for _ in team.run_streaming("hello"):
                    pass

            assert len(captured_multiplexer) == 1
            assert captured_multiplexer[0] is not None

        asyncio.run(run())

    def test_run_streaming_calls_on_envelope_callback(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)

            from nexau.archs.llm.llm_aggregators.events import TextMessageContentEvent

            received: list[object] = []

            async def fake_run(message: str) -> str:
                assert team._multiplexer is not None
                team._multiplexer.emit(
                    agent_id="leader",
                    event=TextMessageContentEvent(message_id="m1", delta="world"),
                    role_name="leader",
                )
                team._multiplexer.close()
                team._multiplexer = None
                return "done"

            with patch.object(team, "run", side_effect=fake_run):
                async for envelope in team.run_streaming("hello", on_envelope=received.append):
                    pass

            assert len(received) == 1

        asyncio.run(run())


class TestAgentTeamStopAll:
    def test_stop_all_calls_force_stop_on_leader(self):
        async def run():
            team = make_team()
            leader = make_mock_agent(name="leader")
            team._leader_agent = leader

            await team.stop_all()

            leader.executor.force_stop.assert_called_once()

        asyncio.run(run())

    def test_stop_all_stops_watchdog(self):
        async def run():
            team = make_team()
            watchdog = MagicMock()
            team._watchdog = watchdog

            await team.stop_all()

            watchdog.stop.assert_called_once()

        asyncio.run(run())

    def test_stop_all_clears_teammate_agents(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            team = make_team(engine=engine)
            await team.initialize()

            for i in range(3):
                agent = make_mock_agent(name="worker")
                done_future: Future[None] = Future()
                done_future.set_result(None)
                team._teammate_agents[f"worker-{i}"] = agent
                team._teammate_futures[f"worker-{i}"] = done_future

            await team.stop_all()

            assert len(team._teammate_agents) == 0
            assert len(team._teammate_futures) == 0

        asyncio.run(run())

    def test_stop_all_no_leader_does_not_raise(self):
        async def run():
            team = make_team()
            team._leader_agent = None
            await team.stop_all()

        asyncio.run(run())
