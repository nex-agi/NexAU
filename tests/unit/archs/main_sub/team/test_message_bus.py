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

"""Unit tests for TeamMessageBus."""

import asyncio

from nexau.archs.main_sub.team.message_bus import TeamMessageBus
from nexau.archs.session.models.team_message import TeamMessageModel
from nexau.archs.session.orm import InMemoryDatabaseEngine


def make_bus(engine: InMemoryDatabaseEngine) -> TeamMessageBus:
    return TeamMessageBus(
        engine=engine,
        user_id="u1",
        session_id="s1",
        team_id="t1",
    )


class TestTeamMessageBusSend:
    def test_send_persists_message(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            msg = await bus.send(from_agent_id="leader", to_agent_id="agent1", content="hello")

            assert msg.from_agent_id == "leader"
            assert msg.to_agent_id == "agent1"
            assert msg.content == "hello"
            assert not msg.delivered

        asyncio.run(run())

    def test_send_calls_delivery_callback(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            delivered: list[tuple[str, str, str]] = []

            bus.set_agent_delivery(
                deliver_message=lambda to, content, frm: delivered.append((to, content, frm)),
                get_broadcast_recipients=lambda: [],
            )

            await bus.send(from_agent_id="leader", to_agent_id="agent1", content="ping")

            assert delivered == [("agent1", "ping", "leader")]

        asyncio.run(run())

    def test_send_without_delivery_callback(self):
        """send() works fine when no delivery callback is wired."""

        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            # No set_agent_delivery call — should not raise
            msg = await bus.send(from_agent_id="a", to_agent_id="b", content="ok")
            assert msg.message_id

        asyncio.run(run())


class TestTeamMessageBusBroadcast:
    def test_broadcast_persists_with_no_to_agent(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            msg = await bus.broadcast(from_agent_id="leader", content="all hands")

            assert msg.to_agent_id is None
            assert msg.content == "all hands"

        asyncio.run(run())

    def test_broadcast_delivers_to_all_except_sender(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            delivered: list[str] = []

            bus.set_agent_delivery(
                deliver_message=lambda to, content, frm: delivered.append(to),
                get_broadcast_recipients=lambda: ["leader", "agent1", "agent2"],
            )

            await bus.broadcast(from_agent_id="leader", content="broadcast msg")

            # leader should be excluded
            assert "leader" not in delivered
            assert "agent1" in delivered
            assert "agent2" in delivered

        asyncio.run(run())

    def test_broadcast_without_delivery_callback(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            msg = await bus.broadcast(from_agent_id="leader", content="hello")
            assert msg.message_id

        asyncio.run(run())


class TestTeamMessageBusDrain:
    def test_drain_returns_direct_messages(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            await bus.send(from_agent_id="leader", to_agent_id="agent1", content="task assigned")

            msgs = await bus.drain(agent_id="agent1")

            assert len(msgs) == 1
            assert msgs[0].content == "task assigned"
            assert msgs[0].delivered

        asyncio.run(run())

    def test_drain_marks_messages_delivered(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            await bus.send(from_agent_id="leader", to_agent_id="agent1", content="msg")

            # First drain
            msgs = await bus.drain(agent_id="agent1")
            assert len(msgs) == 1

            # Second drain — already delivered, should be empty
            msgs2 = await bus.drain(agent_id="agent1")
            assert len(msgs2) == 0

        asyncio.run(run())

    def test_drain_includes_broadcast_messages(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            await bus.broadcast(from_agent_id="leader", content="broadcast")

            msgs = await bus.drain(agent_id="agent1")
            assert len(msgs) == 1
            assert msgs[0].content == "broadcast"

        asyncio.run(run())

    def test_drain_excludes_own_broadcast(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            await bus.broadcast(from_agent_id="leader", content="broadcast")

            # leader drains — should not receive own broadcast
            msgs = await bus.drain(agent_id="leader")
            assert len(msgs) == 0

        asyncio.run(run())

    def test_drain_empty_when_no_messages(self):
        async def run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models([TeamMessageModel])
            bus = make_bus(engine)

            msgs = await bus.drain(agent_id="agent1")
            assert msgs == []

        asyncio.run(run())
