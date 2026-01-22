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

"""Tests for AgentRunActionModel and AgentRunActionService."""

import asyncio

import pytest

from nexau.archs.session import AgentRunActionKey, AgentRunActionService
from nexau.archs.session.models import AgentRunActionModel, RunActionType
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.core.messages import Message, Role, TextBlock


@pytest.fixture
def engine():
    """Create an in-memory database engine."""
    return InMemoryDatabaseEngine()


@pytest.fixture
def service(engine):
    """Create an AgentRunActionService instance."""
    return AgentRunActionService(engine=engine)


@pytest.fixture
def action_key():
    """Create a test action key."""
    return AgentRunActionKey(
        user_id="test_user",
        session_id="test_session",
        agent_id="test_agent",
    )


def test_create_append_action():
    """Test creating an APPEND action record."""
    messages = [
        Message.user("Hello"),
        Message.assistant("Hi there!"),
    ]

    record = AgentRunActionModel.create_append(
        user_id="u1",
        session_id="s1",
        agent_id="agent1",
        run_id="run_001",
        root_run_id="run_001",
        messages=messages,
        agent_name="test_agent",
    )

    assert record.action_type == RunActionType.APPEND
    assert record.append_messages == messages
    assert record.replace_messages is None
    assert record.undo_before_run_id is None
    assert record.run_id == "run_001"
    assert record.action_id is not None  # Auto-generated


def test_create_undo_action():
    """Test creating an UNDO action record."""
    record = AgentRunActionModel.create_undo(
        user_id="u1",
        session_id="s1",
        agent_id="agent1",
        run_id="run_003",
        root_run_id="run_001",
        undo_before_run_id="run_002",
        agent_name="test_agent",
    )

    assert record.action_type == RunActionType.UNDO
    assert record.undo_before_run_id == "run_002"
    assert record.append_messages is None
    assert record.replace_messages is None
    assert record.run_id == "run_003"
    assert record.action_id is not None  # Auto-generated


def test_create_replace_action():
    """Test creating a REPLACE action record."""
    messages = [
        Message.user("Compacted message 1"),
        Message.assistant("Compacted message 2"),
    ]

    record = AgentRunActionModel.create_replace(
        user_id="u1",
        session_id="s1",
        agent_id="agent1",
        run_id="run_004",
        root_run_id="run_001",
        messages=messages,
        agent_name="test_agent",
    )

    assert record.action_type == RunActionType.REPLACE
    assert record.replace_messages == messages
    assert record.append_messages is None
    assert record.undo_before_run_id is None
    assert record.run_id == "run_004"
    assert record.action_id is not None  # Auto-generated


def test_load_messages_empty_list(service, action_key, engine):
    """Test loading messages when no actions exist."""

    async def run():
        await engine.setup_models([AgentRunActionModel])
        messages = await service.load_messages(key=action_key)
        assert messages == []

    asyncio.run(run())


def test_load_messages_single_append(service, action_key, engine):
    """Test loading messages from single APPEND action."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        msg1 = Message.user("Hello")
        msg2 = Message.assistant("Hi!")

        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg1, msg2],
        )

        messages = await service.load_messages(key=action_key)
        assert len(messages) == 2
        assert messages[0].content == msg1.content
        assert messages[1].content == msg2.content

    asyncio.run(run())


def test_load_messages_multiple_appends(service, action_key, engine):
    """Test loading messages from multiple APPEND actions."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Hello")],
        )

        await service.persist_append(
            key=action_key,
            run_id="run_002",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.assistant("Hi!"), Message.user("How are you?")],
        )

        messages = await service.load_messages(key=action_key)
        assert len(messages) == 3

    asyncio.run(run())


def test_load_messages_with_undo(service, action_key, engine):
    """Test loading messages with UNDO action removes correct range."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create 3 APPEND actions
        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Message 1")],
        )

        await service.persist_append(
            key=action_key,
            run_id="run_002",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Message 2")],
        )

        await service.persist_append(
            key=action_key,
            run_id="run_003",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Message 3")],
        )

        # UNDO before run_002 (removes run_002 and run_003)
        await service.persist_undo(
            key=action_key,
            run_id="run_004",
            root_run_id="run_001",
            undo_before_run_id="run_002",
        )

        messages = await service.load_messages(key=action_key)
        # Should only have message from run_001
        assert len(messages) == 1
        assert messages[0].content[0].text == "Message 1"

    asyncio.run(run())


def test_load_messages_with_replace(service, action_key, engine):
    """Test loading messages with REPLACE action discards previous history."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Create 2 APPEND actions
        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Old message 1")],
        )

        await service.persist_append(
            key=action_key,
            run_id="run_002",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Old message 2")],
        )

        # REPLACE with compacted messages
        await service.persist_replace(
            key=action_key,
            run_id="run_003",
            root_run_id="run_001",
            agent_name="test_agent",
            messages=[Message.user("Compacted summary")],
        )

        # Add another APPEND after REPLACE
        await service.persist_append(
            key=action_key,
            run_id="run_004",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("New message")],
        )

        messages = await service.load_messages(key=action_key)
        # Should have compacted summary + new message
        assert len(messages) == 2
        assert messages[0].content[0].text == "Compacted summary"
        assert messages[1].content[0].text == "New message"

    asyncio.run(run())


def test_load_messages_filters_system_messages(service, action_key, engine):
    """Test that system messages are filtered out during load."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        system_msg = Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")])
        user_msg = Message.user("Hello")

        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[system_msg, user_msg],
        )

        messages = await service.load_messages(key=action_key)
        # System message should be filtered
        assert len(messages) == 1
        assert messages[0].content[0].text == "Hello"

    asyncio.run(run())


def test_load_messages_deduplicates_by_id(service, action_key, engine):
    """Test that messages with same ID are last-write-wins."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        msg1 = Message.user("Old")
        msg2 = Message.user("New")  # Different instance, same ID
        msg2.id = msg1.id  # Force same ID

        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg1],
        )

        await service.persist_append(
            key=action_key,
            run_id="run_002",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[msg2],
        )

        messages = await service.load_messages(key=action_key)
        assert len(messages) == 1
        assert messages[0].content[0].text == "New"

    asyncio.run(run())


def test_service_persist_append(service, action_key, engine):
    """Test persisting an APPEND action via service."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        messages = [Message.user("Hello"), Message.assistant("Hi!")]

        record = await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=messages,
        )

        assert record.action_type == RunActionType.APPEND
        assert len(record.append_messages) == 2

    asyncio.run(run())


def test_service_persist_append_filters_system_messages(service, action_key, engine):
    """Test that persist_append filters out system messages."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        system_msg = Message(role=Role.SYSTEM, content=[TextBlock(text="System")])
        user_msg = Message.user("Hello")

        record = await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[system_msg, user_msg],
        )

        # System message should be filtered
        assert len(record.append_messages) == 1
        assert record.append_messages[0].role == Role.USER

    asyncio.run(run())


def test_service_persist_append_empty_raises_error(service, action_key, engine):
    """Test that persisting empty APPEND raises ValueError."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        with pytest.raises(ValueError, match="Cannot persist APPEND action with no messages"):
            await service.persist_append(
                key=action_key,
                run_id="run_001",
                root_run_id="run_001",
                parent_run_id=None,
                agent_name="test_agent",
                messages=[],
            )

    asyncio.run(run())


def test_service_persist_undo(service, action_key, engine):
    """Test persisting an UNDO action via service."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        record = await service.persist_undo(
            key=action_key,
            run_id="run_003",
            root_run_id="run_001",
            undo_before_run_id="run_002",
            agent_name="test_agent",
        )

        assert record.action_type == RunActionType.UNDO
        assert record.undo_before_run_id == "run_002"

    asyncio.run(run())


def test_service_persist_replace(service, action_key, engine):
    """Test persisting a REPLACE action via service."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        messages = [Message.user("Compacted")]

        record = await service.persist_replace(
            key=action_key,
            run_id="run_004",
            root_run_id="run_001",
            agent_name="test_agent",
            messages=messages,
        )

        assert record.action_type == RunActionType.REPLACE
        assert len(record.replace_messages) == 1

    asyncio.run(run())


def test_service_persist_replace_empty_raises_error(service, action_key, engine):
    """Test that persisting empty REPLACE raises ValueError."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        with pytest.raises(ValueError, match="Cannot persist REPLACE action with no messages"):
            await service.persist_replace(
                key=action_key,
                run_id="run_004",
                root_run_id="run_001",
                agent_name="test_agent",
                messages=[],
            )

    asyncio.run(run())


def test_service_load_messages_empty(service, action_key, engine):
    """Test loading messages when no actions exist."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        messages = await service.load_messages(key=action_key)
        assert messages == []

    asyncio.run(run())


def test_service_load_messages_with_actions(service, action_key, engine):
    """Test loading and rebuilding messages from multiple actions."""

    async def run():
        await engine.setup_models([AgentRunActionModel])

        # Persist multiple actions
        await service.persist_append(
            key=action_key,
            run_id="run_001",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Message 1")],
        )

        await service.persist_append(
            key=action_key,
            run_id="run_002",
            root_run_id="run_001",
            parent_run_id=None,
            agent_name="test_agent",
            messages=[Message.user("Message 2")],
        )

        # Load messages
        messages = await service.load_messages(key=action_key)
        assert len(messages) == 2

    asyncio.run(run())
