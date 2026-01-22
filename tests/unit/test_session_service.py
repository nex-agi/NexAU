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

"""Tests for SessionManager abstraction.

Requirements: 4.1
"""

from __future__ import annotations

import asyncio

from nexau.archs.session import (
    AgentModel,
    AgentRunActionKey,
    AgentRunActionModel,
    InMemoryDatabaseEngine,
    RunActionType,
    SessionManager,
    SessionModel,
)
from nexau.archs.session.orm import AndFilter, ComparisonFilter
from nexau.core.messages import Message, Role


class TestSessionManagerInitialization:
    """Tests for SessionManager initialization."""

    def test_init_with_valid_engine(self) -> None:
        """Test SessionManager initializes correctly with a valid engine."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        assert service is not None
        # SessionManager stores engine internally
        assert service._engine is engine


class TestSessionManagerPropertyAccess:
    """Tests for SessionManager property access."""

    def test_agent_run_action_property(self) -> None:
        """Test agent_run_action property returns the service."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        assert service.agent_run_action is not None

    def test_agent_lock_property(self) -> None:
        """Test agent_lock property returns the service."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        assert service.agent_lock is not None


class TestSessionManagerEngineOperations:
    """Tests for engine operations through SessionManager."""

    def test_session_operations(self) -> None:
        """Test CRUD operations for SessionModel."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            # Initialize models first
            await engine.setup_models([SessionModel])

            # Create a session
            from nexau.archs.main_sub.agent_context import GlobalStorage

            session = SessionModel(user_id="u1", session_id="s1", storage=GlobalStorage())
            session.context = {"key": "value"}
            await engine.create(session)

            # Verify it was saved using Filter DSL syntax
            filters = AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", "u1"),
                    ComparisonFilter.eq("session_id", "s1"),
                ]
            )
            loaded = await engine.find_first(SessionModel, filters=filters)
            assert loaded is not None
            assert loaded.context == {"key": "value"}

        asyncio.run(run())

    def test_agent_operations(self) -> None:
        """Test CRUD operations for AgentModel."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            # Initialize models first
            await engine.setup_models([AgentModel])

            # Create an agent
            agent = AgentModel(user_id="u1", session_id="s1", agent_id="a1")
            agent.agent_name = "test_agent"
            await engine.create(agent)

            # Verify it was saved using Filter DSL syntax
            filters = AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", "u1"),
                    ComparisonFilter.eq("session_id", "s1"),
                    ComparisonFilter.eq("agent_id", "a1"),
                ]
            )
            loaded = await engine.find_first(AgentModel, filters=filters)
            assert loaded is not None
            assert loaded.agent_name == "test_agent"

        asyncio.run(run())

    def test_run_action_operations(self) -> None:
        """Test CRUD operations for AgentRunActionModel."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            # Initialize models first
            await engine.setup_models([AgentRunActionModel])

            # Create a run action record
            record = AgentRunActionModel(
                user_id="u1",
                session_id="s1",
                agent_id="a1",
                run_id="run_001",
                root_run_id="run_001",
                action_type=RunActionType.APPEND,
                append_messages=[Message.user("hello")],
            )
            await engine.create(record)

            # Verify it was saved using Filter DSL syntax
            filters = AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", "u1"),
                    ComparisonFilter.eq("session_id", "s1"),
                    ComparisonFilter.eq("agent_id", "a1"),
                    ComparisonFilter.eq("run_id", "run_001"),
                ]
            )
            loaded = await engine.find_first(AgentRunActionModel, filters=filters)
            assert loaded is not None
            assert loaded.append_messages is not None
            assert loaded.append_messages[0].role == Role.USER

        asyncio.run(run())


class TestSessionManagerHistoryAPI:
    def test_persist_messages_creates_append_action(self) -> None:
        """Test that persist_append creates an APPEND action."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()
            key = AgentRunActionKey(user_id="u1", session_id="s1", agent_id="a1")

            msg = Message.user("hello")
            record = await service.agent_run_action.persist_append(
                key=key,
                run_id="run_001",
                root_run_id="run_001",
                parent_run_id=None,
                agent_name="test_agent",
                messages=[msg],
            )

            assert record is not None

            # Verify the action was created
            filters = AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", "u1"),
                    ComparisonFilter.eq("session_id", "s1"),
                    ComparisonFilter.eq("agent_id", "a1"),
                    ComparisonFilter.eq("run_id", "run_001"),
                ]
            )
            loaded = await engine.find_first(AgentRunActionModel, filters=filters)
            assert loaded is not None
            assert loaded.action_type == RunActionType.APPEND
            assert loaded.append_messages is not None
            assert len(loaded.append_messages) == 1

        asyncio.run(run())

    def test_persist_messages_multiple_runs(self) -> None:
        """Test persisting messages across multiple runs."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()
            key = AgentRunActionKey(user_id="u1", session_id="s1", agent_id="a1")

            # First run
            msg1 = Message.user("hello")
            await service.agent_run_action.persist_append(
                key=key,
                run_id="run_001",
                root_run_id="run_001",
                parent_run_id=None,
                agent_name="test_agent",
                messages=[msg1],
            )

            # Second run
            msg2 = Message.user("world")
            await service.agent_run_action.persist_append(
                key=key,
                run_id="run_002",
                root_run_id="run_001",
                parent_run_id=None,
                agent_name="test_agent",
                messages=[msg2],
            )

            # Load all messages
            loaded = await service.agent_run_action.load_messages(key=key)
            assert len(loaded) == 2

        asyncio.run(run())

    def test_load_messages_returns_persisted_messages(self) -> None:
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()
            key = AgentRunActionKey(user_id="u1", session_id="s1", agent_id="a1")

            user_msg = Message.user("hello")
            await service.agent_run_action.persist_append(
                key=key,
                run_id="run_001",
                root_run_id="run_001",
                parent_run_id=None,
                agent_name="test_agent",
                messages=[user_msg],
            )

            loaded = await service.agent_run_action.load_messages(key=key)
            assert any(m.id == user_msg.id for m in loaded)

        asyncio.run(run())

    def test_replace_messages_creates_replace_action(self) -> None:
        """Test that persist_replace creates a REPLACE action."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()
            key = AgentRunActionKey(user_id="u1", session_id="s1", agent_id="a1")

            # First persist some messages
            msg = Message.user("hello")
            await service.agent_run_action.persist_append(
                key=key,
                run_id="run_001",
                root_run_id="run_001",
                parent_run_id=None,
                agent_name="test_agent",
                messages=[msg],
            )

            # Replace with compacted version
            replaced = Message.user("hi")
            await service.agent_run_action.persist_replace(
                key=key,
                run_id="run_002",
                root_run_id="run_001",
                messages=[replaced],
                parent_run_id=None,
                agent_name="test_agent",
            )

            # Load messages - should only have the replaced one
            loaded = await service.agent_run_action.load_messages(key=key)
            assert len(loaded) == 1
            assert loaded[0].get_text_content() == "hi"

        asyncio.run(run())


class TestSessionManagerDataIsolation:
    """Tests for data isolation between model types."""

    def test_data_isolation_between_model_types(self) -> None:
        """Test that data in different model types is isolated."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            # Initialize models first
            await engine.setup_models([SessionModel, AgentModel])

            # Create session data
            from nexau.archs.main_sub.agent_context import GlobalStorage

            session = SessionModel(user_id="u1", session_id="s1", storage=GlobalStorage())
            session.context = {"session": "data"}
            await engine.create(session)

            # Create agent data
            agent = AgentModel(user_id="u1", session_id="s1", agent_id="a1")
            agent.agent_name = "agent_data"
            await engine.create(agent)

            # Verify data is isolated using Filter DSL syntax
            session_filters = AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", "u1"),
                    ComparisonFilter.eq("session_id", "s1"),
                ]
            )
            loaded_session = await engine.find_first(SessionModel, filters=session_filters)

            agent_filters = AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", "u1"),
                    ComparisonFilter.eq("session_id", "s1"),
                    ComparisonFilter.eq("agent_id", "a1"),
                ]
            )
            loaded_agent = await engine.find_first(AgentModel, filters=agent_filters)

            assert loaded_session is not None
            assert loaded_agent is not None
            assert loaded_session.context == {"session": "data"}
            assert loaded_agent.agent_name == "agent_data"

        asyncio.run(run())


class TestSessionManagerAgentRegistration:
    """Tests for agent registration methods."""

    def test_register_agent_reuses_existing_root_agent_id(self) -> None:
        """Test that register_agent reuses existing root_agent_id when is_root=True."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            # First registration - creates root agent
            first_agent_id, _ = await service.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="root_agent",
                is_root=True,
            )

            # Second registration - should reuse the same root_agent_id
            second_agent_id, _ = await service.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="root_agent",
                is_root=True,
            )

            assert first_agent_id == second_agent_id

        asyncio.run(run())

    def test_register_agent_with_explicit_id(self) -> None:
        """Test register_agent with explicitly provided agent_id."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            agent_id, _ = await service.register_agent(
                user_id="u1",
                session_id="s1",
                agent_id="explicit_agent_id",
                agent_name="test_agent",
                is_root=False,
            )

            assert agent_id == "explicit_agent_id"

        asyncio.run(run())

    def test_register_non_root_agent(self) -> None:
        """Test registering a non-root agent."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            # Register non-root agent without explicit ID
            agent_id, _ = await service.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="sub_agent",
                is_root=False,
            )

            # Should generate a new ID
            assert agent_id is not None
            assert len(agent_id) > 0

        asyncio.run(run())


class TestSessionManagerAgentMetadata:
    """Tests for agent metadata operations."""

    def test_get_agent(self) -> None:
        """Test get_agent method."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            # Register an agent
            agent_id, _ = await service.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="test_agent",
                is_root=True,
            )

            # Get the agent
            agent = await service.get_agent(
                user_id="u1",
                session_id="s1",
                agent_id=agent_id,
            )

            assert agent is not None
            assert agent.agent_name == "test_agent"

        asyncio.run(run())

    def test_get_agent_not_found(self) -> None:
        """Test get_agent returns None for nonexistent agent."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            agent = await service.get_agent(
                user_id="u1",
                session_id="s1",
                agent_id="nonexistent",
            )

            assert agent is None

        asyncio.run(run())

    def test_update_agent_metadata(self) -> None:
        """Test update_agent_metadata method."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            # Register an agent
            agent_id, _ = await service.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="original_name",
                is_root=True,
            )

            # Update metadata
            updated = await service.update_agent_metadata(
                user_id="u1",
                session_id="s1",
                agent_id=agent_id,
                agent_name="updated_name",
            )

            assert updated is not None
            assert updated.agent_name == "updated_name"

        asyncio.run(run())

    def test_update_agent_metadata_not_found(self) -> None:
        """Test update_agent_metadata returns None for nonexistent agent."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            result = await service.update_agent_metadata(
                user_id="u1",
                session_id="s1",
                agent_id="nonexistent",
                agent_name="new_name",
            )

            assert result is None

        asyncio.run(run())


class TestSessionManagerInitModels:
    """Tests for setup_models idempotency."""

    def test_setup_models_idempotent(self) -> None:
        """Test that setup_models is idempotent."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            # First call
            await service.setup_models()
            assert service._models_initialized is True

            # Second call should be a no-op
            await service.setup_models()
            assert service._models_initialized is True

        asyncio.run(run())


class TestSessionManagerContextAndStorage:
    """Tests for session context and storage management."""

    def test_update_session_context_replace(self) -> None:
        """Test update_session_context replaces context entirely."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            # Initial context
            await service.update_session_context(
                user_id="u1",
                session_id="s1",
                context={"key1": "value1"},
            )

            # Replace context (full replacement is now the default behavior)
            session = await service.update_session_context(
                user_id="u1",
                session_id="s1",
                context={"key2": "value2"},
            )

            assert session.context == {"key2": "value2"}
            assert "key1" not in session.context

        asyncio.run(run())

    def test_update_session_storage_with_global_storage(self) -> None:
        """Test update_session_storage with GlobalStorage input."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            from nexau.archs.main_sub.agent_context import GlobalStorage

            gs = GlobalStorage()
            gs.set("key", "value")

            session = await service.update_session_storage(
                user_id="u1",
                session_id="s1",
                storage=gs,
            )

            assert session.storage.get("key") == "value"

        asyncio.run(run())

    def test_update_session_storage_replaces_existing(self) -> None:
        """Test update_session_storage replaces existing storage."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            from nexau.archs.main_sub.agent_context import GlobalStorage

            # Initial storage
            gs1 = GlobalStorage()
            gs1.set("key1", "value1")
            await service.update_session_storage(
                user_id="u1",
                session_id="s1",
                storage=gs1,
            )

            # Replace with new storage
            gs2 = GlobalStorage()
            gs2.set("key2", "value2")
            session = await service.update_session_storage(
                user_id="u1",
                session_id="s1",
                storage=gs2,
            )

            # Old key should be gone, new key should exist
            assert session.storage.get("key1") is None
            assert session.storage.get("key2") == "value2"

        asyncio.run(run())

    def test_update_session_state_combined(self) -> None:
        """Test update_session_state updates both context and storage in one call."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            from nexau.archs.main_sub.agent_context import GlobalStorage

            gs = GlobalStorage()
            gs.set("storage_key", "storage_value")

            session = await service.update_session_state(
                user_id="u1",
                session_id="s1",
                context={"context_key": "context_value"},
                storage=gs,
            )

            # Both context and storage should be updated
            assert session.context == {"context_key": "context_value"}
            assert session.storage.get("storage_key") == "storage_value"

        asyncio.run(run())

    def test_get_session(self) -> None:
        """Test get_session method."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            # Create session through update_session_context
            await service.update_session_context(
                user_id="u1",
                session_id="s1",
                context={"key": "value"},
            )

            # Get the session
            session = await service.get_session(user_id="u1", session_id="s1")

            assert session is not None
            assert session.context == {"key": "value"}

        asyncio.run(run())

    def test_get_session_not_found(self) -> None:
        """Test get_session returns None for nonexistent session."""
        engine = InMemoryDatabaseEngine()
        service = SessionManager(engine=engine)

        async def run() -> None:
            await service.setup_models()

            session = await service.get_session(user_id="u1", session_id="nonexistent")
            assert session is None

        asyncio.run(run())
