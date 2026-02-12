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

"""Unit tests for AgentContext and GlobalStorage."""

import pytest

from nexau.archs.main_sub.agent_context import (
    AgentContext,
    GlobalStorage,
    get_context,
    get_context_dict,
    get_context_variables,
    merge_context_variables,
)


class TestAgentContext:
    """Test cases for AgentContext class."""

    def test_initialization_with_context(self):
        """Test AgentContext initialization with context."""
        ctx = AgentContext(context={"key": "value"})
        assert ctx.context == {"key": "value"}
        assert ctx._context_modified is False
        assert ctx._modification_callbacks == []

    def test_initialization_without_context(self):
        """Test AgentContext initialization without context."""
        ctx = AgentContext()
        assert ctx.context == {}

    def test_context_manager_enter_exit(self):
        """Test AgentContext as context manager."""
        ctx = AgentContext({"test": "value"})

        with ctx as entered_ctx:
            assert entered_ctx is ctx
            assert get_context() is ctx

        # After exit, context should be cleared
        assert get_context() is None

    def test_context_manager_restores_previous_context(self):
        """Test that exiting context manager restores previous context."""
        outer_ctx = AgentContext({"outer": "context"})
        inner_ctx = AgentContext({"inner": "context"})

        with outer_ctx:
            assert get_context().context == {"outer": "context"}

            with inner_ctx:
                assert get_context().context == {"inner": "context"}

            # After inner exit, outer context should be restored
            restored = get_context()
            assert restored is not None
            assert restored.context == {"outer": "context"}

        # After outer exit, no context
        assert get_context() is None

    def test_context_manager_exit_with_no_original_context(self):
        """Test exit when there was no original context."""
        ctx = AgentContext({"test": "value"})
        ctx._original_context = None

        # Manually call __exit__ to test the None case
        ctx.__exit__(None, None, None)

        assert get_context() is None

    def test_context_manager_exit_with_empty_original_context(self):
        """Test exit when original context was empty dict."""
        ctx = AgentContext({"test": "value"})
        ctx._original_context = {}

        ctx.__exit__(None, None, None)

        # Should clear context when original was empty
        assert get_context() is None

    def test_update_context(self):
        """Test update_context method."""
        ctx = AgentContext({"key1": "value1"})
        ctx.update_context({"key2": "value2"})

        assert ctx.context == {"key1": "value1", "key2": "value2"}
        assert ctx._context_modified is True

    def test_get_context_value(self):
        """Test get_context_value method."""
        ctx = AgentContext({"key": "value"})

        assert ctx.get_context_value("key") == "value"
        assert ctx.get_context_value("missing") is None
        assert ctx.get_context_value("missing", "default") == "default"

    def test_set_context_value(self):
        """Test set_context_value method."""
        ctx = AgentContext()
        ctx.set_context_value("key", "value")

        assert ctx.context["key"] == "value"
        assert ctx._context_modified is True

    def test_mark_modified_triggers_callbacks(self):
        """Test _mark_modified triggers registered callbacks."""
        ctx = AgentContext()
        callback_called = []

        def callback():
            callback_called.append(True)

        ctx.add_modification_callback(callback)
        ctx._mark_modified()

        assert callback_called == [True]
        assert ctx._context_modified is True

    def test_mark_modified_swallows_callback_exceptions(self):
        """Test _mark_modified swallows exceptions from callbacks."""
        ctx = AgentContext()

        def failing_callback():
            raise RuntimeError("Callback error")

        ctx.add_modification_callback(failing_callback)

        # Should not raise
        ctx._mark_modified()
        assert ctx._context_modified is True

    def test_add_and_remove_modification_callback(self):
        """Test adding and removing modification callbacks."""
        ctx = AgentContext()

        def callback():
            pass

        ctx.add_modification_callback(callback)
        assert callback in ctx._modification_callbacks

        ctx.remove_modification_callback(callback)
        assert callback not in ctx._modification_callbacks

    def test_remove_nonexistent_callback(self):
        """Test removing a callback that doesn't exist."""
        ctx = AgentContext()

        def callback():
            pass

        # Should not raise
        ctx.remove_modification_callback(callback)

    def test_is_modified(self):
        """Test is_modified method."""
        ctx = AgentContext()
        assert ctx.is_modified() is False

        ctx._context_modified = True
        assert ctx.is_modified() is True

    def test_reset_modification_flag(self):
        """Test reset_modification_flag method."""
        ctx = AgentContext()
        ctx._context_modified = True

        ctx.reset_modification_flag()
        assert ctx._context_modified is False

    def test_get_context_variables(self):
        """Test get_context_variables returns copy of context."""
        ctx = AgentContext({"key": "value"})
        variables = ctx.get_context_variables()

        assert variables == {"key": "value"}
        # Should be a copy
        variables["new_key"] = "new_value"
        assert "new_key" not in ctx.context

    def test_merge_context_variables(self):
        """Test merge_context_variables method."""
        ctx = AgentContext({"ctx_key": "ctx_value"})
        existing = {"existing_key": "existing_value", "ctx_key": "old_value"}

        merged = ctx.merge_context_variables(existing)

        # Context variables should override existing
        assert merged == {
            "existing_key": "existing_value",
            "ctx_key": "ctx_value",
        }


class TestModuleLevelFunctions:
    """Test cases for module-level context functions."""

    def test_get_context_returns_none_without_context(self):
        """Test get_context returns None when no context is set."""
        # Ensure no context is set
        import nexau.archs.main_sub.agent_context as ctx_module

        ctx_module._current_context = None

        assert get_context() is None

    def test_get_context_dict_raises_without_context(self):
        """Test get_context_dict raises RuntimeError without context."""
        import nexau.archs.main_sub.agent_context as ctx_module

        ctx_module._current_context = None

        with pytest.raises(RuntimeError, match="No agent context available"):
            get_context_dict()

    def test_get_context_dict_with_context(self):
        """Test get_context_dict returns context dict."""
        ctx = AgentContext({"key": "value"})
        with ctx:
            result = get_context_dict()
            assert result == {"key": "value"}

    def test_get_context_variables_without_context(self):
        """Test get_context_variables returns empty dict without context."""
        import nexau.archs.main_sub.agent_context as ctx_module

        ctx_module._current_context = None

        assert get_context_variables() == {}

    def test_get_context_variables_with_context(self):
        """Test get_context_variables returns context variables."""
        ctx = AgentContext({"key": "value"})
        with ctx:
            result = get_context_variables()
            assert result == {"key": "value"}

    def test_merge_context_variables_without_context(self):
        """Test merge_context_variables returns existing without context."""
        import nexau.archs.main_sub.agent_context as ctx_module

        ctx_module._current_context = None

        existing = {"key": "value"}
        result = merge_context_variables(existing)

        assert result == {"key": "value"}
        assert result is existing

    def test_merge_context_variables_with_context(self):
        """Test merge_context_variables merges with context."""
        ctx = AgentContext({"ctx_key": "ctx_value"})
        with ctx:
            existing = {"existing_key": "existing_value"}
            result = merge_context_variables(existing)

            assert result == {
                "existing_key": "existing_value",
                "ctx_key": "ctx_value",
            }


class TestGlobalStorage:
    """Test cases for GlobalStorage class."""

    def test_initialization(self):
        """Test GlobalStorage initialization."""
        storage = GlobalStorage()
        assert storage._storage == {}
        assert storage._locks == {}

    def test_set_and_get(self):
        """Test set and get methods."""
        storage = GlobalStorage()
        storage.set("key", "value")

        assert storage.get("key") == "value"
        assert storage.get("missing") is None
        assert storage.get("missing", "default") == "default"

    def test_set_with_key_lock(self):
        """Test set with key-specific lock."""
        storage = GlobalStorage()
        storage._get_lock("key")  # Create lock for key

        storage.set("key", "value")
        assert storage.get("key") == "value"

    def test_get_with_key_lock(self):
        """Test get with key-specific lock."""
        storage = GlobalStorage()
        storage._get_lock("key")  # Create lock for key
        storage._storage["key"] = "value"

        assert storage.get("key") == "value"

    def test_update(self):
        """Test update method."""
        storage = GlobalStorage()
        storage.update({"key1": "value1", "key2": "value2"})

        assert storage.get("key1") == "value1"
        assert storage.get("key2") == "value2"

    def test_update_with_existing_key_locks(self):
        """Test update when some keys have locks."""
        storage = GlobalStorage()
        storage._get_lock("key1")  # Create lock for key1

        storage.update({"key1": "value1", "key2": "value2"})

        assert storage.get("key1") == "value1"
        assert storage.get("key2") == "value2"

    def test_delete_existing_key(self):
        """Test delete removes existing key."""
        storage = GlobalStorage()
        storage.set("key", "value")

        result = storage.delete("key")

        assert result is True
        assert storage.get("key") is None

    def test_delete_nonexistent_key(self):
        """Test delete returns False for nonexistent key."""
        storage = GlobalStorage()

        result = storage.delete("missing")

        assert result is False

    def test_delete_with_key_lock(self):
        """Test delete with key-specific lock."""
        storage = GlobalStorage()
        storage._get_lock("key")  # Create lock
        storage._storage["key"] = "value"

        result = storage.delete("key")

        assert result is True
        assert storage.get("key") is None

    def test_delete_nonexistent_with_key_lock(self):
        """Test delete nonexistent key with lock returns False."""
        storage = GlobalStorage()
        storage._get_lock("key")  # Create lock but no value

        result = storage.delete("key")
        assert result is False

    def test_keys(self):
        """Test keys method."""
        storage = GlobalStorage()
        storage.set("key1", "value1")
        storage.set("key2", "value2")

        keys = storage.keys()
        assert sorted(keys) == ["key1", "key2"]

    def test_items(self):
        """Test items method."""
        storage = GlobalStorage()
        storage.set("key1", "value1")
        storage.set("key2", "value2")

        items = storage.items()
        assert sorted(items) == [("key1", "value1"), ("key2", "value2")]

    def test_clear(self):
        """Test clear method."""
        storage = GlobalStorage()
        storage.set("key", "value")
        storage._get_lock("key")

        storage.clear()

        assert storage._storage == {}
        assert storage._locks == {}

    def test_to_dict(self):
        """Test to_dict method."""
        storage = GlobalStorage()
        storage.set("key1", "value1")
        storage.set("key2", "value2")

        result = storage.to_dict()
        assert result == {"key1": "value1", "key2": "value2"}

    def test_lock_key_context_manager(self):
        """Test lock_key context manager."""
        storage = GlobalStorage()
        storage.set("key", "initial")

        with storage.lock_key("key") as ctx:
            assert ctx is storage
            storage.set("key", "modified")

        assert storage.get("key") == "modified"

    def test_lock_multiple_context_manager(self):
        """Test lock_multiple context manager."""
        storage = GlobalStorage()
        storage.set("key1", "value1")
        storage.set("key2", "value2")

        with storage.lock_multiple("key2", "key1") as ctx:
            assert ctx is storage
            storage.set("key1", "modified1")
            storage.set("key2", "modified2")

        assert storage.get("key1") == "modified1"
        assert storage.get("key2") == "modified2"

    def test_validate_with_global_storage(self):
        """Test _validate with GlobalStorage instance."""
        storage = GlobalStorage()
        storage.set("key", "value")

        result = GlobalStorage._validate(storage)
        assert result is storage

    def test_validate_with_dict(self):
        """Test _validate with dict."""
        result = GlobalStorage._validate({"key": "value"})

        assert isinstance(result, GlobalStorage)
        assert result.get("key") == "value"

    def test_validate_with_other_type(self):
        """Test _validate with other type returns empty GlobalStorage."""
        result = GlobalStorage._validate("not a dict")

        assert isinstance(result, GlobalStorage)
        assert result._storage == {}

    def test_serialize_with_global_storage(self):
        """Test _serialize with GlobalStorage instance."""
        storage = GlobalStorage()
        storage.set("key", "value")

        result = GlobalStorage._serialize(storage)
        assert result == {"key": "value"}

    def test_serialize_with_dict(self):
        """Test _serialize with dict."""
        result = GlobalStorage._serialize({"key": "value"})
        assert result == {"key": "value"}

    def test_serialize_with_other_type(self):
        """Test _serialize with other type returns empty dict."""
        result = GlobalStorage._serialize("not a dict")
        assert result == {}

    def test_pydantic_schema(self):
        """Test __get_pydantic_core_schema__ returns valid schema."""
        schema = GlobalStorage.__get_pydantic_core_schema__(GlobalStorage, None)

        # Verify the schema has required keys for json_or_python_schema
        assert "json_schema" in schema
        assert "python_schema" in schema
        assert "serialization" in schema


class TestAgentInitSessionState:
    """Tests for Agent._init_session_state method.

    These tests verify the global_storage initialization precedence:
    1. User-provided storage (override mode)
    2. Session-restored storage (restore mode)
    3. Empty storage (default)
    """

    def test_override_mode_user_provided_storage(self):
        """Test that user-provided global_storage is used directly (override mode)."""
        from nexau.archs.main_sub.agent import Agent
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import SessionManager
        from nexau.archs.session.orm import InMemoryDatabaseEngine

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        # Pre-populate session with some storage
        async def setup_session():
            await session_manager.setup_models()
            gs = GlobalStorage()
            gs.set("session_key", "session_value")
            await session_manager.update_session_storage(
                user_id="test_user",
                session_id="test_session",
                storage=gs,
            )

        import asyncio

        asyncio.run(setup_session())

        # Create agent with user-provided storage
        user_storage = GlobalStorage()
        user_storage.set("user_key", "user_value")

        config = AgentConfig(name="test_agent")
        agent = Agent(
            config=config,
            global_storage=user_storage,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
        )

        # User-provided storage should be used, session storage ignored
        assert agent.global_storage.get("user_key") == "user_value"
        assert agent.global_storage.get("session_key") is None

    def test_restore_mode_from_session(self):
        """Test that storage is restored from session when not provided."""
        from nexau.archs.main_sub.agent import Agent
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import SessionManager
        from nexau.archs.session.orm import InMemoryDatabaseEngine

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        # Pre-populate session with storage
        async def setup_session():
            await session_manager.setup_models()
            gs = GlobalStorage()
            gs.set("restored_key", "restored_value")
            gs.set("another_key", 42)
            await session_manager.update_session_storage(
                user_id="test_user",
                session_id="test_session",
                storage=gs,
            )

        import asyncio

        asyncio.run(setup_session())

        # Create agent without providing global_storage
        config = AgentConfig(name="test_agent")
        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
        )

        # Storage should be restored from session
        assert agent.global_storage.get("restored_key") == "restored_value"
        assert agent.global_storage.get("another_key") == 42

    def test_default_empty_storage_no_session(self):
        """Test that empty storage is created when no session exists."""
        from nexau.archs.main_sub.agent import Agent
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import SessionManager
        from nexau.archs.session.orm import InMemoryDatabaseEngine

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        # Create agent without any pre-existing session
        config = AgentConfig(name="test_agent")
        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="new_user",
            session_id="new_session",
        )

        # Should have empty storage (except for skill_registry which is always set)
        storage_keys = set(agent.global_storage.to_dict().keys())
        # skill_registry is set during agent init, so we exclude it
        user_keys = storage_keys - {"skill_registry"}
        assert user_keys == set()

    def test_override_mode_empty_user_storage(self):
        """Test that empty user-provided storage still overrides session storage."""
        from nexau.archs.main_sub.agent import Agent
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import SessionManager
        from nexau.archs.session.orm import InMemoryDatabaseEngine

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        # Pre-populate session with storage
        async def setup_session():
            await session_manager.setup_models()
            gs = GlobalStorage()
            gs.set("session_key", "should_be_ignored")
            await session_manager.update_session_storage(
                user_id="test_user",
                session_id="test_session",
                storage=gs,
            )

        import asyncio

        asyncio.run(setup_session())

        # Create agent with empty user-provided storage
        empty_storage = GlobalStorage()
        config = AgentConfig(name="test_agent")
        agent = Agent(
            config=config,
            global_storage=empty_storage,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
        )

        # Empty user storage should override session storage
        assert agent.global_storage.get("session_key") is None
        # skill_registry is set during agent init, so we exclude it
        storage_keys = set(agent.global_storage.to_dict().keys())
        user_keys = storage_keys - {"skill_registry"}
        assert user_keys == set()

    def test_agent_id_reuse_for_root_agent(self):
        """Test that root agent reuses existing agent_id from session."""
        from nexau.archs.main_sub.agent import Agent
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import SessionManager
        from nexau.archs.session.orm import InMemoryDatabaseEngine

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        config = AgentConfig(name="test_agent")

        # Create first agent
        agent1 = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
            is_root=True,
        )
        first_agent_id = agent1.agent_id

        # Create second agent with same session
        agent2 = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
            is_root=True,
        )

        # Should reuse the same agent_id
        assert agent2.agent_id == first_agent_id

    def test_non_root_agent_gets_new_id(self):
        """Test that non-root agent gets a new agent_id."""
        from nexau.archs.main_sub.agent import Agent
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import SessionManager
        from nexau.archs.session.orm import InMemoryDatabaseEngine

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        config = AgentConfig(name="test_agent")

        # Create root agent
        root_agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
            is_root=True,
        )

        # Create non-root agent
        sub_agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="test_session",
            is_root=False,
        )

        # Non-root agent should have different agent_id
        assert sub_agent.agent_id != root_agent.agent_id


class TestAgentContextFromSources:
    def test_empty_sources(self):
        ctx = AgentContext.from_sources()
        assert ctx.context == {}

    def test_initial_context_only(self):
        ctx = AgentContext.from_sources(initial_context={"a": "1"})
        assert ctx.context == {"a": "1"}

    def test_template_overrides_initial(self):
        ctx = AgentContext.from_sources(
            initial_context={"k": "old"},
            template={"k": "new"},
        )
        assert ctx.context["k"] == "new"

    def test_legacy_context_triggers_deprecation_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx = AgentContext.from_sources(legacy_context={"x": "1"})
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert ctx.context == {"x": "1"}

    def test_no_warning_without_legacy_context(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AgentContext.from_sources(
                initial_context={"a": "1"},
                template={"b": "2"},
            )
            assert len(w) == 0

    def test_priority_order(self):
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ctx = AgentContext.from_sources(
                initial_context={"a": "init", "b": "init"},
                legacy_context={"b": "legacy", "c": "legacy"},
                template={"c": "template", "d": "template"},
            )
        assert ctx.context == {
            "a": "init",
            "b": "legacy",
            "c": "template",
            "d": "template",
        }
