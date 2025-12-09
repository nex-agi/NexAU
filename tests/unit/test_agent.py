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

"""
Unit tests for agent components.
"""

from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent, create_agent
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig, ExecutionConfig
from nexau.archs.tool import Tool
from nexau.archs.tracer.core import BaseTracer


class TestAgent:
    """Test cases for Agent class."""

    def test_agent_initialization(self, agent_config, execution_config, global_storage):
        """Test agent initialization."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config.max_iterations = execution_config.max_iterations
            agent_config.max_context_tokens = execution_config.max_context_tokens
            agent_config.max_running_subagents = execution_config.max_running_subagents
            agent_config.retry_attempts = execution_config.retry_attempts
            agent_config.timeout = execution_config.timeout
            agent_config.tool_call_mode = execution_config.tool_call_mode

            agent = Agent(agent_config, global_storage)

            expected_exec_config = ExecutionConfig.from_agent_config(agent_config)

            assert agent.config == agent_config
            assert agent.global_storage == global_storage
            assert agent.exec_config == expected_exec_config
            assert agent.openai_client is not None
            assert agent.history == []
            assert agent.queued_messages == []

    def test_agent_initialization_no_external_client(self, agent_config, global_storage):
        """Test agent initialization when OpenAI client creation fails."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.side_effect = Exception("API Error")

            agent = Agent(agent_config, global_storage)

            assert agent.openai_client is None

    def test_add_tool(self, agent_config, global_storage, sample_tool):
        """Test adding tools to agent."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            agent.add_tool(sample_tool)

            assert sample_tool.name in agent.tool_registry
            assert sample_tool in agent.config.tools

    def test_add_tool_updates_executor_payload_openai(self, agent_config, global_storage):
        """add_tool should propagate structured payload for OpenAI mode."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config.tool_call_mode = "openai"
            agent = Agent(agent_config, global_storage)

            tool = Tool(
                name="dynamic",
                description="dyn tool",
                input_schema={"type": "object", "properties": {}},
                implementation=lambda: {"result": "ok"},
            )

            agent.add_tool(tool)

            assert agent.executor.tool_executor.tool_registry["dynamic"] is tool
            assert agent.executor.structured_tool_payload[-1]["function"]["name"] == "dynamic"

    def test_add_tool_updates_executor_payload_anthropic(self, global_storage):
        """add_tool should propagate structured payload for Anthropic mode."""
        with patch("nexau.archs.main_sub.agent.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = Mock()

            anthropic_llm = LLMConfig(
                model="claude-3-opus-20240229",
                base_url="https://api.anthropic.com",
                api_key="test-key",
                api_type="anthropic_chat_completion",
            )
            agent_config = AgentConfig(
                name="test_agent",
                agent_id="test_agent_123",
                system_prompt="",
                tools=[],
                sub_agents=[],
                llm_config=anthropic_llm,
                tool_call_mode="anthropic",
            )

            agent = Agent(agent_config, global_storage)

            tool = Tool(
                name="dynamic",
                description="dyn tool",
                input_schema={"type": "object", "properties": {}},
                implementation=lambda: {"result": "ok"},
            )

            agent.add_tool(tool)

            assert agent.executor.tool_executor.tool_registry["dynamic"] is tool
            assert agent.executor.structured_tool_payload[-1]["name"] == "dynamic"

    def test_add_sub_agent(self, agent_config, global_storage):
        """Test adding sub-agents to agent."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            def mock_sub_agent_factory():
                return create_agent(name="sub_agent", llm_config=LLMConfig())

            agent.add_sub_agent("test_sub", mock_sub_agent_factory)

            assert "test_sub" in agent.config.sub_agent_factories

    def test_enqueue_message(self, agent_config, global_storage):
        """Test enqueuing messages."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            test_message = {"role": "user", "content": "Test message"}

            agent.enqueue_message(test_message)

            assert test_message in agent.executor.queued_messages

    def test_agent_cleanup(self, agent_config, global_storage):
        """Test agent cleanup."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            with patch.object(agent.executor, "cleanup") as mock_cleanup:
                agent.stop()

                mock_cleanup.assert_called_once()

    def test_agent_injects_tracer_into_global_storage(self, agent_config, global_storage):
        """Tracer set on config should be visible via shared global storage."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            tracer = Mock(spec=BaseTracer)
            agent_payload = agent_config.model_dump()
            agent_payload["tracers"] = [tracer]
            config_with_tracer = AgentConfig(**agent_payload)

            Agent(config_with_tracer, global_storage)

            assert global_storage.get("tracer") is tracer

    def test_agent_preserves_existing_global_tracer(self, agent_config, global_storage):
        """Nested agents should reuse tracer already stored in global storage without mutating config."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            parent_tracer = Mock(spec=BaseTracer)
            global_storage.set("tracer", parent_tracer)

            agent_payload = agent_config.model_dump()
            child_tracer = Mock(spec=BaseTracer)
            agent_payload["tracers"] = [child_tracer]
            config_with_tracer = AgentConfig(**agent_payload)

            Agent(config_with_tracer, global_storage)

            assert global_storage.get("tracer") is parent_tracer
            assert config_with_tracer.resolved_tracer is child_tracer

    def test_initialize_mcp_tools_success(self, agent_config, global_storage):
        """Test successful MCP tools initialization."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock the sync_initialize_mcp_tools function
            mock_mcp_tool = Mock()
            mock_mcp_tool.name = "test_mcp_tool"

            with patch("nexau.archs.tool.builtin.sync_initialize_mcp_tools") as mock_sync_init:
                mock_sync_init.return_value = [mock_mcp_tool]

                # Add MCP servers to config
                agent_config.mcp_servers = [{"name": "test_server", "type": "stdio", "command": "python", "args": ["server.py"]}]

                agent = Agent(agent_config, global_storage)

                # Verify MCP tools were added
                assert mock_mcp_tool in agent.config.tools
                assert "test_mcp_tool" in agent.tool_registry
                mock_sync_init.assert_called_once_with(agent_config.mcp_servers)

    def test_initialize_mcp_tools_import_error(self, agent_config, global_storage):
        """Test MCP tools initialization with import error."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock import error
            with patch("nexau.archs.tool.builtin.sync_initialize_mcp_tools", side_effect=ImportError("MCP not available")):
                agent_config.mcp_servers = [{"name": "test_server"}]

                # Should not raise, but log error
                agent = Agent(agent_config, global_storage)

                # MCP tools should not be added
                assert len(agent.config.tools) == 0

    def test_initialize_mcp_tools_general_error(self, agent_config, global_storage):
        """Test MCP tools initialization with general error."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock general exception
            with patch("nexau.archs.tool.builtin.sync_initialize_mcp_tools", side_effect=Exception("Connection failed")):
                agent_config.mcp_servers = [{"name": "test_server"}]

                # Should not raise, but log error
                agent = Agent(agent_config, global_storage)

                # MCP tools should not be added
                assert len(agent.config.tools) == 0

    def test_run_basic(self, agent_config, global_storage):
        """Test basic agent run."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            # Mock executor.execute
            with patch.object(agent.executor, "execute") as mock_execute:
                mock_execute.return_value = (
                    "Test response",
                    [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "Test message"},
                        {"role": "assistant", "content": "Test response"},
                    ],
                )

                response = agent.run("Test message")

                assert response == "Test response"
                assert len(agent.history) > 0
                assert agent.history[-1]["role"] == "assistant"
                assert agent.history[-1]["content"] == "Test response"
                mock_execute.assert_called_once()

    def test_run_with_history(self, agent_config, global_storage):
        """Test agent run with existing history."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            existing_history = [{"role": "user", "content": "Previous message"}, {"role": "assistant", "content": "Previous response"}]

            with patch.object(agent.executor, "execute") as mock_execute:
                mock_execute.return_value = (
                    "New response",
                    [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "Previous message"},
                        {"role": "assistant", "content": "Previous response"},
                        {"role": "user", "content": "New message"},
                        {"role": "assistant", "content": "New response"},
                    ],
                )

                response = agent.run("New message", history=existing_history)

                assert response == "New response"
                # Should include system prompt + existing history + new message
                call_args = mock_execute.call_args[0][0]
                assert any(msg["content"] == "Previous message" for msg in call_args)

    def test_run_with_context_state_config(self, agent_config, global_storage):
        """Test agent run with context, state, and config."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Set initial values in agent config
            agent_config.initial_context = {"initial_ctx": "value"}
            agent_config.initial_state = {"initial_state": "value"}
            agent_config.initial_config = {"initial_config": "value"}

            agent = Agent(agent_config, global_storage)

            with patch.object(agent.executor, "execute") as mock_execute:
                mock_execute.return_value = (
                    "Response",
                    [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "Message"},
                        {"role": "assistant", "content": "Response"},
                    ],
                )

                response = agent.run(
                    "Message", context={"runtime_ctx": "value"}, state={"runtime_state": "value"}, config={"runtime_config": "value"}
                )

                assert response == "Response"
                # Verify merged context, state, and config were used
                mock_execute.assert_called_once()

    def test_run_with_error_handler(self, agent_config, global_storage):
        """Test agent run with error handler."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            error_handler_called = []

            def custom_error_handler(error, agent, context):
                error_handler_called.append(True)
                return f"Error handled: {str(error)}"

            agent_config.error_handler = custom_error_handler
            agent = Agent(agent_config, global_storage)

            with patch.object(agent.executor, "execute", side_effect=Exception("Test error")):
                response = agent.run("Message")

                assert "Error handled: Test error" in response
                assert len(error_handler_called) == 1
                assert agent.history[-1]["role"] == "assistant"

    def test_run_without_error_handler(self, agent_config, global_storage):
        """Test agent run without error handler."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            with patch.object(agent.executor, "execute", side_effect=Exception("Test error")):
                with pytest.raises(Exception, match="Test error"):
                    agent.run("Message")

    def test_run_with_parent_agent_state(self, agent_config, global_storage, mock_executor):
        """Test agent run with parent agent state."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(agent_config, global_storage)

            parent_state = AgentState(
                agent_name="parent_agent",
                agent_id="parent_123",
                context=AgentContext(),
                global_storage=global_storage,
                executor=mock_executor,
            )

            with patch.object(agent.executor, "execute") as mock_execute:
                mock_execute.return_value = (
                    "Response",
                    [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "Message"},
                        {"role": "assistant", "content": "Response"},
                    ],
                )

                response = agent.run("Message", parent_agent_state=parent_state)

                assert response == "Response"
                # Verify parent_agent_state was passed to execute
                call_args = mock_execute.call_args[0]
                agent_state = call_args[1]
                assert agent_state.parent_agent_state == parent_state


class TestCreateAgent:
    """Test cases for create_agent factory function."""

    def test_create_agent_minimal(self):
        """Test creating agent with minimal parameters."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = create_agent(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
            )

            assert agent.config.name == "test_agent"
            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert isinstance(agent.config.agent_id, str)
            assert len(agent.config.agent_id) > 0

    def test_create_agent_with_dict_llm_config(self):
        """Test creating agent with dictionary LLM config."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = create_agent(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.5),
            )

            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert agent.config.llm_config.temperature == 0.5

    def test_create_agent_with_llm_kwargs(self):
        """Test creating agent with LLM kwargs."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = create_agent(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.3, max_tokens=2000),
                temperature=0.3,
                max_tokens=2000,
            )

            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert agent.config.llm_config.temperature == 0.3
            assert agent.config.llm_config.max_tokens == 2000

    def test_create_agent_missing_llm_config(self):
        """Test creating agent without LLM config."""
        with pytest.raises(ValueError, match="llm_config is required"):
            create_agent(name="test_agent")

    def test_create_agent_with_tools(self, sample_tool):
        """Test creating agent with tools."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = create_agent(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[sample_tool],
            )

            assert len(agent.config.tools) == 1
            assert agent.config.tools[0].name == "sample_tool"

    def test_create_agent_with_mcp_servers(self):
        """Test creating agent with MCP servers."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock the MCP initialization to prevent actual server startup
            with patch("nexau.archs.main_sub.agent.Agent._initialize_mcp_tools"):
                mcp_servers = [{"name": "test_server", "type": "stdio", "command": "python", "args": ["server.py"]}]

                agent = create_agent(
                    name="test_agent",
                    llm_config=LLMConfig(model="gpt-4o-mini"),
                    mcp_servers=mcp_servers,
                )

                assert len(agent.config.mcp_servers) == 1
                assert agent.config.mcp_servers[0]["name"] == "test_server"

    def test_create_agent_with_hooks(self):
        """Test creating agent with hooks."""

        def mock_hook(*args, **kwargs):
            pass

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = create_agent(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                after_model_hooks=[mock_hook],
                before_model_hooks=[mock_hook],
                after_tool_hooks=[mock_hook],
            )

            assert len(agent.config.after_model_hooks) == 1
            assert len(agent.config.before_model_hooks) == 1
            assert len(agent.config.after_tool_hooks) == 1


class TestAgentState:
    """Test cases for AgentState."""

    def test_agent_state_initialization(self, mock_executor):
        """Test agent state initialization."""
        context = AgentContext({"test": "value"})
        global_storage = GlobalStorage()

        state = AgentState(
            agent_name="test_agent", agent_id="test_id_123", context=context, global_storage=global_storage, executor=mock_executor
        )

        assert state.agent_name == "test_agent"
        assert state.agent_id == "test_id_123"
        assert state.context == context
        assert state.global_storage == global_storage
        assert state.parent_agent_state is None

    def test_agent_state_with_parent(self, mock_executor):
        """Test agent state with parent state."""
        parent_state = AgentState(
            agent_name="parent",
            agent_id="parent_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            executor=mock_executor,
        )

        child_state = AgentState(
            agent_name="child",
            agent_id="child_456",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            parent_agent_state=parent_state,
            executor=mock_executor,
        )

        assert child_state.parent_agent_state == parent_state

    def test_get_set_context_value(self, mock_executor):
        """Test getting and setting context values."""
        state = AgentState(
            agent_name="test",
            agent_id="test_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            executor=mock_executor,
        )

        state.set_context_value("test_key", "test_value")
        assert state.get_context_value("test_key") == "test_value"
        assert state.get_context_value("missing", "default") == "default"

    def test_get_set_global_value(self, mock_executor):
        """Test getting and setting global values."""
        state = AgentState(
            agent_name="test",
            agent_id="test_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            executor=mock_executor,
        )

        state.set_global_value("global_key", "global_value")
        assert state.get_global_value("global_key") == "global_value"
        assert state.get_global_value("missing", "default") == "default"

    def test_add_tool_delegates_to_executor(self, agent_state, sample_tool, mock_executor):
        """add_tool should forward to the executor for registration."""
        agent_state.add_tool(sample_tool)

        mock_executor.add_tool.assert_called_once_with(sample_tool)

    def test_string_representations(self, mock_executor):
        """Test string representations of agent state."""
        state = AgentState(
            agent_name="test_agent",
            agent_id="test_id_123",
            context=AgentContext({"key": "value"}),
            global_storage=GlobalStorage(),
            executor=mock_executor,
        )

        repr_str = repr(state)
        str_repr = str(state)

        # Test __repr__ contains agent info
        assert "test_agent" in repr_str
        assert "test_id_123" in repr_str

        # Test __str__ contains agent name and key counts
        assert "test_agent" in str_repr
        assert "1 context keys" in str_repr
        assert "0 global keys" in str_repr


class TestAgentContext:
    """Test cases for AgentContext."""

    def test_agent_context_initialization(self):
        """Test agent context initialization."""
        context = AgentContext({"initial": "value"})

        assert context.context == {"initial": "value"}
        assert not context.is_modified()

    def test_context_enter_exit(self):
        """Test context manager functionality."""
        context = AgentContext({"test": "value"})

        # Test entering context
        result = context.__enter__()
        assert result == context

        # Test exiting context (no previous context)
        context.__exit__(None, None, None)
        # Should not raise any exceptions

    def test_update_context(self):
        """Test updating context."""
        context = AgentContext({"initial": "value"})

        context.update_context({"new_key": "new_value", "initial": "updated"})

        assert context.context["initial"] == "updated"
        assert context.context["new_key"] == "new_value"
        assert context.is_modified()

    def test_get_set_context_value(self):
        """Test getting and setting individual context values."""
        context = AgentContext()

        context.set_context_value("test_key", "test_value")
        assert context.get_context_value("test_key") == "test_value"
        assert context.get_context_value("missing", "default") == "default"
        assert context.is_modified()

    def test_merge_context_variables(self):
        """Test merging context variables."""
        context = AgentContext({"context_key": "context_value"})

        existing = {"existing_key": "existing_value"}
        merged = context.merge_context_variables(existing)

        assert merged["existing_key"] == "existing_value"
        assert merged["context_key"] == "context_value"
        # Context should take priority
        merged_conflict = context.merge_context_variables({"context_key": "conflict"})
        assert merged_conflict["context_key"] == "context_value"

    def test_modification_callbacks(self):
        """Test modification callbacks."""
        context = AgentContext()
        callback_called = []

        def test_callback():
            callback_called.append(True)

        context.add_modification_callback(test_callback)
        context.set_context_value("test", "value")

        assert len(callback_called) == 1
        assert context.is_modified()

        # Test removing callback
        context.remove_modification_callback(test_callback)
        context.set_context_value("test2", "value2")

        assert len(callback_called) == 1  # Should not increase

    def test_reset_modification_flag(self):
        """Test resetting modification flag."""
        context = AgentContext()

        context.set_context_value("test", "value")
        assert context.is_modified()

        context.reset_modification_flag()
        assert not context.is_modified()


class TestGlobalStorage:
    """Test cases for GlobalStorage."""

    def test_global_storage_initialization(self):
        """Test global storage initialization."""
        storage = GlobalStorage()

        assert storage._storage == {}
        assert storage._locks == {}

    def test_set_get_values(self):
        """Test setting and getting values."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        assert storage.get("key1") == "value1"
        assert storage.get("missing", "default") == "default"

    def test_update_values(self):
        """Test updating multiple values."""
        storage = GlobalStorage()

        storage.update({"key1": "value1", "key2": "value2"})
        assert storage.get("key1") == "value1"
        assert storage.get("key2") == "value2"

    def test_delete_values(self):
        """Test deleting values."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        assert storage.delete("key1") is True
        assert storage.delete("missing") is False
        assert storage.get("key1") is None

    def test_keys_items_methods(self):
        """Test keys and items methods."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        storage.set("key2", "value2")

        assert set(storage.keys()) == {"key1", "key2"}
        assert set(storage.items()) == {("key1", "value1"), ("key2", "value2")}

    def test_clear_storage(self):
        """Test clearing storage."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        storage.clear()

        assert storage.keys() == []
        assert storage._storage == {}

    def test_lock_key_context_manager(self):
        """Test key-specific locking."""
        storage = GlobalStorage()

        with storage.lock_key("test_key"):
            storage.set("test_key", "value")

        assert storage.get("test_key") == "value"

    def test_lock_multiple_context_manager(self):
        """Test multiple key locking."""
        storage = GlobalStorage()

        with storage.lock_multiple("key1", "key2"):
            storage.set("key1", "value1")
            storage.set("key2", "value2")

        assert storage.get("key1") == "value1"
        assert storage.get("key2") == "value2"
