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
Unit tests for SubAgentManager class.
"""

import threading
from unittest.mock import MagicMock, Mock, patch

import pytest

from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager


class TestSubAgentManager:
    """Test cases for SubAgentManager class."""

    @pytest.fixture
    def mock_sub_agent(self):
        """Create a mock sub-agent."""
        sub_agent = Mock()
        sub_agent.config = Mock()
        sub_agent.config.agent_id = "sub_agent_123"
        sub_agent.run = Mock(return_value="sub agent result")
        sub_agent.stop = Mock()
        return sub_agent

    @pytest.fixture
    def sub_agent_factory(self, mock_sub_agent):
        """Create a sub-agent factory function."""
        return Mock(return_value=mock_sub_agent)

    @pytest.fixture
    def sub_agent_factories(self, sub_agent_factory):
        """Create a dictionary of sub-agent factories."""
        return {"test_sub_agent": sub_agent_factory}

    @pytest.fixture
    def subagent_manager(self, sub_agent_factories):
        """Create a SubAgentManager instance."""
        return SubAgentManager(agent_name="parent_agent", sub_agent_factories=sub_agent_factories)

    def test_initialization(self, sub_agent_factories):
        """Test SubAgentManager initialization."""
        manager = SubAgentManager(agent_name="test_agent", sub_agent_factories=sub_agent_factories)

        assert manager.agent_name == "test_agent"
        assert manager.sub_agent_factories == sub_agent_factories
        assert manager.langfuse_client is None
        assert manager.global_storage is None
        assert manager.xml_parser is not None
        assert isinstance(manager._shutdown_event, threading.Event)
        assert manager.running_sub_agents == {}

    def test_initialization_with_optional_params(self, sub_agent_factories):
        """Test SubAgentManager initialization with optional parameters."""
        mock_langfuse = Mock()
        mock_storage = GlobalStorage()

        manager = SubAgentManager(
            agent_name="test_agent",
            sub_agent_factories=sub_agent_factories,
            langfuse_client=mock_langfuse,
            global_storage=mock_storage,
        )

        assert manager.langfuse_client == mock_langfuse
        assert manager.global_storage == mock_storage

    def test_call_sub_agent_not_found(self, subagent_manager):
        """Test calling a non-existent sub-agent."""
        with pytest.raises(ValueError, match="Sub-agent 'nonexistent' not found"):
            subagent_manager.call_sub_agent("nonexistent", "test message")

    def test_call_sub_agent_during_shutdown(self, subagent_manager):
        """Test calling sub-agent when manager is shutting down."""
        subagent_manager.shutdown()

        with pytest.raises(RuntimeError, match="Agent 'parent_agent' is shutting down"):
            subagent_manager.call_sub_agent("test_sub_agent", "test message")

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_success_no_context(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test successful sub-agent call without context."""
        mock_get_context.return_value = None

        result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        assert result == "sub agent result"
        mock_sub_agent.run.assert_called_once()
        call_args = mock_sub_agent.run.call_args
        assert call_args[0][0] == "test message"
        assert call_args[1]["context"] is None
        assert call_args[1]["parent_agent_state"] is None

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_success_with_context(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test successful sub-agent call with context."""
        mock_context = Mock()
        mock_context.context = {"key": "value"}
        mock_get_context.return_value = mock_context

        result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        assert result == "sub agent result"
        mock_sub_agent.run.assert_called_once()
        call_args = mock_sub_agent.run.call_args
        assert call_args[0][0] == "test message"
        assert call_args[1]["context"] == {"key": "value"}

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_explicit_context(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test sub-agent call with explicitly provided context."""
        mock_context = Mock()
        mock_context.context = {"default": "context"}
        mock_get_context.return_value = mock_context
        explicit_context = {"explicit": "context"}

        result = subagent_manager.call_sub_agent("test_sub_agent", "test message", context=explicit_context)

        assert result == "sub agent result"
        call_args = mock_sub_agent.run.call_args
        # Explicit context should be used
        assert call_args[1]["context"] == explicit_context

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_parent_state(self, mock_get_context, subagent_manager, mock_sub_agent, agent_state):
        """Test sub-agent call with parent agent state."""
        mock_get_context.return_value = None

        result = subagent_manager.call_sub_agent("test_sub_agent", "test message", parent_agent_state=agent_state)

        assert result == "sub agent result"
        call_args = mock_sub_agent.run.call_args
        assert call_args[1]["parent_agent_state"] == agent_state

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_global_storage(self, mock_get_context, sub_agent_factories):
        """Test sub-agent call with global storage."""
        mock_storage = GlobalStorage()
        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"

        # Factory that accepts global_storage parameter
        def factory_with_storage(global_storage=None):
            assert global_storage == mock_storage
            return mock_sub_agent

        factories = {"test_sub": factory_with_storage}
        manager = SubAgentManager(agent_name="parent", sub_agent_factories=factories, global_storage=mock_storage)

        mock_get_context.return_value = None
        result = manager.call_sub_agent("test_sub", "message")

        assert result == "result"
        mock_sub_agent.run.assert_called_once()

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_global_storage_fallback(self, mock_get_context, sub_agent_factories):
        """Test sub-agent call with global storage fallback when factory doesn't support it."""
        mock_storage = GlobalStorage()
        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"
        mock_sub_agent.executor = Mock()

        # Factory that doesn't accept global_storage parameter
        def factory_without_storage():
            return mock_sub_agent

        factories = {"test_sub": factory_without_storage}
        manager = SubAgentManager(agent_name="parent", sub_agent_factories=factories, global_storage=mock_storage)

        mock_get_context.return_value = None
        result = manager.call_sub_agent("test_sub", "message")

        assert result == "result"
        # Global storage should be set after creation
        assert mock_sub_agent.global_storage == mock_storage
        assert mock_sub_agent.executor.global_storage == mock_storage

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_global_storage_with_subagent_manager(self, mock_get_context, sub_agent_factories):
        """Test global storage propagation to nested subagent manager."""
        mock_storage = GlobalStorage()
        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"
        mock_sub_agent.executor = Mock()
        mock_sub_agent.executor.subagent_manager = Mock()

        # Factory that raises TypeError
        def factory_with_error():
            return mock_sub_agent

        factories = {"test_sub": factory_with_error}
        manager = SubAgentManager(agent_name="parent", sub_agent_factories=factories, global_storage=mock_storage)

        mock_get_context.return_value = None
        result = manager.call_sub_agent("test_sub", "message")

        assert result == "result"
        # Global storage should be propagated to nested subagent manager
        assert mock_sub_agent.executor.subagent_manager.global_storage == mock_storage

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_langfuse(self, mock_get_context, sub_agent_factories):
        """Test sub-agent call with Langfuse tracing."""
        mock_langfuse = Mock()
        mock_context_manager = MagicMock()
        mock_langfuse.start_as_current_generation.return_value = mock_context_manager

        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"

        factories = {"test_sub": Mock(return_value=mock_sub_agent)}
        manager = SubAgentManager(agent_name="parent", sub_agent_factories=factories, langfuse_client=mock_langfuse)

        mock_agent_context = Mock()
        mock_agent_context.context = {"key": "value"}
        mock_get_context.return_value = mock_agent_context

        result = manager.call_sub_agent("test_sub", "message")

        assert result == "result"
        mock_sub_agent.run.assert_called_once_with(
            "message",
            context={"key": "value"},
            parent_agent_state=None,
        )

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_shares_langfuse_trace(
        self,
        mock_get_context,
        sub_agent_factories,
    ):
        """Child agents should reuse parent's Langfuse trace and client."""

        mock_langfuse = Mock()
        mock_context_manager = MagicMock()
        mock_langfuse.start_as_current_generation.return_value = mock_context_manager

        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"
        mock_sub_agent.executor = Mock()
        mock_sub_agent.executor.subagent_manager = Mock()

        factories = {"test_sub": Mock(return_value=mock_sub_agent)}
        manager = SubAgentManager(
            agent_name="parent",
            sub_agent_factories=factories,
            langfuse_client=mock_langfuse,
        )

        mock_agent_context = Mock()
        mock_agent_context.context = {"key": "value"}
        mock_get_context.return_value = mock_agent_context

        parent_state = Mock()
        parent_state.langfuse_trace_id = "trace-abc"
        parent_state.langfuse_span_id = "parent-span"

        result = manager.call_sub_agent(
            "test_sub",
            "message",
            parent_agent_state=parent_state,
        )

        assert result == "result"
        mock_sub_agent.run.assert_called_once_with(
            "message",
            context={"key": "value"},
            parent_agent_state=parent_state,
        )
        assert mock_sub_agent.langfuse_trace_id == "trace-abc"
        assert mock_sub_agent.langfuse_client == mock_langfuse
        assert mock_sub_agent.executor.langfuse_client == mock_langfuse
        assert mock_sub_agent.executor.subagent_manager.langfuse_client == mock_langfuse

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_langfuse_error(self, mock_get_context, sub_agent_factories):
        """Test sub-agent call when Langfuse tracing fails."""
        mock_langfuse = Mock()
        mock_langfuse.start_as_current_generation.side_effect = Exception("Langfuse error")

        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"

        factories = {"test_sub": Mock(return_value=mock_sub_agent)}
        manager = SubAgentManager(agent_name="parent", sub_agent_factories=factories, langfuse_client=mock_langfuse)

        mock_agent_context = Mock()
        mock_agent_context.context = {"key": "value"}
        mock_get_context.return_value = mock_agent_context

        # Should fall back to running without Langfuse
        result = manager.call_sub_agent("test_sub", "message")

        assert result == "result"
        # Sub-agent should still run
        assert mock_sub_agent.run.call_count == 1

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_execution_error(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test sub-agent call when execution fails."""
        mock_get_context.return_value = None
        mock_sub_agent.run.side_effect = Exception("Execution error")

        with pytest.raises(Exception, match="Execution error"):
            subagent_manager.call_sub_agent("test_sub_agent", "test message")

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_running_agents_tracking(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test that running sub-agents are tracked and cleaned up."""
        mock_get_context.return_value = None

        # Before call
        assert len(subagent_manager.running_sub_agents) == 0

        result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        # After call, should be cleaned up
        assert len(subagent_manager.running_sub_agents) == 0
        assert result == "sub agent result"

    def test_shutdown(self, subagent_manager, mock_sub_agent):
        """Test shutdown method."""
        # Add a running sub-agent
        subagent_manager.running_sub_agents["sub_123"] = mock_sub_agent

        subagent_manager.shutdown()

        assert subagent_manager._shutdown_event.is_set()
        mock_sub_agent.stop.assert_called_once()

    def test_shutdown_with_error(self, subagent_manager):
        """Test shutdown when sub-agent stop raises error."""
        mock_sub_agent = Mock()
        mock_sub_agent.stop.side_effect = Exception("Stop error")
        subagent_manager.running_sub_agents["sub_123"] = mock_sub_agent

        # Should not raise, just log error
        subagent_manager.shutdown()

        assert subagent_manager._shutdown_event.is_set()
        mock_sub_agent.stop.assert_called_once()

    def test_shutdown_no_running_agents(self, subagent_manager):
        """Test shutdown when no agents are running."""
        subagent_manager.shutdown()

        assert subagent_manager._shutdown_event.is_set()
        # Should not raise

    def test_add_sub_agent(self, subagent_manager):
        """Test adding a sub-agent factory."""
        new_factory = Mock()

        subagent_manager.add_sub_agent("new_agent", new_factory)

        assert "new_agent" in subagent_manager.sub_agent_factories
        assert subagent_manager.sub_agent_factories["new_agent"] == new_factory

    def test_add_sub_agent_overwrite(self, subagent_manager):
        """Test overwriting an existing sub-agent factory."""
        new_factory = Mock()

        subagent_manager.add_sub_agent("test_sub_agent", new_factory)

        assert subagent_manager.sub_agent_factories["test_sub_agent"] == new_factory

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_without_global_storage(self, mock_get_context, sub_agent_factories):
        """Test sub-agent call without global storage."""
        mock_sub_agent = Mock()
        mock_sub_agent.config.agent_id = "sub_123"
        mock_sub_agent.run.return_value = "result"

        factories = {"test_sub": Mock(return_value=mock_sub_agent)}
        manager = SubAgentManager(agent_name="parent", sub_agent_factories=factories, global_storage=None)

        mock_get_context.return_value = None
        result = manager.call_sub_agent("test_sub", "message")

        assert result == "result"
        # Factory should be called without global_storage
        factories["test_sub"].assert_called_once_with()

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_multiple_times(self, mock_get_context, subagent_manager):
        """Test calling sub-agent multiple times."""
        mock_get_context.return_value = None

        result1 = subagent_manager.call_sub_agent("test_sub_agent", "message 1")
        result2 = subagent_manager.call_sub_agent("test_sub_agent", "message 2")

        assert result1 == "sub agent result"
        assert result2 == "sub agent result"
        # Each call should create a new sub-agent instance
        assert len(subagent_manager.running_sub_agents) == 0
