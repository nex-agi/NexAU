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

"""Unit tests for CLI subagent adapter."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager
from nexau.cli.cli_subagent_adapter import (
    CLIEnabledSubAgentManager,
    attach_cli_manager,
    attach_cli_to_agent,
)


class TestCLIEnabledSubAgentManager:
    """Test cases for CLIEnabledSubAgentManager class."""

    def test_initialization(self):
        """Test CLIEnabledSubAgentManager initialization."""
        progress_hook = Mock()
        tool_hook = Mock()
        event_callback = Mock()

        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            global_storage=None,
            progress_hook=progress_hook,
            tool_hook=tool_hook,
            event_callback=event_callback,
        )

        assert manager.agent_name == "test_agent"
        assert manager.cli_progress_hook == progress_hook
        assert manager.cli_tool_hook == tool_hook
        assert manager.event_callback == event_callback

    def test_from_existing(self):
        """Test creating CLIEnabledSubAgentManager from existing SubAgentManager."""
        # Create original manager
        original_manager = SubAgentManager(
            agent_name="original_agent",
            sub_agents={"sub1": MagicMock(spec=AgentConfig)},
            global_storage={"key": "value"},
        )
        original_manager.running_sub_agents = {"test": Mock()}
        original_manager.session_manager = Mock()
        original_manager.user_id = "user123"
        original_manager.session_id = "session456"

        progress_hook = Mock()
        tool_hook = Mock()
        event_callback = Mock()

        # Create CLI-enabled manager from existing
        cli_manager = CLIEnabledSubAgentManager.from_existing(
            original_manager,
            progress_hook,
            tool_hook,
            event_callback,
        )

        assert cli_manager.agent_name == "original_agent"
        assert cli_manager.sub_agents == original_manager.sub_agents
        assert cli_manager.global_storage == original_manager.global_storage
        assert cli_manager.running_sub_agents == original_manager.running_sub_agents
        assert cli_manager.session_manager == original_manager.session_manager
        assert cli_manager.user_id == original_manager.user_id
        assert cli_manager.session_id == original_manager.session_id
        assert cli_manager.cli_progress_hook == progress_hook
        assert cli_manager.cli_tool_hook == tool_hook
        assert cli_manager.event_callback == event_callback

    def test_from_existing_with_shutdown_event_set(self):
        """Test creating manager from existing with shutdown event already set."""
        original_manager = SubAgentManager(
            agent_name="original_agent",
            sub_agents={},
        )
        original_manager._shutdown_event.set()

        cli_manager = CLIEnabledSubAgentManager.from_existing(
            original_manager,
            progress_hook=None,
            tool_hook=None,
            event_callback=None,
        )

        assert cli_manager._shutdown_event.is_set()

    def test_emit_event_with_callback(self):
        """Test _emit_event calls the event_callback."""
        event_callback = Mock()
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            event_callback=event_callback,
        )

        payload = {"key": "value"}
        manager._emit_event("test_event", payload)

        event_callback.assert_called_once_with("test_event", payload)

    def test_emit_event_without_callback(self):
        """Test _emit_event does nothing without callback."""
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            event_callback=None,
        )

        # Should not raise
        manager._emit_event("test_event", {"key": "value"})

    def test_emit_event_swallows_exceptions(self):
        """Test _emit_event swallows callback exceptions."""
        event_callback = Mock(side_effect=RuntimeError("callback error"))
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            event_callback=event_callback,
        )

        # Should not raise despite callback error
        manager._emit_event("test_event", {"key": "value"})

    def test_inject_cli_hooks_skips_already_injected(self):
        """Test _inject_cli_hooks skips if already injected."""
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            progress_hook=Mock(),
            tool_hook=Mock(),
        )

        sub_agent = Mock()
        sub_agent._cli_hooks_injected = True

        # Track middleware calls
        middleware_manager = Mock()
        middleware_manager.middlewares = Mock()
        middleware_manager.middlewares.insert = Mock()
        sub_agent.executor.middleware_manager = middleware_manager

        manager._inject_cli_hooks(sub_agent)

        # Should not have called insert since already injected
        middleware_manager.middlewares.insert.assert_not_called()

    def test_inject_cli_hooks_with_progress_hook(self):
        """Test _inject_cli_hooks injects progress hook middleware."""
        progress_hook = Mock()
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            progress_hook=progress_hook,
            tool_hook=None,
        )

        sub_agent = Mock()
        sub_agent._cli_hooks_injected = False
        middleware_manager = Mock()
        middleware_manager.middlewares = []
        sub_agent.executor.middleware_manager = middleware_manager
        sub_agent.executor.subagent_manager = None

        manager._inject_cli_hooks(sub_agent)

        assert len(middleware_manager.middlewares) == 1
        assert middleware_manager.middlewares[0].name == "cli_progress_hook"
        assert sub_agent._cli_hooks_injected is True

    def test_inject_cli_hooks_with_tool_hook(self):
        """Test _inject_cli_hooks injects tool hook middleware."""
        tool_hook = Mock()
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            progress_hook=None,
            tool_hook=tool_hook,
        )

        sub_agent = Mock()
        sub_agent._cli_hooks_injected = False
        middleware_manager = Mock()
        middleware_manager.middlewares = []
        sub_agent.executor.middleware_manager = middleware_manager
        sub_agent.executor.subagent_manager = None

        manager._inject_cli_hooks(sub_agent)

        assert len(middleware_manager.middlewares) == 1
        assert middleware_manager.middlewares[0].name == "cli_tool_hook"
        assert sub_agent._cli_hooks_injected is True

    def test_inject_cli_hooks_without_middleware_manager(self):
        """Test _inject_cli_hooks handles missing middleware_manager."""
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            progress_hook=Mock(),
            tool_hook=Mock(),
        )

        sub_agent = Mock()
        sub_agent._cli_hooks_injected = False
        sub_agent.executor.middleware_manager = None
        sub_agent.executor.subagent_manager = None

        # Should not raise
        manager._inject_cli_hooks(sub_agent)
        assert sub_agent._cli_hooks_injected is True

    def test_inject_cli_hooks_propagates_to_nested_manager(self):
        """Test _inject_cli_hooks propagates to nested sub-agent managers."""
        event_callback = Mock()
        progress_hook = Mock()
        tool_hook = Mock()

        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            progress_hook=progress_hook,
            tool_hook=tool_hook,
            event_callback=event_callback,
        )

        # Create nested manager
        nested_manager = SubAgentManager(
            agent_name="nested_agent",
            sub_agents={},
        )

        sub_agent = Mock()
        sub_agent._cli_hooks_injected = False
        sub_agent.executor.middleware_manager = Mock()
        sub_agent.executor.middleware_manager.middlewares = []
        sub_agent.executor.subagent_manager = nested_manager
        sub_agent.executor.batch_processor = None

        manager._inject_cli_hooks(sub_agent)

        # Verify nested manager was replaced with CLI-enabled version
        assert isinstance(sub_agent.executor.subagent_manager, CLIEnabledSubAgentManager)

    def test_inject_cli_hooks_updates_batch_processor(self):
        """Test _inject_cli_hooks updates batch_processor's subagent_manager."""
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
            event_callback=Mock(),
        )

        nested_manager = SubAgentManager(
            agent_name="nested_agent",
            sub_agents={},
        )

        batch_processor = Mock()
        batch_processor.subagent_manager = nested_manager

        sub_agent = Mock()
        sub_agent._cli_hooks_injected = False
        sub_agent.executor.middleware_manager = Mock()
        sub_agent.executor.middleware_manager.middlewares = []
        sub_agent.executor.subagent_manager = nested_manager
        sub_agent.executor.batch_processor = batch_processor

        manager._inject_cli_hooks(sub_agent)

        # Verify batch_processor's manager was also updated
        assert isinstance(batch_processor.subagent_manager, CLIEnabledSubAgentManager)

    def test_call_sub_agent_raises_on_shutdown(self):
        """Test call_sub_agent raises RuntimeError when shutting down."""
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={"sub1": MagicMock(spec=AgentConfig)},
        )
        manager._shutdown_event.set()

        with pytest.raises(RuntimeError, match="shutting down"):
            manager.call_sub_agent(
                sub_agent_name="sub1",
                message="test message",
            )

    def test_call_sub_agent_raises_on_unknown_agent(self):
        """Test call_sub_agent raises ValueError for unknown agent."""
        manager = CLIEnabledSubAgentManager(
            agent_name="test_agent",
            sub_agents={},
        )

        with pytest.raises(ValueError, match="not found"):
            manager.call_sub_agent(
                sub_agent_name="unknown_agent",
                message="test message",
            )

    @patch("nexau.cli.cli_subagent_adapter.Agent")
    @patch("nexau.cli.cli_subagent_adapter.get_context")
    def test_call_sub_agent_success(self, mock_get_context, mock_agent_class):
        """Test successful call_sub_agent execution."""
        mock_config = MagicMock(spec=AgentConfig)
        mock_config.name = "test_sub_agent"

        event_callback = Mock()
        manager = CLIEnabledSubAgentManager(
            agent_name="parent_agent",
            sub_agents={"sub1": mock_config},
            event_callback=event_callback,
        )

        # Setup mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "agent_123"
        mock_agent.agent_name = "sub1"
        mock_agent.config.name = "test_sub_agent"
        mock_agent.executor.middleware_manager = Mock()
        mock_agent.executor.middleware_manager.middlewares = []
        mock_agent.executor.subagent_manager = None
        mock_agent._cli_hooks_injected = False
        mock_agent.run.return_value = "Result from sub-agent"
        mock_agent_class.return_value = mock_agent

        # Mock context
        mock_ctx = Mock()
        mock_ctx.context = {"ctx_key": "ctx_value"}
        mock_get_context.return_value = mock_ctx

        result = manager.call_sub_agent(
            sub_agent_name="sub1",
            message="Hello sub-agent",
        )

        assert "Result from sub-agent" in result
        assert "agent_123" in result
        mock_agent.run.assert_called_once()

        # Verify events were emitted
        assert event_callback.call_count == 2  # start and complete
        start_call = event_callback.call_args_list[0]
        assert start_call[0][0] == "start"
        complete_call = event_callback.call_args_list[1]
        assert complete_call[0][0] == "complete"

    @patch("nexau.cli.cli_subagent_adapter.Agent")
    @patch("nexau.cli.cli_subagent_adapter.get_context")
    def test_call_sub_agent_with_explicit_context(self, mock_get_context, mock_agent_class):
        """Test call_sub_agent with explicitly provided context."""
        mock_config = MagicMock(spec=AgentConfig)

        manager = CLIEnabledSubAgentManager(
            agent_name="parent_agent",
            sub_agents={"sub1": mock_config},
        )

        mock_agent = Mock()
        mock_agent.agent_id = "agent_123"
        mock_agent.config.name = "sub1"
        mock_agent.executor.middleware_manager = None
        mock_agent.executor.subagent_manager = None
        mock_agent._cli_hooks_injected = False
        mock_agent.run.return_value = "Result"
        mock_agent_class.return_value = mock_agent

        explicit_context = {"explicit": "context"}
        manager.call_sub_agent(
            sub_agent_name="sub1",
            message="Test",
            context=explicit_context,
        )

        # Should not call get_context since context was provided
        mock_get_context.assert_not_called()
        mock_agent.run.assert_called_once()
        call_kwargs = mock_agent.run.call_args[1]
        assert call_kwargs["context"] == explicit_context

    @patch("nexau.cli.cli_subagent_adapter.Agent")
    @patch("nexau.cli.cli_subagent_adapter.get_context")
    def test_call_sub_agent_error_emits_event(self, mock_get_context, mock_agent_class):
        """Test call_sub_agent emits error event on exception."""
        mock_config = MagicMock(spec=AgentConfig)

        event_callback = Mock()
        manager = CLIEnabledSubAgentManager(
            agent_name="parent_agent",
            sub_agents={"sub1": mock_config},
            event_callback=event_callback,
        )

        mock_agent = Mock()
        mock_agent.agent_id = "agent_123"
        mock_agent.config.name = "sub1"
        mock_agent.executor.middleware_manager = None
        mock_agent.executor.subagent_manager = None
        mock_agent._cli_hooks_injected = False
        mock_agent.run.side_effect = RuntimeError("Execution failed")
        mock_agent_class.return_value = mock_agent
        mock_get_context.return_value = None

        with pytest.raises(RuntimeError, match="Execution failed"):
            manager.call_sub_agent(
                sub_agent_name="sub1",
                message="Test",
            )

        # Verify error event was emitted
        error_call = [c for c in event_callback.call_args_list if c[0][0] == "error"]
        assert len(error_call) == 1
        assert "Execution failed" in error_call[0][0][1]["error"]


class TestAttachCLIManager:
    """Test cases for attach_cli_manager function."""

    def test_returns_none_for_none_input(self):
        """Test attach_cli_manager returns None for None input."""
        result = attach_cli_manager(None, None, None, None)
        assert result is None

    def test_returns_same_if_already_cli_enabled(self):
        """Test attach_cli_manager returns same manager if already CLI-enabled."""
        cli_manager = CLIEnabledSubAgentManager(
            agent_name="test",
            sub_agents={},
        )

        result = attach_cli_manager(cli_manager, Mock(), Mock(), Mock())
        assert result is cli_manager

    def test_wraps_regular_manager(self):
        """Test attach_cli_manager wraps regular SubAgentManager."""
        regular_manager = SubAgentManager(
            agent_name="test",
            sub_agents={},
        )

        progress_hook = Mock()
        tool_hook = Mock()
        event_callback = Mock()

        result = attach_cli_manager(
            regular_manager,
            progress_hook,
            tool_hook,
            event_callback,
        )

        assert isinstance(result, CLIEnabledSubAgentManager)
        assert result.cli_progress_hook == progress_hook
        assert result.cli_tool_hook == tool_hook
        assert result.event_callback == event_callback


class TestAttachCLIToAgent:
    """Test cases for attach_cli_to_agent function."""

    def test_skips_already_attached(self):
        """Test attach_cli_to_agent skips if already attached."""
        agent = Mock()
        agent._cli_hooks_attached = True

        # Track middleware calls
        middleware_manager = Mock()
        middleware_manager.middlewares = Mock()
        middleware_manager.middlewares.insert = Mock()
        agent.executor.middleware_manager = middleware_manager

        attach_cli_to_agent(agent, Mock(), Mock(), Mock())

        # Should not have called insert since already attached
        middleware_manager.middlewares.insert.assert_not_called()

    def test_attaches_progress_hook(self):
        """Test attach_cli_to_agent attaches progress hook middleware."""
        agent = Mock()
        agent._cli_hooks_attached = False
        middleware_manager = Mock()
        middleware_manager.middlewares = []
        agent.executor.middleware_manager = middleware_manager
        agent.executor.subagent_manager = None
        agent.executor.batch_processor = None

        progress_hook = Mock()

        attach_cli_to_agent(agent, progress_hook, None, None)

        assert len(middleware_manager.middlewares) == 1
        assert middleware_manager.middlewares[0].name == "cli_progress_hook"
        assert agent._cli_hooks_attached is True

    def test_attaches_tool_hook(self):
        """Test attach_cli_to_agent attaches tool hook middleware."""
        agent = Mock()
        agent._cli_hooks_attached = False
        middleware_manager = Mock()
        middleware_manager.middlewares = []
        agent.executor.middleware_manager = middleware_manager
        agent.executor.subagent_manager = None
        agent.executor.batch_processor = None

        tool_hook = Mock()

        attach_cli_to_agent(agent, None, tool_hook, None)

        assert len(middleware_manager.middlewares) == 1
        assert middleware_manager.middlewares[0].name == "cli_tool_hook"
        assert agent._cli_hooks_attached is True

    def test_handles_missing_middleware_manager(self):
        """Test attach_cli_to_agent handles missing middleware_manager."""
        agent = Mock()
        agent._cli_hooks_attached = False
        agent.executor.middleware_manager = None
        agent.executor.subagent_manager = None
        agent.executor.batch_processor = None

        # Should not raise
        attach_cli_to_agent(agent, Mock(), Mock(), Mock())
        assert agent._cli_hooks_attached is True

    def test_wraps_subagent_manager(self):
        """Test attach_cli_to_agent wraps subagent_manager."""
        agent = Mock()
        agent._cli_hooks_attached = False
        agent.executor.middleware_manager = None

        regular_manager = SubAgentManager(
            agent_name="test",
            sub_agents={},
        )
        agent.executor.subagent_manager = regular_manager
        agent.executor.batch_processor = None

        event_callback = Mock()

        attach_cli_to_agent(agent, None, None, event_callback)

        assert isinstance(agent.executor.subagent_manager, CLIEnabledSubAgentManager)

    def test_updates_batch_processor_subagent_manager(self):
        """Test attach_cli_to_agent updates batch_processor's subagent_manager."""
        agent = Mock()
        agent._cli_hooks_attached = False
        agent.executor.middleware_manager = None

        regular_manager = SubAgentManager(
            agent_name="test",
            sub_agents={},
        )
        agent.executor.subagent_manager = regular_manager

        batch_processor = Mock()
        batch_processor.subagent_manager = regular_manager
        agent.executor.batch_processor = batch_processor

        attach_cli_to_agent(agent, None, None, Mock())

        assert isinstance(batch_processor.subagent_manager, CLIEnabledSubAgentManager)
