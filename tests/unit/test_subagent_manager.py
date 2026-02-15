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
from unittest.mock import Mock, patch

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
        sub_agent.run = Mock(return_value="sub agent result")
        sub_agent.stop = Mock()
        sub_agent.agent_id = "mock_sub_agent_id"
        return sub_agent

    @pytest.fixture
    def sub_agent_config(self, agent_config):
        """AgentConfig for a sub-agent."""
        return agent_config.model_copy(update={"name": "test_sub_agent"})

    @pytest.fixture
    def sub_agents(self, sub_agent_config):
        """Create a dictionary of sub-agent configs."""
        return {"test_sub_agent": sub_agent_config}

    @pytest.fixture
    def subagent_manager(self, sub_agents):
        """Create a SubAgentManager instance."""
        return SubAgentManager(agent_name="parent_agent", sub_agents=sub_agents)

    def test_initialization(self, sub_agents):
        """Test SubAgentManager initialization."""
        manager = SubAgentManager(agent_name="test_agent", sub_agents=sub_agents)

        assert manager.agent_name == "test_agent"
        assert manager.sub_agents == sub_agents
        assert manager.global_storage is None
        assert manager.xml_parser is not None
        assert isinstance(manager._shutdown_event, threading.Event)
        assert manager.running_sub_agents == {}
        assert manager.session_manager is None
        assert manager.user_id is None
        assert manager.session_id is None

    def test_initialization_with_optional_params(self, sub_agents):
        """Test SubAgentManager initialization with optional parameters."""
        mock_storage = GlobalStorage()

        manager = SubAgentManager(
            agent_name="test_agent",
            sub_agents=sub_agents,
            global_storage=mock_storage,
        )

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
    def test_call_sub_agent_success_no_context(
        self,
        mock_get_context,
        subagent_manager,
        sub_agent_config,
        mock_sub_agent,
    ):
        """Test successful sub-agent call without context."""
        mock_get_context.return_value = None

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        assert result.startswith("sub agent result")
        assert "Sub-agent finished" in result
        mock_agent_cls.assert_called_once_with(
            config=sub_agent_config,
            global_storage=subagent_manager.global_storage,
            session_manager=subagent_manager.session_manager,
            user_id=subagent_manager.user_id,
            session_id=subagent_manager.session_id,
            is_root=False,
        )
        mock_sub_agent.run.assert_called_once()
        call_args = mock_sub_agent.run.call_args
        assert call_args[1]["message"] == "test message"
        assert call_args[1]["context"] is None
        assert call_args[1]["parent_agent_state"] is None

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_success_with_context(
        self,
        mock_get_context,
        subagent_manager,
        sub_agent_config,
        mock_sub_agent,
    ):
        """Test successful sub-agent call with context."""
        mock_context = Mock()
        mock_context.context = {"key": "value"}
        mock_get_context.return_value = mock_context

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        assert result.startswith("sub agent result")
        assert "Sub-agent finished" in result
        mock_agent_cls.assert_called_once_with(
            config=sub_agent_config,
            global_storage=subagent_manager.global_storage,
            session_manager=subagent_manager.session_manager,
            user_id=subagent_manager.user_id,
            session_id=subagent_manager.session_id,
            is_root=False,
        )
        mock_sub_agent.run.assert_called_once()
        call_args = mock_sub_agent.run.call_args
        print(call_args)
        assert call_args[1]["message"] == "test message"
        assert call_args[1]["context"] == {"key": "value"}
        assert call_args[1]["context"] is not mock_context.context

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_explicit_context(
        self,
        mock_get_context,
        subagent_manager,
        mock_sub_agent,
    ):
        """Test sub-agent call with explicitly provided context."""
        mock_context = Mock()
        mock_context.context = {"default": "context"}
        mock_get_context.return_value = mock_context
        explicit_context = {"explicit": "context"}

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent(
                "test_sub_agent",
                "test message",
                context=explicit_context,
            )

        assert result.startswith("sub agent result")
        assert "Sub-agent finished" in result
        call_args = mock_sub_agent.run.call_args
        assert call_args[1]["context"] == explicit_context

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_parent_state(
        self,
        mock_get_context,
        subagent_manager,
        mock_sub_agent,
        agent_state,
    ):
        """Test sub-agent call with parent agent state."""
        mock_get_context.return_value = None

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent(
                "test_sub_agent",
                "test message",
                parent_agent_state=agent_state,
            )

        assert result.startswith("sub agent result")
        call_args = mock_sub_agent.run.call_args
        assert call_args[1]["parent_agent_state"] == agent_state

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_with_global_storage(
        self,
        mock_get_context,
        sub_agents,
        sub_agent_config,
        mock_sub_agent,
    ):
        """Test sub-agent creation with global storage."""
        mock_storage = GlobalStorage()
        manager = SubAgentManager(agent_name="parent", sub_agents=sub_agents, global_storage=mock_storage)

        mock_get_context.return_value = None
        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = manager.call_sub_agent("test_sub_agent", "message")

        assert result.startswith("sub agent result")
        mock_agent_cls.assert_called_once_with(
            config=sub_agent_config,
            global_storage=mock_storage,
            session_manager=manager.session_manager,
            user_id=manager.user_id,
            session_id=manager.session_id,
            is_root=False,
        )

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_execution_error(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test sub-agent call when execution fails."""
        mock_get_context.return_value = None
        mock_sub_agent.run.side_effect = Exception("Execution error")

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            with pytest.raises(Exception, match="Execution error"):
                subagent_manager.call_sub_agent("test_sub_agent", "test message")
        assert subagent_manager.running_sub_agents == {}

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_missing_agent_id(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test sub-agent call when agent_id is missing."""
        mock_get_context.return_value = None
        mock_sub_agent.agent_id = None

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        assert result.startswith("sub agent result")
        assert subagent_manager.running_sub_agents == {}

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_running_agents_tracking(self, mock_get_context, subagent_manager, mock_sub_agent):
        """Test that running sub-agents are tracked and cleaned up."""
        mock_get_context.return_value = None
        mock_sub_agent.agent_id = "sub_agent_123"
        mock_sub_agent.run.side_effect = lambda *args, **kwargs: (
            "sub agent result"
            if subagent_manager.running_sub_agents.get("sub_agent_123") is mock_sub_agent
            else pytest.fail("running_sub_agents missing while sub-agent is executing")
        )

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            assert len(subagent_manager.running_sub_agents) == 0
            result = subagent_manager.call_sub_agent("test_sub_agent", "test message")

        assert len(subagent_manager.running_sub_agents) == 0
        assert result.startswith("sub agent result")

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_recall_by_agent_id(
        self,
        mock_get_context,
        subagent_manager,
        sub_agent_config,
        mock_sub_agent,
    ):
        """Test sub-agent recall using agent_id creates agent with that ID."""
        mock_get_context.return_value = None
        mock_sub_agent.run.return_value = "recalled result"
        mock_sub_agent.agent_id = "recall-agent-1"

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent(
                "test_sub_agent",
                "follow-up message",
                sub_agent_id="recall-agent-1",
            )
            # Agent is created with the specified agent_id
            mock_agent_cls.assert_called_once_with(
                agent_id="recall-agent-1",
                config=sub_agent_config,
                global_storage=subagent_manager.global_storage,
                session_manager=subagent_manager.session_manager,
                user_id=subagent_manager.user_id,
                session_id=subagent_manager.session_id,
                is_root=False,
            )

        assert result.startswith("recalled result")
        assert "recall-agent-1" in result

    def test_shutdown(self, subagent_manager, mock_sub_agent):
        """Test shutdown method."""
        subagent_manager.running_sub_agents["sub_123"] = mock_sub_agent

        subagent_manager.shutdown()

        assert subagent_manager._shutdown_event.is_set()
        mock_sub_agent.sync_cleanup.assert_called_once()

    def test_shutdown_with_error(self, subagent_manager):
        """Test shutdown when sub-agent stop raises error."""
        mock_sub_agent = Mock()
        mock_sub_agent.sync_cleanup.side_effect = Exception("Stop error")
        subagent_manager.running_sub_agents["sub_123"] = mock_sub_agent

        subagent_manager.shutdown()

        assert subagent_manager._shutdown_event.is_set()
        mock_sub_agent.sync_cleanup.assert_called_once()

    def test_shutdown_no_running_agents(self, subagent_manager):
        """Test shutdown when no agents are running."""
        subagent_manager.shutdown()

        assert subagent_manager._shutdown_event.is_set()

    def test_add_sub_agent(self, subagent_manager, sub_agent_config):
        """Test adding a sub-agent config."""
        new_config = sub_agent_config.model_copy(update={"name": "new_agent", "agent_id": "new_agent_123"})

        subagent_manager.add_sub_agent("new_agent", new_config)

        assert "new_agent" in subagent_manager.sub_agents
        assert subagent_manager.sub_agents["new_agent"] == new_config

    def test_add_sub_agent_overwrite(self, subagent_manager, sub_agent_config):
        """Test overwriting an existing sub-agent config."""
        new_config = sub_agent_config.model_copy(update={"name": "override_agent", "agent_id": "override_agent_123"})

        subagent_manager.add_sub_agent("test_sub_agent", new_config)

        assert subagent_manager.sub_agents["test_sub_agent"] == new_config

    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_passes_custom_llm_client_provider(
        self,
        mock_get_context,
        subagent_manager,
        mock_sub_agent,
    ):
        """Custom LLM provider should be passed through to sub-agent run."""
        mock_get_context.return_value = None
        custom_provider = Mock(name="custom_provider")

        with patch("nexau.archs.main_sub.agent.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = mock_sub_agent
            result = subagent_manager.call_sub_agent(
                "test_sub_agent",
                "message",
                custom_llm_client_provider=custom_provider,
            )

        assert result.startswith("sub agent result")
        assert "Sub-agent finished" in result
        call_args = mock_sub_agent.run.call_args
        assert call_args[1]["custom_llm_client_provider"] is custom_provider


class TestSubAgentManagerParallelExecutionId:
    """Test parallel_execution_id storage and handling in SubAgentManager."""

    @pytest.fixture
    def mock_sub_agent(self):
        """Create a mock sub-agent."""
        sub_agent = Mock()
        sub_agent.config = Mock()
        sub_agent.run = Mock(return_value="sub agent result")
        sub_agent.stop = Mock()
        sub_agent.agent_id = "mock_sub_agent_id"
        return sub_agent

    @pytest.fixture
    def sub_agent_config(self, agent_config):
        """AgentConfig for a sub-agent."""
        return agent_config.model_copy(update={"name": "test_sub_agent"})

    @pytest.fixture
    def sub_agents(self, sub_agent_config):
        """Create a dictionary of sub-agent configs."""
        return {"test_sub_agent": sub_agent_config}

    @pytest.fixture
    def subagent_manager(self, sub_agents):
        """Create a SubAgentManager instance."""
        return SubAgentManager(agent_name="parent_agent", sub_agents=sub_agents)

    @patch("nexau.archs.main_sub.agent.Agent")
    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_stores_parallel_execution_id_in_parent_state(
        self,
        mock_get_context,
        mock_agent_cls,
        subagent_manager,
        mock_sub_agent,
        agent_state,
    ):
        """Test that parallel_execution_id is stored in parent_agent_state.global_storage."""
        mock_get_context.return_value = None
        mock_agent_cls.return_value = mock_sub_agent

        test_exec_id = "test-parallel-exec-id-555"

        # Call sub_agent with parallel_execution_id
        subagent_manager.call_sub_agent(
            sub_agent_name="test_sub_agent",
            message="test message",
            parent_agent_state=agent_state,
            parallel_execution_id=test_exec_id,
        )

        # Verify parallel_execution_id was stored in global_storage
        stored_id = agent_state.global_storage.get("parallel_execution_id")
        assert stored_id == test_exec_id

    @patch("nexau.archs.main_sub.agent.Agent")
    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_without_parallel_execution_id_no_storage(
        self,
        mock_get_context,
        mock_agent_cls,
        subagent_manager,
        mock_sub_agent,
        agent_state,
    ):
        """Test that calling without parallel_execution_id doesn't store it in global_storage."""
        mock_get_context.return_value = None
        mock_agent_cls.return_value = mock_sub_agent

        # Call sub_agent without parallel_execution_id
        subagent_manager.call_sub_agent(
            sub_agent_name="test_sub_agent",
            message="test message",
            parent_agent_state=agent_state,
        )

        # Verify parallel_execution_id was not stored
        stored_id = agent_state.global_storage.get("parallel_execution_id")
        assert stored_id is None

    @patch("nexau.archs.main_sub.agent.Agent")
    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_without_parent_state_no_error(
        self,
        mock_get_context,
        mock_agent_cls,
        subagent_manager,
        mock_sub_agent,
    ):
        """Test that parallel_execution_id parameter works without parent_agent_state."""
        mock_get_context.return_value = None
        mock_agent_cls.return_value = mock_sub_agent

        # Call sub_agent with parallel_execution_id but without parent_agent_state
        # Should not raise error
        result = subagent_manager.call_sub_agent(
            sub_agent_name="test_sub_agent",
            message="test message",
            parallel_execution_id="test-exec-id-666",
        )

        # Should complete successfully
        assert "sub agent result" in result

    @patch("nexau.archs.main_sub.agent.Agent")
    @patch("nexau.archs.main_sub.agent_context.get_context")
    def test_call_sub_agent_parallel_execution_id_overwrites_previous(
        self,
        mock_get_context,
        mock_agent_cls,
        subagent_manager,
        mock_sub_agent,
        agent_state,
    ):
        """Test that parallel_execution_id can be overwritten with new value."""
        mock_get_context.return_value = None
        mock_agent_cls.return_value = mock_sub_agent

        # First call with exec_id_1
        subagent_manager.call_sub_agent(
            sub_agent_name="test_sub_agent",
            message="message 1",
            parent_agent_state=agent_state,
            parallel_execution_id="exec-id-1",
        )

        stored_id = agent_state.global_storage.get("parallel_execution_id")
        assert stored_id == "exec-id-1"

        # Second call with exec_id_2 should overwrite
        subagent_manager.call_sub_agent(
            sub_agent_name="test_sub_agent",
            message="message 2",
            parent_agent_state=agent_state,
            parallel_execution_id="exec-id-2",
        )

        stored_id = agent_state.global_storage.get("parallel_execution_id")
        assert stored_id == "exec-id-2"
