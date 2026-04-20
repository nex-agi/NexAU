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

"""Coverage improvement tests for SubAgentManager async path.

Targets uncovered paths in:
- nexau/archs/main_sub/execution/subagent_manager.py (call_sub_agent_async, add_sub_agent)
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager


class TestSubAgentManagerAsync:
    @pytest.fixture
    def sub_agent_config(self, agent_config):
        return agent_config.model_copy(update={"name": "test_sub_agent"})

    @pytest.fixture
    def sub_agents(self, sub_agent_config):
        return {"test_sub_agent": sub_agent_config}

    @pytest.fixture
    def subagent_manager(self, sub_agents):
        return SubAgentManager(agent_name="parent_agent", sub_agents=sub_agents)

    @pytest.mark.anyio
    async def test_call_sub_agent_async_not_found(self, subagent_manager):
        with pytest.raises(ValueError, match="Sub-agent 'nonexistent' not found"):
            await subagent_manager.call_sub_agent_async("nonexistent", "test message")

    @pytest.mark.anyio
    async def test_call_sub_agent_async_during_shutdown(self, subagent_manager):
        subagent_manager.shutdown()
        with pytest.raises(RuntimeError, match="shutting down"):
            await subagent_manager.call_sub_agent_async("test_sub_agent", "test message")

    @pytest.mark.anyio
    @patch("nexau.archs.main_sub.agent_context.get_context")
    async def test_call_sub_agent_async_success(self, mock_get_context, subagent_manager, sub_agent_config):
        mock_get_context.return_value = None
        mock_sub_agent = Mock()
        mock_sub_agent.agent_name = "test_sub_agent"
        mock_sub_agent.agent_id = "sub-async-id"
        mock_sub_agent.run_async = AsyncMock(return_value="async result")

        with patch("nexau.archs.main_sub.agent.Agent.create", new_callable=AsyncMock, return_value=mock_sub_agent):
            result = await subagent_manager.call_sub_agent_async("test_sub_agent", "test message")

        assert "[sub_agent_id: sub-async-id]" in result
        assert "async result" in result
        assert "Sub-agent finished" in result
        assert subagent_manager.running_sub_agents == {}

    @pytest.mark.anyio
    @patch("nexau.archs.main_sub.agent_context.get_context")
    async def test_call_sub_agent_async_with_context(self, mock_get_context, subagent_manager):
        mock_context = Mock()
        mock_context.context = {"key": "value"}
        mock_get_context.return_value = mock_context

        mock_sub_agent = Mock()
        mock_sub_agent.agent_name = "test_sub_agent"
        mock_sub_agent.agent_id = "sub-async-id-2"
        mock_sub_agent.run_async = AsyncMock(return_value="result with context")

        with patch("nexau.archs.main_sub.agent.Agent.create", new_callable=AsyncMock, return_value=mock_sub_agent):
            result = await subagent_manager.call_sub_agent_async("test_sub_agent", "msg")

        assert "result with context" in result
        call_args = mock_sub_agent.run_async.call_args
        assert call_args[1]["context"] == {"key": "value"}

    @pytest.mark.anyio
    @patch("nexau.archs.main_sub.agent_context.get_context")
    async def test_call_sub_agent_async_error_cleanup(self, mock_get_context, subagent_manager):
        mock_get_context.return_value = None
        mock_sub_agent = Mock()
        mock_sub_agent.agent_name = "test_sub_agent"
        mock_sub_agent.agent_id = "sub-fail"
        mock_sub_agent.run_async = AsyncMock(side_effect=RuntimeError("async fail"))

        with patch("nexau.archs.main_sub.agent.Agent.create", new_callable=AsyncMock, return_value=mock_sub_agent):
            with pytest.raises(RuntimeError, match="async fail"):
                await subagent_manager.call_sub_agent_async("test_sub_agent", "msg")

        assert subagent_manager.running_sub_agents == {}

    @pytest.mark.anyio
    @patch("nexau.archs.main_sub.agent_context.get_context")
    async def test_call_sub_agent_async_with_parallel_execution_id(self, mock_get_context, subagent_manager, agent_state):
        mock_get_context.return_value = None
        mock_sub_agent = Mock()
        mock_sub_agent.agent_name = "test_sub_agent"
        mock_sub_agent.agent_id = "sub-parallel"
        mock_sub_agent.run_async = AsyncMock(return_value="parallel result")

        with patch("nexau.archs.main_sub.agent.Agent.create", new_callable=AsyncMock, return_value=mock_sub_agent):
            result = await subagent_manager.call_sub_agent_async(
                "test_sub_agent",
                "msg",
                parent_agent_state=agent_state,
                parallel_execution_id="exec-123",
            )

        assert "parallel result" in result
        stored_id = agent_state.global_storage.get("parallel_execution_id")
        assert stored_id == "exec-123"
