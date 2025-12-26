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

"""Unit tests for the recall_sub_agent tool."""

from unittest.mock import Mock

from nexau.archs.tool.builtin.recall_sub_agent_tool import recall_sub_agent


def test_recall_sub_agent_requires_agent_state():
    """Calling without agent_state should return an error payload."""
    result = recall_sub_agent("sub_agent", "message", "agent-id", agent_state=None)

    assert result["status"] == "error"
    assert "Agent state not available" in result["error"]


def test_recall_sub_agent_requires_executor(agent_state):
    """AgentState must expose an executor reference."""
    agent_state._executor = None

    result = recall_sub_agent("sub_agent", "message", "agent-id", agent_state=agent_state)

    assert result["status"] == "error"
    assert "Executor not available" in result["error"]


def test_recall_sub_agent_requires_subagent_manager(agent_state):
    """Executor must expose a subagent_manager."""
    agent_state._executor = Mock(subagent_manager=None)

    result = recall_sub_agent("sub_agent", "message", "agent-id", agent_state=agent_state)

    assert result["status"] == "error"
    assert "Sub-agent manager not available" in result["error"]


def test_recall_sub_agent_calls_subagent_manager(agent_state):
    """Successful recall delegates to SubAgentManager.call_sub_agent."""
    subagent_manager = Mock()
    subagent_manager.call_sub_agent.return_value = "result text"
    executor = Mock(subagent_manager=subagent_manager)
    agent_state._executor = executor

    result = recall_sub_agent("sub_agent", "message", "agent-id", agent_state=agent_state)

    assert result["status"] == "success"
    assert result["result"] == "result text"
    subagent_manager.call_sub_agent.assert_called_once_with(
        "sub_agent",
        "message",
        "agent-id",
        parent_agent_state=agent_state,
    )


def test_recall_sub_agent_handles_subagent_exception(agent_state):
    """Errors from SubAgentManager are captured in the payload."""
    subagent_manager = Mock()
    subagent_manager.call_sub_agent.side_effect = ValueError("bad sub-agent id")
    executor = Mock(subagent_manager=subagent_manager)
    agent_state._executor = executor

    result = recall_sub_agent("sub_agent", "message", "agent-id", agent_state=agent_state)

    assert result["status"] == "error"
    assert result["error"] == "bad sub-agent id"
    assert result["error_type"] == "ValueError"
