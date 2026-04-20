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

"""Unit tests for Agent builtin tool.

RFC-0015: 合并 Sub-agent 工具为统一 Agent 工具

Tests cover:
- Create mode (sub_agent_id=None): delegates to SubAgentManager.call_sub_agent()
- Resume mode (sub_agent_id provided): delegates with existing sub_agent_id
- _extract_sub_agent_id() helper function
- Error cases (no agent_state, no subagent_manager, exception from manager)
"""

from unittest.mock import MagicMock

import pytest

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.tool.builtin.agent_tool import _extract_sub_agent_id, call_sub_agent


class TestExtractSubAgentId:
    """Test _extract_sub_agent_id helper function."""

    def test_extract_id_from_prefix(self):
        """Should extract sub_agent_id from [sub_agent_id: ...] prefix."""
        text = "[sub_agent_id: abc-123] Some result text"
        assert _extract_sub_agent_id(text) == "abc-123"

    def test_extract_id_with_whitespace(self):
        """Should capture sub_agent_id including trailing whitespace inside brackets."""
        text = "[sub_agent_id:   xyz-456  ] Result"
        # The regex captures [^\]]+ which includes trailing whitespace
        assert _extract_sub_agent_id(text) == "xyz-456  "

    def test_extract_id_no_prefix(self):
        """Should return None when no [sub_agent_id: ...] prefix."""
        text = "No prefix here"
        assert _extract_sub_agent_id(text) is None

    def test_extract_id_empty_string(self):
        """Should return None for empty string."""
        assert _extract_sub_agent_id("") is None

    def test_extract_id_prefix_in_middle(self):
        """Should return None when prefix is not at the start."""
        text = "Some text [sub_agent_id: abc-123] not at start"
        assert _extract_sub_agent_id(text) is None

    def test_extract_id_uuid_format(self):
        """Should extract UUID-format sub_agent_id."""
        text = "[sub_agent_id: 550e8400-e29b-41d4-a716-446655440000] Result"
        assert _extract_sub_agent_id(text) == "550e8400-e29b-41d4-a716-446655440000"


class TestCallSubAgentCreateMode:
    """RFC-0015: Agent tool create mode (sub_agent_id=None)."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock AgentState with a SubAgentManager."""
        agent_state = MagicMock(spec=AgentState)
        mock_manager = MagicMock()
        # Simulate SubAgentManager.call_sub_agent() return format
        mock_manager.call_sub_agent.return_value = (
            "[sub_agent_id: abc-123] Research complete. "
            "Sub-agent finished (sub_agent_name: researcher, "
            "sub_agent_id: abc-123. Use the Agent tool with this sub_agent_id to resume if needed)."
        )
        agent_state.subagent_manager = mock_manager
        return agent_state

    def test_create_mode_success(self, mock_agent_state):
        """Should call manager without sub_agent_id for create mode."""
        result = call_sub_agent(
            sub_agent_name="researcher",
            message="Analyze data",
            agent_state=mock_agent_state,
        )

        assert result["status"] == "success"
        assert result["sub_agent_name"] == "researcher"
        assert result["sub_agent_id"] == "abc-123"
        assert result["message"] == "Analyze data"
        assert "Research complete" in result["result"]

        # 1. 验证 SubAgentManager.call_sub_agent 被调用时 sub_agent_id=None
        mock_agent_state.subagent_manager.call_sub_agent.assert_called_once_with(
            "researcher",
            "Analyze data",
            None,
            parent_agent_state=mock_agent_state,
        )

    def test_create_mode_passes_none_sub_agent_id(self, mock_agent_state):
        """Should pass sub_agent_id=None when not provided."""
        result = call_sub_agent(
            sub_agent_name="worker",
            message="Do work",
            agent_state=mock_agent_state,
        )

        assert result["status"] == "success"
        call_args = mock_agent_state.subagent_manager.call_sub_agent.call_args
        # 2. 确认 sub_agent_id 参数为 None（创建新子代理）
        assert call_args[0][2] is None

    def test_empty_string_sub_agent_id_normalized_to_none(self, mock_agent_state):
        """Empty string sub_agent_id should be normalized to None (create mode).

        LLM 常将可选字符串参数发送为空字符串 ""，需统一为 None 以避免
        duplicate primary key 错误（agent_id='' 与已有记录冲突）。
        """
        # 3. LLM 发送 sub_agent_id="" 的场景
        result = call_sub_agent(
            sub_agent_name="researcher",
            message="Explore the repo",
            sub_agent_id="",
            agent_state=mock_agent_state,
        )

        assert result["status"] == "success"
        call_args = mock_agent_state.subagent_manager.call_sub_agent.call_args
        # 确认空字符串被规范化为 None（走创建新子代理路径）
        assert call_args[0][2] is None


class TestCallSubAgentResumeMode:
    """RFC-0015: Agent tool resume mode (sub_agent_id provided)."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock AgentState with a SubAgentManager."""
        agent_state = MagicMock(spec=AgentState)
        mock_manager = MagicMock()
        mock_manager.call_sub_agent.return_value = (
            "[sub_agent_id: existing-456] Continued work. "
            "Sub-agent finished (sub_agent_name: worker, "
            "sub_agent_id: existing-456. Use the Agent tool with this sub_agent_id to resume if needed)."
        )
        agent_state.subagent_manager = mock_manager
        return agent_state

    def test_resume_mode_success(self, mock_agent_state):
        """Should call manager with sub_agent_id for resume mode."""
        result = call_sub_agent(
            sub_agent_name="worker",
            message="Continue the task",
            sub_agent_id="existing-456",
            agent_state=mock_agent_state,
        )

        assert result["status"] == "success"
        assert result["sub_agent_name"] == "worker"
        assert result["sub_agent_id"] == "existing-456"
        assert result["message"] == "Continue the task"

        # 3. 验证 SubAgentManager.call_sub_agent 被调用时 sub_agent_id 为传入值
        mock_agent_state.subagent_manager.call_sub_agent.assert_called_once_with(
            "worker",
            "Continue the task",
            "existing-456",
            parent_agent_state=mock_agent_state,
        )

    def test_resume_mode_preserves_sub_agent_id(self, mock_agent_state):
        """Should pass the provided sub_agent_id to the manager."""
        call_sub_agent(
            sub_agent_name="researcher",
            message="Follow up",
            sub_agent_id="uuid-resume-id",
            agent_state=mock_agent_state,
        )

        call_args = mock_agent_state.subagent_manager.call_sub_agent.call_args
        assert call_args[0][2] == "uuid-resume-id"


class TestCallSubAgentErrorCases:
    """RFC-0015: Agent tool error handling."""

    def test_no_agent_state(self):
        """Should return error when agent_state is None."""
        # 4. 验证 agent_state 为 None 时的错误返回
        result = call_sub_agent(
            sub_agent_name="researcher",
            message="Analyze data",
            agent_state=None,
        )

        assert result["status"] == "error"
        assert "Agent state not available" in result["error"]

    def test_no_subagent_manager(self):
        """Should return error when subagent_manager is None."""
        agent_state = MagicMock(spec=AgentState)
        agent_state.subagent_manager = None

        # 5. 验证 subagent_manager 为 None 时的错误返回
        result = call_sub_agent(
            sub_agent_name="researcher",
            message="Analyze data",
            agent_state=agent_state,
        )

        assert result["status"] == "error"
        assert "Sub-agent manager not available" in result["error"]

    def test_manager_raises_exception(self):
        """Should return error dict when manager raises an exception."""
        agent_state = MagicMock(spec=AgentState)
        mock_manager = MagicMock()
        mock_manager.call_sub_agent.side_effect = RuntimeError(
            "[sub_agent_id: failed-789] Sub-agent 'researcher' (id: failed-789) failed: timeout"
        )
        agent_state.subagent_manager = mock_manager

        # 6. 验证异常时返回 error 格式（含 sub_agent_id）
        result = call_sub_agent(
            sub_agent_name="researcher",
            message="Analyze data",
            agent_state=agent_state,
        )

        assert result["status"] == "error"
        assert result["sub_agent_name"] == "researcher"
        assert result["sub_agent_id"] == "failed-789"
        assert "timeout" in result["error"]
        assert result["error_type"] == "RuntimeError"

    def test_manager_raises_value_error(self):
        """Should return error dict when sub-agent name not found."""
        agent_state = MagicMock(spec=AgentState)
        mock_manager = MagicMock()
        mock_manager.call_sub_agent.side_effect = ValueError("Sub-agent 'unknown' not found")
        agent_state.subagent_manager = mock_manager

        # 7. 验证子代理不存在时的错误返回
        result = call_sub_agent(
            sub_agent_name="unknown",
            message="Do something",
            agent_state=agent_state,
        )

        assert result["status"] == "error"
        assert result["sub_agent_name"] == "unknown"
        assert result["sub_agent_id"] is None
        assert "not found" in result["error"]
        assert result["error_type"] == "ValueError"

    def test_exception_without_sub_agent_id_prefix(self):
        """Should handle exceptions without [sub_agent_id: ...] prefix."""
        agent_state = MagicMock(spec=AgentState)
        mock_manager = MagicMock()
        mock_manager.call_sub_agent.side_effect = ConnectionError("Network failure")
        agent_state.subagent_manager = mock_manager

        # 8. 验证异常消息不含 [sub_agent_id:] 前缀时的处理
        result = call_sub_agent(
            sub_agent_name="worker",
            message="Do work",
            agent_state=agent_state,
        )

        assert result["status"] == "error"
        assert result["sub_agent_id"] is None
        assert "Network failure" in result["error"]
        assert result["error_type"] == "ConnectionError"
