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

"""Coverage improvement tests for nexau/archs/main_sub/execution/llm_caller.py.

Targets uncovered paths: _normalize_token_ids, _compact_scalar,
_raw_message_metadata, _ensure_tool_results, session_id injection,
call_llm_async, _call_once_sync.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import (
    LLMCaller,
    _compact_scalar,
    _ensure_tool_results,
    _normalize_token_ids,
    _raw_message_metadata,
)
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock


@pytest.fixture(autouse=True)
def mock_openai_module():
    """Mock the openai module to prevent any real API calls."""
    with patch("nexau.archs.main_sub.execution.llm_caller.openai") as mock_openai:
        mock_openai.OpenAI.side_effect = RuntimeError("Real OpenAI client cannot be instantiated in tests")
        yield mock_openai


# ---------------------------------------------------------------------------
# _normalize_token_ids
# ---------------------------------------------------------------------------


class TestNormalizeTokenIds:
    def test_valid_list(self):
        result = _normalize_token_ids([1, 2, 3], context="stop_token_ids")
        assert result == [1, 2, 3]

    def test_string_ints(self):
        result = _normalize_token_ids(["1", "2"], context="stop_token_ids")
        assert result == [1, 2]

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _normalize_token_ids("not a list", context="test")

    def test_non_integer_item_raises(self):
        with pytest.raises(ValueError, match="non-integer"):
            _normalize_token_ids([1, "abc"], context="test")


# ---------------------------------------------------------------------------
# _compact_scalar
# ---------------------------------------------------------------------------


class TestCompactScalar:
    def test_short_value(self):
        assert _compact_scalar("hello") == "hello"

    def test_long_value_truncated(self):
        text = "a" * 500
        result = _compact_scalar(text, max_chars=256)
        assert len(result) < 500
        assert "truncated" in result


# ---------------------------------------------------------------------------
# _raw_message_metadata
# ---------------------------------------------------------------------------


class TestRawMessageMetadata:
    def test_none_payload(self):
        result = _raw_message_metadata(None)
        assert result["raw_type"] == "NoneType"

    def test_dict_payload(self):
        payload = {
            "id": "abc",
            "model": "gpt-4",
            "finish_reason": "stop",
            "choices": [{"finish_reason": "stop"}],
        }
        result = _raw_message_metadata(payload)
        assert result["id"] == "abc"
        assert result["model"] == "gpt-4"
        assert result["choices_count"] == 1
        assert result["choice_finish_reason"] == "stop"

    def test_object_payload(self):
        payload = SimpleNamespace(
            id="x",
            model="m",
            object=None,
            role=None,
            finish_reason=None,
            stop_reason=None,
            status=None,
            choices=[SimpleNamespace(finish_reason="length")],
        )
        result = _raw_message_metadata(payload)
        assert result["id"] == "x"
        assert result["choices_count"] == 1
        assert result["choice_finish_reason"] == "length"

    def test_empty_choices_in_dict(self):
        result = _raw_message_metadata({"choices": []})
        assert result["choices_count"] == 0


# ---------------------------------------------------------------------------
# _ensure_tool_results
# ---------------------------------------------------------------------------


class TestEnsureToolResults:
    def test_no_tool_use_returns_unchanged(self):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="Hello")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="Hi")]),
        ]
        result = _ensure_tool_results(msgs)
        assert result is msgs

    def test_all_results_present_returns_unchanged(self):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="Hello")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="t1", name="foo", input={})],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="t1", content="result")],
            ),
        ]
        result = _ensure_tool_results(msgs)
        assert result is msgs

    def test_missing_result_injects_synthetic(self):
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="Hello")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="t1", name="foo", input={})],
            ),
            # Missing tool result for t1
        ]
        result = _ensure_tool_results(msgs)
        assert len(result) == 3
        assert result[2].role == Role.TOOL
        tool_result_block = result[2].content[0]
        assert isinstance(tool_result_block, ToolResultBlock)
        assert tool_result_block.is_error is True

    def test_prefix_match_tool_result(self):
        """Tool result ID starts with the tool use ID (prefix match)."""
        msgs = [
            Message(role=Role.USER, content=[TextBlock(text="Hello")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="call", name="foo", input={})],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="call_abc123", content="result")],
            ),
        ]
        result = _ensure_tool_results(msgs)
        # Should match via prefix, no synthetic injection needed
        assert result is msgs


# ---------------------------------------------------------------------------
# LLMCaller._get_tracer
# ---------------------------------------------------------------------------


class TestLLMCallerGetTracer:
    def test_returns_tracer_from_global_storage(self, mock_llm_config):
        mock_tracer = Mock()
        mock_storage = Mock()
        mock_storage.get.return_value = mock_tracer
        caller = LLMCaller(Mock(), mock_llm_config, global_storage=mock_storage)
        assert caller._get_tracer() is mock_tracer

    def test_returns_none_without_storage(self, mock_llm_config):
        caller = LLMCaller(Mock(), mock_llm_config)
        assert caller._get_tracer() is None


# ---------------------------------------------------------------------------
# LLMCaller.call_llm — session_id injection
# ---------------------------------------------------------------------------


class TestLLMCallerSessionId:
    def test_session_id_openai_chat(self, mock_llm_config, mock_openai_client):
        mock_llm_config.api_type = "openai_chat_completion"
        caller = LLMCaller(mock_openai_client, mock_llm_config, session_id="sess-123")

        # We can't fully test call_llm without heavy mocking,
        # but we test that session_id is stored correctly
        assert caller.session_id == "sess-123"

    def test_session_id_anthropic(self, mock_openai_client):
        config = LLMConfig(
            model="claude-3",
            base_url="https://api.anthropic.com",
            api_key="test-key",
            api_type="anthropic_chat_completion",
        )
        caller = LLMCaller(mock_openai_client, config, session_id="sess-456")
        assert caller.session_id == "sess-456"


# ---------------------------------------------------------------------------
# LLMCaller.call_llm — RuntimeError without client
# ---------------------------------------------------------------------------


class TestLLMCallerCallLlm:
    def test_raises_without_client(self, mock_llm_config):
        caller = LLMCaller(None, mock_llm_config)
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hello")])]
        with pytest.raises(RuntimeError, match="OpenAI client is not available"):
            caller.call_llm(msgs)


# ---------------------------------------------------------------------------
# LLMCaller.call_llm_async
# ---------------------------------------------------------------------------


class TestLLMCallerCallLlmAsync:
    @pytest.mark.anyio
    async def test_call_llm_async_returns_none_on_force_stop(self, mock_llm_config, mock_openai_client):
        from nexau.archs.main_sub.execution.stop_reason import AgentStopReason

        caller = LLMCaller(mock_openai_client, mock_llm_config)
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hello")])]
        result = await caller.call_llm_async(
            msgs,
            force_stop_reason=AgentStopReason.CONTEXT_TOKEN_LIMIT,
        )
        assert result is None
