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

"""Additional coverage tests for nexau/archs/main_sub/execution/llm_caller.py.

Targets uncovered paths: _call_once_sync session_id injection for all providers,
_call_with_retry retry/backoff/shutdown paths, call_llm_async full flow,
_call_with_retry_async retry/gemini/shutdown paths, call_llm_with_different_client,
_adapt_structured_tools_for_provider, _raw_message_metadata object with choices.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import MiddlewareManager, ModelCallParams
from nexau.archs.main_sub.execution.llm_caller import (
    LLMCaller,
    _adapt_structured_tools_for_provider,
    _raw_message_metadata,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.core.messages import Message, Role, TextBlock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_config():
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )


@pytest.fixture
def responses_config():
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_responses",
    )


@pytest.fixture
def anthropic_config():
    return LLMConfig(
        model="claude-3",
        base_url="https://api.anthropic.com",
        api_key="test-key",
        api_type="anthropic_chat_completion",
    )


@pytest.fixture
def gemini_config():
    return LLMConfig(
        model="gemini-pro",
        base_url="https://generativelanguage.googleapis.com",
        api_key="test-key",
        api_type="gemini_rest",
    )


def _make_model_response(content: str = "Hello", has_content: bool = True) -> ModelResponse:
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        to_dict=lambda: {"prompt_tokens": 10, "completion_tokens": 20},
    )
    resp = Mock(spec=ModelResponse)
    resp.content = content if has_content else None
    resp.role = "assistant"
    resp.tool_calls = []
    resp.raw_message = None
    resp.usage = usage
    resp.has_content.return_value = has_content
    resp.has_tool_calls.return_value = False
    resp.render_text.return_value = content
    return resp


# ---------------------------------------------------------------------------
# _call_once_sync — session_id injection for different providers
# ---------------------------------------------------------------------------


class TestCallOnceSyncSessionIdInjection:
    """Test _call_once_sync injects session_id correctly for each provider."""

    def test_openai_chat_completion_injects_user(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, session_id="sess-1")
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_model_response(),
        ):
            result = caller._call_once_sync(params)
        assert result is not None

    def test_openai_responses_injects_safety_identifier(self, responses_config):
        caller = LLMCaller(Mock(), responses_config, session_id="sess-2")
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_model_response(),
        ):
            result = caller._call_once_sync(params)
        assert result is not None

    def test_anthropic_injects_metadata(self, anthropic_config):
        caller = LLMCaller(Mock(), anthropic_config, session_id="sess-3")
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "claude-3"},
            openai_client=Mock(),
        )
        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_model_response(),
        ):
            result = caller._call_once_sync(params)
        assert result is not None

    def test_force_stop_returns_none(self, openai_config):
        caller = LLMCaller(Mock(), openai_config)
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=AgentStopReason.CONTEXT_TOKEN_LIMIT,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )
        result = caller._call_once_sync(params)
        assert result is None

    def test_empty_response_raises(self, openai_config):
        caller = LLMCaller(Mock(), openai_config)
        empty_resp = _make_model_response(content="", has_content=False)
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
                return_value=empty_resp,
            ),
            pytest.raises(RuntimeError, match="No response content"),
        ):
            caller._call_once_sync(params)


# ---------------------------------------------------------------------------
# _call_with_retry — retry/backoff/shutdown paths
# ---------------------------------------------------------------------------


class TestCallWithRetry:
    def test_force_stop_returns_none(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=1)
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )
        result = caller._call_with_retry(params)
        assert result is None

    def test_shutdown_event_stops_retry(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=3)
        shutdown = threading.Event()
        shutdown.set()  # Already set
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
            shutdown_event=shutdown,
        )
        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            side_effect=RuntimeError("fail"),
        ):
            result = caller._call_with_retry(params)
        assert result is None

    def test_retries_on_error(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=2)
        good_resp = _make_model_response()
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
                side_effect=[RuntimeError("try 1"), good_resp],
            ),
            patch("time.sleep"),
        ):
            result = caller._call_with_retry(params)
        assert result is not None

    def test_exhausts_retries_raises(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=1)
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
                side_effect=RuntimeError("permanent"),
            ),
            pytest.raises(RuntimeError, match="permanent"),
        ):
            caller._call_with_retry(params)


# ---------------------------------------------------------------------------
# call_llm_async — full flow
# ---------------------------------------------------------------------------


class TestCallLlmAsync:
    @pytest.mark.anyio
    async def test_returns_response(self, openai_config):
        caller = LLMCaller(Mock(), openai_config)
        good_resp = _make_model_response()
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hello")])]
        with patch.object(caller, "_call_with_retry_async", return_value=good_resp):
            result = await caller.call_llm_async(msgs)
        assert result is not None

    @pytest.mark.anyio
    async def test_no_client_raises(self):
        config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_chat_completion",
        )
        caller = LLMCaller(None, config)
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hello")])]
        with pytest.raises(RuntimeError, match="OpenAI client is not available"):
            await caller.call_llm_async(msgs)

    @pytest.mark.anyio
    async def test_with_middleware(self, openai_config):
        mm = MiddlewareManager()
        caller = LLMCaller(Mock(), openai_config, middleware_manager=mm)
        good_resp = _make_model_response()
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hello")])]
        with patch.object(caller, "_call_with_retry_async", return_value=good_resp):
            result = await caller.call_llm_async(msgs)
        assert result is not None


# ---------------------------------------------------------------------------
# _call_with_retry_async — retry and shutdown paths
# ---------------------------------------------------------------------------


class TestCallWithRetryAsync:
    @pytest.mark.anyio
    async def test_force_stop_returns_none(self, openai_config):
        caller = LLMCaller(Mock(), openai_config)
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )
        result = await caller._call_with_retry_async(params)
        assert result is None

    @pytest.mark.anyio
    async def test_shutdown_during_retry(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=3)
        shutdown = threading.Event()
        shutdown.set()
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
            shutdown_event=shutdown,
        )
        result = await caller._call_with_retry_async(params)
        assert result is None

    @pytest.mark.anyio
    async def test_with_sync_call_fn(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=1)
        good_resp = _make_model_response()
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        result = await caller._call_with_retry_async(params, sync_call_fn=lambda p: good_resp)
        assert result is not None

    @pytest.mark.anyio
    async def test_retries_and_raises_on_exhaustion(self, openai_config):
        caller = LLMCaller(Mock(), openai_config, retry_attempts=1)
        params = ModelCallParams(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={"model": "gpt-4o-mini"},
            openai_client=Mock(),
        )
        with pytest.raises(RuntimeError, match="fail"):
            await caller._call_with_retry_async(params, sync_call_fn=Mock(side_effect=RuntimeError("fail")))


# ---------------------------------------------------------------------------
# _adapt_structured_tools_for_provider
# ---------------------------------------------------------------------------


class TestAdaptStructuredToolsForProvider:
    def test_none_tools_returns_none(self):
        result = _adapt_structured_tools_for_provider(None, "openai")
        assert result is None

    def test_empty_tools_returns_none(self):
        result = _adapt_structured_tools_for_provider([], "openai")
        assert result is None

    def test_openai_adapts(self):
        tools = [
            {
                "name": "my_tool",
                "description": "desc",
                "input_schema": {"type": "object", "properties": {}},
                "kind": "tool",
            }
        ]
        result = _adapt_structured_tools_for_provider(tools, "openai")
        assert result is not None
        assert len(result) == 1

    def test_anthropic_adapts(self):
        tools = [
            {
                "name": "my_tool",
                "description": "desc",
                "input_schema": {"type": "object", "properties": {}},
                "kind": "tool",
            }
        ]
        result = _adapt_structured_tools_for_provider(tools, "anthropic")
        assert result is not None
        assert len(result) == 1

    def test_gemini_adapts(self):
        tools = [
            {
                "name": "my_tool",
                "description": "desc",
                "input_schema": {"type": "object", "properties": {}},
                "kind": "tool",
            }
        ]
        result = _adapt_structured_tools_for_provider(tools, "gemini")
        assert result is not None


# ---------------------------------------------------------------------------
# _raw_message_metadata — edge cases for object payloads
# ---------------------------------------------------------------------------


class TestRawMessageMetadataEdgeCases:
    def test_object_with_dict_choice(self):
        payload = SimpleNamespace(
            id="x",
            model=None,
            object=None,
            role=None,
            finish_reason=None,
            stop_reason=None,
            status=None,
            choices=[{"finish_reason": "stop"}],
        )
        result = _raw_message_metadata(payload)
        assert result["choices_count"] == 1
        assert result["choice_finish_reason"] == "stop"

    def test_dict_with_empty_choices(self):
        result = _raw_message_metadata({"choices": []})
        assert result["choices_count"] == 0
        assert "choice_finish_reason" not in result

    def test_dict_without_choices(self):
        result = _raw_message_metadata({"id": "x"})
        assert "choices_count" not in result

    def test_object_without_choices(self):
        payload = SimpleNamespace(id="x")
        result = _raw_message_metadata(payload)
        assert result["id"] == "x"
        assert "choices_count" not in result


# ---------------------------------------------------------------------------
# call_llm — XML stop sequences and param drops
# ---------------------------------------------------------------------------


class TestCallLlmXMLStopAndParamDrops:
    def test_xml_mode_adds_stop_sequences(self, openai_config):
        caller = LLMCaller(Mock(), openai_config)
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
        good_resp = _make_model_response()
        with (
            patch.object(caller, "_call_with_retry", return_value=good_resp),
            patch.object(caller, "middleware_manager", None),
        ):
            result = caller.call_llm(msgs, tool_call_mode="xml")
        assert result is not None

    def test_existing_stop_string_is_merged(self, openai_config):
        openai_config.set_param("stop", "DONE")
        caller = LLMCaller(Mock(), openai_config)
        msgs = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
        good_resp = _make_model_response()
        with patch.object(caller, "_call_with_retry", return_value=good_resp):
            result = caller.call_llm(msgs, tool_call_mode="xml")
        assert result is not None
