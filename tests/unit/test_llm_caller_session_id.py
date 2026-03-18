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

"""Tests for session_id injection into LLM provider payloads.

Verifies that LLMCaller correctly injects session_id as:
- OpenAI Chat Completion / Responses API: ``user`` field
- Anthropic: ``metadata.user_id`` field
- Gemini REST: skipped (no such field)
"""

from unittest.mock import patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.core.messages import Message


@pytest.fixture(autouse=True)
def mock_openai_module():
    """Mock the openai module to prevent any real API calls."""
    with patch("nexau.archs.main_sub.execution.llm_caller.openai") as mock_openai:
        mock_openai.OpenAI.side_effect = RuntimeError("Real OpenAI client cannot be instantiated in tests")
        yield mock_openai


def _make_llm_config(api_type: str) -> LLMConfig:
    return LLMConfig(
        model="test-model",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        temperature=0.1,
        max_tokens=100,
        api_type=api_type,
    )


def _make_mock_model_response() -> ModelResponse:
    return ModelResponse(content="test response", role="assistant", tool_calls=[])


class TestSessionIdInitialization:
    """Test session_id is stored and defaults correctly."""

    def test_default_session_id_is_none(self, mock_openai_client):
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=_make_llm_config("openai_chat_completion"),
        )
        assert caller.session_id is None

    def test_session_id_is_stored(self, mock_openai_client):
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=_make_llm_config("openai_chat_completion"),
            session_id="sess-abc-123",
        )
        assert caller.session_id == "sess-abc-123"


class TestOpenAIChatCompletionSessionId:
    """Test session_id injection for openai_chat_completion api_type."""

    def test_user_field_injected(self, mock_openai_client, agent_state):
        """session_id should appear as 'user' in kwargs passed to the OpenAI client."""
        llm_config = _make_llm_config("openai_chat_completion")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-openai-001",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]  # positional arg: kwargs
        assert kwargs_passed["user"] == "sess-openai-001"

    def test_user_field_not_injected_when_session_id_is_none(self, mock_openai_client, agent_state):
        """When session_id is None, no 'user' field should be added."""
        llm_config = _make_llm_config("openai_chat_completion")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert "user" not in kwargs_passed

    def test_existing_user_field_not_overwritten(self, mock_openai_client, agent_state):
        """If 'user' already exists in api_params (via extra_params), it should not be overwritten."""
        llm_config = _make_llm_config("openai_chat_completion")
        llm_config.extra_params["user"] = "explicit-user"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-should-not-win",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert kwargs_passed["user"] == "explicit-user"


class TestOpenAIResponsesSessionId:
    """Test session_id injection for openai_responses api_type.

    The Responses API deprecates ``user`` in favor of ``safety_identifier``
    and ``prompt_cache_key``.  All three are set for backward-compat.
    """

    def test_session_id_fields_injected(self, mock_openai_client, agent_state):
        """safety_identifier and prompt_cache_key should be set; user should NOT
        be injected (deprecated and rejected by some backends)."""
        llm_config = _make_llm_config("openai_responses")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-responses-001",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert "user" not in kwargs_passed
        assert kwargs_passed["safety_identifier"] == "sess-responses-001"
        assert kwargs_passed["prompt_cache_key"] == "sess-responses-001"

    def test_existing_prompt_cache_key_not_overwritten(self, mock_openai_client, agent_state):
        """prompt_cache_key set by llm_config (e.g. Agent.__init__) must survive."""
        llm_config = _make_llm_config("openai_responses")
        llm_config.extra_params["prompt_cache_key"] = "agent-uuid-key"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-responses-002",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        # prompt_cache_key from llm_config takes precedence
        assert kwargs_passed["prompt_cache_key"] == "agent-uuid-key"
        # safety_identifier still defaults to session_id
        assert kwargs_passed["safety_identifier"] == "sess-responses-002"

    def test_existing_safety_identifier_not_overwritten(self, mock_openai_client, agent_state):
        """safety_identifier set by llm_config must survive."""
        llm_config = _make_llm_config("openai_responses")
        llm_config.extra_params["safety_identifier"] = "explicit-safety-id"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-responses-003",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert kwargs_passed["safety_identifier"] == "explicit-safety-id"
        # prompt_cache_key still defaults to session_id
        assert kwargs_passed["prompt_cache_key"] == "sess-responses-003"

    def test_all_fields_preset_by_config(self, mock_openai_client, agent_state):
        """When llm_config sets safety_identifier and prompt_cache_key, neither should be overwritten."""
        llm_config = _make_llm_config("openai_responses")
        llm_config.extra_params["safety_identifier"] = "cfg-safety"
        llm_config.extra_params["prompt_cache_key"] = "cfg-cache"

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-should-not-win",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert kwargs_passed["safety_identifier"] == "cfg-safety"
        assert kwargs_passed["prompt_cache_key"] == "cfg-cache"

    def test_no_fields_injected_when_session_id_is_none(self, mock_openai_client, agent_state):
        llm_config = _make_llm_config("openai_responses")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert "user" not in kwargs_passed
        assert "safety_identifier" not in kwargs_passed


class TestAnthropicSessionId:
    """Test session_id injection for anthropic_chat_completion api_type."""

    def test_metadata_user_id_injected(self, mock_openai_client, agent_state):
        """session_id should appear as metadata.user_id for Anthropic."""
        llm_config = _make_llm_config("anthropic_chat_completion")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-anthropic-001",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert kwargs_passed["metadata"] == {"user_id": "sess-anthropic-001"}

    def test_metadata_not_injected_when_session_id_is_none(self, mock_openai_client, agent_state):
        llm_config = _make_llm_config("anthropic_chat_completion")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert "metadata" not in kwargs_passed

    def test_existing_metadata_merged(self, mock_openai_client, agent_state):
        """If metadata already exists, user_id should be merged without overwriting."""
        llm_config = _make_llm_config("anthropic_chat_completion")
        llm_config.extra_params["metadata"] = {"user_id": "explicit-user", "custom_key": "value"}

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-should-not-win",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        # Existing user_id should be preserved, custom_key should remain
        assert kwargs_passed["metadata"]["user_id"] == "explicit-user"
        assert kwargs_passed["metadata"]["custom_key"] == "value"


class TestGeminiRestSessionId:
    """Test that session_id is NOT injected for gemini_rest api_type."""

    def test_no_user_or_metadata_for_gemini(self, mock_openai_client, agent_state):
        llm_config = _make_llm_config("gemini_rest")
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
            retry_attempts=1,
            session_id="sess-gemini-001",
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=_make_mock_model_response(),
        ) as mock_call:
            caller.call_llm(
                [Message.user("Hello")],
                max_tokens=100,
                force_stop_reason=AgentStopReason.SUCCESS,
                agent_state=agent_state,
            )

        kwargs_passed = mock_call.call_args[0][2]
        assert "user" not in kwargs_passed
        assert "metadata" not in kwargs_passed
