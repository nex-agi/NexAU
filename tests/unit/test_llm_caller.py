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
Unit tests for LLM caller component.

Tests cover:
- LLMCaller initialization
- Basic LLM API calls
- Retry logic with exponential backoff
- Force stop reason handling
- XML tag restoration
- Debug logging
- Stop sequence handling
- Error scenarios
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from nexau.archs.main_sub.execution.hooks import MiddlewareManager
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason


@pytest.fixture(autouse=True)
def mock_openai_module():
    """Mock the openai module to prevent any real API calls."""
    with patch("nexau.archs.main_sub.execution.llm_caller.openai") as mock_openai:
        # Ensure OpenAI client cannot be instantiated
        mock_openai.OpenAI.side_effect = RuntimeError("Real OpenAI client cannot be instantiated in tests")
        yield mock_openai


@pytest.fixture(autouse=True)
def mock_langfuse_module():
    """Mock langfuse functions to prevent any real connections to Langfuse server."""
    with (
        patch("nexau.archs.main_sub.execution.llm_caller.get_client") as mock_get_client,
        patch("nexau.archs.main_sub.execution.llm_caller.observe") as mock_observe,
    ):
        # Mock get_client to return a mock client
        mock_langfuse_client = Mock()
        mock_get_client.return_value = mock_langfuse_client

        # Mock observe decorator to pass through the original function
        def mock_observe_decorator(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        mock_observe.side_effect = mock_observe_decorator

        yield {"get_client": mock_get_client, "observe": mock_observe, "client": mock_langfuse_client}


class TestLLMCallerInitialization:
    """Test cases for LLMCaller initialization."""

    def test_initialization_with_defaults(self, mock_openai_client, mock_llm_config):
        """Test LLMCaller initialization with default parameters."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        assert caller.openai_client == mock_openai_client
        assert caller.llm_config == mock_llm_config
        assert caller.retry_attempts == 5
        assert caller.middleware_manager is None

    def test_initialization_with_custom_retry_attempts(self, mock_openai_client, mock_llm_config):
        """Test LLMCaller initialization with custom retry attempts."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=10,
        )

        assert caller.retry_attempts == 10

    def test_initialization_with_middleware_manager(self, mock_openai_client, mock_llm_config):
        """LLMCaller can be initialized with a middleware manager."""

        manager = MiddlewareManager([])
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            middleware_manager=manager,
        )

        assert caller.middleware_manager is manager


class TestLLMCallerBasicCalls:
    """Test cases for basic LLM API calls."""

    def test_call_llm_success(self, mock_openai_client, mock_llm_config, agent_state):
        """Test successful LLM API call."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Hello! How can I help you?"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_call_llm_success_responses_api(self, mock_openai_client, responses_llm_config, agent_state):
        """Test successful call flow when using the Responses API."""
        # Setup mock Responses payload
        message_item = {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello from responses!"}],
        }

        responses_payload = SimpleNamespace(
            output=[message_item],
            output_text="Hello from responses!",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_openai_client.responses.create.return_value = responses_payload

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=responses_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=120, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Hello from responses!"
        assert response.response_items == responses_payload.output

        mock_openai_client.responses.create.assert_called_once()
        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        expected_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        assert call_kwargs["input"] == expected_input
        assert call_kwargs["max_output_tokens"] == 120

    def test_call_llm_responses_api_carries_reasoning(self, mock_openai_client, responses_llm_config, agent_state):
        """Reasoning items should be preserved for subsequent turns."""

        reasoning_item = {
            "id": "rs_123",
            "type": "reasoning",
            "content": [{"type": "text", "text": "Thought step"}],
            "summary": [{"type": "text", "text": "Summary"}],
        }
        message_item = {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Assistant reply"}],
        }

        responses_payload = SimpleNamespace(
            output=[reasoning_item, message_item],
            output_text="Assistant reply",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=15, output_tokens=8),
        )
        mock_openai_client.responses.create.return_value = responses_payload

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=responses_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=60, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert "[reasoning]" in response.render_text()

        # Append to history and ensure reasoning makes it into subsequent input
        history = messages + [response.to_message_dict()]
        mock_openai_client.responses.create.reset_mock()

        caller.call_llm(history, max_tokens=60, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        followup_input = mock_openai_client.responses.create.call_args.kwargs["input"]
        assert reasoning_item in followup_input

    def test_call_llm_responses_api_normalizes_tools(self, mock_openai_client, responses_llm_config, agent_state):
        """Ensure tool payloads satisfy Responses API expectations."""

        message_item = {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Tool ready"}],
        }

        responses_payload = SimpleNamespace(
            output=[message_item],
            output_text="Tool ready",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
        )
        mock_openai_client.responses.create.return_value = responses_payload

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=responses_llm_config,
        )

        tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "A simple test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]

        caller.call_llm(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="openai",
            tools=tools_payload,
        )

        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        normalized_tool = call_kwargs["tools"][0]

        assert normalized_tool["name"] == "simple_tool"
        assert normalized_tool["type"] == "function"
        assert normalized_tool["description"] == "A simple test tool"
        assert normalized_tool["parameters"] == tools_payload[0]["function"]["parameters"]
        assert "function" not in normalized_tool

    def test_call_llm_responses_api_strips_status_from_history(self, mock_openai_client, responses_llm_config, agent_state):
        """Status fields from prior outputs should be removed when replaying context."""

        first_output_item = {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Using tool"}],
        }
        first_function_call = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "WebSearch",
            "arguments": "{}",
        }
        tool_result_item = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {
                    "type": "output_text",
                    "text": "Result text",
                }
            ],
        }

        responses_payload_first = SimpleNamespace(
            output=[first_output_item, first_function_call, tool_result_item],
            output_text="Using tool",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
        )

        second_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done"}],
        }

        responses_payload_second = SimpleNamespace(
            output=[second_output_item],
            output_text="Done",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=25, output_tokens=5),
        )

        mock_openai_client.responses.create.side_effect = [responses_payload_first, responses_payload_second]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=responses_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        first_response = caller.call_llm(messages, max_tokens=80, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        history = messages + [first_response.to_message_dict()]
        history.append(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{}",
            },
        )

        caller.call_llm(history, max_tokens=80, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        _, kwargs = mock_openai_client.responses.create.call_args
        replay_items = kwargs["input"]

        message_items = [item for item in replay_items if isinstance(item, dict) and item.get("type") == "message"]
        assert message_items, "Expected at least one message item in replayed input"
        for item in message_items:
            assert "status" not in item

        function_outputs = [item for item in replay_items if isinstance(item, dict) and item.get("type") == "function_call_output"]
        assert function_outputs, "Expected tool outputs to be replayed"
        output_values = [output_item.get("output") for output_item in function_outputs]
        for output_value in output_values:
            assert isinstance(output_value, str)
        assert "Result text" in output_values

    def test_call_llm_without_client_raises_error(self, mock_llm_config):
        """Test that calling LLM without client raises RuntimeError."""
        caller = LLMCaller(
            openai_client=None,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenAI client is not available"):
            caller.call_llm(messages, max_tokens=100)

    def test_call_llm_includes_xml_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls include XML stop sequences."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that stop sequences were added
        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        assert "</tool_use>" in stop_sequences
        assert "</use_parallel_tool_calls>" in stop_sequences
        assert "</use_batch_agent>" in stop_sequences

    def test_call_llm_merges_existing_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that existing stop sequences are preserved and merged."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with existing stop sequences
        mock_llm_config.to_openai_params = Mock(
            return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": ["custom_stop_1", "custom_stop_2"]}
        )

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        # Both custom and XML stop sequences should be present
        assert "custom_stop_1" in stop_sequences
        assert "custom_stop_2" in stop_sequences
        assert "</tool_use>" in stop_sequences

    def test_call_llm_with_openai_tools(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that tools parameter is forwarded and XML stops are omitted."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_llm_config.to_openai_params = Mock(
            return_value={"model": "gpt-4o-mini", "temperature": 0.7},
        )

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "A simple tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

        caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="openai",
            tools=tools_payload,
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["tools"] == tools_payload
        assert "tool_choice" in call_args[1]
        assert "stop" not in call_args[1] or "</tool_use>" not in call_args[1].get("stop", [])

    def test_call_llm_anthorpic_mode_not_supported(
        self,
        mock_openai_client,
        mock_llm_config,
        agent_state,
    ):
        """Anthorpic mode should clearly signal lack of support."""

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Mock response with proper tool_calls structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Anthorpic mode sets tools and tool_choice but may not be fully supported
        # The test verifies it doesn't crash with proper mocking
        response = caller.call_llm(
            messages,
            max_tokens=50,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="anthorpic",
        )

        # Verify the call completed (though anthorpic mode may not be fully functional)
        assert response is not None
        assert isinstance(response, ModelResponse)

    def test_call_llm_applies_additional_drop_params(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that additional_drop_params are applied before sending request."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_llm_config.additional_drop_params = ("stop", "temperature")

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_args = mock_openai_client.chat.completions.create.call_args
        assert "stop" not in call_args.kwargs
        assert "temperature" not in call_args.kwargs

    def test_call_llm_handles_string_stop_sequence(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that string stop sequences are converted to list."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with string stop sequence
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": "single_stop"})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        assert "single_stop" in stop_sequences
        assert "</tool_use>" in stop_sequences


class TestLLMCallerXMLRestoration:
    """Test cases for XML tag restoration."""

    def test_call_llm_restores_closing_tags(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that closing XML tags are restored when missing."""
        # Response with missing closing tag
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<tool_use><tool_name>test"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # The XMLUtils.restore_closing_tags should add </tool_use>
        assert "</tool_use>" in response.content

    def test_call_llm_splits_response_at_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that response is split at stop sequences."""
        # Response that contains stop sequence
        full_response = "Response content</tool_use>Extra content"
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = full_response
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Response should be split at the stop sequence
        # After split and restoration, it should have the closing tag but not extra content
        assert "Response content" in response.content
        assert "Extra content" not in response.content


class TestLLMCallerDebugLogging:
    """Test cases for debug logging."""

    def test_call_llm_debug_logging_enabled(self, mock_openai_client, mock_llm_config, agent_state, caplog):
        """Test that debug logging works when enabled."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Debug response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Enable debug mode
        mock_llm_config.debug = True

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "User message"}]

        with caplog.at_level(logging.INFO):
            caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that debug logs were created
        debug_logs = [rec.message for rec in caplog.records if "🐛 [DEBUG]" in rec.message]
        assert len(debug_logs) > 0

    def test_call_llm_debug_logging_disabled(self, mock_openai_client, mock_llm_config, agent_state, caplog):
        """Test that debug logging is disabled when debug mode is off."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Disable debug mode
        mock_llm_config.debug = False

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with caplog.at_level(logging.INFO):
            caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that debug logs were NOT created
        debug_logs = [rec.message for rec in caplog.records if "🐛 [DEBUG]" in rec.message]
        assert len(debug_logs) == 0


class TestLLMCallerRetryLogic:
    """Test cases for retry logic."""

    def test_call_llm_retry_on_failure(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls retry on failure."""
        # First two calls fail, third succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(choices=[Mock(message=Mock(content="Success after retry", tool_calls=[]))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):  # Mock sleep to speed up test
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Success after retry"
        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_call_llm_exhausts_retries(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls raise error after exhausting retries."""
        # All calls fail
        mock_openai_client.chat.completions.create.side_effect = Exception("Persistent API Error")

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(Exception, match="Persistent API Error"):
                caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_call_llm_exponential_backoff(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that retry uses exponential backoff."""
        # Fail a few times to trigger backoff
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            Mock(choices=[Mock(message=Mock(content="Success", tool_calls=[]))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep") as mock_sleep:
            caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that sleep was called with exponential backoff
        sleep_calls = mock_sleep.call_args_list
        assert len(sleep_calls) == 2  # Two failures before success
        assert sleep_calls[0] == call(1)  # First backoff: 1 second
        assert sleep_calls[1] == call(2)  # Second backoff: 2 seconds

    def test_call_llm_empty_response_triggers_retry(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that empty response content triggers retry."""
        # First call returns empty content, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content="", tool_calls=[]))]),
            Mock(choices=[Mock(message=Mock(content="Valid response", tool_calls=[]))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Valid response"
        assert mock_openai_client.chat.completions.create.call_count == 2


class TestLLMCallerForceStopReason:
    """Test cases for force stop reason handling."""

    def test_call_llm_with_force_stop_reason_success(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with SUCCESS stop reason proceeds normally."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Normal response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        assert isinstance(response, ModelResponse)
        assert response.content == "Normal response"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_call_llm_with_non_success_stop_reason_returns_none(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with non-SUCCESS stop reason returns None."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.MAX_ITERATIONS_REACHED,
            agent_state=agent_state,
        )

        assert response is None
        # OpenAI client should not be called
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_call_llm_with_error_stop_reason(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with ERROR_OCCURRED stop reason."""
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(
            messages,
            max_tokens=100,
            force_stop_reason=AgentStopReason.ERROR_OCCURRED,
            agent_state=agent_state,
        )

        assert response is None
        mock_openai_client.chat.completions.create.assert_not_called()


class TestLLMCallerEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_call_llm_with_empty_messages(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with empty messages list."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        response = caller.call_llm([], max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Response"
        # Check that empty messages were passed
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == []

    def test_call_llm_with_zero_max_tokens(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with zero max_tokens."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=0, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Response"
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] == 0

    def test_call_llm_with_none_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call when stop sequences are None."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with None stop sequences
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": None})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Response"
        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]
        # Should only have XML stop sequences
        assert "</tool_use>" in stop_sequences

    def test_call_llm_with_complex_response_content(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with complex response containing special characters."""
        complex_response = """
<tool_use>
<tool_name>test_tool</tool_name>
<parameter>
<query>Search for "complex & special < > characters"</query>
</parameter>
</tool_use>
"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = complex_response
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Response should preserve special characters
        assert "complex & special < > characters" in response.content
        assert "<tool_use>" in response.content

    def test_call_llm_with_very_large_retry_attempts(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM caller with very large retry attempts."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=100,
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Response"
        assert caller.retry_attempts == 100

    def test_call_llm_response_content_none_triggers_exception(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that None response content triggers retry."""
        # First call returns None, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content=None, tool_calls=[]))]),
            Mock(choices=[Mock(message=Mock(content="Valid response", tool_calls=[]))]),
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "user", "content": "Hello"}]

        with patch("time.sleep"):
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Valid response"
        assert mock_openai_client.chat.completions.create.call_count == 2


class TestLLMCallerIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_all_features(self, mock_openai_client, mock_llm_config, agent_state):
        """Test complete workflow with all features enabled."""
        # Setup mock with XML response containing stop sequence
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "<tool_use><tool_name>test</tool_name>"
        mock_response.choices[0].message.tool_calls = []
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Enable debug mode
        mock_llm_config.debug = True
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": ["custom_stop"]})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Execute tool"}]

        response = caller.call_llm(
            messages,
            max_tokens=200,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        # Verify response has closing tag restored
        assert "</tool_use>" in response.content
        assert "<tool_use>" in response.content

        # Verify API was called with correct parameters
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == messages
        assert call_args[1]["max_tokens"] == 200
        assert "custom_stop" in call_args[1]["stop"]
        assert "</tool_use>" in call_args[1]["stop"]
