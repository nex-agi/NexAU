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

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import MiddlewareManager, ModelCallParams
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload
from nexau.core.serializers.openai_responses import prepare_openai_responses_api_input
from nexau.core.usage import TokenUsage


@pytest.fixture(autouse=True)
def mock_openai_module():
    """Mock the openai module to prevent any real API calls."""
    with patch("nexau.archs.main_sub.execution.llm_caller.openai") as mock_openai:
        # Ensure OpenAI client cannot be instantiated
        mock_openai.OpenAI.side_effect = RuntimeError("Real OpenAI client cannot be instantiated in tests")
        yield mock_openai


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

    def test_initialization_with_retry_backoff_cap_and_callback(self, mock_openai_client, mock_llm_config):
        """Test LLMCaller stores retry backoff cap and retry callback."""
        callback = Mock()
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_backoff_max_seconds=7,
            on_retry=callback,
        )

        assert caller.retry_backoff_max_seconds == 7
        assert caller.on_retry is callback

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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Hello! How can I help you?"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_call_llm_chat_completion_strips_response_items(self, mock_openai_client, mock_llm_config, agent_state):
        """Ensure Responses-specific fields are removed before chat completion calls."""

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        reasoning_item = {
            "type": "reasoning",
            "id": "rs_reasoning_stream",
            "summary": [{"type": "summary_text", "text": "cached summary"}],
            "content": [{"type": "text", "text": "chain-of-thought"}],
        }

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Cached reply", "response_items": [reasoning_item]},
            ],
        )

        mock_openai_client.chat.completions.create.return_value.choices[0].message.tool_calls = []

        response = caller.call_llm(messages, max_tokens=50, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)

        sent_messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert all("response_items" not in msg for msg in sent_messages if isinstance(msg, dict))
        # Original history should remain intact for future Responses API use
        assert messages[1].metadata["response_items"][0]["id"] == "rs_reasoning_stream"

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

        messages = [Message.user("Hello")]
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
        assert call_kwargs["parallel_tool_calls"] is True

    def test_call_llm_success_responses_api_honors_parallel_tool_calls_override(self, mock_openai_client, agent_state):
        """Responses API should preserve explicit parallel_tool_calls=False from LLMConfig extra kwargs."""
        responses_payload = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Hello from responses!"}],
                }
            ],
            output_text="Hello from responses!",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_openai_client.responses.create.return_value = responses_payload

        llm_config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            temperature=0.1,
            max_tokens=1000,
            api_type="openai_responses",
            parallel_tool_calls=False,
        )
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=llm_config,
        )

        messages = [Message.user("Hello")]
        response = caller.call_llm(messages, max_tokens=120, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        assert call_kwargs["parallel_tool_calls"] is False

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

        messages = [Message.user("Hello")]
        response = caller.call_llm(messages, max_tokens=60, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert "[reasoning]" in response.render_text()

        # Append to history and ensure reasoning makes it into subsequent input
        history = messages + [response.to_ump_message()]
        mock_openai_client.responses.create.reset_mock()

        caller.call_llm(history, max_tokens=60, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        followup_input = mock_openai_client.responses.create.call_args.kwargs["input"]
        # RFC-0014: serializer strips `content` from reasoning items; only `summary` is kept
        expected_reasoning = {
            "type": "reasoning",
            "summary": [{"type": "text", "text": "Summary"}],
        }
        assert expected_reasoning in followup_input

    def test_model_response_from_completion_reasoning_shape(self):
        """Chat Completions responses can carry reasoning_content directly on the assistant message."""

        response = ModelResponse.from_openai_message(
            {
                "role": "assistant",
                "content": "35 × 11 = 385\nFinal: 34",
                "reasoning_content": "Compute 35 * 11 first, then divide by 13.",
            },
            usage={"prompt_tokens": 43, "completion_tokens": 82, "total_tokens": 125},
        )

        assert response.content == "35 × 11 = 385\nFinal: 34"
        assert response.reasoning_content == "Compute 35 * 11 first, then divide by 13."
        assert response.response_items == []
        assert response.usage.total_tokens == 125

    def test_model_response_preserves_empty_completion_reasoning_content(self):
        """DeepSeek requires explicit blank reasoning_content to be echoed on later turns."""

        response = ModelResponse.from_openai_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            },
        )

        assert response.reasoning_content == ""
        assert response.to_message_dict()["reasoning_content"] == ""
        assert serialize_ump_to_openai_chat_payload([response.to_ump_message()]) == [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            }
        ]

    def test_model_response_from_responses_reasoning_shape(self):
        """Responses API reasoning items should preserve summary text and encrypted replay artifacts."""

        response = ModelResponse.from_openai_response(
            {
                "output": [
                    {
                        "id": "rs_real",
                        "type": "reasoning",
                        "encrypted_content": "encrypted_blob",
                        "summary": [{"type": "summary_text", "text": "Verified the arithmetic before answering."}],
                    },
                    {
                        "id": "msg_real",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Final: 34"}],
                    },
                ],
                "usage": {
                    "input_tokens": 43,
                    "output_tokens": 73,
                    "total_tokens": 116,
                    "output_tokens_details": {"reasoning_tokens": 38},
                },
            }
        )

        assert response.content == "Final: 34"
        assert response.reasoning_content == "Verified the arithmetic before answering."
        assert response.response_items[0]["type"] == "reasoning"
        assert response.response_items[0]["encrypted_content"] == "encrypted_blob"
        assert response.usage.reasoning_tokens == 38

    def test_model_response_from_responses_preserves_empty_reasoning_content(self):
        """Responses API blank reasoning summaries should remain replayable as empty strings."""

        response = ModelResponse.from_openai_response(
            {
                "output": [
                    {
                        "id": "rs_blank",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": ""}],
                    },
                    {
                        "id": "msg_real",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Final: 34"}],
                    },
                ],
            }
        )

        assert response.content == "Final: 34"
        assert response.reasoning_content == ""
        assert response.to_message_dict()["reasoning_content"] == ""

    def test_prepare_responses_input_reuses_encrypted_reasoning_response_items(self):
        """Stored Responses API reasoning items should be replayed with encrypted_content intact."""
        prepared, instructions = prepare_openai_responses_api_input(
            [
                {
                    "role": "assistant",
                    "content": "Final: 34",
                    "response_items": [
                        {
                            "id": "rs_real",
                            "type": "reasoning",
                            "encrypted_content": "encrypted_blob",
                            "summary": [{"type": "summary_text", "text": "Verified the arithmetic before answering."}],
                        },
                        {
                            "id": "msg_real",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "Final: 34"}],
                        },
                    ],
                }
            ]
        )

        assert instructions is None
        assert prepared == [
            {
                "type": "reasoning",
                "encrypted_content": "encrypted_blob",
                "summary": [{"type": "summary_text", "text": "Verified the arithmetic before answering."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Final: 34"}],
            },
        ]

    def test_prepare_responses_input_reconstructs_reasoning_from_completion_reasoning_content(self):
        """Chat-completions reasoning_content should become replayable Responses reasoning input."""
        prepared, instructions = prepare_openai_responses_api_input(
            [
                {
                    "role": "assistant",
                    "content": "Final: 34",
                    "reasoning_content": "Compute 35 * 11 first, then divide by 13.",
                }
            ]
        )

        assert instructions is None
        # RFC-0014: reasoning 在 message 之前（与 Responses API 原生顺序一致）
        assert prepared == [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Compute 35 * 11 first, then divide by 13."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Final: 34"}],
            },
        ]

    def test_prepare_responses_api_input_preserves_assistant_phase(self):
        """Assistant message phase should be forwarded to Responses API input."""
        prepared, instructions = prepare_openai_responses_api_input(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Working...", "phase": "commentary"},
                {"role": "assistant", "content": "Done", "phase": "final_answer"},
            ]
        )

        assert instructions is None
        assistant_messages = [item for item in prepared if item.get("type") == "message" and item.get("role") == "assistant"]
        assert assistant_messages[0]["phase"] == "commentary"
        assert assistant_messages[1]["phase"] == "final_answer"

    def test_prepare_responses_api_input_user_multimodal_content(self):
        """User message with list content (text + image_url) should produce input_text + input_image parts."""
        prepared, _instructions = prepare_openai_responses_api_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe these"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                    ],
                },
            ]
        )

        assert len(prepared) == 1
        msg = prepared[0]
        assert msg["type"] == "message"
        assert msg["role"] == "user"
        parts = msg["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "input_text", "text": "describe these"}
        assert parts[1] == {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}

    def test_prepare_responses_api_input_user_image_with_detail(self):
        """Image part with explicit detail should forward the detail field."""

        prepared, _ = prepare_openai_responses_api_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}},
                    ],
                },
            ]
        )

        parts = prepared[0]["content"]
        assert len(parts) == 2
        img_part = parts[1]
        assert img_part["type"] == "input_image"
        assert img_part["image_url"] == "https://example.com/img.png"
        assert img_part["detail"] == "high"

    def test_prepare_responses_api_input_user_string_content_unchanged(self):
        """Plain string content for user messages should still work (regression test)."""

        prepared, _ = prepare_openai_responses_api_input([{"role": "user", "content": "just text"}])

        assert len(prepared) == 1
        msg = prepared[0]
        assert msg["content"] == [{"type": "input_text", "text": "just text"}]

    def test_prepare_responses_api_input_assistant_multimodal_content(self):
        """Assistant message with list content should convert parts using output_text type."""

        prepared, _ = prepare_openai_responses_api_input(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here is the image analysis"},
                    ],
                },
            ]
        )

        assert len(prepared) == 1
        msg = prepared[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "output_text", "text": "Here is the image analysis"}]

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
            messages=[Message.user("Hello")],
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

    def test_call_llm_generate_with_token_uses_client_response_tokens(self, agent_state):
        """generate_with_token should parse OpenAI-like payloads from the client."""
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        token_trace_session = Mock()
        token_trace_session.token_ids = [1, 2, 3]
        token_trace_session.build_generate_with_token_kwargs.return_value = {
            "model": "token-model",
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 32},
        }
        token_client = Mock()
        token_client.generate_with_token.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "tool call incoming",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": '{"city":"Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"input_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            "nexrl_train": {
                "prompt_tokens": [1, 2, 3],
                "response_tokens": [4, 5, 6],
                "response_logprobs": [-0.1, -0.2, -0.3],
            },
        }

        caller = LLMCaller(
            openai_client=token_client,
            llm_config=llm_config,
        )
        response = caller.call_llm(
            [Message.user("hello")],
            max_tokens=32,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="openai",
            tools=[],
            token_trace_session=token_trace_session,
        )

        assert isinstance(response, ModelResponse)
        assert response.content == "tool call incoming"
        assert response.output_token_ids == [4, 5, 6]
        assert response.tool_calls[0].name == "lookup_weather"
        token_trace_session.sync_external_messages.assert_called_once()
        token_trace_session.record_round.assert_called_once()
        token_client.generate_with_token.assert_called_once_with(
            model="token-model",
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 32},
            tools=[],
        )

    def test_call_llm_generate_with_token_forwards_structured_tools(self, agent_state):
        """generate_with_token should forward structured tools from model call params."""
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        token_trace_session = Mock()
        token_trace_session.token_ids = [1, 2, 3]
        token_trace_session.build_generate_with_token_kwargs.return_value = {
            "model": "token-model",
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 32},
        }
        token_client = Mock()
        token_client.generate_with_token.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"input_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "nexrl_train": {
                "prompt_tokens": [1, 2, 3],
                "response_tokens": [4],
                "response_logprobs": [-0.1],
            },
        }
        tools_payload = [
            {
                "name": "lookup_weather",
                "description": "Look up weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["city"],
                },
                "kind": "tool",
            }
        ]

        caller = LLMCaller(
            openai_client=token_client,
            llm_config=llm_config,
        )
        response = caller.call_llm(
            [Message.user("hello")],
            max_tokens=32,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="openai",
            tools=tools_payload,
            token_trace_session=token_trace_session,
        )

        assert isinstance(response, ModelResponse)
        token_client.generate_with_token.assert_called_once_with(
            model="token-model",
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 32},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

    def test_call_llm_generate_with_token_detokenizes_when_text_missing(self, agent_state):
        """generate_with_token should detokenize response tokens when text is missing."""
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        token_trace_session = Mock()
        token_trace_session.token_ids = [11, 12]
        token_trace_session.build_generate_with_token_kwargs.return_value = {
            "model": "token-model",
            "input_ids": [11, 12],
        }
        token_client = Mock()
        token_client.generate_with_token.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"input_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            "nexrl_train": {
                "prompt_tokens": [11, 12],
                "response_tokens": [13, 14],
                "response_logprobs": [-0.1, -0.2],
            },
        }
        token_trace_session.detokenize.return_value = "decoded output"

        caller = LLMCaller(
            openai_client=token_client,
            llm_config=llm_config,
        )
        response = caller.call_llm(
            [Message.user("hello")],
            max_tokens=16,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            token_trace_session=token_trace_session,
        )

        assert isinstance(response, ModelResponse)
        assert response.content == "decoded output"
        assert response.output_token_ids == [13, 14]
        token_trace_session.detokenize.assert_called_once_with([13, 14])
        token_client.generate_with_token.assert_called_once_with(
            model="token-model",
            input_ids=[11, 12],
        )

    def test_call_llm_generate_with_token_uses_nexrl_train_usage(self, agent_state):
        """generate_with_token should read token ids from nexrl_train."""
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        token_trace_session = Mock()
        token_trace_session.token_ids = [21, 22, 23]
        token_trace_session.build_generate_with_token_kwargs.return_value = {
            "model": "token-model",
            "input_ids": [21, 22, 23],
        }
        token_client = Mock()
        token_client.generate_with_token.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "openai-like output"},
                    "finish_reason": {"type": "length", "length": 2},
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 1,
            },
            "nexrl_train": {
                "prompt_tokens": [21, 22, 23],
                "response_tokens": [31, 32],
                "response_logprobs": [-0.1, -0.2],
            },
        }

        caller = LLMCaller(
            openai_client=token_client,
            llm_config=llm_config,
        )
        response = caller.call_llm(
            [Message.user("hello")],
            max_tokens=16,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            token_trace_session=token_trace_session,
        )

        assert isinstance(response, ModelResponse)
        assert response.content == "openai-like output"
        assert response.output_token_ids == [31, 32]
        assert response.usage.input_tokens == 2
        assert response.usage.completion_tokens == 2
        assert response.usage.total_tokens == 5
        assert response.usage.cache_read_tokens == 1
        assert response.raw_message["choices"][0]["finish_reason"]["type"] == "length"
        token_trace_session.record_round.assert_called_once()
        token_client.generate_with_token.assert_called_once_with(
            model="token-model",
            input_ids=[21, 22, 23],
        )

    def test_call_llm_generate_with_token_supports_raw_generate_payload(self, agent_state):
        """generate_with_token should accept raw `/generate` payloads without OpenAI choices."""
        llm_config = LLMConfig(
            model="token-model",
            base_url="http://token-gateway",
            api_key="test-key",
            api_type="generate_with_token",
        )
        token_trace_session = Mock()
        token_trace_session.token_ids = [151644, 872, 198]
        token_trace_session.build_generate_with_token_kwargs.return_value = {
            "model": "token-model",
            "input_ids": [151644, 872, 198],
        }
        token_client = Mock()
        token_client.generate_with_token.return_value = {
            "text": "你好，请帮我分析一下。",
            "meta_info": {
                "finish_reason": {"type": "length", "length": 32},
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 10,
                "output_token_logprobs": [[-0.1, 31, None], [-0.2, 32, None]],
            },
            "output_ids": [31, 32],
            "nexrl_train": {
                "prompt_tokens": [151644, 872, 198],
                "response_tokens": [31, 32],
                "response_logprobs": [-0.1, -0.2],
            },
        }

        caller = LLMCaller(
            openai_client=token_client,
            llm_config=llm_config,
        )
        response = caller.call_llm(
            [Message.user("hello")],
            max_tokens=16,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            token_trace_session=token_trace_session,
        )

        assert isinstance(response, ModelResponse)
        assert response.content == "你好，请帮我分析一下。"
        assert response.output_token_ids == [31, 32]
        assert response.usage.input_tokens == 0
        assert response.usage.completion_tokens == 2
        assert response.usage.total_tokens == 5
        assert response.usage.cache_read_tokens == 10
        assert response.raw_message["meta_info"]["finish_reason"]["type"] == "length"
        token_trace_session.record_round.assert_called_once()
        token_client.generate_with_token.assert_called_once_with(
            model="token-model",
            input_ids=[151644, 872, 198],
        )

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

        messages = [Message.user("Hello")]
        first_response = caller.call_llm(messages, max_tokens=80, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        history = messages + [first_response.to_ump_message()]
        history.append(Message(role=Role.TOOL, content=[ToolResultBlock(tool_use_id="call_1", content="{}", is_error=False)]))

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

    def test_call_llm_responses_api_tool_output_images_preserved_in_function_call_output(
        self, mock_openai_client, responses_llm_config, agent_state
    ):
        """Responses API supports multimodal function_call_output.output arrays; preserve images there."""

        responses_payload = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ],
            output_text="ok",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
        )
        mock_openai_client.responses.create.return_value = responses_payload

        caller = LLMCaller(openai_client=mock_openai_client, llm_config=responses_llm_config)

        # This tool message gets converted to OpenAI Chat payload dicts via the chat serializer,
        # which represent images as {"type":"image_url","image_url":{"url":"...","detail":"high"}} parts.
        history = [
            Message.user("Hello"),
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content=[
                            TextBlock(text="Here is the image"),
                            ImageBlock(url="https://example.com/a.jpg", detail="high"),
                        ],
                    ),
                ],
            ),
        ]

        caller.call_llm(history, max_tokens=80, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]

        function_outputs = [item for item in input_items if isinstance(item, dict) and item.get("type") == "function_call_output"]
        assert function_outputs
        output_payload = function_outputs[0].get("output")
        assert isinstance(output_payload, list)
        assert any(isinstance(p, dict) and p.get("type") == "input_image" for p in output_payload)
        assert any(isinstance(p, dict) and p.get("type") == "input_text" for p in output_payload)

    def test_call_llm_responses_api_sanitizes_response_items_multimodal_function_call_output(
        self, mock_openai_client, responses_llm_config, agent_state
    ) -> None:
        """Cover response_items sanitizer: output_text->input_text, image_url string/map, skip non-mapping, omit auto detail."""

        responses_payload = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ],
            output_text="ok",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_openai_client.responses.create.return_value = responses_payload

        caller = LLMCaller(openai_client=mock_openai_client, llm_config=responses_llm_config)

        response_items = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    "not a mapping",
                    {"type": "output_text", "text": "Result text"},
                    {"type": "image_url", "image_url": "https://example.com/a.png"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/b.png", "detail": "auto"}},
                ],
            }
        ]

        history = [
            Message.user("Hello"),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="cached")], metadata={"response_items": response_items}),
        ]

        caller.call_llm(history, max_tokens=80, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        call_kwargs = mock_openai_client.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]
        function_outputs = [item for item in input_items if isinstance(item, dict) and item.get("type") == "function_call_output"]
        assert function_outputs

        output_payload = function_outputs[0].get("output")
        assert isinstance(output_payload, list)

        # output_text -> input_text
        assert {"type": "input_text", "text": "Result text"} in output_payload

        # image_url (string) -> input_image
        assert {"type": "input_image", "image_url": "https://example.com/a.png"} in output_payload

        # image_url with detail=auto should omit detail field
        assert {"type": "input_image", "image_url": "https://example.com/b.png"} in output_payload
        assert not any(
            isinstance(p, dict) and p.get("type") == "input_image" and p.get("image_url") == "https://example.com/b.png" and "detail" in p
            for p in output_payload
        )

    def test_call_llm_without_client_raises_error(self, mock_llm_config):
        """Test that calling LLM without client raises RuntimeError."""
        caller = LLMCaller(
            openai_client=None,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]

        with pytest.raises(RuntimeError, match="OpenAI client is not available"):
            caller.call_llm(messages, max_tokens=100)

    def test_call_llm_includes_xml_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that LLM calls include XML stop sequences."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
        caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        # Check that stop sequences were added
        call_args = mock_openai_client.chat.completions.create.call_args
        stop_sequences = call_args[1]["stop"]

        assert "</tool_use>" in stop_sequences
        assert "</use_parallel_tool_calls>" in stop_sequences

    def test_call_llm_merges_existing_stop_sequences(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that existing stop sequences are preserved and merged."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with existing stop sequences
        mock_llm_config.to_openai_params = Mock(
            return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": ["custom_stop_1", "custom_stop_2"]}
        )

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_llm_config.to_openai_params = Mock(
            return_value={"model": "gpt-4o-mini", "temperature": 0.7},
        )

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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

    def test_legacy_tool_call_mode_alias_uses_provider_selected_by_api_type(
        self,
        mock_openai_client,
        mock_llm_config,
        agent_state,
    ):
        """Legacy aliases should normalize to structured and still follow ``api_type``."""

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_llm_config.api_type = "openai_chat_completion"
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini"})

        caller = LLMCaller(openai_client=mock_openai_client, llm_config=mock_llm_config)

        caller.call_llm(
            [Message.user("Hello")],
            max_tokens=50,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="anthropic",
            tools=[
                {
                    "name": "simple_tool",
                    "description": "A simple tool",
                    "input_schema": {"properties": {}},
                    "kind": "tool",
                }
            ],
        )

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "A simple tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        assert call_args.kwargs["tool_choice"] == "auto"

    def test_call_llm_anthropic_mode_not_supported(
        self,
        mock_openai_client,
        mock_llm_config,
        agent_state,
    ):
        """anthropic mode should clearly signal lack of support."""

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]

        # Mock response with proper tool_calls structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # anthropic mode sets tools and tool_choice but may not be fully supported
        # The test verifies it doesn't crash with proper mocking
        response = caller.call_llm(
            messages,
            max_tokens=50,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
            tool_call_mode="anthropic",
        )

        # Verify the call completed (though anthropic mode may not be fully functional)
        assert response is not None
        assert isinstance(response, ModelResponse)

    def test_call_llm_applies_additional_drop_params(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that additional_drop_params are applied before sending request."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_llm_config.additional_drop_params = ("stop", "temperature")

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with string stop sequence
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": "single_stop"})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Enable debug mode
        mock_llm_config.debug = True

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
            ],
        )

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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Disable debug mode
        mock_llm_config.debug = False

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]

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
        success_response = Mock(choices=[Mock(message=Mock(content="Success after retry", tool_calls=[]))])
        success_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            success_response,
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [Message.user("Hello")]

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

        messages = [Message.user("Hello")]

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(Exception, match="Persistent API Error"):
                caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert mock_openai_client.chat.completions.create.call_count == 3

    def test_call_llm_exponential_backoff(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that retry uses exponential backoff."""
        # Fail a few times to trigger backoff
        success_response = Mock(choices=[Mock(message=Mock(content="Success", tool_calls=[]))])
        success_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            success_response,
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [Message.user("Hello")]

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
        empty_response = Mock(choices=[Mock(message=Mock(content="", tool_calls=[], reasoning_content=None))])
        empty_response.usage = {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        valid_response = Mock(choices=[Mock(message=Mock(content="Valid response", tool_calls=[], reasoning_content=None))])
        valid_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.side_effect = [
            empty_response,
            valid_response,
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [Message.user("Hello")]

        with patch("time.sleep"):
            response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Valid response"
        assert mock_openai_client.chat.completions.create.call_count == 2

    def test_empty_response_logs_only_sanitized_raw_message_metadata(self, mock_openai_client, mock_llm_config, agent_state, caplog):
        """Empty response logs should avoid leaking raw response content."""
        unsafe_raw_message = {
            "model": "gpt-4o-mini",
            "choices": [{"finish_reason": "length", "message": {"content": "TOP_SECRET_PAYLOAD"}}],
            "content": "TOP_SECRET_PAYLOAD",
        }
        empty_model_response = ModelResponse(
            content="",
            role="assistant",
            usage=TokenUsage(),
            raw_message=unsafe_raw_message,
        )

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=1,
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.call_llm_with_different_client",
            return_value=empty_model_response,
        ):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(Exception, match="No response content or tool calls"):
                    caller.call_llm(
                        [Message.user("Hello")],
                        max_tokens=100,
                        force_stop_reason=AgentStopReason.SUCCESS,
                        agent_state=agent_state,
                    )

        all_logs = "\n".join(record.message for record in caplog.records)
        assert "raw_message_meta" in all_logs
        assert "gpt-4o-mini" in all_logs
        assert "TOP_SECRET_PAYLOAD" not in all_logs


class TestLLMCallerForceStopReason:
    """Test cases for force stop reason handling."""

    def test_call_llm_with_force_stop_reason_success(self, mock_openai_client, mock_llm_config, agent_state):
        """Test LLM call with SUCCESS stop reason proceeds normally."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Normal response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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

        messages = [Message.user("Hello")]
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

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Setup llm_config with None stop sequences
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": None})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
        )

        messages = [Message.user("Hello")]
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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=100,
        )

        messages = [Message.user("Hello")]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert isinstance(response, ModelResponse)
        assert response.content == "Response"
        assert caller.retry_attempts == 100

    def test_call_llm_response_content_none_triggers_exception(self, mock_openai_client, mock_llm_config, agent_state):
        """Test that None response content triggers retry."""
        # First call returns None, second succeeds
        none_response = Mock(choices=[Mock(message=Mock(content=None, tool_calls=[], reasoning_content=None))])
        none_response.usage = {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        valid_response = Mock(choices=[Mock(message=Mock(content="Valid response", tool_calls=[], reasoning_content=None))])
        valid_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.side_effect = [
            none_response,
            valid_response,
        ]

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = [Message.user("Hello")]

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
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Enable debug mode
        mock_llm_config.debug = True
        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7, "stop": ["custom_stop"]})

        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=3,
        )

        messages = messages_from_legacy_openai_chat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Execute tool"},
            ],
        )

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
        assert call_args[1]["messages"] == serialize_ump_to_openai_chat_payload(messages, tool_image_policy="inject_user_message")
        assert call_args[1]["max_tokens"] == 200
        assert "custom_stop" in call_args[1]["stop"]
        assert "</tool_use>" in call_args[1]["stop"]

    def test_call_llm_preserves_empty_reasoning_content_in_history(self, mock_openai_client, mock_llm_config, agent_state):
        """DeepSeek thinking mode requires prior assistant reasoning_content to be echoed, even blank."""

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "follow up"
        mock_response.choices[0].message.tool_calls = []
        mock_response.choices[0].message.reasoning_content = None
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_llm_config.to_openai_params = Mock(return_value={"model": "gpt-4o-mini", "temperature": 0.7})
        caller = LLMCaller(
            openai_client=mock_openai_client,
            llm_config=mock_llm_config,
            retry_attempts=1,
        )

        prior_assistant = ModelResponse.from_openai_message(
            {
                "role": "assistant",
                "content": "previous answer",
                "reasoning_content": "",
            },
        ).to_ump_message()
        messages = [Message.user("first"), prior_assistant, Message.user("again")]

        caller.call_llm(
            messages,
            max_tokens=200,
            force_stop_reason=AgentStopReason.SUCCESS,
            agent_state=agent_state,
        )

        sent_messages = mock_openai_client.chat.completions.create.call_args[1]["messages"]
        assert sent_messages[1]["role"] == "assistant"
        assert sent_messages[1]["reasoning_content"] == ""


class TestAnthropicCacheControl:
    """Tests for cache_control on Anthropic system blocks."""

    def test_cache_control_applied_based_on_cache_flag(self):
        """Test that cache_control is only added to system blocks with _cache=True."""
        from unittest.mock import MagicMock

        from nexau.archs.main_sub.execution.llm_caller import call_llm_with_anthropic_chat_completion

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="response")]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-3"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        kwargs = {
            "messages": [
                {"role": "system", "content": "static prompt"},
                {"role": "system", "content": "dynamic prompt"},
                {"role": "user", "content": "hello"},
            ],
            "model": "claude-3",
            "max_tokens": 100,
        }
        params = ModelCallParams(
            messages=messages_from_legacy_openai_chat(kwargs["messages"]),
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            openai_client=None,
            llm_config=None,
            retry_attempts=1,
            shutdown_event=None,
        )

        # Patch the Anthropic serializer to return system blocks with _cache flags
        with patch(
            "nexau.core.adapters.anthropic_messages.serialize_ump_to_anthropic_messages_payload",
            return_value=(
                [
                    {"type": "text", "text": "static prompt", "_cache": True},
                    {"type": "text", "text": "dynamic prompt", "_cache": False},
                ],
                [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            ),
        ):
            call_llm_with_anthropic_chat_completion(mock_client, kwargs, model_call_params=params)

        call_args = mock_client.messages.create.call_args
        system_blocks = call_args[1]["system"]

        # First block (cache=True) should have cache_control
        assert system_blocks[0].get("cache_control") == {"type": "ephemeral"}
        assert "_cache" not in system_blocks[0]

        # Second block (cache=False) should NOT have cache_control
        assert "cache_control" not in system_blocks[1]
        assert "_cache" not in system_blocks[1]

    def test_cache_control_defaults_to_true_when_no_flag(self):
        """Test that system blocks without _cache flag default to cached."""
        from unittest.mock import MagicMock

        from nexau.archs.main_sub.execution.llm_caller import call_llm_with_anthropic_chat_completion

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="response")]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-3"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        kwargs = {
            "messages": [
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "hello"},
            ],
            "model": "claude-3",
            "max_tokens": 100,
        }
        params = ModelCallParams(
            messages=messages_from_legacy_openai_chat(kwargs["messages"]),
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            openai_client=None,
            llm_config=None,
            retry_attempts=1,
            shutdown_event=None,
        )

        # System block without _cache flag (legacy single-string prompt)
        with patch(
            "nexau.core.adapters.anthropic_messages.serialize_ump_to_anthropic_messages_payload",
            return_value=(
                [{"type": "text", "text": "prompt"}],
                [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            ),
        ):
            call_llm_with_anthropic_chat_completion(mock_client, kwargs, model_call_params=params)

        call_args = mock_client.messages.create.call_args
        system_blocks = call_args[1]["system"]

        # Should default to cached
        assert system_blocks[0].get("cache_control") == {"type": "ephemeral"}
