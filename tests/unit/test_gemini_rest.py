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
Unit tests for Gemini REST API support.

Tests cover:
- ModelResponse.from_gemini_rest parsing
- convert_tools_to_gemini conversion
- openai_to_gemini_rest_messages conversion
- call_llm_with_gemini_rest function
- GeminiRestStreamAggregator streaming aggregation
- Gemini REST streaming end-to-end
- LLMConfig gemini_rest api_type support
- LLMCaller with gemini_rest api_type
"""

import json
import threading
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution import llm_caller
from nexau.archs.main_sub.execution.llm_caller import (
    GeminiRestStreamAggregator,
    LLMCaller,
    _iter_gemini_sse_chunks,
    call_llm_with_gemini_rest,
    convert_tools_to_gemini,
    openai_to_gemini_rest_messages,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import SpanType
from nexau.core.messages import Message


class TestModelResponseFromGeminiRest:
    """Test cases for ModelResponse.from_gemini_rest."""

    def test_basic_text_response(self):
        """Test parsing a basic text response from Gemini."""
        response_json = {
            "candidates": [{"content": {"parts": [{"text": "Hello, how can I help you?"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 8, "totalTokenCount": 18},
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert result.content == "Hello, how can I help you?"
        assert result.role == "assistant"
        assert result.reasoning_content is None
        assert result.thought_signature is None
        assert len(result.tool_calls) == 0
        assert result.usage["input_tokens"] == 10
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 18

    def test_response_with_thinking(self):
        """Test parsing a response with thought content."""
        response_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Let me think about this...", "thought": True}, {"text": "The answer is 42."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 20, "thoughtsTokenCount": 10, "totalTokenCount": 45},
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert result.content == "The answer is 42."
        assert result.reasoning_content == "Let me think about this..."
        assert result.usage["reasoning_tokens"] == 10

    def test_response_with_thought_signature(self):
        """Test parsing a response with thought signature."""
        response_json = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Here is the answer.", "thoughtSignature": "sig_abc123"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert result.content == "Here is the answer."
        assert result.thought_signature == "sig_abc123"

    def test_response_with_function_call(self):
        """Test parsing a response with function call."""
        response_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "San Francisco", "unit": "celsius"}}}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 15, "totalTokenCount": 35},
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert result.content is None
        assert len(result.tool_calls) == 1

        tool_call = result.tool_calls[0]
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "San Francisco", "unit": "celsius"}
        assert tool_call.call_type == "function"

    def test_response_with_multiple_function_calls(self):
        """Test parsing a response with multiple function calls."""
        response_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I'll check the weather in both cities."},
                            {"functionCall": {"name": "get_weather", "args": {"location": "San Francisco"}}},
                            {"functionCall": {"name": "get_weather", "args": {"location": "New York"}}},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 25, "totalTokenCount": 55},
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert result.content == "I'll check the weather in both cities."
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].arguments["location"] == "San Francisco"
        assert result.tool_calls[1].arguments["location"] == "New York"

    def test_response_with_empty_function_args(self):
        """Test parsing a response with empty function arguments."""
        response_json = {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": "get_current_time", "args": {}}}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_current_time"
        assert result.tool_calls[0].arguments == {}

    def test_invalid_response_missing_candidates(self):
        """Test error handling for missing candidates."""
        response_json = {"usageMetadata": {}}

        with pytest.raises(ValueError, match="Invalid Gemini response"):
            ModelResponse.from_gemini_rest(response_json)

    def test_invalid_response_empty(self):
        """Test error handling for empty response."""
        with pytest.raises(ValueError, match="Invalid Gemini response"):
            ModelResponse.from_gemini_rest({})

    def test_invalid_response_none(self):
        """Test error handling for None response."""
        with pytest.raises(ValueError, match="Invalid Gemini response"):
            ModelResponse.from_gemini_rest(None)

    def test_usage_metadata_missing(self):
        """Test parsing response without usage metadata."""
        response_json = {
            "candidates": [{"content": {"parts": [{"text": "Response without usage"}], "role": "model"}, "finishReason": "STOP"}]
        }

        result = ModelResponse.from_gemini_rest(response_json)

        assert result.content == "Response without usage"
        assert result.usage["input_tokens"] == 0
        assert result.usage["completion_tokens"] == 0
        assert result.usage["total_tokens"] == 0

    def test_to_message_dict_with_thought_signature(self):
        """Test that thought_signature is preserved in to_message_dict."""
        response_json = {
            "candidates": [
                {"content": {"parts": [{"text": "Answer", "thoughtSignature": "sig_test"}], "role": "model"}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }

        result = ModelResponse.from_gemini_rest(response_json)
        message_dict = result.to_message_dict()

        assert message_dict["thought_signature"] == "sig_test"


class TestConvertToolsToGemini:
    """Test cases for convert_tools_to_gemini function."""

    def test_convert_single_tool(self):
        """Test converting a single OpenAI tool definition."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = convert_tools_to_gemini(openai_tools)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get the current weather for a location"
        assert result[0]["parameters"]["type"] == "object"
        assert "location" in result[0]["parameters"]["properties"]

    def test_convert_multiple_tools(self):
        """Test converting multiple OpenAI tool definitions."""
        openai_tools = [
            {
                "type": "function",
                "function": {"name": "tool1", "description": "First tool", "parameters": {"type": "object", "properties": {}}},
            },
            {
                "type": "function",
                "function": {"name": "tool2", "description": "Second tool", "parameters": {"type": "object", "properties": {}}},
            },
        ]

        result = convert_tools_to_gemini(openai_tools)

        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"

    def test_convert_tool_without_description(self):
        """Test converting a tool without description."""
        openai_tools = [{"type": "function", "function": {"name": "simple_tool", "parameters": {"type": "object", "properties": {}}}}]

        result = convert_tools_to_gemini(openai_tools)

        assert result[0]["description"] == ""

    def test_convert_tool_without_parameters(self):
        """Test converting a tool without parameters."""
        openai_tools = [{"type": "function", "function": {"name": "no_params_tool", "description": "A tool without parameters"}}]

        result = convert_tools_to_gemini(openai_tools)

        assert result[0]["parameters"] == {"type": "object", "properties": {}}

    def test_convert_non_function_tool_skipped(self):
        """Test that non-function tools are skipped."""
        openai_tools = [
            {"type": "retrieval", "retrieval": {}},
            {
                "type": "function",
                "function": {
                    "name": "valid_tool",
                    "description": "A valid function tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_gemini(openai_tools)

        assert len(result) == 1
        assert result[0]["name"] == "valid_tool"

    def test_convert_empty_tools_list(self):
        """Test converting an empty tools list."""
        result = convert_tools_to_gemini([])
        assert result == []

    def test_sanitize_strips_nested_additional_properties(self):
        """Test that additionalProperties and $schema are stripped recursively from nested objects."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_todos",
                    "description": "Write todos",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "List of todos",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "status": {"type": "string", "enum": ["pending", "done"]},
                                    },
                                    "required": ["description", "status"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["todos"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                },
            }
        ]

        result = convert_tools_to_gemini(openai_tools)

        params = result[0]["parameters"]
        # Top-level: no additionalProperties or $schema
        assert "additionalProperties" not in params
        assert "$schema" not in params
        # Nested items: no additionalProperties
        items = params["properties"]["todos"]["items"]
        assert "additionalProperties" not in items
        assert "required" in items
        assert items["properties"]["status"]["enum"] == ["pending", "done"]


class TestOpenaiToGeminiRestMessages:
    """Test cases for openai_to_gemini_rest_messages function."""

    def test_convert_user_message(self):
        """Test converting a user message."""
        messages = [{"role": "user", "content": "Hello, world!"}]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello, world!"
        assert system_instruction is None

    def test_convert_assistant_message(self):
        """Test converting an assistant message."""
        messages = [{"role": "assistant", "content": "I am here to help."}]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"][0]["text"] == "I am here to help."

    def test_convert_system_message(self):
        """Test converting a system message."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 0
        assert system_instruction is not None
        assert system_instruction["parts"][0]["text"] == "You are a helpful assistant."

    def test_convert_conversation(self):
        """Test converting a full conversation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 3
        assert system_instruction["parts"][0]["text"] == "You are helpful."
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"

    def test_convert_assistant_with_tool_calls(self):
        """Test converting an assistant message with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}}
                ],
            }
        ]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "model"
        # First part should be text, second should be functionCall
        assert contents[0]["parts"][0]["text"] == "Let me check that."
        assert "functionCall" in contents[0]["parts"][1]
        assert contents[0]["parts"][1]["functionCall"]["name"] == "get_weather"
        assert contents[0]["parts"][1]["functionCall"]["args"] == {"location": "NYC"}

    def test_convert_assistant_with_thought_signature(self):
        """Test converting an assistant message with thought signature."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "thought_signature": "sig_123",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}}],
            }
        ]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 1
        # Check that thought signature is attached to first function call
        function_call_part = None
        for part in contents[0]["parts"]:
            if "functionCall" in part:
                function_call_part = part
                break

        assert function_call_part is not None
        assert function_call_part.get("thoughtSignature") == "sig_123"

    def test_convert_tool_result_message(self):
        """Test converting a tool result message."""
        messages = [{"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "The weather is sunny, 72F."}]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert "functionResponse" in contents[0]["parts"][0]
        assert contents[0]["parts"][0]["functionResponse"]["name"] == "get_weather"
        assert contents[0]["parts"][0]["functionResponse"]["response"]["result"] == "The weather is sunny, 72F."

    def test_convert_multiple_tool_results_merged(self):
        """Test that multiple tool results in sequence are merged into one user turn."""
        messages = [
            {"role": "user", "content": "Check weather in two cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "LA"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "NYC: 72F"},
            {"role": "tool", "tool_call_id": "call_2", "name": "get_weather", "content": "LA: 85F"},
        ]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        # Should have: user message, model with tool calls, user with function responses
        assert len(contents) == 3

        # Last turn should have both function responses
        last_turn = contents[-1]
        assert last_turn["role"] == "user"
        function_responses = [p for p in last_turn["parts"] if "functionResponse" in p]
        assert len(function_responses) == 2

    def test_convert_tool_call_with_dict_arguments(self):
        """Test converting tool calls where arguments is already a dict."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "test"},  # Already a dict
                        },
                    }
                ],
            }
        ]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        fc_part = next(p for p in contents[0]["parts"] if "functionCall" in p)
        assert fc_part["functionCall"]["args"] == {"query": "test"}

    def test_convert_empty_content(self):
        """Test converting messages with empty content."""
        messages = [{"role": "user", "content": ""}]

        contents, system_instruction = openai_to_gemini_rest_messages(messages)

        # Empty content should result in empty parts or no text part
        assert len(contents) == 1
        assert contents[0]["role"] == "user"


class TestCallLLMWithGeminiRest:
    """Test cases for call_llm_with_gemini_rest function."""

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_successful_call(self, mock_post):
        """Test a successful Gemini REST API call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini!"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        llm_config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

        result = call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        assert result.content == "Hello from Gemini!"
        mock_post.assert_called_once()

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_with_tools(self, mock_post):
        """Test Gemini REST API call with tools."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10, "totalTokenCount": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        llm_config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        kwargs = {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        }

        result = call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

        # Verify tools were included in request
        call_args = mock_post.call_args
        request_body = call_args.kwargs["json"]
        assert "tools" in request_body

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_with_thinking_budget(self, mock_post):
        """Test Gemini REST API call with thinking config via extra_params."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Thinking...", "thought": True}, {"text": "Result"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15, "thoughtsTokenCount": 5, "totalTokenCount": 30},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # thinkingConfig is passed via extra_params (LLMConfig **kwargs)
        llm_config = LLMConfig(
            model="gemini-2.5-flash-thinking",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            api_type="gemini_rest",
            thinkingConfig={"thoughtBudgetTokens": 1024},
        )

        kwargs = {"messages": [{"role": "user", "content": "Think about this"}]}

        call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        # Verify thinking config from extra_params was included in request
        call_args = mock_post.call_args
        request_body = call_args.kwargs["json"]
        assert "thinkingConfig" in request_body["generationConfig"]
        assert request_body["generationConfig"]["thinkingConfig"]["thoughtBudgetTokens"] == 1024

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_with_system_instruction(self, mock_post):
        """Test Gemini REST API call with system instruction."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "I am helpful!"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5, "totalTokenCount": 20},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        llm_config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        kwargs = {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi"}]}

        call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        call_args = mock_post.call_args
        request_body = call_args.kwargs["json"]
        assert "systemInstruction" in request_body
        assert request_body["systemInstruction"]["parts"][0]["text"] == "You are a helpful assistant."

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_api_error(self, mock_post):
        """Test error handling for API errors."""
        mock_post.side_effect = Exception("API Error")

        llm_config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

        with pytest.raises(Exception, match="API Error"):
            call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

    def test_call_without_llm_config(self):
        """Test error handling when llm_config is not provided."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}

        with pytest.raises(ValueError, match="llm_config is required"):
            call_llm_with_gemini_rest(kwargs, llm_config=None)

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_uses_default_base_url_when_none(self, mock_post):
        """Test that default Gemini base URL is used when base_url is None after init."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Create LLMConfig with explicit base_url, then set it to None to test fallback
        llm_config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )
        # Manually set to None to test the fallback behavior in call_llm_with_gemini_rest
        llm_config.base_url = None

        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

        call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        call_args = mock_post.call_args
        url = call_args.args[0]
        assert "generativelanguage.googleapis.com" in url


class TestLLMConfigGeminiRest:
    """Test cases for LLMConfig with gemini_rest api_type."""

    def test_gemini_rest_api_type(self):
        """Test LLMConfig with gemini_rest api_type."""
        config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        assert config.api_type == "gemini_rest"
        assert config.model == "gemini-2.5-flash"

    def test_to_openai_params_with_thinking_budget(self):
        """Test to_openai_params includes thinking budget for gemini_rest."""
        config = LLMConfig(
            model="gemini-2.5-flash-thinking",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            api_type="gemini_rest",
            thinking_budget=2048,
            include_thoughts=True,
        )

        params = config.to_openai_params()

        # thinking_budget and include_thoughts are passed as extra_params
        # and should be included in to_openai_params for gemini_rest
        assert params.get("thinking_budget") == 2048
        assert params.get("include_thoughts") is True

    def test_to_openai_params_without_thinking_budget(self):
        """Test to_openai_params without thinking budget for gemini_rest."""
        config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        params = config.to_openai_params()

        assert "thinking_budget" not in params


class TestLLMCallerGeminiRest:
    """Test cases for LLMCaller with gemini_rest api_type."""

    @pytest.fixture
    def gemini_llm_config(self):
        """LLM config for Gemini REST API."""
        return LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_llm_with_gemini_rest_api_type(self, mock_post, gemini_llm_config, agent_state):
        """Test LLMCaller uses gemini_rest when api_type is gemini_rest."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini response"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        caller = LLMCaller(
            openai_client=None,  # No OpenAI client needed for gemini_rest
            llm_config=gemini_llm_config,
        )

        messages = [Message.user("Hello")]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response is not None
        assert response.content == "Gemini response"
        mock_post.assert_called_once()

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_call_llm_without_client_does_not_raise_for_gemini_rest(self, mock_post, gemini_llm_config, agent_state):
        """Test that gemini_rest doesn't require OpenAI client."""
        caller = LLMCaller(openai_client=None, llm_config=gemini_llm_config)

        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Should not raise RuntimeError about missing OpenAI client
        messages = [Message.user("Hello")]
        response = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert response is not None


class TestGeminiRestIntegration:
    """Integration tests for Gemini REST API support."""

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_full_conversation_flow(self, mock_post, agent_state):
        """Test a full conversation flow with Gemini REST API."""
        # First response: assistant with tool call
        first_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me check the weather.", "thoughtSignature": "sig_001"},
                            {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 15, "totalTokenCount": 35},
        }

        # Second response: final answer
        second_response = {
            "candidates": [
                {"content": {"parts": [{"text": "The weather in NYC is sunny and 72F."}], "role": "model"}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 10, "totalTokenCount": 50},
        }

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = [first_response, second_response]
        mock_post.return_value = mock_response

        llm_config = LLMConfig(
            model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com", api_key="test-key", api_type="gemini_rest"
        )

        caller = LLMCaller(openai_client=None, llm_config=llm_config)

        # First turn
        messages = [Message.user("What's the weather in NYC?")]
        first_result = caller.call_llm(messages, max_tokens=100, force_stop_reason=AgentStopReason.SUCCESS, agent_state=agent_state)

        assert first_result.content == "Let me check the weather."
        assert len(first_result.tool_calls) == 1
        assert first_result.thought_signature == "sig_001"

        # The thought_signature should be preserved in message dict
        message_dict = first_result.to_message_dict()
        assert message_dict["thought_signature"] == "sig_001"


# ---------------------------------------------------------------------------
# Helpers for streaming tests
# ---------------------------------------------------------------------------


def _make_sse_response(chunks: list[dict]) -> Mock:
    """Create a mock requests.Response that yields SSE data lines."""
    lines: list[bytes] = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}".encode())
        lines.append(b"")
    mock_resp = Mock()
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.raise_for_status = Mock()
    return mock_resp


# ---------------------------------------------------------------------------
# GeminiRestStreamAggregator unit tests
# ---------------------------------------------------------------------------


class TestGeminiRestStreamAggregator:
    """Test cases for GeminiRestStreamAggregator."""

    def test_basic_text_streaming(self):
        """Test text concatenation across multiple chunks."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]})
        agg.consume({"candidates": [{"content": {"parts": [{"text": "world!"}]}}]})

        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        assert "".join(text_parts) == "Hello world!"

    def test_thinking_content_streaming(self):
        """Test thought parts go to reasoning, regular text to content."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"text": "Let me think...", "thought": True}]}}]})
        agg.consume({"candidates": [{"content": {"parts": [{"text": "The answer is 42."}]}}]})

        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        reasoning = [p for p in parts if p.get("thought")]
        content = [p for p in parts if "text" in p and not p.get("thought")]
        assert len(reasoning) == 1
        assert reasoning[0]["text"] == "Let me think..."
        assert len(content) == 1
        assert content[0]["text"] == "The answer is 42."

    def test_thought_signature_streaming(self):
        """Test thoughtSignature is captured from chunks."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"text": "Answer", "thoughtSignature": "sig_abc"}]}}]})

        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        sig_parts = [p for p in parts if "thoughtSignature" in p]
        assert len(sig_parts) == 1
        assert sig_parts[0]["thoughtSignature"] == "sig_abc"

    def test_function_call_streaming(self):
        """Test functionCall parts are collected."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}]}}]})

        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        fc_parts = [p for p in parts if "functionCall" in p]
        assert len(fc_parts) == 1
        assert fc_parts[0]["functionCall"]["name"] == "get_weather"

    def test_multiple_function_calls_streaming(self):
        """Test multiple function calls across chunks."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"functionCall": {"name": "tool_a", "args": {}}}]}}]})
        agg.consume({"candidates": [{"content": {"parts": [{"functionCall": {"name": "tool_b", "args": {}}}]}}]})

        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        fc_parts = [p for p in parts if "functionCall" in p]
        assert len(fc_parts) == 2
        assert fc_parts[0]["functionCall"]["name"] == "tool_a"
        assert fc_parts[1]["functionCall"]["name"] == "tool_b"

    def test_mixed_content_streaming(self):
        """Test thinking + text + function call + signature in one stream."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"text": "Thinking...", "thought": True}]}}]})
        agg.consume({"candidates": [{"content": {"parts": [{"text": "I'll search.", "thoughtSignature": "sig_1"}]}}]})
        agg.consume({"candidates": [{"content": {"parts": [{"functionCall": {"name": "search", "args": {"q": "test"}}}]}}]})

        result = agg.finalize()
        resp = ModelResponse.from_gemini_rest(result)
        assert resp.reasoning_content == "Thinking..."
        assert resp.content == "I'll search."
        assert resp.thought_signature == "sig_1"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"

    def test_usage_metadata_from_final_chunk(self):
        """Test that the last chunk's usageMetadata wins."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]})
        agg.consume(
            {
                "candidates": [{"content": {"parts": [{"text": "!"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            }
        )

        result = agg.finalize()
        assert result["usageMetadata"]["promptTokenCount"] == 10
        assert result["usageMetadata"]["totalTokenCount"] == 15

    def test_empty_stream_raises(self):
        """Test finalize raises RuntimeError with no chunks consumed."""
        agg = GeminiRestStreamAggregator()
        with pytest.raises(RuntimeError, match="No stream chunks"):
            agg.finalize()

    def test_finalize_compatible_with_from_gemini_rest(self):
        """Test round-trip: finalize() output -> ModelResponse.from_gemini_rest()."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]})
        agg.consume(
            {
                "candidates": [{"content": {"parts": [{"text": " world"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
            }
        )

        result = agg.finalize()
        resp = ModelResponse.from_gemini_rest(result)
        assert resp.content == "Hello world"
        assert resp.usage["input_tokens"] == 5
        assert resp.usage["total_tokens"] == 8

    def test_empty_candidates_skipped(self):
        """Test chunks with empty candidates are silently skipped."""
        agg = GeminiRestStreamAggregator()
        agg.consume({"candidates": []})
        agg.consume({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})

        result = agg.finalize()
        parts = result["candidates"][0]["content"]["parts"]
        assert any(p.get("text") == "OK" for p in parts)


# ---------------------------------------------------------------------------
# _iter_gemini_sse_chunks unit tests
# ---------------------------------------------------------------------------


class TestIterGeminiSseChunks:
    """Test cases for _iter_gemini_sse_chunks SSE parser."""

    def test_basic_parsing(self):
        """Test parsing well-formed SSE lines."""
        mock_resp = _make_sse_response([{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}])
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1
        assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "hi"

    def test_multiple_chunks(self):
        """Test parsing multiple SSE data lines."""
        mock_resp = _make_sse_response([{"a": 1}, {"b": 2}])
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 2

    def test_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = iter([b"", b'data: {"ok": true}', b"", b""])
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1

    def test_skips_non_data_lines(self):
        """Test that non-data SSE fields (event:, id:, retry:) are skipped."""
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = iter(
            [
                b"event: message",
                b'data: {"ok": true}',
                b"id: 123",
            ]
        )
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1

    def test_malformed_json_skipped(self):
        """Test that malformed JSON lines are skipped with a warning."""
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = iter(
            [
                b"data: {invalid json}",
                b'data: {"ok": true}',
            ]
        )
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1
        assert chunks[0]["ok"] is True

    def test_data_without_space(self):
        """Test parsing SSE lines with 'data:' (no space after colon)."""
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = iter([b'data:{"ok": true}', b'data: {"also": true}'])
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 2
        assert chunks[0]["ok"] is True
        assert chunks[1]["also"] is True

    def test_json_array_fallback(self):
        """Test fallback to JSON array parsing when no SSE data: lines found."""
        json_array = json.dumps(
            [
                {"candidates": [{"content": {"parts": [{"text": "a"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "b"}]}}]},
            ]
        )
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = iter([line.encode() for line in json_array.splitlines()])
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 2
        assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "a"

    def test_json_single_object_fallback(self):
        """Test fallback to single JSON object when no SSE data: lines found."""
        obj = {"candidates": [{"content": {"parts": [{"text": "single"}]}}]}
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = iter([json.dumps(obj).encode()])
        chunks = list(_iter_gemini_sse_chunks(mock_resp))
        assert len(chunks) == 1
        assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "single"


# call_llm_with_gemini_rest streaming end-to-end tests
# ---------------------------------------------------------------------------


class TestCallLLMWithGeminiRestStreaming:
    """Test cases for call_llm_with_gemini_rest streaming path."""

    @pytest.fixture
    def gemini_llm_config(self):
        return LLMConfig(
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            api_type="gemini_rest",
        )

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_basic_text(self, mock_post, gemini_llm_config):
        """Test end-to-end streaming returns correct ModelResponse."""
        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]},
                {
                    "candidates": [{"content": {"parts": [{"text": "world!"}]}}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
                },
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        result = call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config)

        assert result.content == "Hello world!"
        assert result.usage["input_tokens"] == 5

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_url_uses_stream_endpoint(self, mock_post, gemini_llm_config):
        """Test URL contains :streamGenerateContent with alt=sse."""
        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]},
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config)

        url = mock_post.call_args.args[0]
        assert ":streamGenerateContent" in url
        assert "alt=sse" in url

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_thinking(self, mock_post, gemini_llm_config):
        """Test streaming with thinking content."""
        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "Hmm...", "thought": True}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "42"}]}}]},
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Think"}], "stream": True}
        result = call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config)

        assert result.reasoning_content == "Hmm..."
        assert result.content == "42"

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_function_calls(self, mock_post, gemini_llm_config):
        """Test streaming with function calls."""
        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"functionCall": {"name": "search", "args": {"q": "test"}}}]}}]},
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Search"}], "stream": True}
        result = call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_api_error(self, mock_post, gemini_llm_config):
        """Test HTTP error propagation during streaming."""
        import requests as req

        mock_post.side_effect = req.exceptions.ConnectionError("Connection refused")

        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        with pytest.raises(req.exceptions.ConnectionError):
            call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config)

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_shutdown_event(self, mock_post, gemini_llm_config):
        """Test shutdown_event interrupts streaming early."""
        from nexau.archs.main_sub.execution.hooks import ModelCallParams

        # Two chunks, but shutdown fires after first
        shutdown_ev = threading.Event()
        call_count = 0

        original_response = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "First"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "Second"}]}}]},
            ]
        )

        # Wrap iter_lines to set shutdown after first data line
        original_iter = list(original_response.iter_lines.return_value)

        def iter_with_shutdown():
            nonlocal call_count
            for line in original_iter:
                yield line
                if line and line.startswith(b"data:"):
                    call_count += 1
                    if call_count >= 1:
                        shutdown_ev.set()

        original_response.iter_lines.return_value = iter_with_shutdown()
        mock_post.return_value = original_response

        params = ModelCallParams(
            messages=[],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            openai_client=None,
            llm_config=gemini_llm_config,
            retry_attempts=1,
            shutdown_event=shutdown_ev,
        )

        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        result = call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config, model_call_params=params)

        # Should have partial content (only first chunk)
        assert result.content == "First"

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_via_llm_config(self, mock_post):
        """Test llm_config.stream = True triggers streaming."""
        llm_config = LLMConfig(
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            api_type="gemini_rest",
            stream=True,
        )

        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "Streamed"}]}}]},
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Hi"}]}
        result = call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        assert result.content == "Streamed"
        url = mock_post.call_args.args[0]
        assert ":streamGenerateContent" in url

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_custom_base_url(self, mock_post):
        """Test custom base URL uses correct streaming endpoint."""
        llm_config = LLMConfig(
            model="gemini-2.5-flash",
            base_url="https://custom-proxy.example.com",
            api_key="test-key",
            api_type="gemini_rest",
        )

        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]},
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        url = mock_post.call_args.args[0]
        assert url.startswith("https://custom-proxy.example.com/models/gemini-2.5-flash:streamGenerateContent")
        assert "alt=sse" in url

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_passes_stream_true_to_requests(self, mock_post, gemini_llm_config):
        """Test requests.post is called with stream=True."""
        mock_post.return_value = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]},
            ]
        )

        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config)

        assert mock_post.call_args.kwargs["stream"] is True


class TestGeminiRestEventAggregator:
    """Test cases for GeminiRestEventAggregator event emission."""

    @pytest.fixture
    def events_captured(self):
        return []

    @pytest.fixture
    def aggregator(self, events_captured):
        from nexau.archs.llm.llm_aggregators.gemini_rest import GeminiRestEventAggregator

        return GeminiRestEventAggregator(
            on_event=lambda e: events_captured.append(e),
            run_id="test_run_1",
        )

    def test_basic_text_emits_start_content_end(self, aggregator, events_captured):
        """Text chunks should emit START → CONTENT → END."""
        from nexau.archs.llm.llm_aggregators.events import (
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
        )

        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]})
        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": " world"}]}, "finishReason": "STOP"}]})

        assert len(events_captured) == 4
        assert isinstance(events_captured[0], TextMessageStartEvent)
        assert events_captured[0].role == "assistant"
        assert events_captured[0].run_id == "test_run_1"
        assert isinstance(events_captured[1], TextMessageContentEvent)
        assert events_captured[1].delta == "Hello"
        assert isinstance(events_captured[2], TextMessageContentEvent)
        assert events_captured[2].delta == " world"
        assert isinstance(events_captured[3], TextMessageEndEvent)

    def test_thinking_emits_thinking_events(self, aggregator, events_captured):
        """Thinking parts (thought=true) should emit thinking lifecycle events."""
        from nexau.archs.llm.llm_aggregators.events import (
            ThinkingTextMessageContentEvent,
            ThinkingTextMessageEndEvent,
            ThinkingTextMessageStartEvent,
        )

        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "Let me think...", "thought": True}]}}]})
        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "done thinking", "thought": True}]}, "finishReason": "STOP"}]})

        # START(msg) + START(thinking) + CONTENT(thinking) + CONTENT(thinking) + END(thinking) + END(msg)
        thinking_events = [e for e in events_captured if "Thinking" in type(e).__name__]
        assert len(thinking_events) == 4
        assert isinstance(thinking_events[0], ThinkingTextMessageStartEvent)
        assert thinking_events[0].run_id == "test_run_1"
        assert isinstance(thinking_events[1], ThinkingTextMessageContentEvent)
        assert thinking_events[1].delta == "Let me think..."
        assert isinstance(thinking_events[2], ThinkingTextMessageContentEvent)
        assert thinking_events[2].delta == "done thinking"
        assert isinstance(thinking_events[3], ThinkingTextMessageEndEvent)

    def test_function_call_emits_tool_events(self, aggregator, events_captured):
        """Function calls should emit TextMessageStart + ToolCall events + TextMessageEnd."""
        from nexau.archs.llm.llm_aggregators.events import (
            TextMessageEndEvent,
            TextMessageStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallStartEvent,
        )

        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}]},
                    "finishReason": "STOP",
                }
            ]
        }
        aggregator.aggregate(chunk)

        assert isinstance(events_captured[0], TextMessageStartEvent)

        tool_events = [e for e in events_captured if "ToolCall" in type(e).__name__]
        assert len(tool_events) == 3
        assert isinstance(tool_events[0], ToolCallStartEvent)
        assert tool_events[0].tool_call_name == "get_weather"
        assert tool_events[0].tool_call_id == "gemini_tc_0"
        assert isinstance(tool_events[1], ToolCallArgsEvent)
        assert json.loads(tool_events[1].delta) == {"city": "Tokyo"}
        assert isinstance(tool_events[2], ToolCallEndEvent)

        assert isinstance(events_captured[-1], TextMessageEndEvent)

    def test_mixed_thinking_text_and_function_call(self, aggregator, events_captured):
        """Mixed content should emit all event types in correct order."""
        # Chunk 1: thinking
        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}]}}]})
        # Chunk 2: text + function call + finish
        aggregator.aggregate(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "I'll help."},
                                {"functionCall": {"name": "search", "args": {"q": "test"}}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )

        type_names = [type(e).__name__ for e in events_captured]
        assert "TextMessageStartEvent" in type_names
        assert "ThinkingTextMessageStartEvent" in type_names
        assert "ThinkingTextMessageContentEvent" in type_names
        assert "TextMessageContentEvent" in type_names
        assert "ToolCallStartEvent" in type_names
        assert "ToolCallArgsEvent" in type_names
        assert "ToolCallEndEvent" in type_names
        assert "ThinkingTextMessageEndEvent" in type_names
        assert "TextMessageEndEvent" in type_names

    def test_empty_candidates_ignored(self, aggregator, events_captured):
        """Chunks with no candidates should be silently ignored."""
        aggregator.aggregate({"candidates": []})
        aggregator.aggregate({})
        assert len(events_captured) == 0

    def test_no_content_with_finish_reason(self, aggregator, events_captured):
        """A chunk with finishReason but no content should still emit end events if started."""
        from nexau.archs.llm.llm_aggregators.events import TextMessageEndEvent, TextMessageStartEvent

        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]})
        aggregator.aggregate({"candidates": [{"finishReason": "STOP"}]})

        assert isinstance(events_captured[0], TextMessageStartEvent)
        assert isinstance(events_captured[-1], TextMessageEndEvent)

    def test_clear_resets_state(self, aggregator, events_captured):
        """clear() should reset all state so the aggregator can be reused."""
        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "first"}]}, "finishReason": "STOP"}]})
        first_msg_id = events_captured[0].message_id
        events_captured.clear()

        aggregator.clear()
        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "second"}]}, "finishReason": "STOP"}]})
        second_msg_id = events_captured[0].message_id

        assert first_msg_id != second_msg_id

    def test_multiple_function_calls_get_unique_ids(self, aggregator, events_captured):
        """Each function call should get a unique tool_call_id."""
        from nexau.archs.llm.llm_aggregators.events import ToolCallStartEvent

        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "tool_a", "args": {}}},
                            {"functionCall": {"name": "tool_b", "args": {}}},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        aggregator.aggregate(chunk)

        start_events = [e for e in events_captured if isinstance(e, ToolCallStartEvent)]
        assert len(start_events) == 2
        assert start_events[0].tool_call_id != start_events[1].tool_call_id

    def test_message_id_consistency(self, aggregator, events_captured):
        """All events from the same stream should share the same message_id."""
        from nexau.archs.llm.llm_aggregators.events import (
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
        )

        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "a"}]}}]})
        aggregator.aggregate({"candidates": [{"content": {"parts": [{"text": "b"}]}, "finishReason": "STOP"}]})

        start = [e for e in events_captured if isinstance(e, TextMessageStartEvent)][0]
        contents = [e for e in events_captured if isinstance(e, TextMessageContentEvent)]
        end = [e for e in events_captured if isinstance(e, TextMessageEndEvent)][0]

        assert all(c.message_id == start.message_id for c in contents)
        assert end.message_id == start.message_id


class TestGeminiRestMiddlewareIntegration:
    """Test Gemini REST chunk handling in AgentEventsMiddleware."""

    @pytest.fixture
    def events_captured(self):
        return []

    @pytest.fixture
    def middleware(self, events_captured):
        from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware

        return AgentEventsMiddleware(session_id="test_session", on_event=lambda e: events_captured.append(e))

    @pytest.fixture
    def model_call_params(self):
        state = Mock()
        state.run_id = "run_gemini_1"
        params = Mock(spec=["agent_state"])
        params.agent_state = state
        return params

    def test_gemini_chunk_triggers_aggregator(self, middleware, events_captured, model_call_params):
        """Gemini dict chunks should be detected and processed by the aggregator."""
        from nexau.archs.llm.llm_aggregators.events import TextMessageContentEvent, TextMessageStartEvent

        chunk = {"candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}, "finishReason": "STOP"}]}
        middleware.stream_chunk(chunk, model_call_params)

        assert len(events_captured) >= 2
        assert isinstance(events_captured[0], TextMessageStartEvent)
        assert isinstance(events_captured[1], TextMessageContentEvent)
        assert events_captured[1].delta == "Hello from Gemini"

    def test_gemini_chunk_returns_chunk_unchanged(self, middleware, model_call_params):
        """stream_chunk should return the Gemini chunk unchanged."""
        chunk = {"candidates": [{"content": {"parts": [{"text": "test"}]}}]}
        result = middleware.stream_chunk(chunk, model_call_params)
        assert result is chunk

    def test_gemini_aggregator_lazy_init(self, middleware, model_call_params):
        """GeminiRestEventAggregator should be lazily initialized."""
        assert middleware._gemini_rest_aggregator is None
        chunk = {"candidates": [{"content": {"parts": [{"text": "init"}]}}]}
        middleware.stream_chunk(chunk, model_call_params)
        assert middleware._gemini_rest_aggregator is not None

    def test_is_gemini_rest_chunk_type_guard(self):
        """is_gemini_rest_chunk should correctly identify Gemini chunks."""
        from nexau.archs.main_sub.execution.middleware.agent_events_middleware import is_gemini_rest_chunk

        assert is_gemini_rest_chunk({"candidates": [{"content": {"parts": []}}]}) is True
        assert is_gemini_rest_chunk({"candidates": []}) is True
        assert is_gemini_rest_chunk({"other_key": "value"}) is False
        assert is_gemini_rest_chunk("not a dict") is False
        assert is_gemini_rest_chunk(42) is False


# ---------------------------------------------------------------------------
# Gemini REST streaming with tracing (lines 2126-2164 of llm_caller.py)
# ---------------------------------------------------------------------------


class TestCallLLMWithGeminiRestStreamingTracing:
    """Test the tracing branch of do_stream_request in call_llm_with_gemini_rest."""

    @pytest.fixture
    def gemini_llm_config(self):
        return LLMConfig(
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            api_type="gemini_rest",
        )

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_tracer_creates_span(self, mock_post, gemini_llm_config):
        """Tracing path creates a span with correct name, type, and inputs."""
        mock_post.return_value = _make_sse_response([{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}])
        tracer = InMemoryTracer()
        kwargs = {"messages": [{"role": "user", "content": "hello"}], "stream": True}

        with TraceContext(tracer, "parent", SpanType.AGENT):
            result = call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config, tracer=tracer)

        assert result.content == "hi"
        llm_spans = [s for s in tracer.spans.values() if s.name == "Gemini REST streamGenerateContent"]
        assert len(llm_spans) == 1
        assert llm_spans[0].type == SpanType.LLM
        assert "contents" in llm_spans[0].inputs

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_tracer_records_outputs(self, mock_post, gemini_llm_config):
        """Tracing path records finalized aggregator result as span outputs."""
        mock_post.return_value = _make_sse_response([{"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}])
        tracer = InMemoryTracer()
        kwargs = {"messages": [{"role": "user", "content": "hi"}], "stream": True}

        with TraceContext(tracer, "parent", SpanType.AGENT):
            call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config, tracer=tracer)

        llm_spans = [s for s in tracer.spans.values() if s.name == "Gemini REST streamGenerateContent"]
        assert len(llm_spans) == 1
        assert "candidates" in llm_spans[0].outputs

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_tracer_records_time_to_first_token_ms(self, mock_post, gemini_llm_config, monkeypatch: pytest.MonkeyPatch):
        """Tracing path records time_to_first_token_ms attribute on the span."""
        it = iter([1000.0, 1000.3])
        monkeypatch.setattr(llm_caller.time, "time", lambda: next(it))

        mock_post.return_value = _make_sse_response([{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}])
        tracer = InMemoryTracer()
        kwargs = {"messages": [{"role": "user", "content": "hello"}], "stream": True}

        with TraceContext(tracer, "parent", SpanType.AGENT):
            call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config, tracer=tracer)

        llm_spans = [s for s in tracer.spans.values() if s.name == "Gemini REST streamGenerateContent"]
        assert len(llm_spans) == 1
        assert abs(float(llm_spans[0].attributes["time_to_first_token_ms"]) - 300.0) < 1e-6

    @patch("nexau.archs.main_sub.execution.llm_caller.requests.post")
    def test_streaming_with_tracer_shutdown_event_interrupts(self, mock_post, gemini_llm_config):
        """Shutdown event mid-stream still ends the span and returns partial content."""
        from nexau.archs.main_sub.execution.hooks import ModelCallParams

        shutdown_ev = threading.Event()
        call_count = 0

        original_response = _make_sse_response(
            [
                {"candidates": [{"content": {"parts": [{"text": "First"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "Second"}]}}]},
            ]
        )
        original_iter = list(original_response.iter_lines.return_value)

        def iter_with_shutdown():
            nonlocal call_count
            for line in original_iter:
                yield line
                if line and line.startswith(b"data:"):
                    call_count += 1
                    if call_count >= 1:
                        shutdown_ev.set()

        original_response.iter_lines.return_value = iter_with_shutdown()
        mock_post.return_value = original_response

        params = ModelCallParams(
            messages=[],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            openai_client=None,
            llm_config=gemini_llm_config,
            retry_attempts=1,
            shutdown_event=shutdown_ev,
        )

        tracer = InMemoryTracer()
        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

        with TraceContext(tracer, "parent", SpanType.AGENT):
            result = call_llm_with_gemini_rest(kwargs, llm_config=gemini_llm_config, tracer=tracer, model_call_params=params)

        assert result.content == "First"
        llm_spans = [s for s in tracer.spans.values() if s.name == "Gemini REST streamGenerateContent"]
        assert len(llm_spans) == 1
