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
- LLMConfig gemini_rest api_type support
- LLMCaller with gemini_rest api_type
"""

from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import (
    LLMCaller,
    call_llm_with_gemini_rest,
    convert_tools_to_gemini,
    openai_to_gemini_rest_messages,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
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
        """Test Gemini REST API call with thinking budget."""
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

        llm_config = LLMConfig(
            model="gemini-2.5-flash-thinking",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            api_type="gemini_rest",
            thinking_budget=1024,
        )

        kwargs = {"messages": [{"role": "user", "content": "Think about this"}]}

        call_llm_with_gemini_rest(kwargs, llm_config=llm_config)

        # Verify thinking config was included
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
