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
Unit tests for OpenAI chat completion aggregator.
"""

from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice as ChatCompletionChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
)

from nexau.archs.llm.llm_aggregators import OpenAIChatCompletionAggregator
from nexau.archs.llm.llm_aggregators.openai_chat_completion.openai_chat_completion_aggregator import _ToolCallAggregator


class TestOpenAIChatCompletionAggregator:
    """Test cases for OpenAI chat completion aggregator."""

    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        assert aggregator._on_event == mock_on_event
        assert aggregator._choice_aggregators == {}
        assert aggregator._value.id == ""
        assert aggregator._value.created == 0
        assert aggregator._value.model == ""
        assert aggregator._value.service_tier is None
        assert aggregator._value.system_fingerprint is None
        assert aggregator._value.usage is None

    def test_aggregate_single_content_chunk(self):
        """Test aggregating a single content chunk."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk)

        # Check metadata
        assert aggregator._value.id == "chatcmpl-123"
        assert aggregator._value.created == 1234567890
        assert aggregator._value.model == "gpt-4o-mini"

        # Check choice aggregator was created
        assert 0 in aggregator._choice_aggregators

    def test_aggregate_multiple_content_chunks(self):
        """Test aggregating multiple content chunks."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        chunk2 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=" World!"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        aggregator.aggregate(chunk2)

        result = aggregator.build()

        assert result.id == "chatcmpl-123"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello World!"
        assert result.choices[0].finish_reason == "stop"

        # Verify TextMessageContentEvent was also emitted
        content_events = [call for call in mock_on_event.call_args_list if call[0][0].__class__.__name__ == "TextMessageContentEvent"]
        assert len(content_events) == 2  # One for "Hello", one for " World!"
        for event_call in content_events:
            event = event_call[0][0]
            assert event.message_id == "chatcmpl-123"
            assert event.delta in ["Hello", " World!"]
            assert event.timestamp is not None  # timestamp should be set

    def test_aggregate_with_tool_calls(self):
        """Test aggregating chunks with tool calls."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction

        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc123",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        chunk2 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(arguments='{"location": "Beijing"}'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        aggregator.aggregate(chunk2)

        result = aggregator.build()

        assert len(result.choices) == 1
        choice = result.choices[0]
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        tool_call = choice.message.tool_calls[0]
        assert tool_call.id == "call_abc123"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"location": "Beijing"}'

    def test_clear_resets_state(self):
        """Test that clear() resets aggregator state."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk)
        aggregator.clear()

        assert aggregator._choice_aggregators == {}
        assert aggregator._value.id == ""
        assert aggregator._value.created == 0
        assert aggregator._value.model == ""
        assert aggregator._value.service_tier is None
        assert aggregator._value.system_fingerprint is None
        assert aggregator._value.usage is None

    def test_aggregation_with_different_finish_reasons(self):
        """Test aggregating chunks with different finish reasons."""
        finish_reasons = ["stop", "length", "tool_calls", "content_filter", "function_call"]

        for reason in finish_reasons:
            mock_on_event = Mock()
            aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

            chunk = ChatCompletionChunk(
                id="chatcmpl-123",
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChoiceDelta(role="assistant", content="Hello"),
                        finish_reason=reason,
                    )
                ],
                created=1234567890,
                model="gpt-4o-mini",
                object="chat.completion.chunk",
            )

            aggregator.aggregate(chunk)
            result = aggregator.build()

            assert result.choices[0].finish_reason == reason

    def test_aggregation_with_refusal(self):
        """Test aggregating chunks with refusal content."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=None, refusal="I cannot"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        chunk2 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(refusal=" answer that."),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        aggregator.aggregate(chunk2)
        result = aggregator.build()

        assert result.choices[0].message.refusal == "I cannot answer that."

    def test_aggregation_with_system_fingerprint(self):
        """Test aggregating chunks with system fingerprint."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            system_fingerprint="fp_abc123",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk)
        result = aggregator.build()

        assert result.system_fingerprint == "fp_abc123"

    def test_aggregation_with_usage(self):
        """Test aggregating chunks with usage statistics."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        from openai.types.completion_usage import CompletionUsage

        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk)
        result = aggregator.build()

        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_aggregation_multiple_tool_calls(self):
        """Test aggregating multiple tool calls in different chunks."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction

        # First chunk with first tool call
        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc123",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"location": "Beijing"}'),
                            ),
                            ChoiceDeltaToolCall(
                                index=1,
                                id="call_def456",
                                function=ChoiceDeltaToolCallFunction(name="get_time", arguments='{"tz": "Asia/Shanghai"}'),
                            ),
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        result = aggregator.build()

        # Verify two tool calls
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) >= 1

    def test_build_without_valid_chunks_raises_error(self):
        """Test that build() raises RuntimeError when no valid chunks were received."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        # Try to build without aggregating any chunks
        with pytest.raises(RuntimeError, match="Chat completion stream did not receive any valid chunks"):
            aggregator.build()

    def test_aggregation_with_logprobs(self):
        """Test aggregating chunks with logprobs content."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        from openai.types.chat.chat_completion_chunk import ChoiceLogprobs
        from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                    logprobs=ChoiceLogprobs(
                        content=[ChatCompletionTokenLogprob(token="Hello", logprob=-0.5, bytes=[72, 101, 108, 108, 111], top_logprobs=[])]
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        chunk2 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=" World!"),
                    finish_reason=None,
                    logprobs=ChoiceLogprobs(
                        content=[
                            ChatCompletionTokenLogprob(
                                token=" World!", logprob=-0.7, bytes=[32, 87, 111, 114, 108, 100, 33], top_logprobs=[]
                            )
                        ]
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        aggregator.aggregate(chunk2)
        result = aggregator.build()

        # Verify logprobs were aggregated
        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert len(logprobs.content) == 2
        assert logprobs.content[0].token == "Hello"
        assert logprobs.content[1].token == " World!"

    def test_aggregation_with_refusal_logprobs(self):
        """Test aggregating chunks with refusal logprobs."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        from openai.types.chat.chat_completion_chunk import ChoiceLogprobs
        from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=None, refusal="I cannot"),
                    finish_reason=None,
                    logprobs=ChoiceLogprobs(
                        refusal=[
                            ChatCompletionTokenLogprob(token="I", logprob=-0.1, bytes=[73], top_logprobs=[]),
                            ChatCompletionTokenLogprob(
                                token=" cannot", logprob=-0.2, bytes=[32, 99, 97, 110, 110, 111, 116], top_logprobs=[]
                            ),
                        ]
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        chunk2 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(refusal=" answer that."),
                    finish_reason="stop",
                    logprobs=ChoiceLogprobs(
                        refusal=[
                            ChatCompletionTokenLogprob(
                                token=" answer", logprob=-0.3, bytes=[32, 97, 110, 115, 119, 101, 114], top_logprobs=[]
                            ),
                            ChatCompletionTokenLogprob(token=" that.", logprob=-0.4, bytes=[32, 116, 104, 97, 116, 46], top_logprobs=[]),
                        ]
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        aggregator.aggregate(chunk2)
        result = aggregator.build()

        # Verify refusal logprobs were aggregated
        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.refusal is not None
        assert len(logprobs.refusal) == 4

    def test_multiple_tool_calls_with_argument_chunks(self):
        """Test aggregating multiple tool calls where arguments arrive in multiple chunks."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction

        # First chunk: tool call 0 with initial arguments
        chunk1 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc123",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"location":'),
                            ),
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        # Second chunk: continue tool call 0 arguments
        chunk2 = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(arguments=' "Beijing"}'),
                            ),
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk1)
        aggregator.aggregate(chunk2)
        result = aggregator.build()

        # Verify tool call arguments were aggregated correctly
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) >= 1
        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call.id == "call_abc123"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"location": "Beijing"}'

        # Verify that ToolCallStartEvent was emitted with parent_message_id
        tool_call_start_events = [call for call in mock_on_event.call_args_list if call[0][0].__class__.__name__ == "ToolCallStartEvent"]
        assert len(tool_call_start_events) >= 1
        for event_call in tool_call_start_events:
            event = event_call[0][0]
            assert event.tool_call_id == "call_abc123"
            assert event.tool_call_name == "get_weather"
            assert event.parent_message_id == "chatcmpl-123"
            assert event.timestamp is not None  # timestamp should be set

        # Verify ToolCallArgsEvent was emitted (for compatibility)
        tool_call_args_events = [call for call in mock_on_event.call_args_list if call[0][0].__class__.__name__ == "ToolCallArgsEvent"]
        assert len(tool_call_args_events) >= 1
        for event_call in tool_call_args_events:
            event = event_call[0][0]
            assert event.tool_call_id == "call_abc123"
            assert event.timestamp is not None  # timestamp should be set
            # Verify that delta contains the arguments
            assert event.delta in ['{"location":', ' "Beijing"}']

    def test_aggregation_with_service_tier(self):
        """Test aggregating chunks with service tier information."""
        mock_on_event = Mock()
        aggregator = OpenAIChatCompletionAggregator(on_event=mock_on_event, run_id="test-run")

        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            service_tier="default",
            object="chat.completion.chunk",
        )

        aggregator.aggregate(chunk)
        result = aggregator.build()

        assert result.service_tier == "default"

    def test_choice_aggregator_build_without_aggregation_raises_error(self):
        """Test that ChoiceAggregator build() raises error when never aggregated."""
        mock_on_event = Mock()

        # Import the internal _ChoiceAggregator class for white-box testing
        from nexau.archs.llm.llm_aggregators.openai_chat_completion.openai_chat_completion_aggregator import _ChoiceAggregator

        aggregator = _ChoiceAggregator(index=0, message_id="msg-123", on_event=mock_on_event, run_id="test-run")

        # Try to build without aggregating any content
        with pytest.raises(RuntimeError, match="Choice 0 was never aggregated with any content"):
            aggregator.build()

    def test_choice_aggregator_clear_resets_state(self):
        """Test that ChoiceAggregator clear() resets state."""
        mock_on_event = Mock()

        # Import the internal _ChoiceAggregator class for white-box testing
        from nexau.archs.llm.llm_aggregators.openai_chat_completion.openai_chat_completion_aggregator import _ChoiceAggregator

        aggregator = _ChoiceAggregator(index=0, message_id="msg-123", on_event=mock_on_event, run_id="test-run")

        # Aggregate some content first
        aggregator.aggregate(
            ChatCompletionChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content="Hello"),
                finish_reason=None,
            )
        )

        # Clear should reset all state
        aggregator.clear()

        # Verify state is reset (using _value for all checks)
        assert aggregator._value.message.content is None
        assert aggregator._value.message.role == "assistant"
        assert aggregator._value.message.refusal is None
        assert aggregator._value.message.tool_calls is None
        assert aggregator._value.finish_reason == "stop"
        assert aggregator._value.logprobs is None
        assert aggregator._tool_call_aggregators == {}

    def test_tool_call_aggregator_build_without_start_raises_error(self):
        """Test that ToolCallAggregator build() raises error when not started."""
        mock_on_event = Mock()

        aggregator = _ToolCallAggregator(on_event=mock_on_event, parent_message_id="msg-123")

        # Don't aggregate anything, or aggregate without ID and name
        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

        aggregator.aggregate(ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='{"test": true}')))

        # Try to build without receiving ID and name
        with pytest.raises(ValueError, match="Tool call aggregator never received valid tool call data"):
            aggregator.build()

    def test_tool_call_aggregator_clear_resets_state(self):
        """Test that ToolCallAggregator clear() resets state."""
        mock_on_event = Mock()

        aggregator = _ToolCallAggregator(on_event=mock_on_event, parent_message_id="msg-123")

        # First set up aggregator with valid data
        from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

        aggregator.aggregate(
            ChoiceDeltaToolCall(
                index=0, id="call_abc123", function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"location": "Beijing"}')
            )
        )

        # Clear should reset all state
        aggregator.clear()

        # Verify state is reset
        assert aggregator._value.id == ""
        assert aggregator._value.function.name == ""
        assert aggregator._value.function.arguments == ""
        assert not aggregator._started
        assert aggregator._value.type == "function"
        # Verify parent_message_id is preserved (not reset on clear)
        assert aggregator._parent_message_id == "msg-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
