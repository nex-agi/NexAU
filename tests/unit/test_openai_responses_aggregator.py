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
Comprehensive unit tests for OpenAI Responses aggregator.
Target: 100% code coverage
"""

from unittest.mock import Mock

import pytest
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_content_part_done_event import (
    ResponseContentPartDoneEvent,
)
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_reasoning_summary_part_added_event import (
    ResponseReasoningSummaryPartAddedEvent,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    ResponseReasoningSummaryPartDoneEvent,
)
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_reasoning_summary_text_done_event import (
    ResponseReasoningSummaryTextDoneEvent,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent

from nexau.archs.llm.llm_aggregators.openai_responses.openai_responses_aggregator import (
    OpenAIResponsesAggregator,
    _FunctionCallItemAggregator,
    _MessageItemAggregator,
    _ReasoningItemAggregator,
)


class TestOpenAIResponsesAggregator:
    """Test cases for OpenAI responses aggregator main class."""

    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        assert aggregator._on_event == mock_on_event
        assert aggregator._output_aggregators == {}
        assert aggregator._value.id == ""
        assert aggregator._value.created_at == 0
        assert aggregator._value.model == ""
        assert aggregator._value.output == []
        assert aggregator._value.parallel_tool_calls is True

    def test_response_created_event(self):
        """Test response.created event handling."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        created_event = ResponseCreatedEvent(
            type="response.created",
            response={
                "id": "resp_123",
                "model": "gpt-4",
                "created_at": 1234567890,
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            sequence_number=0,
        )
        aggregator.aggregate(created_event)

        assert aggregator._value.id == "resp_123"
        assert aggregator._value.model == "gpt-4"
        assert aggregator._value.created_at == 1234567890

    def test_output_item_added_invalid_index(self):
        """Test that output_item.added with invalid index raises ValueError."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Try to add item at index 1 when list is empty
        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=1,  # Invalid - should be 0
            sequence_number=0,
        )

        with pytest.raises(ValueError, match="Invalid output_index"):
            aggregator.aggregate(event)

    def test_output_item_added_message(self):
        """Test adding message output item."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(event)

        assert len(aggregator._output_aggregators) == 1
        assert "msg_1" in aggregator._output_aggregators
        assert isinstance(aggregator._output_aggregators["msg_1"], _MessageItemAggregator)

    def test_output_item_added_message_missing_id(self):
        """Test that message without ID raises ValueError."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        msg_item = ResponseOutputMessage(
            id="",  # Missing ID
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=0,
            sequence_number=0,
        )

        with pytest.raises(ValueError, match="Message item must have an ID"):
            aggregator.aggregate(event)

    def test_output_item_added_reasoning(self):
        """Test adding reasoning output item."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        reasoning_item = ResponseReasoningItem(
            id="reason_1",
            type="reasoning",
            summary=[],
        )
        event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=reasoning_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(event)

        assert len(aggregator._output_aggregators) == 1
        assert "reason_1" in aggregator._output_aggregators
        assert isinstance(aggregator._output_aggregators["reason_1"], _ReasoningItemAggregator)

    def test_output_item_added_function_call(self):
        """Test adding function_call output item."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="func_1",
            name="test_function",
            arguments="",
        )
        event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=func_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(event)

        assert len(aggregator._output_aggregators) == 1
        assert "func_1" in aggregator._output_aggregators
        assert isinstance(aggregator._output_aggregators["func_1"], _FunctionCallItemAggregator)

    def test_output_item_added_function_call_missing_id(self):
        """Test that function_call without ID raises ValueError."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        func_item = ResponseFunctionToolCall(
            id="",  # Missing ID
            type="function_call",
            call_id="",
            name="test_function",
            arguments="",
        )
        event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=func_item,
            output_index=0,
            sequence_number=0,
        )

        with pytest.raises(ValueError, match="Function call item must have an ID"):
            aggregator.aggregate(event)

    def test_output_item_done(self):
        """Test output_item.done event handling."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # First add a message
        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        add_event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Then mark as done
        msg_done = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[],
        )
        done_event = ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=msg_done,
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(done_event)

        # Verify aggregator still exists
        assert "msg_1" in aggregator._output_aggregators

    def test_message_content_events_dispatch(self):
        """Test that message content events are dispatched correctly."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add message
        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        add_event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Test content part added event
        content_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(content_event)

        # Verify event was dispatched to message aggregator
        msg_aggregator = aggregator._output_aggregators["msg_1"]
        assert isinstance(msg_aggregator, _MessageItemAggregator)
        assert len(msg_aggregator._value.content) == 1

        # Test text delta event
        delta_event = ResponseTextDeltaEvent(
            type="response.output_text.delta",
            delta="Hello",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=2,
        )
        aggregator.aggregate(delta_event)

        # Test text done event
        done_event = ResponseTextDoneEvent(
            type="response.output_text.done",
            text="Hello",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=3,
        )
        aggregator.aggregate(done_event)

    def test_reasoning_events_dispatch(self):
        """Test that reasoning events are dispatched correctly."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add reasoning
        reasoning_item = ResponseReasoningItem(
            id="reason_1",
            type="reasoning",
            summary=[],
        )
        add_event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=reasoning_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Test summary part added
        summary_added = ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            part={"type": "summary_text", "text": ""},
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(summary_added)

        reason_aggregator = aggregator._output_aggregators["reason_1"]
        assert isinstance(reason_aggregator, _ReasoningItemAggregator)
        assert len(reason_aggregator._value.summary) == 1

    def test_function_call_events_dispatch(self):
        """Test that function_call events are dispatched correctly."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add function call
        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="func_1",
            name="test_func",
            arguments="",
        )
        add_event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=func_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Test arguments delta
        delta_event = ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            delta='{"key": "value"}',
            item_id="func_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        func_aggregator = aggregator._output_aggregators["func_1"]
        assert isinstance(func_aggregator, _FunctionCallItemAggregator)
        assert func_aggregator._value.arguments == '{"key": "value"}'

    def test_response_completed_event(self):
        """Test response.completed event handling."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        completed_event = ResponseCompletedEvent(
            type="response.completed",
            response={
                "id": "resp_123",
                "model": "gpt-4",
                "created_at": 1234567890,
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            sequence_number=0,
        )
        aggregator.aggregate(completed_event)

        # Should update _value with completed response
        assert aggregator._value.id == "resp_123"

    def test_build(self):
        """Test building final response."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add and complete a message
        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        add_event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        content_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "test", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(content_event)

        final_response = aggregator.build()

        assert len(final_response.output) == 1
        assert final_response.output[0].id == "msg_1"

    def test_clear(self):
        """Test clear() method."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add a message
        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        add_event = ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=msg_item,
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        assert len(aggregator._output_aggregators) == 1

        aggregator.clear()

        assert aggregator._output_aggregators == {}
        assert aggregator._value.output == []
        assert aggregator._value.id == ""


class TestMessageItemAggregator:
    """Test cases for _MessageItemAggregator."""

    def test_message_aggregator_initialization(self):
        """Test message aggregator initialization."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        assert aggregator._item_id == "msg_1"
        assert aggregator.output_index == 0
        assert aggregator._on_event == mock_on_event
        assert aggregator._started is False
        assert aggregator._value.content == []
        assert aggregator._value.type == "message"

    def test_message_type_event(self):
        """Test handling message type event."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        msg_item = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[{"type": "output_text", "text": "test", "annotations": []}],
        )
        aggregator.aggregate(msg_item)

        assert aggregator._value.content[0].text == "test"

    def test_content_part_added_output_text(self):
        """Test content_part.added with output_text."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(event)

        assert len(aggregator._value.content) == 1
        assert aggregator._value.content[0].type == "output_text"
        assert mock_on_event.called  # Start event should be emitted

    def test_content_part_added_refusal(self):
        """Test content_part.added with refusal."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "refusal", "refusal": ""},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(event)

        assert len(aggregator._value.content) == 1
        assert aggregator._value.content[0].type == "refusal"

    def test_content_part_added_invalid_index(self):
        """Test that invalid content_index raises ValueError."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=1,  # Invalid - should be 0
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )

        with pytest.raises(ValueError, match="Invalid content_index"):
            aggregator.aggregate(event)

    def test_output_text_delta(self):
        """Test output_text.delta event."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        # First add content part
        add_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Then stream text
        delta_event = ResponseTextDeltaEvent(
            type="response.output_text.delta",
            delta="Hello",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        assert aggregator._value.content[0].text == "Hello"
        assert mock_on_event.call_count == 2  # Start + Content

    def test_output_text_delta_refusal(self):
        """Test output_text.delta with refusal content."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        # Add refusal content
        add_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "refusal", "refusal": ""},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Stream refusal text
        delta_event = ResponseTextDeltaEvent(
            type="response.output_text.delta",
            delta="I cannot",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        assert aggregator._value.content[0].refusal == "I cannot"

    def test_output_text_done(self):
        """Test output_text.done event."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        # Add content and stream deltas
        add_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        delta_event = ResponseTextDeltaEvent(
            type="response.output_text.delta",
            delta="Hello",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        # Finalize text
        done_event = ResponseTextDoneEvent(
            type="response.output_text.done",
            text="Hello World",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=2,
        )
        aggregator.aggregate(done_event)

        assert aggregator._value.content[0].text == "Hello World"
        # Start + 1x Content + End = 3 calls
        assert mock_on_event.call_count == 3

    def test_content_part_done(self):
        """Test content_part.done event (no-op)."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        done_event = ResponseContentPartDoneEvent(
            type="response.content_part.done",
            part={"type": "output_text", "text": "test", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )

        # Should not raise or emit events
        aggregator.aggregate(done_event)
        assert mock_on_event.call_count == 0

    def test_build(self):
        """Test build() method."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        # Process some events
        add_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        delta_event = ResponseTextDeltaEvent(
            type="response.output_text.delta",
            delta="test",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            logprobs=[],
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        result = aggregator.build()

        assert result.id == "msg_1"
        assert result.content[0].text == "test"

    def test_clear(self):
        """Test clear() method."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(item_id="msg_1", output_index=0, on_event=mock_on_event, run_id="test-run")

        # Process events
        add_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "output_text", "text": "", "annotations": []},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        assert aggregator._started is True
        assert len(aggregator._value.content) == 1

        aggregator.clear()

        assert aggregator._started is False
        assert aggregator._value.content == []


class TestReasoningItemAggregator:
    """Test cases for _ReasoningItemAggregator."""

    def test_reasoning_aggregator_initialization(self):
        """Test reasoning aggregator initialization."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        assert aggregator._item_id == "reason_1"
        assert aggregator.output_index == 0
        assert aggregator._on_event == mock_on_event
        assert aggregator._value.summary == []
        assert aggregator._value.type == "reasoning"

    def test_reasoning_type_event(self):
        """Test handling reasoning type event."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        reasoning_item = ResponseReasoningItem(
            id="reason_1",
            type="reasoning",
            summary=[{"type": "summary_text", "text": "test"}],
            content=None,
        )
        aggregator.aggregate(reasoning_item)

        assert len(aggregator._value.summary) == 1
        assert aggregator._value.summary[0].text == "test"

    def test_summary_part_added(self):
        """Test reasoning_summary_part.added event."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        event = ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            part={"type": "summary_text", "text": ""},
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(event)

        assert len(aggregator._value.summary) == 1
        assert aggregator._value.summary[0].text == ""
        assert mock_on_event.called  # Start event should be emitted

    def test_summary_part_added_invalid_index(self):
        """Test that invalid summary_index raises ValueError."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        event = ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            part={"type": "summary_text", "text": ""},
            summary_index=1,  # Invalid - should be 0
            item_id="reason_1",
            output_index=0,
            sequence_number=0,
        )

        with pytest.raises(ValueError, match="Invalid summary_index"):
            aggregator.aggregate(event)

    def test_summary_text_delta(self):
        """Test reasoning_summary_text.delta event."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        # Add summary part first
        add_event = ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            part={"type": "summary_text", "text": ""},
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Stream text
        delta_event = ResponseReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            delta="Let me think",
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            logprobs=None,
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        assert aggregator._value.summary[0].text == "Let me think"
        assert mock_on_event.call_count == 2  # Start + Content

    def test_summary_text_done(self):
        """Test reasoning_summary_text.done event."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        # Add and stream
        add_event = ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            part={"type": "summary_text", "text": ""},
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        delta_event = ResponseReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            delta="test",
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            logprobs=None,
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        # Finalize
        done_event = ResponseReasoningSummaryTextDoneEvent(
            type="response.reasoning_summary_text.done",
            text="test completed",
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            logprobs=None,
            sequence_number=2,
        )
        aggregator.aggregate(done_event)

        assert aggregator._value.summary[0].text == "test completed"
        assert mock_on_event.call_count == 3  # Start + Content + End

    def test_summary_part_done(self):
        """Test reasoning_summary_part.done event (no-op)."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        done_event = ResponseReasoningSummaryPartDoneEvent(
            type="response.reasoning_summary_part.done",
            part={"type": "summary_text", "text": "test"},
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            sequence_number=0,
        )

        # Should not raise
        aggregator.aggregate(done_event)

    def test_build_and_clear(self):
        """Test build() and clear() methods."""
        mock_on_event = Mock()
        aggregator = _ReasoningItemAggregator(
            item_id="reason_1", output_index=0, on_event=mock_on_event, run_id="test-run", response_id="resp_123"
        )

        # Process events
        add_event = ResponseReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            part={"type": "summary_text", "text": ""},
            summary_index=0,
            item_id="reason_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Build
        result = aggregator.build()
        assert result.id == "reason_1"
        assert len(result.summary) == 1

        # Clear
        aggregator.clear()
        assert aggregator._value.summary == []


class TestFunctionCallItemAggregator:
    """Test cases for _FunctionCallItemAggregator."""

    def test_function_call_aggregator_initialization(self):
        """Test function call aggregator initialization."""
        mock_on_event = Mock()
        aggregator = _FunctionCallItemAggregator(item_id="func_1", output_index=0, on_event=mock_on_event, response_id="resp_123")

        assert aggregator._item_id == "func_1"
        assert aggregator.output_index == 0
        assert aggregator._on_event == mock_on_event
        assert aggregator._started is False
        assert aggregator._value.arguments == ""

    def test_function_call_type_event(self):
        """Test handling function_call type event."""
        mock_on_event = Mock()
        aggregator = _FunctionCallItemAggregator(item_id="func_1", output_index=0, on_event=mock_on_event, response_id="resp_123")

        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="func_1",
            name="test_func",
            arguments='{"key": "value"}',
        )
        aggregator.aggregate(func_item)

        assert aggregator._value.name == "test_func"
        assert aggregator._started is True
        assert mock_on_event.called  # Start event should be emitted

    def test_arguments_delta(self):
        """Test function_call_arguments.delta event."""
        mock_on_event = Mock()
        aggregator = _FunctionCallItemAggregator(item_id="func_1", output_index=0, on_event=mock_on_event, response_id="resp_123")

        # First call
        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="func_1",
            name="test_func",
            arguments="",
        )
        aggregator.aggregate(func_item)

        # Stream arguments
        delta_event = ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            delta='{"key": "value"}',
            item_id="func_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        assert aggregator._value.arguments == '{"key": "value"}'
        assert mock_on_event.call_count == 2  # Start + Args

    def test_arguments_done(self):
        """Test function_call_arguments.done event."""
        mock_on_event = Mock()
        aggregator = _FunctionCallItemAggregator(item_id="func_1", output_index=0, on_event=mock_on_event, response_id="resp_123")

        # Initialize
        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="func_1",
            name="test_func",
            arguments="",
        )
        aggregator.aggregate(func_item)

        # Stream some arguments
        delta_event = ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            delta='{"partial": true}',
            item_id="func_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(delta_event)

        # Finalize
        done_event = ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            name="test_func",
            arguments='{"complete": true}',
            item_id="func_1",
            output_index=0,
            sequence_number=2,
        )
        aggregator.aggregate(done_event)

        assert aggregator._value.arguments == '{"complete": true}'
        assert mock_on_event.call_count == 3  # Start + Args + End

    def test_build_and_clear(self):
        """Test build() and clear() methods."""
        mock_on_event = Mock()
        aggregator = _FunctionCallItemAggregator(item_id="func_1", output_index=0, on_event=mock_on_event, response_id="resp_123")

        # Process events
        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="func_1",
            name="test_func",
            arguments='{"key": "value"}',
        )
        aggregator.aggregate(func_item)

        # Build
        result = aggregator.build()
        assert result.name == "test_func"
        assert result.arguments == '{"key": "value"}'

        # Clear
        aggregator.clear()
        assert aggregator._started is False
        assert aggregator._value.arguments == ""


class TestIntegration:
    """Integration tests for complete flows."""

    def test_complete_message_flow(self):
        """Test complete message aggregation flow."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # 1. Response created
        aggregator.aggregate(
            ResponseCreatedEvent(
                type="response.created",
                response={
                    "id": "resp_123",
                    "model": "gpt-4",
                    "created_at": 1234567890,
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
                sequence_number=0,
            )
        )

        # 2. Add message
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(id="msg_1", type="message", role="assistant", status="in_progress", content=[]),
                output_index=0,
                sequence_number=1,
            )
        )

        # 3. Add content part
        aggregator.aggregate(
            ResponseContentPartAddedEvent(
                type="response.content_part.added",
                part={"type": "output_text", "text": "", "annotations": []},
                content_index=0,
                item_id="msg_1",
                output_index=0,
                sequence_number=2,
            )
        )

        # 4. Stream text
        aggregator.aggregate(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                delta="Hello World",
                content_index=0,
                item_id="msg_1",
                output_index=0,
                logprobs=[],
                sequence_number=3,
            )
        )

        # 5. Text done
        aggregator.aggregate(
            ResponseTextDoneEvent(
                type="response.output_text.done",
                text="Hello World",
                content_index=0,
                item_id="msg_1",
                output_index=0,
                logprobs=[],
                sequence_number=4,
            )
        )

        # 6. Content part done
        aggregator.aggregate(
            ResponseContentPartDoneEvent(
                type="response.content_part.done",
                part={"type": "output_text", "text": "Hello World", "annotations": []},
                content_index=0,
                item_id="msg_1",
                output_index=0,
                sequence_number=5,
            )
        )

        # 7. Message done
        aggregator.aggregate(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseOutputMessage(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[{"type": "output_text", "text": "Hello World", "annotations": []}],
                ),
                output_index=0,
                sequence_number=6,
            )
        )

        # Build
        result = aggregator.build()
        assert result.output[0].content[0].text == "Hello World"

    def test_multiple_output_items(self):
        """Test multiple output items in one response."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add message at index 0
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(id="msg_1", type="message", role="assistant", status="in_progress", content=[]),
                output_index=0,
                sequence_number=0,
            )
        )

        # Add reasoning at index 1
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseReasoningItem(id="reason_1", type="reasoning", summary=[]),
                output_index=1,
                sequence_number=1,
            )
        )

        # Add function call at index 2
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseFunctionToolCall(
                    id="func_1",
                    type="function_call",
                    call_id="func_1",
                    name="test",
                    arguments="",
                ),
                output_index=2,
                sequence_number=2,
            )
        )

        assert len(aggregator._output_aggregators) == 3
        assert "msg_1" in aggregator._output_aggregators
        assert "reason_1" in aggregator._output_aggregators
        assert "func_1" in aggregator._output_aggregators

        # Verify build works
        result = aggregator.build()
        assert len(result.output) == 3

    def test_reusability_after_clear(self):
        """Test that aggregator can be reused after clear()."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # First stream
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(id="msg_1", type="message", role="assistant", status="in_progress", content=[]),
                output_index=0,
                sequence_number=0,
            )
        )

        aggregator.clear()

        # Second stream - should work the same
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(id="msg_2", type="message", role="assistant", status="in_progress", content=[]),
                output_index=0,
                sequence_number=0,
            )
        )

        assert len(aggregator._output_aggregators) == 1
        assert "msg_2" in aggregator._output_aggregators


class TestAdditionalCoverage:
    """Additional test cases to improve coverage."""

    def test_output_item_done_reasoning(self):
        """Test output_item.done event for reasoning item."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add reasoning item
        reasoning_item = ResponseReasoningItem(
            id="reason_1",
            type="reasoning",
            summary=[],
        )
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=reasoning_item,
                output_index=0,
                sequence_number=0,
            )
        )

        # Mark as done with final state
        done_reasoning = ResponseReasoningItem(
            id="reason_1",
            type="reasoning",
            summary=[{"type": "summary_text", "text": "Final summary"}],
        )
        aggregator.aggregate(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=done_reasoning,
                output_index=0,
                sequence_number=1,
            )
        )

        # Verify aggregator updated
        reason_agg = aggregator._output_aggregators["reason_1"]
        assert isinstance(reason_agg, _ReasoningItemAggregator)

    def test_output_item_done_function_call(self):
        """Test output_item.done event for function_call item."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add function call item
        func_item = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="call_1",
            name="test_func",
            arguments="",
        )
        aggregator.aggregate(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=func_item,
                output_index=0,
                sequence_number=0,
            )
        )

        # Mark as done with final state
        done_func = ResponseFunctionToolCall(
            id="func_1",
            type="function_call",
            call_id="call_1",
            name="test_func",
            arguments='{"key": "value"}',
        )
        aggregator.aggregate(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=done_func,
                output_index=0,
                sequence_number=1,
            )
        )

        # Verify aggregator updated
        func_agg = aggregator._output_aggregators["func_1"]
        assert isinstance(func_agg, _FunctionCallItemAggregator)

    def test_output_item_done_function_call_missing_id(self):
        """Test output_item.done event for function_call with missing ID raises."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add a placeholder to output list
        aggregator._value.output.append(
            ResponseFunctionToolCall(
                id="",
                type="function_call",
                call_id="",
                name="test",
                arguments="",
            )
        )

        # Try to mark as done with missing ID
        done_func = ResponseFunctionToolCall(
            id="",  # Missing ID
            type="function_call",
            call_id="",
            name="test_func",
            arguments="{}",
        )

        with pytest.raises(ValueError, match="Function call item must have an ID"):
            aggregator.aggregate(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=done_func,
                    output_index=0,
                    sequence_number=0,
                )
            )

    def test_output_item_done_message_missing_id(self):
        """Test output_item.done event for message with missing ID raises."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Add a placeholder to output list
        aggregator._value.output.append(
            ResponseOutputMessage(
                id="",
                type="message",
                role="assistant",
                status="completed",
                content=[],
            )
        )

        # Try to mark as done with missing ID
        done_msg = ResponseOutputMessage(
            id="",  # Missing ID
            type="message",
            role="assistant",
            status="completed",
            content=[],
        )

        with pytest.raises(ValueError, match="Message item must have an ID"):
            aggregator.aggregate(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=done_msg,
                    output_index=0,
                    sequence_number=0,
                )
            )

    def test_output_item_done_unknown_type(self):
        """Test output_item.done event for unknown item type updates output directly."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Manually add an item to output list first

        # Use a mock unknown item that's not message/reasoning/function_call
        class UnknownItem:
            type = "unknown_type"
            id = "unknown_1"

        aggregator._value.output.append(UnknownItem())

        # Create a mock done event with unknown item type
        # Since we can't easily create unknown types, we test via direct call
        # This tests line 167-169 where unknown types get directly assigned
        mock_done_item = Mock()
        mock_done_item.type = "unknown_type"
        mock_done_item.id = "unknown_1"

        mock_event = Mock()
        mock_event.type = "response.output_item.done"
        mock_event.item = mock_done_item
        mock_event.output_index = 0

        # Aggregate should handle gracefully
        aggregator.aggregate(mock_event)

        # Verify the output was updated
        assert aggregator._value.output[0] is mock_done_item

    def test_message_refusal_done(self):
        """Test response.refusal.done event handling."""
        mock_on_event = Mock()
        aggregator = _MessageItemAggregator(
            item_id="msg_1",
            output_index=0,
            on_event=mock_on_event,
            run_id="test-run",
        )

        # Add refusal content part first
        from openai.types.responses.response_content_part_added_event import (
            ResponseContentPartAddedEvent,
        )

        add_event = ResponseContentPartAddedEvent(
            type="response.content_part.added",
            part={"type": "refusal", "refusal": ""},
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
        )
        aggregator.aggregate(add_event)

        # Now handle refusal.done
        from openai.types.responses.response_refusal_done_event import (
            ResponseRefusalDoneEvent,
        )

        done_event = ResponseRefusalDoneEvent(
            type="response.refusal.done",
            refusal="I cannot help with that",
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=1,
        )
        aggregator.aggregate(done_event)

        # Verify refusal was set and event emitted
        assert aggregator._value.content[0].refusal == "I cannot help with that"
        # Start + End events should have been emitted
        assert mock_on_event.call_count == 2

    def test_output_item_added_unknown_type(self):
        """Test output_item.added event for unknown item type is ignored."""
        mock_on_event = Mock()
        aggregator = OpenAIResponsesAggregator(on_event=mock_on_event, run_id="test-run")

        # Create a mock unknown item type
        class UnknownItem:
            type = "unknown_type"
            id = "unknown_1"

        mock_event = Mock()
        mock_event.type = "response.output_item.added"
        mock_event.item = UnknownItem()
        mock_event.output_index = 0

        # Aggregate should handle gracefully by adding to output but not creating aggregator
        aggregator.aggregate(mock_event)

        # Verify item was added to output but no aggregator created
        assert len(aggregator._value.output) == 1
        assert len(aggregator._output_aggregators) == 0
