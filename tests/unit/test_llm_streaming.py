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

"""Unit tests for stream aggregation helpers."""

from nexau.archs.main_sub.execution.llm_caller import (
    AnthropicStreamAggregator,
    OpenAIChatStreamAggregator,
    OpenAIResponsesStreamAggregator,
)


def test_openai_chat_stream_aggregator_merges_chunks():
    aggregator = OpenAIChatStreamAggregator()

    aggregator.consume(
        {
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{ "query": "Hel'},
                            },
                        ],
                    },
                },
            ],
        },
    )
    aggregator.consume(
        {
            "choices": [
                {
                    "delta": {
                        "content": [{"type": "output_text", "text": " world"}],
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'lo" }'},
                            },
                        ],
                    },
                },
            ],
        },
    )

    message = aggregator.finalize()

    assert message["content"] == "Hello world"
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "lookup"
    assert tool_call["function"]["arguments"] == '{ "query": "Hello" }'


def test_openai_chat_stream_aggregator_preserves_usage_details():
    aggregator = OpenAIChatStreamAggregator()

    aggregator.consume(
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Token"}],
                    },
                },
            ],
        },
    )
    aggregator.consume(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        },
    )

    message = aggregator.finalize()

    assert message["usage"]["prompt_tokens"] == 10
    assert message["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3


def test_anthropic_stream_aggregator_builds_message_blocks():
    aggregator = AnthropicStreamAggregator()

    aggregator.consume({"type": "message_start", "message": {"role": "assistant", "model": "claude-3"}})
    aggregator.consume({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    aggregator.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}})
    aggregator.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}})
    aggregator.consume({"type": "content_block_stop", "index": 0})

    message = aggregator.finalize()

    assert message["role"] == "assistant"
    assert message["content"][0]["text"] == "Hi there"


def test_anthropic_stream_aggregator_does_not_overwrite_tool_block_on_duplicate_starts():
    """Regression: some stream traces surface duplicate content_block_start events for the same index.

    We should never overwrite a well-formed tool_use block (id/name) with an empty one.
    """
    aggregator = AnthropicStreamAggregator()

    aggregator.consume({"type": "message_start", "message": {"role": "assistant", "model": "claude-3"}})
    aggregator.consume(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "file_read", "input": {}},
        },
    )
    aggregator.consume(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "/tmp/image.png"'},
        },
    )
    # Duplicate start (observed in the wild) with missing name/id should not clobber state.
    aggregator.consume(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": None, "name": None, "input": {}},
        },
    )
    aggregator.consume(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "}"},
        },
    )
    aggregator.consume({"type": "content_block_stop", "index": 0})

    message = aggregator.finalize()

    assert message["content"][0]["type"] == "tool_use"
    assert message["content"][0]["id"] == "toolu_1"
    assert message["content"][0]["name"] == "file_read"
    assert message["content"][0]["input"] == {"file_path": "/tmp/image.png"}


def test_anthropic_stream_aggregator_thinking_delta_accumulates():
    """thinking_delta events should accumulate thinking content in a thinking block."""
    aggregator = AnthropicStreamAggregator()

    aggregator.consume({"type": "message_start", "message": {"role": "assistant", "model": "claude-3"}})
    aggregator.consume({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})
    aggregator.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Step 1: "}})
    aggregator.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "analyze the problem"}})
    aggregator.consume({"type": "content_block_stop", "index": 0})

    message = aggregator.finalize()

    assert len(message["content"]) == 1
    block = message["content"][0]
    assert block["type"] == "thinking"
    assert block["thinking"] == "Step 1: analyze the problem"


def test_anthropic_stream_aggregator_thinking_then_text():
    """A thinking block followed by a text block should both appear in the final message."""
    aggregator = AnthropicStreamAggregator()

    aggregator.consume({"type": "message_start", "message": {"role": "assistant", "model": "claude-3"}})
    # Thinking block
    aggregator.consume({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})
    aggregator.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think"}})
    aggregator.consume({"type": "content_block_stop", "index": 0})
    # Text block
    aggregator.consume({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}})
    aggregator.consume({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The answer is 42"}})
    aggregator.consume({"type": "content_block_stop", "index": 1})

    message = aggregator.finalize()

    assert len(message["content"]) == 2
    assert message["content"][0]["type"] == "thinking"
    assert message["content"][0]["thinking"] == "Let me think"
    assert message["content"][1]["type"] == "text"
    assert message["content"][1]["text"] == "The answer is 42"


def test_anthropic_stream_aggregator_thinking_delta_without_block_start():
    """thinking_delta before content_block_start should still create a thinking block."""
    aggregator = AnthropicStreamAggregator()

    aggregator.consume({"type": "message_start", "message": {"role": "assistant"}})
    # Delta arrives before block_start (edge case)
    aggregator.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}})
    aggregator.consume({"type": "content_block_stop", "index": 0})

    message = aggregator.finalize()

    assert message["content"][0]["type"] == "thinking"
    assert message["content"][0]["thinking"] == "hmm"


def test_openai_responses_stream_aggregator_reconstructs_items():
    aggregator = OpenAIResponsesStreamAggregator()

    aggregator.consume(
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
        },
    )
    aggregator.consume(
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
    )
    aggregator.consume(
        {"type": "response.output_text.delta", "item_id": "msg_1", "content_index": 0, "delta": "Answer: 42"},
    )
    aggregator.consume(
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "tool_1", "call_id": "tc_1", "name": "compute", "arguments": ""},
        },
    )
    aggregator.consume(
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool_1",
            "delta": '{"value":',
        },
    )
    aggregator.consume(
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool_1",
            "delta": " 42}",
        },
    )
    aggregator.consume(
        {
            "type": "response.output_item.added",
            "item": {
                "type": "reasoning",
                "id": "rs_reason_1",
                "summary": [{"type": "summary_text", "text": ""}],
            },
        },
    )
    aggregator.consume(
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_reason_1",
            "summary_index": 0,
            "delta": "Checked prior calculations",
        },
    )
    aggregator.consume(
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "model": "gpt-4.1", "usage": {"input_tokens": 10, "output_tokens": 5}},
        },
    )

    response_payload = aggregator.finalize()

    assert response_payload["model"] == "gpt-4.1"
    assert response_payload["output"][0]["content"][0]["text"] == "Answer: 42"
    assert response_payload["output"][1]["arguments"] == '{"value": 42}'
    assert response_payload["output"][2]["type"] == "reasoning"
    assert response_payload["output"][2]["id"] == "rs_reason_1"
