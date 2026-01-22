#!/usr/bin/env python3
"""
Generate test data for LLM aggregator logging.

Runs all OpenAI API modes and saves outputs to tests/test_data/llm_aggregators/:
1. Chat Completion (non-streaming)
2. Chat Completion Stream
3. Responses (non-streaming)
4. Responses Stream with Aggregator Events (combined)

Use for debugging, testing, and analysis of aggregator behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam
from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.tool_param import ToolParam

from nexau.archs.llm.llm_aggregators import (
    OpenAIChatCompletionAggregator,
    OpenAIResponsesAggregator,
)
from nexau.archs.llm.llm_aggregators.events import Event

# Output directory: tests/test_data/llm_aggregators/
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_data" / "llm_aggregators"

# Load environment variables from .env file
load_dotenv()

# Common configuration
CHAT_COMPLETION_MODEL = "nex-agi/deepseek-v3.1-nex-1"
RESPONSES_MODEL = "gpt-5.2"

# Load API configuration from environment variables
API_BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")

if not API_BASE_URL or not API_KEY:
    raise ValueError("Missing required environment variables. Please set LLM_BASE_URL and LLM_API_KEY in your .env file.")

# Configure OpenAI client
client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Test prompt and tools
# Prompt designed to test multiple scenarios:
# 1. Direct text response (greeting and personal opinion)
# 2. Tool calls (weather lookup and calculation)
# 3. Mixed content with both text and function calls
# 4. Thinking/reasoning content (if supported by model)
TEST_PROMPT = """Please briefly say hello and introduce yourself, then tell me your favorite color and why.

Next, help me with these tasks:
1. Look up today's weather in Beijing
2. Calculate 25 * 47

Finally, summarize what you've done."""

TOOLS: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a basic mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. 2+2, 10*5",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

RESPONSES_TOOLS: list[ToolParam] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Perform a basic mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. 2+2, 10*5",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _log_to_jsonl(
    data: Event | Response | ChatCompletion | ChatCompletionChunk | ResponseStreamEvent,
    log_path: Path,
) -> None:
    """Append data to a JSONL file silently (no console output)."""
    json_obj = data.model_dump() if hasattr(data, "model_dump") else data
    json_str = json.dumps(json_obj, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json_str + "\n")
        f.flush()


def _log_to_json(
    data: Event | Response | ChatCompletion,
    log_path: Path,
) -> None:
    """Save data to a JSON file (not JSONL) silently (no console output)."""
    json_obj = data.model_dump() if hasattr(data, "model_dump") else data
    json_str = json.dumps(json_obj, ensure_ascii=False, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json_str + "\n")


def _create_event_logger(log_path: Path):
    """Create an event logger that saves aggregator events to JSONL."""

    def log_event(event: Event) -> None:
        _log_to_jsonl(event, log_path)

    return log_event


def _example_1_chat_completion_non_streaming(out: Path) -> None:
    """
    Example 1: Chat Completion (non-streaming)
    Saves the final ChatCompletion object.
    """
    print("\n" + "=" * 80)
    print("Example 1: Chat Completion (non-streaming)")
    print("=" * 80)

    log_path = out / "chat_completion_non_stream.json"

    completion = client.chat.completions.create(
        model=CHAT_COMPLETION_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        tools=TOOLS,
    )

    _log_to_json(completion, log_path)
    print(f"✓ Chat completion saved to: {log_path}")
    print(f"  - ID: {completion.id}")
    print(f"  - Model: {completion.model}")
    print(f"  - Choices: {len(completion.choices)}")
    print(f"  - Finish reason: {completion.choices[0].finish_reason if completion.choices else 'N/A'}")


def _example_2_chat_completion_stream(out: Path) -> None:
    """
    Example 2: Chat Completion Stream
    Saves all chunks and optionally aggregates them.
    """
    print("\n" + "=" * 80)
    print("Example 2: Chat Completion Stream")
    print("=" * 80)

    chunk_log_path = out / "chat_completion_stream_chunks.jsonl"
    aggregated_log_path = out / "chat_completion_stream_aggregated.json"
    events_log_path = out / "chat_completion_stream_events.jsonl"

    stream = client.chat.completions.create(
        model=CHAT_COMPLETION_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        tools=TOOLS,
        stream=True,
    )

    event_logger = _create_event_logger(events_log_path)
    aggregator = OpenAIChatCompletionAggregator(on_event=event_logger, run_id="example-run")

    chunk_count = 0
    print("Processing stream chunks...")

    for chunk in stream:
        chunk_count += 1
        _log_to_jsonl(chunk, chunk_log_path)
        aggregator.aggregate(chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()
    print("✓ Stream processing complete:")
    print(f"  - Chunks saved to: {chunk_log_path}")
    print(f"  - Total chunks: {chunk_count}")

    final_completion = aggregator.build()
    _log_to_json(final_completion, aggregated_log_path)
    print(f"  - Aggregated completion saved to: {aggregated_log_path}")
    print(f"  - Final content: {final_completion.choices[0].message.content}")
    print(f"  - Events logged to: {events_log_path}")


def _example_3_responses_non_streaming(out: Path) -> None:
    """
    Example 3: Responses (non-streaming)
    Saves the final Response object.
    """
    print("\n" + "=" * 80)
    print("Example 3: Responses (non-streaming)")
    print("=" * 80)

    log_path = out / "responses_non_stream.json"

    response = client.responses.create(
        model=RESPONSES_MODEL,
        reasoning={"effort": "high"},
        input=[{"role": "user", "type": "message", "content": TEST_PROMPT}],
        tools=RESPONSES_TOOLS,
    )

    _log_to_json(response, log_path)
    print(f"✓ Response saved to: {log_path}")
    print(f"  - ID: {response.id}")
    print(f"  - Model: {response.model}")
    print(f"  - Output: {len(response.output)} items")


def _example_4_responses_stream(out: Path) -> None:
    """
    Example 4: Responses Stream
    Demonstrates streaming with both raw events and aggregated output.
    """
    print("\n" + "=" * 80)
    print("Example 4: Responses Stream")
    print("=" * 80)

    event_log_path = out / "responses_stream.jsonl"
    aggregator_event_log_path = out / "responses_stream_aggregator_events.jsonl"
    aggregated_log_path = out / "responses_stream_aggregated.json"

    stream = client.responses.create(
        model=RESPONSES_MODEL,
        reasoning={"effort": "high"},
        input=[{"role": "user", "type": "message", "content": TEST_PROMPT}],
        tools=RESPONSES_TOOLS,
        stream=True,
    )

    event_logger = _create_event_logger(aggregator_event_log_path)
    aggregator = OpenAIResponsesAggregator(on_event=event_logger, run_id="example-run")

    event_count = 0
    print("Processing stream events...")

    for event in stream:
        event_count += 1
        _log_to_jsonl(event, event_log_path)
        aggregator.aggregate(event)
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.function_call_arguments.delta":
            print(event.delta, end="", flush=True)

    print()
    print("✓ Stream processing complete:")
    print(f"  - Raw events saved to: {event_log_path}")
    print(f"  - Total events processed: {event_count}")

    final_response = aggregator.build()
    _log_to_json(final_response, aggregated_log_path)
    print(f"  - Aggregated response saved to: {aggregated_log_path}")
    print(f"  - Aggregator events logged to: {aggregator_event_log_path}")
    print(f"  - Output items: {len(final_response.output)}")

    print("\nAggregator emits these event types:")
    for event_type, desc in [
        ("TEXT_MESSAGE_START", "Message begins"),
        ("TEXT_MESSAGE_CONTENT", "Text content chunk received"),
        ("TEXT_MESSAGE_END", "Message complete"),
        ("THINKING_TEXT_MESSAGE_START", "Thinking/reasoning content begins"),
        ("THINKING_TEXT_MESSAGE_CONTENT", "Thinking content chunk"),
        ("THINKING_TEXT_MESSAGE_END", "Thinking content complete"),
        ("TOOL_CALL_START", "Tool call begins"),
        ("TOOL_CALL_ARGS", "Tool call arguments chunk"),
        ("TOOL_CALL_END", "Tool call complete"),
    ]:
        print(f"  - {event_type:35s} {desc}")


def _print_summary(out: Path) -> None:
    """Print summary of all log files."""
    print("\n" + "=" * 80)
    print("SUMMARY - Generated Log Files")
    print("=" * 80)
    print(f"\nOutput directory: {out}")

    log_files = [
        ("chat_completion_non_stream.json", "Chat Completion (non-streaming)"),
        ("chat_completion_stream_chunks.jsonl", "Chat Completion Stream (chunks)"),
        ("chat_completion_stream_aggregated.json", "Chat Completion Stream (aggregated)"),
        ("chat_completion_stream_events.jsonl", "Chat Completion Stream (events)"),
        ("responses_non_stream.json", "Responses (non-streaming)"),
        ("responses_stream.jsonl", "Responses Stream (raw events)"),
        ("responses_stream_aggregator_events.jsonl", "Responses Stream (aggregator events)"),
        ("responses_stream_aggregated.json", "Responses Stream (aggregated)"),
    ]
    print("\nLog files:")
    for name, desc in log_files:
        print(f"  - {name:50s} ({desc})")

    print("\nJSON files (single objects, pretty-printed):")
    for name in [
        "chat_completion_non_stream.json",
        "chat_completion_stream_aggregated.json",
        "responses_non_stream.json",
        "responses_stream_aggregated.json",
    ]:
        print(f"  - {out / name}")

    print("\nJSONL files (multiple objects, one per line):")
    for name in [
        "chat_completion_stream_chunks.jsonl",
        "chat_completion_stream_events.jsonl",
        "responses_stream.jsonl",
        "responses_stream_aggregator_events.jsonl",
    ]:
        print(f"  - {out / name}")

    print("\nTo view a JSONL file:")
    print("  cat <file.jsonl> | python -m json.tool")
    print("\nTo format JSONL (multiple objects per file):")
    print("  python -c \"import json; [print(json.dumps(json.loads(line), indent=2)) for line in open('file.jsonl')]\"")
    print("  or: jq . <file.jsonl")


def main() -> None:
    """Run all examples and write logs to tests/test_data/llm_aggregators/."""
    out = _OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("\nComprehensive OpenAI API Logging Example (test data generator)")
    print("This will create JSONL logs for all API modes.")
    print(f"Output: {out}")
    print("=" * 80)

    log_files = [
        out / "chat_completion_non_stream.json",
        out / "chat_completion_stream_chunks.jsonl",
        out / "chat_completion_stream_aggregated.json",
        out / "chat_completion_stream_events.jsonl",
        out / "responses_non_stream.json",
        out / "responses_stream.jsonl",
        out / "responses_stream_aggregator_events.jsonl",
        out / "responses_stream_aggregated.json",
    ]
    for p in log_files:
        if p.exists():
            p.unlink()
            print(f"Cleaned up existing: {p.name}")

    try:
        _example_1_chat_completion_non_streaming(out)
        _example_2_chat_completion_stream(out)
        _example_3_responses_non_streaming(out)
        _example_4_responses_stream(out)
        _print_summary(out)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Make sure the OpenAI API server is running at the configured URL.")
        raise


if __name__ == "__main__":
    main()
