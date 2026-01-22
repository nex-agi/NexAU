#!/usr/bin/env python3
"""
Comprehensive example demonstrating logging for all OpenAI API modes:
1. Chat Completion (non-streaming)
2. Chat Completion Stream
3. Responses (non-streaming)
4. Responses Stream with Aggregator Events (combined)

This example shows how to save all data to JSONL files for debugging, testing,
and analysis purposes.
"""

import json
import os

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
TEST_PROMPT = """请先简单打个招呼并介绍一下自己,然后告诉我你最喜欢的颜色是什么以及为什么?

接着帮我做以下任务:
1. 查询北京今天的天气
2. 计算 25 * 47 的结果

最后,总结一下你刚才完成的工作。"""

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
                "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g. 2+2, 10*5"}},
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
            "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g. 2+2, 10*5"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def log_to_jsonl(data: Event | Response | ChatCompletion | ChatCompletionChunk | ResponseStreamEvent, log_file: str) -> None:
    """Append data to a JSONL file silently (no console output)."""
    json_obj = data.model_dump() if hasattr(data, "model_dump") else data
    json_str = json.dumps(json_obj, ensure_ascii=False)

    # Save to file (no console output to reduce noise)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json_str + "\n")
        f.flush()


def log_to_json(data: Event | Response | ChatCompletion, log_file: str) -> None:
    """Save data to a JSON file (not JSONL) silently (no console output)."""
    json_obj = data.model_dump() if hasattr(data, "model_dump") else data
    json_str = json.dumps(json_obj, ensure_ascii=False, indent=2)

    # Save to file (no console output to reduce noise)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(json_str + "\n")


def create_event_logger(log_file: str):
    """Create an event logger that saves aggregator events to JSONL."""

    def log_event(event: Event) -> None:
        log_to_jsonl(event, log_file)

    return log_event


def example_1_chat_completion_non_streaming():
    """
    Example 1: Chat Completion (non-streaming)
    Saves the final ChatCompletion object.
    """
    print("\n" + "=" * 80)
    print("Example 1: Chat Completion (non-streaming)")
    print("=" * 80)

    log_file = "chat_completion_non_stream.json"

    # Make the request
    completion = client.chat.completions.create(
        model=CHAT_COMPLETION_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        tools=TOOLS,
    )

    # Save to JSON
    log_to_json(completion, log_file)
    print(f"✓ Chat completion saved to: {log_file}")
    print(f"  - ID: {completion.id}")
    print(f"  - Model: {completion.model}")
    print(f"  - Choices: {len(completion.choices)}")
    print(f"  - Finish reason: {completion.choices[0].finish_reason if completion.choices else 'N/A'}")


def example_2_chat_completion_stream():
    """
    Example 2: Chat Completion Stream
    Saves all chunks and optionally aggregates them.
    """
    print("\n" + "=" * 80)
    print("Example 2: Chat Completion Stream")
    print("=" * 80)

    chunk_log_file = "chat_completion_stream_chunks.jsonl"
    aggregated_log_file = "chat_completion_stream_aggregated.json"

    # Create stream
    stream = client.chat.completions.create(
        model=CHAT_COMPLETION_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        tools=TOOLS,
        stream=True,
    )

    # Initialize aggregator
    event_logger = create_event_logger("chat_completion_stream_events.jsonl")
    aggregator = OpenAIChatCompletionAggregator(on_event=event_logger, run_id="example-run")

    chunk_count = 0
    print("Processing stream chunks...")

    # Process stream
    for chunk in stream:
        chunk_count += 1

        # Save chunk
        log_to_jsonl(chunk, chunk_log_file)

        # Aggregate chunk
        aggregator.aggregate(chunk)

        # Print progress (show text content as it arrives)
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()
    print("✓ Stream processing complete:")
    print(f"  - Chunks saved to: {chunk_log_file}")
    print(f"  - Total chunks: {chunk_count}")

    # Build and save final aggregated completion
    final_completion = aggregator.build()
    log_to_json(final_completion, aggregated_log_file)
    print(f"  - Aggregated completion saved to: {aggregated_log_file}")
    print(f"  - Final content: {final_completion.choices[0].message.content}")
    print("  - Events logged to: chat_completion_stream_events.jsonl")


def example_3_responses_non_streaming():
    """
    Example 3: Responses (non-streaming)
    Saves the final Response object.
    """
    print("\n" + "=" * 80)
    print("Example 3: Responses (non-streaming)")
    print("=" * 80)

    log_file = "responses_non_stream.json"

    # Make the request
    response = client.responses.create(
        model=RESPONSES_MODEL,
        reasoning={"effort": "high"},
        input=[{"role": "user", "type": "message", "content": TEST_PROMPT}],
        tools=RESPONSES_TOOLS,
    )

    # Save to JSON
    log_to_json(response, log_file)
    print(f"✓ Response saved to: {log_file}")
    print(f"  - ID: {response.id}")
    print(f"  - Model: {response.model}")
    print(f"  - Output: {len(response.output)} items")


def example_4_responses_stream():
    """
    Example 4: Responses Stream
    Demonstrates streaming with both raw events and aggregated output.
    """
    print("\n" + "=" * 80)
    print("Example 4: Responses Stream")
    print("=" * 80)

    event_log_file = "responses_stream.jsonl"
    aggregator_event_log_file = "responses_stream_aggregator_events.jsonl"
    aggregated_log_file = "responses_stream_aggregated.json"

    # Create stream
    stream = client.responses.create(
        model=RESPONSES_MODEL,
        reasoning={"effort": "high"},
        input=[{"role": "user", "type": "message", "content": TEST_PROMPT}],
        tools=RESPONSES_TOOLS,
        stream=True,
    )

    # Initialize aggregator
    event_logger = create_event_logger(aggregator_event_log_file)
    aggregator = OpenAIResponsesAggregator(on_event=event_logger, run_id="example-run")

    event_count = 0
    print("Processing stream events...")

    # Process stream
    for event in stream:
        event_count += 1

        # Save raw event
        log_to_jsonl(event, event_log_file)

        # Aggregate event
        aggregator.aggregate(event)

        # Print progress (show text content or function calls as they arrive)
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.function_call_arguments.delta":
            print(event.delta, end="", flush=True)

    print()
    print("✓ Stream processing complete:")
    print(f"  - Raw events saved to: {event_log_file}")
    print(f"  - Total events processed: {event_count}")

    # Build and save final aggregated response
    final_response = aggregator.build()
    log_to_json(final_response, aggregated_log_file)
    print(f"  - Aggregated response saved to: {aggregated_log_file}")
    print(f"  - Aggregator events logged to: {aggregator_event_log_file}")
    print(f"  - Output items: {len(final_response.output)}")

    # Print event types
    print("\nAggregator emits these event types:")
    response_event_types = [
        ("TEXT_MESSAGE_START", "Message begins"),
        ("TEXT_MESSAGE_CONTENT", "Text content chunk received"),
        ("TEXT_MESSAGE_END", "Message complete"),
        ("THINKING_TEXT_MESSAGE_START", "Thinking/reasoning content begins"),
        ("THINKING_TEXT_MESSAGE_CONTENT", "Thinking content chunk"),
        ("THINKING_TEXT_MESSAGE_END", "Thinking content complete"),
        ("TOOL_CALL_START", "Tool call begins"),
        ("TOOL_CALL_ARGS", "Tool call arguments chunk"),
        ("TOOL_CALL_END", "Tool call complete"),
    ]
    for event_type, description in response_event_types:
        print(f"  - {event_type:35s} {description}")


def print_summary():
    """Print summary of all log files."""
    print("\n" + "=" * 80)
    print("SUMMARY - Generated Log Files")
    print("=" * 80)

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
    for file_name, description in log_files:
        print(f"  - {file_name:50s} ({description})")

    print("\nJSON files (single objects, pretty-printed):")
    json_files = [
        "chat_completion_non_stream.json",
        "chat_completion_stream_aggregated.json",
        "responses_non_stream.json",
        "responses_stream_aggregated.json",
    ]
    for file_name in json_files:
        print(f"  - {file_name}")

    print("\nJSONL files (multiple objects, one per line):")
    jsonl_files = [
        "chat_completion_stream_chunks.jsonl",
        "chat_completion_stream_events.jsonl",
        "responses_stream.jsonl",
        "responses_stream_aggregator_events.jsonl",
    ]
    for file_name in jsonl_files:
        print(f"  - {file_name}")

    print("\nTo view a JSONL file:")
    print("  cat <file.jsonl> | python -m json.tool")
    print("\nFor JSON files (already pretty-printed):")
    print("  cat <file.json>")
    print("\nTo format JSONL (multiple objects per file):")
    print("  python -c \"import json; [print(json.dumps(json.loads(line), indent=2)) for line in open('file.jsonl')]\"")
    print("  or:")
    print("  jq . <file.jsonl")

    print("\nEvent types (chat_completion_stream_events.jsonl):")
    event_types = [
        ("TEXT_MESSAGE_START", "Message begins"),
        ("TEXT_MESSAGE_CONTENT", "Text content chunk received"),
        ("TEXT_MESSAGE_CHUNK", "Unified text chunk"),
        ("TEXT_MESSAGE_END", "Message complete"),
        ("TOOL_CALL_START", "Tool call begins"),
        ("TOOL_CALL_ARGS", "Tool call arguments chunk"),
        ("TOOL_CALL_CHUNK", "Unified tool call chunk"),
        ("TOOL_CALL_END", "Tool call complete"),
    ]
    for event_type, description in event_types:
        print(f"  - {event_type:30s} {description}")

    print("\nEvent types (responses_stream_aggregator_events.jsonl):")
    response_event_types = [
        ("TEXT_MESSAGE_START", "Message begins"),
        ("TEXT_MESSAGE_CONTENT", "Text content chunk"),
        ("TEXT_MESSAGE_END", "Message complete"),
        ("THINKING_TEXT_MESSAGE_START", "Thinking content begins"),
        ("THINKING_TEXT_MESSAGE_CONTENT", "Thinking content chunk"),
        ("THINKING_TEXT_MESSAGE_END", "Thinking content complete"),
        ("TOOL_CALL_START", "Tool call begins"),
        ("TOOL_CALL_ARGS", "Tool call arguments chunk"),
        ("TOOL_CALL_END", "Tool call complete"),
    ]
    for event_type, description in response_event_types:
        print(f"  - {event_type:35s} {description}")


def main():
    """Run all examples."""
    print("\nComprehensive OpenAI API Logging Example")
    print("This will create JSONL logs for all API modes.")
    print("=" * 80)

    # Clean or create fresh log files
    import os

    log_files = [
        "chat_completion_non_stream.json",
        "chat_completion_stream_chunks.jsonl",
        "chat_completion_stream_aggregated.json",
        "chat_completion_stream_events.jsonl",
        "responses_non_stream.json",
        "responses_stream.jsonl",
        "responses_stream_aggregator_events.jsonl",
        "responses_stream_aggregated.json",
    ]
    for log_file in log_files:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"Cleaned up existing: {log_file}")

    try:
        # Run all examples
        example_1_chat_completion_non_streaming()
        example_2_chat_completion_stream()
        example_3_responses_non_streaming()
        example_4_responses_stream()

        print_summary()

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Make sure the OpenAI API server is running at the configured URL.")
        raise


if __name__ == "__main__":
    main()
