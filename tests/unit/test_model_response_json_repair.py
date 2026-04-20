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

import json

from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall


def test_model_tool_call_from_openai_repairs_malformed_arguments_and_serializes_valid_json() -> None:
    """Regression: malformed function.arguments must not be forwarded as invalid JSON in later rounds."""
    malformed = '{"file_path": "/workspace/MiniMax_Research_Report_2026.md"'
    call = {
        "id": "tooluse_gTSISHixQkS6KV-yLIHrbg",
        "type": "function",
        "function": {"name": "Write", "arguments": malformed},
    }

    tool_call = ModelToolCall.from_openai(call)

    assert tool_call.arguments == {"file_path": "/workspace/MiniMax_Research_Report_2026.md"}
    assert isinstance(tool_call.raw_arguments, str)
    assert json.loads(tool_call.raw_arguments) == tool_call.arguments

    openai_dict = tool_call.to_openai_dict()
    assert json.loads(openai_dict["function"]["arguments"]) == {"file_path": "/workspace/MiniMax_Research_Report_2026.md"}


def test_model_response_from_openai_message_rewrites_tool_call_arguments_to_valid_json() -> None:
    """End-to-end for the pattern in the bug report: malformed tool_calls must become safe for next round."""
    malformed = '{"file_path": "/workspace/MiniMax_Research_Report_2026.md"'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "tooluse_gTSISHixQkS6KV-yLIHrbg",
                "type": "function",
                "function": {"name": "Write", "arguments": malformed},
            }
        ],
    }

    response = ModelResponse.from_openai_message(message)
    assert response.role == "assistant"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments["file_path"] == "/workspace/MiniMax_Research_Report_2026.md"

    # This is what gets fed to the provider in the next request.
    next_round_message = response.to_message_dict()
    serialized_args = next_round_message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(serialized_args) == {"file_path": "/workspace/MiniMax_Research_Report_2026.md"}


def test_model_response_from_openai_message_preserves_reasoning_details_verbatim() -> None:
    """OpenRouter requires `reasoning_details` to be echoed back unmodified on later turns.

    The structured list must survive ModelResponse parsing and serialization without being
    flattened into reasoning_content or otherwise mutated.
    """
    details = [
        {
            "type": "reasoning.summary",
            "summary": "Analyzed by decomposition",
            "id": "reasoning-summary-1",
            "format": "anthropic-claude-v1",
            "index": 0,
        },
        {
            "type": "reasoning.text",
            "text": "Step-by-step details here.",
            "signature": None,
            "id": "reasoning-text-1",
            "format": "anthropic-claude-v1",
            "index": 1,
        },
    ]
    message = {
        "role": "assistant",
        "content": "Recommendation follows.",
        "reasoning_details": details,
    }

    response = ModelResponse.from_openai_message(message)

    # reasoning_content is NOT synthesized from details — the two fields are independent.
    assert response.reasoning_content is None
    assert response.reasoning_details == details

    # Next-turn dict echoes the structured list back to the provider unchanged.
    next_round = response.to_message_dict()
    assert next_round["reasoning_details"] == details
    assert "reasoning_content" not in next_round


def test_model_response_from_openai_message_reasoning_content_and_details_coexist() -> None:
    """Providers may send both (DeepSeek-style string + OpenRouter-style list) — keep both."""
    details = [{"type": "reasoning.text", "text": "structured"}]
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "flat string",
        "reasoning_details": details,
    }

    response = ModelResponse.from_openai_message(message)

    assert response.reasoning_content == "flat string"
    assert response.reasoning_details == details


def test_model_response_to_ump_message_carries_reasoning_details_in_metadata() -> None:
    """to_ump_message must stash reasoning_details in Message.metadata for echo-back."""
    details = [{"type": "reasoning.text", "text": "verbatim"}]
    response = ModelResponse(
        role="assistant",
        content="ok",
        reasoning_details=details,
    )

    msg = response.to_ump_message()

    assert msg.metadata["reasoning_details"] == details


def test_tool_call_from_response_item_repairs_malformed_arguments_and_serializes_valid_json() -> None:
    malformed = '{"file_path": "/workspace/MiniMax_Research_Report_2026.md"'
    item = {
        "type": "function",
        "id": "call_1",
        "name": "Write",
        "arguments": malformed,
    }

    tool_call = ModelResponse._tool_call_from_response_item(item)
    assert tool_call is not None
    assert tool_call.arguments == {"file_path": "/workspace/MiniMax_Research_Report_2026.md"}
    assert isinstance(tool_call.raw_arguments, str)
    assert json.loads(tool_call.raw_arguments) == tool_call.arguments


def test_model_tool_call_from_openai_unparseable_arguments_are_wrapped_as_valid_json_object() -> None:
    """Even if repair fails, raw_arguments must still be a valid JSON object string."""
    call = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "Write", "arguments": "not-json"},
    }

    tool_call = ModelToolCall.from_openai(call)

    assert "raw_arguments" in tool_call.arguments
    assert tool_call.arguments["raw_arguments"] == "not-json"
    assert isinstance(tool_call.raw_arguments, str)
    assert json.loads(tool_call.raw_arguments) == {"raw_arguments": "not-json"}
