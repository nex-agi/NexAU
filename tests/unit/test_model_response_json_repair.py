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
