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

"""Data structures for parsed tool calls.

RFC-0015: Removed BatchAgentCall and CallType enum; ExecutableCall is now just ToolCall.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from .model_response import ModelResponse

ToolCallSource = Literal["xml", "structured"]


@dataclass
class ToolCall:
    """Represents a parsed tool call."""

    tool_name: str
    parameters: dict[str, Any]
    raw_content: str | None = None  # Original representation for error reporting/debugging
    tool_call_id: str | None = None
    source: ToolCallSource = "xml"
    parallel_execution_id: str | None = None  # ID for grouping parallel executions

    def __post_init__(self):
        if self.tool_call_id is None:
            self.tool_call_id = "tool_call_" + str(uuid.uuid4())

    @property
    def xml_content(self) -> str | None:  # pragma: no cover - backward compatibility
        return self.raw_content


# Union type for all call types — simplified to ToolCall after BatchAgentCall removal (RFC-0015)
ExecutableCall = ToolCall


@dataclass
class ParsedResponse:
    """Container for all parsed calls from an LLM response.

    RFC-0015: Removed sub_agent_calls and batch_agent_calls fields.
    """

    original_response: str
    tool_calls: list[ToolCall]
    is_parallel_tools: bool = False
    model_response: ModelResponse | None = None

    def get_all_calls(self) -> list[ExecutableCall]:
        """Get all calls in execution order."""
        return list(self.tool_calls)

    def has_calls(self) -> bool:
        """Check if there are any calls to execute."""
        return bool(self.tool_calls)

    def get_call_summary(self) -> str:
        """Get a summary of all calls."""
        if self.tool_calls:
            return f"{len(self.tool_calls)} tool calls"
        return "no calls"
