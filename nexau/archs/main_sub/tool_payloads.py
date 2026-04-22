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

"""Provider-specific structured tool payload builders.

RFC-0015: Removed sub_agents parameter and build_sub_agent_tool_name calls.
Agent is now a regular builtin tool registered in ToolRegistry via
AgentConfig._finalize(), so tool_payloads only needs to iterate over
eager tools — no special sub-agent handling required.
"""

from collections.abc import Sequence
from typing import Any, cast

from anthropic.types import ToolParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.tool.tool import Tool


def build_openai_tool_payload(
    *,
    eager_tools: Sequence[Tool],
) -> list[ChatCompletionToolParam]:
    """Build OpenAI-compatible structured tool payload."""
    tools_payload: list[ChatCompletionToolParam] = []

    for tool in eager_tools:
        tool_spec = tool.to_openai()
        try:
            function_block = cast(Any, tool_spec).get("function")
            if isinstance(function_block, dict):
                function_block["description"] = structured_tool_description(tool)
        except (AttributeError, KeyError, TypeError):
            pass
        tools_payload.append(tool_spec)

    return tools_payload


def build_anthropic_tool_payload(
    *,
    eager_tools: Sequence[Tool],
    tool_streaming: bool = True,
) -> list[ToolParam]:
    """Build Anthropic-compatible structured tool payload.

    Parameters
    ----------
    tool_streaming:
        Forwarded to :meth:`Tool.to_anthropic`.  When *False*,
        ``eager_input_streaming`` is omitted from the tool schema.
    """
    tools_payload: list[ToolParam] = []

    for tool in eager_tools:
        tool_spec = tool.to_anthropic(tool_streaming=tool_streaming)
        try:
            tool_spec["description"] = structured_tool_description(tool)
        except (KeyError, TypeError):
            pass
        tools_payload.append(tool_spec)

    return tools_payload


def structured_tool_description(tool: Tool) -> str:
    """Return the description exposed to structured tool-calling models."""
    if tool.as_skill:
        if not tool.skill_description:
            raise ValueError(
                f"Tool {tool.name} is marked as a skill but has no skill_description",
            )
        return tool.skill_description
    return tool.description or ""
