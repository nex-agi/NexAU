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

"""Provider-specific structured tool payload builders."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from anthropic.types import ToolParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.sub_agent_naming import build_sub_agent_tool_name
from nexau.archs.tool.tool import Tool


def build_openai_tool_payload(
    *,
    eager_tools: Sequence[Tool],
    sub_agents: Mapping[str, AgentConfig] | None = None,
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

    for sub_agent_name, sub_agent_config in _iter_sub_agents(sub_agents):
        tools_payload.append(
            {
                "type": "function",
                "function": {
                    "name": build_sub_agent_tool_name(sub_agent_name),
                    "description": sub_agent_config.description or f"Delegate work to sub-agent '{sub_agent_name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Task or question for the sub-agent.",
                            },
                        },
                        "required": ["message"],
                    },
                },
            },
        )

    return tools_payload


def build_anthropic_tool_payload(
    *,
    eager_tools: Sequence[Tool],
    sub_agents: Mapping[str, AgentConfig] | None = None,
) -> list[ToolParam]:
    """Build Anthropic-compatible structured tool payload."""
    tools_payload: list[ToolParam] = []

    for tool in eager_tools:
        tool_spec = tool.to_anthropic()
        try:
            tool_spec["description"] = structured_tool_description(tool)
        except (KeyError, TypeError):
            pass
        tools_payload.append(tool_spec)

    for sub_agent_name, sub_agent_config in _iter_sub_agents(sub_agents):
        tools_payload.append(
            {
                "name": build_sub_agent_tool_name(sub_agent_name),
                "description": sub_agent_config.description or f"Delegate work to sub-agent '{sub_agent_name}'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Task or question for the sub-agent.",
                        },
                    },
                    "required": ["message"],
                },
            },
        )

    return tools_payload


def _iter_sub_agents(
    sub_agents: Mapping[str, AgentConfig] | None,
) -> Sequence[tuple[str, AgentConfig]]:
    """Return sub-agent items while tolerating test doubles."""
    if isinstance(sub_agents, Mapping):
        return tuple(sub_agents.items())
    return ()


def structured_tool_description(tool: Tool) -> str:
    """Return the description exposed to structured tool-calling models."""
    if tool.as_skill:
        if not tool.skill_description:
            raise ValueError(
                f"Tool {tool.name} is marked as a skill but has no skill_description",
            )
        return tool.skill_description
    return tool.description or ""
