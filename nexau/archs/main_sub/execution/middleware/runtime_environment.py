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

"""Middleware for injecting runtime environment facts into the system prompt."""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import Template

from nexau.core.messages import Message, Role, TextBlock

from ..hooks import BeforeAgentHookInput, HookResult, Middleware

logger = logging.getLogger(__name__)


DEFAULT_RUNTIME_ENVIRONMENT_TEMPLATE = """# Runtime Environment

- Current directory: {{ current_directory or "Unknown" }}
- Operating system: {{ operating_system or "Unknown" }}
- Shell backend: {{ shell_tool_backend or "Unknown" }}"""

_INJECTED_METADATA_KEY = "runtime_environment_injected"


class RuntimeEnvironmentMiddleware(Middleware):
    """Append selected runtime facts to the first system prompt message.

    The middleware intentionally injects only stable platform facts. More
    volatile values such as date, username, or arbitrary environment variables
    should stay in explicit user templates instead of being injected by default.
    """

    def __init__(
        self,
        *,
        template: str = DEFAULT_RUNTIME_ENVIRONMENT_TEMPLATE,
    ) -> None:
        self.template = template

    def before_agent(self, hook_input: BeforeAgentHookInput) -> HookResult:
        """Append rendered runtime facts to the first system message."""

        system_index = self._first_system_message_index(hook_input.messages)
        if system_index is None:
            return HookResult.no_changes()

        system_message = hook_input.messages[system_index]
        if system_message.metadata.get(_INJECTED_METADATA_KEY):
            return HookResult.no_changes()

        rendered = self._render_environment(hook_input)
        if not rendered:
            return HookResult.no_changes()

        updated_messages = list(hook_input.messages)
        updated_messages[system_index] = self._append_to_message(system_message, rendered)

        logger.info("[RuntimeEnvironmentMiddleware] Injected runtime environment into system prompt")
        return HookResult.with_modifications(messages=updated_messages)

    def _render_environment(self, hook_input: BeforeAgentHookInput) -> str:
        values = self._runtime_values(hook_input)
        if not any(values.values()):
            return ""
        return Template(self.template).render(**values).strip()

    @staticmethod
    def _runtime_values(hook_input: BeforeAgentHookInput) -> dict[str, Any]:
        working_directory = hook_input.agent_state.get_context_value("working_directory")
        operating_system = hook_input.agent_state.get_context_value("operating_system")
        shell_tool_backend = hook_input.agent_state.get_context_value("shell_tool_backend")

        return {
            "current_directory": working_directory,
            "working_directory": working_directory,
            "operating_system": operating_system,
            "shell_tool_backend": shell_tool_backend,
            "shell_backend": shell_tool_backend,
        }

    @staticmethod
    def _first_system_message_index(messages: list[Message]) -> int | None:
        for index, message in enumerate(messages):
            if message.role == Role.SYSTEM:
                return index
        return None

    @staticmethod
    def _append_to_message(message: Message, rendered_environment: str) -> Message:
        blocks = list(message.content)
        for index in range(len(blocks) - 1, -1, -1):
            block = blocks[index]
            if isinstance(block, TextBlock):
                blocks[index] = TextBlock(text=f"{block.text.rstrip()}\n\n{rendered_environment}")
                break
        else:
            blocks.append(TextBlock(text=rendered_environment))

        metadata = {**message.metadata, _INJECTED_METADATA_KEY: True}
        return message.model_copy(update={"content": blocks, "metadata": metadata})
