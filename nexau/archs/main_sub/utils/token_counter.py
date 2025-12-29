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

"""Token counting utilities for agents."""

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Final, cast

from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
from nexau.core.messages import Message

logger = logging.getLogger(__name__)

type LegacyOpenAIChatMessage = dict[str, Any]
type TokenCountableMessage = Message | LegacyOpenAIChatMessage

_tiktoken: Any
try:
    import tiktoken as _tiktoken
except ImportError:
    _tiktoken = None

tiktoken: Any | None = _tiktoken
TIKTOKEN_AVAILABLE: Final[bool] = _tiktoken is not None


class TokenCounter:
    """Handles token counting for LLM messages using various strategies."""

    def __init__(self, strategy: str = "tiktoken", model: str = "gpt-4o"):
        """Initialize token counter with specified strategy.

        Args:
            strategy: "tiktoken" or "fallback"
            model: Model name for tiktoken encoding
        """
        self.strategy = strategy
        self.model = model
        self._counter = self._create_counter()

    def _create_counter(
        self,
    ) -> Callable[[Sequence[TokenCountableMessage], list[dict[str, Any]] | None], int]:
        """Create the appropriate token counter based on strategy."""
        if self.strategy == "tiktoken" and TIKTOKEN_AVAILABLE:
            return self._create_tiktoken_counter()
        else:
            if self.strategy == "tiktoken":
                logger.warning(
                    "tiktoken not available, using fallback counter",
                )
            return self._create_fallback_counter()

    def _create_tiktoken_counter(
        self,
    ) -> Callable[[Sequence[TokenCountableMessage], list[dict[str, Any]] | None], int]:
        """Create tiktoken-based counter."""
        if tiktoken is None:
            raise RuntimeError("tiktoken is not available")
        try:
            encoding = tiktoken.encoding_for_model(self.model)

            def tiktoken_message_counter(
                messages: Sequence[TokenCountableMessage],
                tools: list[dict[str, Any]] | None = None,
            ) -> int:
                """Count tokens in messages using tiktoken."""
                legacy_messages: list[LegacyOpenAIChatMessage]
                if messages and isinstance(messages[0], Message):
                    legacy_messages = messages_to_legacy_openai_chat(cast(list[Message], list(messages)))
                else:
                    legacy_messages = cast(list[LegacyOpenAIChatMessage], list(messages))
                total_tokens = 0
                for message in legacy_messages:
                    # Add tokens for role and content
                    total_tokens += len(
                        encoding.encode(
                            message.get("content", ""),
                            allowed_special=set(),
                        ),
                    )
                    reasoning = message.get("reasoning_content", "")
                    if reasoning:
                        total_tokens += len(
                            encoding.encode(
                                reasoning,
                                allowed_special=set(),
                            ),
                        )
                    if tool_calls := message.get("tool_calls"):
                        total_tokens += self._count_tiktoken_tool_calls(
                            tool_calls,
                            encoding,
                        )
                if tools:
                    total_tokens += self._count_tiktoken_tools(tools, encoding)
                return total_tokens

            return tiktoken_message_counter
        except Exception as e:
            logger.warning(
                f"Failed to create tiktoken encoder: {e}, using fallback",
            )
            return self._create_fallback_counter()

    def _create_fallback_counter(
        self,
    ) -> Callable[[Sequence[TokenCountableMessage], list[dict[str, Any]] | None], int]:
        """Create fallback counter using character approximation."""

        def fallback_message_counter(
            messages: Sequence[TokenCountableMessage],
            tools: list[dict[str, Any]] | None = None,
        ) -> int:
            """Fallback token counter using character approximation."""
            legacy_messages: list[LegacyOpenAIChatMessage]
            if messages and isinstance(messages[0], Message):
                legacy_messages = messages_to_legacy_openai_chat(cast(list[Message], list(messages)))
            else:
                legacy_messages = cast(list[LegacyOpenAIChatMessage], list(messages))
            total_tokens = 0
            for message in legacy_messages:
                # Add tokens for role and content using chars/4 approximation
                total_tokens += len(message.get("role", "")) // 4
                total_tokens += len(message.get("content", "")) // 4
                total_tokens += len(message.get("reasoning_content", "")) // 4
                # Add overhead tokens for message formatting
                total_tokens += 4
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    total_tokens += self._count_fallback_tool_calls(tool_calls)
            if tools:
                total_tokens += self._count_fallback_tools(tools)
            return max(total_tokens, 1)  # Ensure at least 1 token

        return fallback_message_counter

    def _count_tiktoken_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        encoding: Any,
    ) -> int:
        """Count tokens contributed by tool calls using tiktoken."""
        total_tokens = 0
        for call in tool_calls:
            # Base overhead per tool call based on OpenAI cookbook guidance
            total_tokens += 3

            function_data = call.get("function")
            if not isinstance(function_data, dict):
                continue

            function_data = cast(dict[str, Any], function_data)

            total_tokens += len(
                encoding.encode(
                    str(function_data.get("name", "")),
                    allowed_special=set(),
                ),
            )

            args_value = function_data.get("arguments", "")
            args_value = cast(dict[str, Any], args_value)
            try:
                args_str = json.dumps(args_value)
            except TypeError:
                args_str = str(args_value)
            total_tokens += len(
                encoding.encode(args_str, allowed_special=set()),
            )

        return total_tokens

    def _count_tiktoken_tools(
        self,
        tools: list[dict[str, Any]],
        encoding: Any,
    ) -> int:
        """Count tokens contributed by tool definitions using tiktoken."""
        total_tokens = 0
        for tool in tools:
            try:
                tool_str = json.dumps(tool, ensure_ascii=False)
            except TypeError:
                tool_str = str(tool)
            total_tokens += len(encoding.encode(tool_str, allowed_special=set()))
        return total_tokens

    def _count_fallback_tool_calls(self, tool_calls: list[dict[str, Any]]) -> int:
        """Approximate tokens contributed by tool calls without tiktoken."""
        total_tokens = 0
        for call in tool_calls:
            total_tokens += 3

            function_data = call.get("function")
            if not isinstance(function_data, dict):
                continue

            function_data = cast(dict[str, Any], function_data)

            total_tokens += len(str(function_data.get("name", ""))) // 4
            args_value = function_data.get("arguments", "")
            args_value = cast(dict[str, Any], args_value)
            try:
                args_str = json.dumps(args_value)
            except TypeError:
                args_str = str(args_value)
            total_tokens += len(args_str) // 4

        return total_tokens

    def _count_fallback_tools(self, tools: list[dict[str, Any]]) -> int:
        """Approximate tokens contributed by tool definitions without tiktoken."""
        total_tokens = 0
        for tool in tools:
            try:
                tool_str = json.dumps(tool)
            except TypeError:
                tool_str = str(tool)
            total_tokens += len(tool_str) // 4
        return total_tokens

    def count_tokens(
        self,
        messages: Sequence[TokenCountableMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count total tokens in a list of messages.

        Args:
            messages: UMP `Message` objects (or legacy OpenAI-style dict messages for backward compatibility).
            tools: Optional list of tool definitions to include in token count

        Returns:
            Total token count
        """
        return self._counter(messages, tools)
