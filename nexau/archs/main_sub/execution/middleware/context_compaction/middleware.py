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

"""Context compaction middleware with customizable trigger and compaction strategies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...hooks import AfterModelHookInput, HookResult, Middleware
from .config import CompactionConfig
from .factory import create_compaction_strategy, create_trigger_strategy

if TYPE_CHECKING:
    from ....utils.token_counter import TokenCounter

from nexau.core.messages import Message, Role, ToolUseBlock

logger = logging.getLogger(__name__)


class ContextCompactionMiddleware(Middleware):
    """Middleware for automatic context compaction with customizable strategies.

    This middleware monitors token usage and automatically compacts the message
    history when certain thresholds are exceeded. Both the trigger logic and
    compaction logic can be customized.

    Default behavior:
    - Trigger: When token usage reaches 75% (25% remaining)
    - Compaction: Sliding window - keeps only the most recent N messages (default: 20)
    - Simple and efficient: No summarization overhead
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize context compaction middleware.

        Args:
            token_counter: Token counter instance for fallback counting
            **kwargs: Configuration parameters validated by CompactionConfig.
                     All configuration including strategies should be passed via kwargs.
        """
        # Initialize statistics
        self._compact_count = 0
        self._total_messages_removed = 0

        # Initialize token counter
        if token_counter is None:
            from ....utils.token_counter import TokenCounter

            self.token_counter = TokenCounter()
        else:
            self.token_counter = token_counter

        # Configuration parameters are required
        if not kwargs:
            raise ValueError(
                "Configuration parameters (kwargs) are required. Provide at least the required config fields or use default values."
            )

        # Pydantic validates the flat YAML/dict here
        config = CompactionConfig(**kwargs)

        self.max_context_tokens = config.max_context_tokens
        self.auto_compact = config.auto_compact

        # Create strategies from config
        self.trigger_strategy = create_trigger_strategy(config)
        self.compaction_strategy = create_compaction_strategy(config)

        logger.info(
            f"[ContextCompactionMiddleware] Initialized: "
            f"max_context_tokens={self.max_context_tokens}, "
            f"auto_compact={self.auto_compact}, "
            f"trigger={self.trigger_strategy.__class__.__name__}, "
            f"compaction={self.compaction_strategy.__class__.__name__}"
        )

    def _get_current_tokens(self, hook_input: AfterModelHookInput) -> int | None:
        """Extract token count from model response usage information.

        Args:
            hook_input: After model hook input containing model response

        Returns:
            Total token count if available, None otherwise
        """
        if not hook_input.model_response or not hook_input.model_response.usage:
            return None

        usage = hook_input.model_response.usage

        # Usage is now in normalized format with total_tokens field
        if "total_tokens" in usage:
            return usage["total_tokens"]

        # Unknown format
        logger.warning(f"[ContextCompactionMiddleware] Unknown usage format: {usage}")
        return None

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        """Check and compact messages after each model call if needed."""
        if not self.auto_compact:
            return HookResult.no_changes()

        messages = hook_input.messages

        # Get token count from model response usage information
        current_tokens = self._get_current_tokens(hook_input)

        # If we couldn't get token count from model response, fall back to token_counter
        if current_tokens is None:
            logger.warning("[ContextCompactionMiddleware] No usage information from model response, falling back to token_counter")
            current_tokens = self.token_counter.count_tokens(messages)

        usage_ratio = current_tokens / self.max_context_tokens

        logger.info(
            f"[ContextCompactionMiddleware] Checking compaction: "
            f"{len(messages)} messages, {current_tokens}/{self.max_context_tokens} tokens ({usage_ratio:.1%})"
        )

        # Check if the last assistant message has tool calls
        # If not, skip compaction to preserve the conversation state
        last_assistant_msg: Message | None = None
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT:
                last_assistant_msg = msg
                break

        if last_assistant_msg is not None:
            has_tool_calls = any(isinstance(block, ToolUseBlock) for block in last_assistant_msg.content)

            if not has_tool_calls:
                logger.info(
                    "[ContextCompactionMiddleware] Last assistant message has no tool calls, "
                    "skipping compaction to preserve conversation state"
                )
                return HookResult.no_changes()
            else:
                logger.info("[ContextCompactionMiddleware] Last assistant message has tool calls, proceeding with compaction check")

        # Check if compaction should be triggered
        should_compact, trigger_reason = self.trigger_strategy.should_compact(
            messages,
            current_tokens,
            self.max_context_tokens,
        )

        if not should_compact:
            logger.info("[ContextCompactionMiddleware] No compaction needed")
            return HookResult.no_changes()

        logger.info(f"[ContextCompactionMiddleware] Compaction triggered: {trigger_reason}")

        # Perform compaction
        original_message_count = len(messages)
        compacted_messages = self._compact_messages(messages)
        compacted_message_count = len(compacted_messages)

        # Update statistics
        self._compact_count += 1
        self._total_messages_removed += original_message_count - compacted_message_count

        logger.info(
            f"[ContextCompactionMiddleware] Compaction complete: "
            f"{original_message_count} -> {compacted_message_count} messages "
            f"({original_message_count - compacted_message_count} removed)"
        )

        return HookResult.with_modifications(messages=compacted_messages)

    def _compact_messages(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Compact messages using the configured strategy."""
        return self.compaction_strategy.compact(messages)
