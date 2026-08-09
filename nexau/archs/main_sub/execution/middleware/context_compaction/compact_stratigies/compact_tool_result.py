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

"""Tool result compaction strategy."""

import json
import logging
from typing import Any, ClassVar

from nexau.core.messages import BlockType, Message, Role, ToolResultBlock, ToolUseBlock

logger = logging.getLogger(__name__)


def _truncate_large_strings(obj: Any, max_length: int = 100) -> Any:
    """Recursively truncates large strings in tool arguments."""
    if isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:50] + f"...<compacted {len(obj)} chars>..." + obj[-50:]
        return obj
    elif isinstance(obj, dict):
        return {k: _truncate_large_strings(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_truncate_large_strings(v, max_length) for v in obj]
    return obj


class ToolResultCompaction:
    """Tool result compaction strategy - compacts old tool execution results.

    An "iteration" is bounded by ASSISTANT messages:
    [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

    - Each ASSISTANT message starts a new iteration
    - USER or FRAMEWORK messages immediately before the ASSISTANT are part of that iteration (optional)
    - TOOL results after the ASSISTANT are part of that iteration (optional)

    This strategy preserves:
    1. System prompt (if keep_system=True)
    2. All user/assistant/framework messages (in full)
    3. Tool results in the last N iterations (in full, controlled by keep_iterations)
    4. Older tool results with compacted content: "Tool call result has been compacted"
    """

    # RFC-0026: stable canonical name persisted into CompactAutoVariant.strategy.
    name: ClassVar[str] = "tool_result"

    def __init__(
        self,
        keep_system: bool = True,
        keep_iterations: int = 3,
        keep_user_rounds: int = 0,
        compactable_tools: frozenset[str] | None = None,
    ):
        """Initialize tool result compaction.

        micro-compact: 增强 ToolResultCompaction，支持工具类型过滤

        Args:
            keep_system: Whether to preserve the system message. Default: True.
            keep_iterations: Number of recent iterations to keep tool results uncompacted. Default: 3.
            keep_user_rounds: Number of recent user rounds to keep uncompacted. Default: 0 (disabled).
                When > 0, uses user rounds mode instead of iterations mode.
            compactable_tools: Tool names eligible for compaction. None = compact all tools (backward compat).
                When set, only ToolResultBlocks whose corresponding ToolUseBlock.name is in this set
                will be compacted; other tool results are preserved intact.

        Raises:
            ValueError: If both keep_iterations != 3 and keep_user_rounds > 0 are set.
            ValueError: If keep_iterations < 1 or keep_user_rounds < 0.
        """
        if keep_iterations != 3 and keep_user_rounds > 0:
            raise ValueError("Cannot set both keep_iterations and keep_user_rounds")

        if keep_iterations < 1:
            raise ValueError(f"keep_iterations must be >= 1, got {keep_iterations}")
        if keep_user_rounds < 0:
            raise ValueError(f"keep_user_rounds must be >= 0, got {keep_user_rounds}")

        self.keep_system = keep_system
        self.keep_iterations = keep_iterations
        self.keep_user_rounds = keep_user_rounds
        self.compactable_tools = compactable_tools

        logger.info(
            f"[ToolResultCompaction] Initialized: keep_system={self.keep_system}, "
            f"keep_iterations={self.keep_iterations}, keep_user_rounds={self.keep_user_rounds}, "
            f"compactable_tools={self.compactable_tools}"
        )

    def compact(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Compact messages by compressing old tool results."""
        logger.info(f"[ToolResultCompaction] Starting compaction on {len(messages)} messages")

        result: list[Message] = []
        start_idx = 0

        # Keep system message if present
        if self.keep_system and messages and messages[0].role == Role.SYSTEM:
            result.append(messages[0])
            start_idx = 1

        # Determine protected message indices based on keep_user_rounds or keep_iterations
        protected_indices: set[int] = set()

        if self.keep_user_rounds > 0:
            # Use keep_user_rounds logic
            user_rounds = self._group_into_user_rounds(messages[start_idx:])
            if len(user_rounds) > self.keep_user_rounds:
                rounds_to_protect = user_rounds[-self.keep_user_rounds :]
            else:
                rounds_to_protect = user_rounds

            # Collect indices of messages in protected user rounds
            for round_msgs in rounds_to_protect:
                for msg in round_msgs:
                    for i in range(start_idx, len(messages)):
                        if messages[i] is msg:
                            protected_indices.add(i)
                            break
        else:
            # Use keep_iterations logic (original behavior)
            iterations = self._group_into_iterations(messages[start_idx:])
            if len(iterations) > self.keep_iterations:
                iterations_to_protect = iterations[-self.keep_iterations :]
            else:
                iterations_to_protect = iterations

            # Collect indices of messages in protected iterations
            for iteration_msgs in iterations_to_protect:
                for msg in iteration_msgs:
                    for i in range(start_idx, len(messages)):
                        if messages[i] is msg:
                            protected_indices.add(i)
                            break

        # micro-compact: 构建 tool_use_id → tool_name 映射，用于 compactable_tools 过滤
        tool_use_id_to_name: dict[str, str] = {}
        if self.compactable_tools is not None:
            for msg in messages:
                if msg.role == Role.ASSISTANT:
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            tool_use_id_to_name[block.id] = block.name

        # Pass 1: Identify which tool_use_ids are being compacted
        compacted_tool_use_ids: set[str] = set()
        for i in range(start_idx, len(messages)):
            msg = messages[i]
            if msg.role == Role.TOOL and i not in protected_indices:
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        should_compact_block = True
                        if self.compactable_tools is not None:
                            tool_name = tool_use_id_to_name.get(block.tool_use_id, "")
                            should_compact_block = tool_name in self.compactable_tools
                        if should_compact_block:
                            compacted_tool_use_ids.add(block.tool_use_id)

        # Pass 2: Rebuild messages, compacting target ToolResultBlocks and ToolUseBlocks
        compacted_count = 0
        for i in range(start_idx, len(messages)):
            msg = messages[i]
            if msg.role == Role.TOOL and i not in protected_indices:
                # Compact this tool result (respecting compactable_tools filter)
                new_blocks: list[BlockType] = []
                any_compacted = False
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.tool_use_id in compacted_tool_use_ids:
                        new_blocks.append(
                            ToolResultBlock(
                                tool_use_id=block.tool_use_id,
                                content="Tool call result has been compacted",
                                is_error=block.is_error,
                            ),
                        )
                        any_compacted = True
                    else:
                        new_blocks.append(block)
                if any_compacted:
                    result.append(msg.model_copy(update={"content": new_blocks}))
                    compacted_count += 1
                else:
                    result.append(msg)
            elif msg.role == Role.ASSISTANT:
                # Truncate ToolUseBlock payloads if their result was compacted
                new_blocks: list[BlockType] = []
                any_compacted = False
                for block in msg.content:
                    if isinstance(block, ToolUseBlock) and block.id in compacted_tool_use_ids:
                        truncated_input = _truncate_large_strings(block.input)
                        truncated_raw_input = block.raw_input
                        if truncated_raw_input is not None:
                            try:
                                parsed = json.loads(truncated_raw_input)
                                truncated_raw_input = json.dumps(_truncate_large_strings(parsed), ensure_ascii=False)
                            except Exception:
                                pass # fallback if parsing fails
                        new_blocks.append(block.model_copy(update={"input": truncated_input, "raw_input": truncated_raw_input}))
                        any_compacted = True
                    else:
                        new_blocks.append(block)
                if any_compacted:
                    result.append(msg.model_copy(update={"content": new_blocks}))
                else:
                    result.append(msg)
            else:
                # Keep message as-is
                result.append(msg)

        logger.info(
            f"[ToolResultCompaction] Compaction complete: "
            f"{len(messages)} messages -> {len(result)} messages "
            f"({compacted_count} tool results compacted)"
        )
        return result

    def _group_into_user_rounds(self, messages: list[Message]) -> list[list[Message]]:
        """Group messages into user rounds.

        A UserRound starts with a USER message and ends with an ASSISTANT message
        that has no tool calls (final response).
        """
        user_rounds: list[list[Message]] = []
        current_round: list[Message] = []

        for msg in messages:
            current_round.append(msg)

            if msg.role == Role.ASSISTANT:
                # Check if this is a final response (no tool calls)
                has_tool_use = any(isinstance(block, ToolUseBlock) for block in msg.content)
                if not has_tool_use and current_round:
                    user_rounds.append(current_round)
                    current_round = []

        # Handle incomplete round at the end
        if current_round:
            user_rounds.append(current_round)

        return user_rounds

    def _group_into_iterations(self, messages: list[Message]) -> list[list[Message]]:
        """Group messages into conversation iterations.

        An iteration is bounded by ASSISTANT messages:
        [USER or FRAMEWORK](optional) -> [ASSISTANT] -> [TOOL results](optional)

        - Each ASSISTANT message starts a new iteration
        - USER or FRAMEWORK messages before the ASSISTANT are part of that iteration (optional)
        - TOOL results after the ASSISTANT are part of that iteration (optional)
        """
        iterations: list[list[Message]] = []
        current_iteration: list[Message] = []

        for msg in messages:
            if msg.role == Role.ASSISTANT:
                # ASSISTANT starts a new iteration
                # Move any preceding USER and FRAMEWORK messages to the new iteration
                prefix_msgs: list[Message] = []
                while current_iteration and current_iteration[-1].role in (Role.USER, Role.FRAMEWORK):
                    prefix_msgs.insert(0, current_iteration.pop())

                if current_iteration:
                    iterations.append(current_iteration)

                current_iteration = prefix_msgs + [msg]
            else:
                # Continue current iteration (user, framework, or tool)
                current_iteration.append(msg)

        # Add the last iteration
        if current_iteration:
            iterations.append(current_iteration)

        return iterations
