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

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolResultCompaction:
    """Tool result compaction strategy - compacts old tool execution results.

    This strategy preserves:
    1. System prompt (if keep_system=True)
    2. All user and assistant messages (in full)
    3. Tool results after the last assistant message (in full)
    4. Older tool results with compacted content: "Tool call result has been compacted"
    """

    def __init__(
        self,
        keep_system: bool = True,
    ):
        """Initialize tool result compaction.

        Args:
            keep_system: Whether to preserve the system message. Default: True.
        """
        self.keep_system = keep_system

        logger.info(f"[ToolResultCompaction] Initialized with keep_system={self.keep_system}")

    def compact(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compact messages by compressing old tool results."""
        logger.info(f"[ToolResultCompaction] Starting compaction on {len(messages)} messages")

        result = []
        start_idx = 0

        # Keep system message if present
        if self.keep_system and messages and messages[0].get("role") == "system":
            result.append(messages[0].copy())
            start_idx = 1

        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(messages) - 1, start_idx - 1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # No assistant message found, return as is
            logger.info("[ToolResultCompaction] No assistant message found, skipping compaction")
            return messages.copy()

        # Process all messages
        compacted_count = 0
        for i in range(start_idx, len(messages)):
            msg = messages[i]
            role = msg.get("role")

            if role == "tool":
                # Check if this tool message is after the last assistant
                if i > last_assistant_idx:
                    # Keep in full
                    result.append(msg.copy())
                else:
                    # Compact this tool result
                    compacted_msg = msg.copy()
                    compacted_msg["content"] = "Tool call result has been compacted"
                    result.append(compacted_msg)
                    compacted_count += 1
            else:
                # Keep all non-tool messages as-is
                result.append(msg.copy())

        logger.info(
            f"[ToolResultCompaction] Compaction complete: "
            f"{len(messages)} messages -> {len(result)} messages "
            f"({compacted_count} tool results compacted)"
        )
        return result
