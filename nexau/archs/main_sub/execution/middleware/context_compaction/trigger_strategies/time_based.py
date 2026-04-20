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

"""Time-based trigger strategy for micro-compact.

micro-compact: 基于 Message 时间戳触发压缩。当最后一条 assistant 消息的 created_at
距当前时间超过阈值（默认 5 分钟，对齐 Anthropic prompt cache 标准 TTL）时触发。
Trigger 本身无状态，判断完全基于 messages 数据。
"""

import logging
from datetime import UTC, datetime, timedelta

from nexau.core.messages import Message, Role

logger = logging.getLogger(__name__)


class TimeBasedTrigger:
    """Trigger compaction when the gap since last assistant message exceeds a threshold.

    micro-compact: 无状态触发器，从 messages 中读取最后一条 assistant 消息的 created_at，
    与当前时间比较。超过 gap_threshold_minutes 即触发。
    """

    def __init__(self, gap_threshold_minutes: float = 5):
        """Initialize time-based trigger.

        Args:
            gap_threshold_minutes: Minutes since last assistant message to trigger compaction.
                Default: 5 (aligned with Anthropic prompt cache standard TTL).
        """
        self.gap_threshold = timedelta(minutes=gap_threshold_minutes)
        logger.info(
            "[TimeBasedTrigger] Initialized with gap_threshold: %d minutes",
            gap_threshold_minutes,
        )

    def should_compact(
        self,
        messages: list[Message],
        current_tokens: int,
        max_context_tokens: int,
    ) -> tuple[bool, str]:
        """Check if compaction should be triggered based on time gap.

        micro-compact: 找到 messages 中最后一条 assistant 消息的 created_at，
        计算 now - created_at，超过阈值即触发。无 assistant 消息或 created_at 为 None 时不触发。
        """
        # 1. 从后往前找最后一条 assistant 消息
        last_assistant_created_at: datetime | None = None
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT:
                last_assistant_created_at = msg.created_at
                break

        # 2. 无 assistant 消息（新会话）或 created_at 为 None → 不触发
        if last_assistant_created_at is None:
            return False, ""

        # 3. 确保 timezone-aware 比较
        now = datetime.now(UTC)
        if last_assistant_created_at.tzinfo is None:
            last_assistant_created_at = last_assistant_created_at.replace(tzinfo=UTC)

        gap = now - last_assistant_created_at
        if gap >= self.gap_threshold:
            gap_minutes = gap.total_seconds() / 60
            threshold_minutes = self.gap_threshold.total_seconds() / 60
            return (
                True,
                f"Time gap {gap_minutes:.1f}min >= threshold {threshold_minutes:.0f}min "
                f"(last assistant message at {last_assistant_created_at.isoformat()})",
            )

        return False, ""
