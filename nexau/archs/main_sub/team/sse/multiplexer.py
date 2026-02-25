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

"""Team SSE multiplexer.

RFC-0002: 多 Agent 事件聚合器
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable

from nexau.archs.llm.llm_aggregators.events import Event

from .envelope import TeamStreamEnvelope


class TeamSSEMultiplexer:
    """Multiplexes events from multiple agents into a single SSE stream.

    RFC-0002: 多 Agent 事件聚合器

    每个 agent 通过 create_event_handler 获取独立的事件回调，
    所有事件汇聚到同一个 asyncio.Queue，由 stream() 统一输出。
    """

    def __init__(
        self,
        *,
        team_id: str,
        on_envelope: Callable[[TeamStreamEnvelope], None] | None = None,
    ) -> None:
        self._team_id = team_id
        self._queue: asyncio.Queue[TeamStreamEnvelope | None] = asyncio.Queue()
        self._on_envelope = on_envelope

    def create_event_handler(self, agent_id: str, role_name: str) -> Callable[[Event], None]:
        """Create an on_event callback for a specific agent.

        RFC-0002: 为指定 agent 创建事件回调

        返回的回调函数将事件包装为 TeamStreamEnvelope 后放入队列。
        """

        def handler(event: Event) -> None:
            envelope = TeamStreamEnvelope(
                team_id=self._team_id,
                agent_id=agent_id,
                role_name=role_name,
                event=event,
            )
            if self._on_envelope is not None:
                self._on_envelope(envelope)
            self._queue.put_nowait(envelope)

        return handler

    def emit(self, agent_id: str, event: Event, *, role_name: str | None = None) -> None:
        """Emit a custom event to the SSE stream.

        RFC-0002: 向 SSE 流发送自定义事件（如用户消息、Agent 间消息）
        """
        envelope = TeamStreamEnvelope(
            team_id=self._team_id,
            agent_id=agent_id,
            role_name=role_name,
            event=event,
        )
        if self._on_envelope is not None:
            self._on_envelope(envelope)
        self._queue.put_nowait(envelope)

    async def stream(self) -> AsyncGenerator[TeamStreamEnvelope, None]:
        """Yield envelopes as they arrive from any agent.

        RFC-0002: 统一输出所有 agent 的事件流

        收到 None 哨兵值时结束迭代。
        """
        while True:
            envelope = await self._queue.get()
            if envelope is None:
                break
            yield envelope

    def close(self) -> None:
        """Signal end of stream.

        RFC-0002: 发送流结束信号

        向队列放入 None 哨兵值，通知 stream() 停止迭代。
        """
        self._queue.put_nowait(None)
