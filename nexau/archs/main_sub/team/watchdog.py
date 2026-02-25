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

"""Teammate execution watchdog.

RFC-0002: 全员空闲检测
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchdogConfig:
    """Idle detection configuration.

    RFC-0002: Watchdog 全员空闲检测配置
    """

    idle_check_interval_seconds: float = 10.0


class TeammateWatchdog:
    """Detects all-idle deadlock among teammates.

    RFC-0002: 全员空闲检测

    Runs as a background asyncio.Task, periodically checking
    all agents for idle deadlock (all waiting with no messages).
    """

    def __init__(
        self,
        *,
        config: WatchdogConfig,
        check_all_idle: Callable[[], bool] | None = None,
        notify_leader: Callable[[str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._start_times: dict[str, float] = {}
        self._stopped = False

        # RFC-0002: 全员空闲检测回调
        self._check_all_idle = check_all_idle
        self._notify_leader = notify_leader

    def stop(self) -> None:
        """Signal the watchdog loop to exit."""
        self._stopped = True

    def register(self, agent_id: str) -> None:
        """Register a teammate for idle monitoring."""
        self._start_times[agent_id] = time.monotonic()

    def unregister(self, agent_id: str) -> None:
        """Unregister a teammate from idle monitoring."""
        self._start_times.pop(agent_id, None)

    async def run(self) -> None:
        """Watchdog loop — runs as background asyncio.Task.

        RFC-0002: Watchdog 主循环

        周期性执行全员空闲检测。
        """
        while not self._stopped:
            await asyncio.sleep(self._config.idle_check_interval_seconds)

            # 全员空闲检测 — 仅在有注册 teammate 时才唤醒 leader
            if self._check_all_idle is not None and self._notify_leader is not None and len(self._start_times) > 0:
                if self._check_all_idle():
                    logger.info("Watchdog: all agents idle, waking leader")
                    self._notify_leader(
                        "[All Idle] All agents are idle. Review task board status and decide next steps — assign new tasks, check results.",
                        "watchdog",
                    )
