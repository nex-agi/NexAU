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

    RFC-0002: 全员空闲检测（状态转换触发）

    Runs as a background asyncio.Task, periodically checking
    all agents for idle deadlock (all waiting with no messages).

    采用「单次通知」策略：每次全员进入空闲后仅通知 leader 一次，
    避免 leader 处理完 watchdog 消息后再次空闲导致无限循环。
    仅当有意义的状态变更（teammate 注册/注销、外部消息到达）发生后
    才重置通知标记，允许下一次通知。
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

        # RFC-0002: 单次通知标记，防止同一空闲周期内重复唤醒 leader
        self._idle_notified = False

    def stop(self) -> None:
        """Signal the watchdog loop to exit."""
        self._stopped = True

    def register(self, agent_id: str) -> None:
        """Register a teammate for idle monitoring.

        重置通知标记：新 teammate 加入意味着可能有新工作。
        """
        self._start_times[agent_id] = time.monotonic()
        self._idle_notified = False

    def unregister(self, agent_id: str) -> None:
        """Unregister a teammate from idle monitoring.

        重置通知标记：teammate 离开意味着状态发生变化。
        """
        self._start_times.pop(agent_id, None)
        self._idle_notified = False

    def reset_idle_notification(self) -> None:
        """Reset idle notification flag on meaningful state changes.

        RFC-0002: 重置空闲通知标记

        当有意义的状态变更发生时调用，允许 watchdog 在下次全员空闲时
        再次通知 leader。典型调用场景：
        - teammate 发送消息（非 watchdog 来源）
        - 任务状态变更
        """
        self._idle_notified = False

    async def run(self) -> None:
        """Watchdog loop — runs as background asyncio.Task.

        RFC-0002: Watchdog 主循环（单次通知策略）

        周期性执行全员空闲检测。每个空闲周期仅通知一次 leader，
        直到 reset_idle_notification() 被调用后才允许再次通知。
        """
        # 每次启动时重置状态，支持跨 run() 复用同一实例
        self._stopped = False
        self._idle_notified = False

        while not self._stopped:
            await asyncio.sleep(self._config.idle_check_interval_seconds)

            # 全员空闲检测 — 仅在有注册 teammate 且尚未通知时才唤醒 leader
            if self._check_all_idle is not None and self._notify_leader is not None and len(self._start_times) > 0:
                if self._check_all_idle() and not self._idle_notified:
                    self._idle_notified = True
                    logger.info("Watchdog: all agents idle, waking leader (one-shot)")
                    self._notify_leader(
                        "[All Idle] All agents are idle. Review task board status and decide next steps — assign new tasks, check results.",
                        "watchdog",
                    )
