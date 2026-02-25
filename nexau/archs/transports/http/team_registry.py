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

"""Team instance registry for HTTP endpoints.

RFC-0002: AgentTeam 实例注册表

Manages AgentTeam instances keyed by (user_id, session_id).
Team configs are registered at server startup from YAML files.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.agent_team import AgentTeam

if TYPE_CHECKING:
    from nexau.archs.main_sub.config import AgentConfig
    from nexau.archs.session.orm import DatabaseEngine
    from nexau.archs.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class TeamRegistry:
    """Registry for managing AgentTeam instances.

    RFC-0002: AgentTeam 实例注册表

    在服务器启动时注册 team 配置（leader + candidates），
    HTTP 请求到达时按 (user_id, session_id) 获取或创建 AgentTeam 实例。
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        session_manager: SessionManager,
    ) -> None:
        self._engine = engine
        self._session_manager = session_manager

        # team config 注册表: config_name -> (leader_config, candidates)
        self._configs: dict[str, tuple[AgentConfig, dict[str, AgentConfig]]] = {}

        # 活跃 team 实例: (user_id, session_id) -> AgentTeam
        self._teams: dict[tuple[str, str], AgentTeam] = {}

    def register_config(
        self,
        name: str,
        *,
        leader_config: AgentConfig,
        candidates: dict[str, AgentConfig],
    ) -> None:
        """Register a team configuration.

        RFC-0002: 注册 team 配置

        Args:
            name: Config name (e.g. "default").
            leader_config: Leader agent configuration.
            candidates: Role name -> agent config mapping.
        """
        self._configs[name] = (leader_config, candidates)
        logger.info("Registered team config: %s (candidates: %s)", name, list(candidates.keys()))

    def get_or_create(
        self,
        user_id: str,
        session_id: str,
        config_name: str = "default",
    ) -> AgentTeam:
        """Get existing or create new AgentTeam instance.

        RFC-0002: 获取或创建 AgentTeam 实例

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            config_name: Registered config name.

        Returns:
            AgentTeam instance.

        Raises:
            ValueError: If config_name is not registered.
        """
        key = (user_id, session_id)

        existing = self._teams.get(key)
        if existing is not None:
            return existing

        if config_name not in self._configs:
            raise ValueError(f"Team config '{config_name}' not registered")

        leader_config, candidates = self._configs[config_name]

        team = AgentTeam(
            leader_config=leader_config,
            candidates=candidates,
            engine=self._engine,
            session_manager=self._session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        self._teams[key] = team
        logger.info("Created team for (%s, %s) with config '%s'", user_id, session_id, config_name)
        return team

    def get(self, user_id: str, session_id: str) -> AgentTeam | None:
        """Get an existing team instance without creating one.

        RFC-0002: 获取已有 team 实例（不创建）

        Args:
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            AgentTeam instance, or None if not found.
        """
        return self._teams.get((user_id, session_id))

    def remove(self, user_id: str, session_id: str) -> None:
        """Remove a team instance from the registry.

        RFC-0002: 移除 team 实例

        Args:
            user_id: User identifier.
            session_id: Session identifier.
        """
        self._teams.pop((user_id, session_id), None)
