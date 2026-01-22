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

"""Service for managing agent run actions (APPEND/UNDO/REPLACE)."""

from __future__ import annotations

from typing import NamedTuple

from nexau.core.messages import Message, Role

from .models import AgentRunActionModel
from .models.agent_run_action_model import RunActionType
from .orm import AndFilter, ComparisonFilter, DatabaseEngine


class AgentRunActionKey(NamedTuple):
    """Key for identifying agent run actions."""

    user_id: str
    session_id: str
    agent_id: str


class AgentRunActionService:
    """Service for managing agent run action history.

    This service provides methods to:
    - Persist APPEND actions (normal run with messages)
    - Persist UNDO actions (revert to before a specific run)
    - Persist REPLACE actions (full state replacement for compaction)
    - Load and rebuild message history from action records

    The service uses event sourcing pattern where all operations are append-only.
    """

    def __init__(self, *, engine: DatabaseEngine) -> None:
        """Initialize the service.

        Args:
            engine: Database engine for persistence
        """
        self._engine = engine
        self._last_created_at_ns: dict[AgentRunActionKey, int] = {}

    async def _run_exists(self, *, key: AgentRunActionKey, run_id: str) -> bool:
        action_filter = AndFilter(
            filters=[
                ComparisonFilter.eq("user_id", key.user_id),
                ComparisonFilter.eq("session_id", key.session_id),
                ComparisonFilter.eq("agent_id", key.agent_id),
                ComparisonFilter.eq("run_id", run_id),
            ]
        )
        return await self._engine.find_first(AgentRunActionModel, filters=action_filter) is not None

    async def persist_append(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        parent_run_id: str | None,
        agent_name: str,
        messages: list[Message],
    ) -> AgentRunActionModel:
        """Persist an APPEND action.

        Args:
            key: Agent run action key (user_id, session_id, agent_id)
            run_id: Current run ID
            root_run_id: Root run ID
            parent_run_id: Parent run ID (for sub-agents)
            agent_name: Agent name
            messages: List of messages produced in this run

        Returns:
            The created action record
        """
        # Filter out system messages
        messages = [m for m in messages if m.role != Role.SYSTEM]
        if not messages:
            raise ValueError("Cannot persist APPEND action with no messages")

        record = AgentRunActionModel.create_append(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            messages=messages,
        )
        return await self._engine.create(record)

    async def persist_undo(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        undo_before_run_id: str,
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Persist an UNDO action.

        Semantics: undo(before=X) removes runs in range [X, undo_run),
        does not affect runs added after the undo.

        Args:
            key: Agent run action key (user_id, session_id, agent_id)
            run_id: Current run ID (for the undo operation itself)
            root_run_id: Root run ID
            undo_before_run_id: Target run to undo to (exclusive)
            parent_run_id: Parent run ID (for sub-agents)
            agent_name: Agent name

        Returns:
            The created action record
        """
        record = AgentRunActionModel.create_undo(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            undo_before_run_id=undo_before_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )
        return await self._engine.create(record)

    async def persist_replace(
        self,
        *,
        key: AgentRunActionKey,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Persist a REPLACE action.

        Used for context compaction - replaces all previous history with
        the provided compacted messages.

        Args:
            key: Agent run action key (user_id, session_id, agent_id)
            run_id: Current run ID
            root_run_id: Root run ID
            messages: Complete compacted message history
            parent_run_id: Parent run ID (for sub-agents)
            agent_name: Agent name

        Returns:
            The created action record
        """
        # Filter out system messages
        messages = [m for m in messages if m.role != Role.SYSTEM]
        if not messages:
            raise ValueError("Cannot persist REPLACE action with no messages")

        record = AgentRunActionModel.create_replace(
            user_id=key.user_id,
            session_id=key.session_id,
            agent_id=key.agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            messages=messages,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )
        return await self._engine.create(record)

    async def _scan_actions_desc(
        self,
        *,
        key: AgentRunActionKey,
        page_size: int,
        cursor: int | None,
    ) -> list[AgentRunActionModel]:
        filters = [
            ComparisonFilter.eq("user_id", key.user_id),
            ComparisonFilter.eq("session_id", key.session_id),
            ComparisonFilter.eq("agent_id", key.agent_id),
        ]
        if cursor is not None:
            filters.append(ComparisonFilter.lt("created_at_ns", cursor))

        action_filter = AndFilter(filters=filters)
        return await self._engine.find_many(
            AgentRunActionModel,
            filters=action_filter,
            order_by=("-created_at_ns", "-action_id"),
            limit=page_size,
        )

    async def load_messages(
        self,
        *,
        key: AgentRunActionKey,
    ) -> list[Message]:
        cursor: int | None = None
        page_size = 200
        skip_until_run_id: str | None = None
        appends_desc: list[AgentRunActionModel] = []
        base_replace: AgentRunActionModel | None = None

        while True:
            page = await self._scan_actions_desc(
                key=key,
                page_size=page_size,
                cursor=cursor,
            )
            if not page:
                break

            stop = False
            for action in page:
                if skip_until_run_id is not None:
                    if action.run_id == skip_until_run_id:
                        skip_until_run_id = None
                    continue

                if action.action_type == RunActionType.UNDO and action.undo_before_run_id:
                    if await self._run_exists(key=key, run_id=action.undo_before_run_id):
                        skip_until_run_id = action.undo_before_run_id
                    continue

                if action.action_type == RunActionType.REPLACE:
                    base_replace = action
                    stop = True
                    break

                if action.action_type == RunActionType.APPEND:
                    appends_desc.append(action)

            if stop:
                break

            cursor = int(page[-1].created_at_ns)

        messages: list[Message] = []
        message_by_id: dict[str, int] = {}

        def apply_messages(msgs: list[Message] | None) -> None:
            if not msgs:
                return
            for msg in msgs:
                if msg.role == Role.SYSTEM:
                    continue
                msg_id = str(msg.id)
                if msg_id not in message_by_id:
                    message_by_id[msg_id] = len(messages)
                    messages.append(msg)
                else:
                    messages[message_by_id[msg_id]] = msg

        if base_replace is not None:
            apply_messages(base_replace.replace_messages)

        for action in reversed(appends_desc):
            apply_messages(action.append_messages)

        return messages
