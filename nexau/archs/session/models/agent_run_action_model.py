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

"""Agent run action model using SQLModel.

This module provides the AgentRunActionModel for storing agent conversation
history actions at run-level granularity with support for append-only operations,
undo, and replace functionality.

Key concepts:
- Run granularity: One action record per agent.run() invocation
- Append-only: All operations append new records, never modify existing ones
- Event sourcing: Rebuild state by replaying action records

Action types:
- APPEND: Normal run, appends incremental messages
- UNDO: Undo to a specific point (removes runs in range [target, undo))
- REPLACE: Full state replacement for context compaction
"""

from __future__ import annotations

import time
from datetime import datetime
from enum import StrEnum

from pydantic import TypeAdapter
from sqlalchemy import Column, Index
from sqlmodel import Field, SQLModel

from nexau.core.messages import Message

from ..id_generator import generate_action_id
from .types import PydanticJson


class RunActionType(StrEnum):
    """Run record action types."""

    APPEND = "append"  # Normal run, appends incremental messages
    UNDO = "undo"  # Undo operation, reverts to before a specific run
    REPLACE = "replace"  # Replace operation, replaces all previous history


# TypeAdapter for list[Message] serialization
_messages_adapter: TypeAdapter[list[Message]] = TypeAdapter(list[Message])


class AgentRunActionModel(SQLModel, table=True):
    """Per-run history record (append-only).

    Record type is determined by action_type field:
    - APPEND: Normal run with messages
    - UNDO: Undo marker to revert to before a specific run
    - REPLACE: Full state replacement for context compaction

    Each action type has its own prefixed payload fields:
    - append_messages: for APPEND action
    - replace_messages: for REPLACE action
    - undo_before_run_id: for UNDO action

    Example:
        >>> record = AgentRunActionModel(
        ...     user_id="u1",
        ...     session_id="s1",
        ...     agent_id="agent_main",
        ...     run_id="run_001",
        ...     root_run_id="run_001",
        ...     action_type=RunActionType.APPEND,
        ...     append_messages=[Message.user("hello")],
        ... )
    """

    __tablename__ = "agent_run_actions"  # type: ignore[assignment]
    __table_args__ = (
        Index(
            "idx_agent_run_actions_created_at_ns",
            "user_id",
            "session_id",
            "agent_id",
            "created_at_ns",
        ),
        Index(
            "idx_agent_run_actions_action_type_created_at_ns",
            "user_id",
            "session_id",
            "agent_id",
            "action_type",
            "created_at_ns",
        ),
    )

    action_id: str = Field(primary_key=True, default_factory=generate_action_id)
    user_id: str = Field(index=True)
    session_id: str = Field(index=True)
    agent_id: str = Field(index=True)
    run_id: str = Field(index=True)

    # === Run tracking ===
    root_run_id: str = Field(index=True)
    parent_run_id: str | None = Field(default=None, index=True)
    agent_name: str = ""

    # === Timestamp ===
    created_at: datetime = Field(default_factory=datetime.now)
    created_at_ns: int = Field(default_factory=time.time_ns, index=True)

    # === Action type ===
    action_type: str = Field(index=True)  # RunActionType value

    # === APPEND action payload ===
    # Incremental messages produced in this run
    append_messages: list[Message] | None = Field(default=None, sa_column=Column(PydanticJson(list[Message])))

    # === REPLACE action payload ===
    # Complete compacted state replacing all previous history
    replace_messages: list[Message] | None = Field(default=None, sa_column=Column(PydanticJson(list[Message])))

    # === UNDO action payload ===
    # Target run_id to undo before
    undo_before_run_id: str | None = None

    @classmethod
    def create_append(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Create an APPEND record from a list of messages.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            run_id: Current run identifier
            root_run_id: Root run identifier
            messages: List of messages produced in this run
            parent_run_id: Parent run identifier (for sub-agents)
            agent_name: Agent name
        """
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.APPEND,
            append_messages=messages,
        )

    @classmethod
    def create_undo(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        undo_before_run_id: str,
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Create an UNDO record.

        Semantics: undo(before=X) removes runs in range [X, undo_run),
        does not affect runs added after the undo.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            run_id: Current run identifier (for the undo operation itself)
            root_run_id: Root run identifier
            undo_before_run_id: Target run to undo to (exclusive)
            parent_run_id: Parent run identifier (for sub-agents)
            agent_name: Agent name
        """
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.UNDO,
            undo_before_run_id=undo_before_run_id,
        )

    @classmethod
    def create_replace(
        cls,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        messages: list[Message],
        parent_run_id: str | None = None,
        agent_name: str = "",
    ) -> AgentRunActionModel:
        """Create a REPLACE record.

        Replace replaces all previous history with the provided messages.
        Used for context compaction.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            run_id: Current run identifier
            root_run_id: Root run identifier
            messages: Compacted message list
            parent_run_id: Parent run identifier (for sub-agents)
            agent_name: Agent name
        """
        return cls(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            action_type=RunActionType.REPLACE,
            replace_messages=messages,
        )
