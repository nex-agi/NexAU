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

"""Session model for multi-agent conversations using SQLModel."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from nexau.archs.main_sub.agent_context import GlobalStorage

from .types import GlobalStorageJson


class SessionModel(SQLModel, table=True):
    """Multi-Agent Session model using SQLModel.

    SessionModel stores session-level data. Agent metadata is stored separately
    in AgentModel, avoiding data redundancy.

    Uses SQLModel Field definitions:
    - Field(primary_key=True) for primary keys
    - Field(default_factory=...) for mutable defaults
    - Field(sa_column=Column(JSON)) for complex types

    Architecture:
        SessionModel (lightweight)          AgentModel (independent)
        ├── session_id                      ├── (user_id, session_id, agent_id)
        ├── user_id                         ├── agent_name
        ├── context                         └── created_at
        ├── storage
        └── root_agent_id

    Example:
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>> from nexau.archs.session.models import SessionModel
        >>>
        >>> engine = InMemoryDatabaseEngine()
        >>> session = SessionModel(user_id="u1", session_id="s1")
        >>> session.context["key"] = "value"
        >>> await engine.create(session)
    """

    __tablename__ = "sessions"  # type: ignore[assignment]

    # Primary key fields (using SQLModel Field with primary_key=True)
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)

    # Timestamps (using Field with default_factory for mutable defaults)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Session-level runtime context (shared across all Agents)
    # e.g., working_directory, username, date, etc.
    context: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Global storage for cross-agent data sharing
    storage: GlobalStorage = Field(default_factory=GlobalStorage, sa_column=Column(GlobalStorageJson()))

    # Root agent ID (for quick access to main agent)
    root_agent_id: str | None = Field(default=None)
