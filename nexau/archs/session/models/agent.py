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

"""Agent model - lightweight agent metadata using SQLModel.

Note: Message history is now stored separately in AgentHistoryRecordModel.
See models/history.py for details.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class AgentModel(SQLModel, table=True):
    """Lightweight agent metadata model using SQLModel.

    AgentModel stores only agent metadata. Conversation history is stored
    separately in AgentHistoryRecordModel.

    Uses SQLModel Field definitions:
    - Field(primary_key=True) for primary keys
    - Field(index=True) for indexes
    - Field(default_factory=datetime.now) for timestamp fields

    This separation enables:
    - Message deduplication across agents/forks
    - Complete history with revert support
    - Efficient fork (just copy message_ids)
    - Lightweight agent loading

    Example:
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>> from nexau.archs.session.models import AgentModel
        >>>
        >>> engine = InMemoryDatabaseEngine()
        >>> agent = AgentModel(user_id="u1", session_id="s1", agent_id="a1")
        >>> agent.agent_name = "main_agent"
        >>> await engine.create(agent)
    """

    __tablename__ = "agents"  # type: ignore[assignment]

    # Primary key fields (using SQLModel Field with primary_key=True)
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True, index=True)
    agent_id: str = Field(primary_key=True)

    # Agent metadata
    agent_name: str = ""

    # Timestamps (using Field with default_factory for datetime)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
