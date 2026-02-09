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

"""Agent lock data models."""

from __future__ import annotations

import time

from sqlmodel import Field, SQLModel


class AgentLockModel(SQLModel, table=True):
    """Agent lock model for preventing concurrent agent execution.

    Lock key: (session_id, agent_id)
    - Same agent_id: serialized execution
    - Different agent_id: concurrent execution (supports sub-agents)

    Expiration-based locking:
    - Lock is valid while expires_at_ns > current time
    - Heartbeat renews lock by extending expires_at_ns
    - No background cleanup needed - expired locks are ignored on query

    Attributes:
        session_id: Session identifier (primary key)
        agent_id: Agent identifier (primary key)
        user_id: User identifier (metadata, optional)
        run_id: Run identifier (metadata, optional)
        holder_id: Unique identifier for this lock acquisition (e.g., "12345:a1b2c3d4")
        acquired_at_ns: Nanosecond timestamp when lock was acquired
        expires_at_ns: Nanosecond timestamp when lock expires (renewed by heartbeat)
    """

    __tablename__ = "agent_locks"  # type: ignore[assignment]

    session_id: str = Field(primary_key=True)
    agent_id: str = Field(primary_key=True)
    user_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    holder_id: str
    acquired_at_ns: int = Field(default_factory=time.time_ns)
    expires_at_ns: int = Field(default_factory=time.time_ns)  # Replaces last_heartbeat_ns
