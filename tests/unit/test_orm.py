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

"""Tests for the ORM layer."""

from __future__ import annotations

import asyncio
from datetime import datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    InMemoryDatabaseEngine,
    get_table_name,
)

# ============================================================================
# Test Models (using SQLModel Field pattern with table=True for SQLAlchemy)
# ============================================================================


class UserModel(SQLModel, table=True):
    """Test user model with composite primary key."""

    __tablename__ = "test_users"  # type: ignore[assignment]

    tenant_id: str = Field(primary_key=True)
    user_id: str = Field(primary_key=True)
    name: str = ""
    email: str = ""
    created_at: datetime | None = None


class OrmTestSessionModel(SQLModel, table=True):
    """Test session model."""

    __tablename__ = "test_sessions"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    context: dict = Field(default_factory=dict, sa_column=Column(JSON))
    agent_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class SimpleModel(SQLModel, table=True):
    """Model with single primary key."""

    __tablename__ = "test_simple"  # type: ignore[assignment]

    id: str = Field(primary_key=True)
    value: int = 0


# ============================================================================
# SQLModel Tests
# ============================================================================


class TestStorageModel:
    """Tests for SQLModel base class with Field pattern."""

    def test_table_name_default(self) -> None:
        """Test that table name is set correctly."""
        table_name = get_table_name(UserModel)
        assert table_name == "test_users"

    def test_mutable_defaults_independence(self) -> None:
        """Test that mutable defaults are independent between instances."""
        session1 = OrmTestSessionModel(user_id="u1", session_id="s1")
        session2 = OrmTestSessionModel(user_id="u2", session_id="s2")

        session1.context["key"] = "value1"
        session1.agent_ids.append("agent1")

        # session2 should not be affected
        assert "key" not in session2.context
        assert "agent1" not in session2.agent_ids


# ============================================================================
# InMemoryDatabaseEngine Tests
# ============================================================================


class TestMemoryEngine:
    """Tests for InMemoryDatabaseEngine."""

    def test_create_and_find_first(self) -> None:
        """Test that create stores and find_first retrieves."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1")
            await engine.create(user)

            # Use Filter DSL syntax
            filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "u1")])
            loaded = await engine.find_first(UserModel, filters=filters)
            assert loaded is not None
            assert loaded.tenant_id == "t1"
            assert loaded.user_id == "u1"
            assert loaded.name == ""

        asyncio.run(run())

    def test_update_existing(self) -> None:
        """Test that update modifies existing model."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="Bob")
            await engine.create(user)

            # Update
            user.name = "Bob Updated"
            await engine.update(user)

            # Verify updated using Filter DSL
            filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "u1")])
            reloaded = await engine.find_first(UserModel, filters=filters)
            assert reloaded is not None
            assert reloaded.name == "Bob Updated"

        asyncio.run(run())

    def test_find_first_nonexistent(self) -> None:
        """Test find_first returns None for non-existent."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "nonexistent")])
            result = await engine.find_first(UserModel, filters=filters)
            assert result is None

        asyncio.run(run())

    def test_delete(self) -> None:
        """Test delete operation."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="ToDelete")
            await engine.create(user)

            filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "u1")])
            count = await engine.delete(UserModel, filters=filters)
            assert count == 1

            loaded = await engine.find_first(UserModel, filters=filters)
            assert loaded is None

        asyncio.run(run())

    def test_delete_nonexistent(self) -> None:
        """Test delete returns 0 for non-existent."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "nonexistent")])
            count = await engine.delete(UserModel, filters=filters)
            assert count == 0

        asyncio.run(run())

    def test_count(self) -> None:
        """Test count operation."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "u1")])
            assert await engine.count(UserModel, filters=filters) == 0

            user = UserModel(tenant_id="t1", user_id="u1")
            await engine.create(user)

            assert await engine.count(UserModel, filters=filters) == 1

        asyncio.run(run())

    def test_find_many(self) -> None:
        """Test find_many with filters."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            # Create users in different tenants
            await engine.create(UserModel(tenant_id="t1", user_id="u1", name="Alice"))
            await engine.create(UserModel(tenant_id="t1", user_id="u2", name="Bob"))
            await engine.create(UserModel(tenant_id="t2", user_id="u3", name="Charlie"))

            # Find by tenant using Filter DSL
            t1_users = await engine.find_many(UserModel, filters=ComparisonFilter.eq("tenant_id", "t1"))
            assert len(t1_users) == 2
            names = {u.name for u in t1_users}
            assert names == {"Alice", "Bob"}

            t2_users = await engine.find_many(UserModel, filters=ComparisonFilter.eq("tenant_id", "t2"))
            assert len(t2_users) == 1
            assert t2_users[0].name == "Charlie"

        asyncio.run(run())

    def test_complex_model(self) -> None:
        """Test with model containing complex fields."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([OrmTestSessionModel])
            session = OrmTestSessionModel(user_id="u1", session_id="s1")
            session.context["key"] = "value"
            session.agent_ids.append("agent1")
            await engine.create(session)

            filters = AndFilter(filters=[ComparisonFilter.eq("user_id", "u1"), ComparisonFilter.eq("session_id", "s1")])
            loaded = await engine.find_first(OrmTestSessionModel, filters=filters)
            assert loaded is not None
            assert loaded.context["key"] == "value"
            assert "agent1" in loaded.agent_ids

        asyncio.run(run())

    def test_upsert_creates_new(self) -> None:
        """Test upsert creates new record if not exists."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="New")
            result, created = await engine.upsert(user)
            assert created is True
            assert result.name == "New"

        asyncio.run(run())

    def test_upsert_updates_existing(self) -> None:
        """Test upsert updates existing record."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="Original")
            await engine.create(user)

            user.name = "Updated"
            result, created = await engine.upsert(user)
            assert created is False
            assert result.name == "Updated"

        asyncio.run(run())

    def test_get_or_create_creates_new(self) -> None:
        """Test get_or_create creates new record if not exists."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="New")
            result, created = await engine.get_or_create(user)
            assert created is True
            assert result.name == "New"

        asyncio.run(run())

    def test_get_or_create_returns_existing(self) -> None:
        """Test get_or_create returns existing record without update."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="Original")
            await engine.create(user)

            # Try to get_or_create with different data
            new_user_data = UserModel(tenant_id="t1", user_id="u1", name="ShouldNotUpdate")
            result, created = await engine.get_or_create(new_user_data)
            assert created is False
            assert result.name == "Original"  # Should still be "Original"

        asyncio.run(run())


# ============================================================================
# Shared Backend Tests
# ============================================================================


class TestSharedBackend:
    """Test that multiple model types can share a backend."""

    def test_shared_memory_backend(self) -> None:
        """Test multiple model types sharing InMemoryDatabaseEngine."""
        engine = InMemoryDatabaseEngine()

        async def run() -> None:
            await engine.setup_models([UserModel, OrmTestSessionModel])
            user = UserModel(tenant_id="t1", user_id="u1", name="Alice")
            await engine.create(user)

            session = OrmTestSessionModel(user_id="u1", session_id="s1")
            session.context["user_name"] = "Alice"
            await engine.create(session)

            # Both should be accessible using Filter DSL
            user_filters = AndFilter(filters=[ComparisonFilter.eq("tenant_id", "t1"), ComparisonFilter.eq("user_id", "u1")])
            loaded_user = await engine.find_first(UserModel, filters=user_filters)

            session_filters = AndFilter(filters=[ComparisonFilter.eq("user_id", "u1"), ComparisonFilter.eq("session_id", "s1")])
            loaded_session = await engine.find_first(OrmTestSessionModel, filters=session_filters)

            assert loaded_user is not None
            assert loaded_session is not None
            assert loaded_user.name == "Alice"
            assert loaded_session.context["user_name"] == "Alice"

        asyncio.run(run())
