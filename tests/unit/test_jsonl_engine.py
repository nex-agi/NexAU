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

"""Tests for JSONLDatabaseEngine."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    JSONLDatabaseEngine,
)

# ============================================================================
# Test Models
# ============================================================================


class JsonlUserModel(SQLModel, table=True):
    """Test user model for JSONL engine."""

    __tablename__ = "jsonl_test_users"  # type: ignore[assignment]

    tenant_id: str = Field(primary_key=True)
    user_id: str = Field(primary_key=True)
    name: str = ""
    email: str = ""
    age: int = 0


class JsonlSessionModel(SQLModel, table=True):
    """Test session model for JSONL engine."""

    __tablename__ = "jsonl_test_sessions"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    context: dict = Field(default_factory=dict, sa_column=Column(JSON))
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class JsonlSimpleModel(SQLModel, table=True):
    """Simple model with single primary key."""

    __tablename__ = "jsonl_test_simple"  # type: ignore[assignment]

    id: str = Field(primary_key=True)
    value: int = 0


# ============================================================================
# JSONLDatabaseEngine Tests
# ============================================================================


class TestJSONLDatabaseEngine:
    """Tests for JSONLDatabaseEngine."""

    def test_init_default_path(self) -> None:
        """Test initialization with default path."""
        engine = JSONLDatabaseEngine()
        assert engine._base_path == Path.home() / ".nexau" / "data"

    def test_init_custom_path(self) -> None:
        """Test initialization with custom path."""
        custom_path = "/tmp/custom_data"
        engine = JSONLDatabaseEngine(base_path=custom_path)
        assert engine._base_path == Path(custom_path)

    def test_init_with_path_object(self) -> None:
        """Test initialization with Path object."""
        custom_path = Path("/tmp/path_data")
        engine = JSONLDatabaseEngine(base_path=custom_path)
        assert engine._base_path == custom_path

    def test_get_file_path(self) -> None:
        """Test file path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)
            file_path = engine._get_file_path("test_table")
            assert file_path == Path(tmpdir) / "test_table.jsonl"

    def test_pk_to_str(self) -> None:
        """Test primary key to string conversion."""
        engine = JSONLDatabaseEngine()
        pk = {"tenant_id": "t1", "user_id": "u1"}
        pk_str = engine._pk_to_str(pk)
        assert pk_str == "tenant_id=t1|user_id=u1"

    def test_pk_to_str_single_key(self) -> None:
        """Test primary key to string with single key."""
        engine = JSONLDatabaseEngine()
        pk = {"id": "123"}
        pk_str = engine._pk_to_str(pk)
        assert pk_str == "id=123"

    def test_create_and_find_first(self) -> None:
        """Test create and find_first operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                user = JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice", age=30)
                await engine.create(user)

                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                loaded = await engine.find_first(JsonlUserModel, filters=filters)
                assert loaded is not None
                assert loaded.tenant_id == "t1"
                assert loaded.user_id == "u1"
                assert loaded.name == "Alice"
                assert loaded.age == 30

            asyncio.run(run())

    def test_create_persists_to_file(self) -> None:
        """Test that create persists data to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                user = JsonlUserModel(tenant_id="t1", user_id="u1", name="Bob")
                await engine.create(user)

                # Verify file exists
                file_path = engine._get_file_path("jsonl_test_users")
                assert file_path.exists()

                # Verify file content
                with open(file_path) as f:
                    lines = f.readlines()
                    assert len(lines) == 1
                    assert "Bob" in lines[0]

            asyncio.run(run())

    def test_update_existing(self) -> None:
        """Test update operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                user = JsonlUserModel(tenant_id="t1", user_id="u1", name="Original")
                await engine.create(user)

                user.name = "Updated"
                await engine.update(user)

                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                loaded = await engine.find_first(JsonlUserModel, filters=filters)
                assert loaded is not None
                assert loaded.name == "Updated"

            asyncio.run(run())

    def test_find_first_nonexistent(self) -> None:
        """Test find_first returns None for non-existent record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "nonexistent"),
                    ]
                )
                result = await engine.find_first(JsonlUserModel, filters=filters)
                assert result is None

            asyncio.run(run())

    def test_find_many(self) -> None:
        """Test find_many with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice", age=25))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob", age=30))
                await engine.create(JsonlUserModel(tenant_id="t2", user_id="u3", name="Charlie", age=35))

                # Find by tenant
                t1_users = await engine.find_many(
                    JsonlUserModel,
                    filters=ComparisonFilter.eq("tenant_id", "t1"),
                )
                assert len(t1_users) == 2
                names = {u.name for u in t1_users}
                assert names == {"Alice", "Bob"}

            asyncio.run(run())

    def test_find_many_with_limit(self) -> None:
        """Test find_many with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob"))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u3", name="Charlie"))

                results = await engine.find_many(
                    JsonlUserModel,
                    filters=ComparisonFilter.eq("tenant_id", "t1"),
                    limit=2,
                )
                assert len(results) == 2

            asyncio.run(run())

    def test_find_many_with_offset(self) -> None:
        """Test find_many with offset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob"))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u3", name="Charlie"))

                results = await engine.find_many(
                    JsonlUserModel,
                    filters=ComparisonFilter.eq("tenant_id", "t1"),
                    offset=1,
                )
                assert len(results) == 2

            asyncio.run(run())

    def test_find_many_with_order_by(self) -> None:
        """Test find_many with order_by."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Charlie", age=35))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Alice", age=25))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u3", name="Bob", age=30))

                results = await engine.find_many(
                    JsonlUserModel,
                    filters=ComparisonFilter.eq("tenant_id", "t1"),
                    order_by="age",
                )
                assert len(results) == 3
                assert results[0].age == 25
                assert results[1].age == 30
                assert results[2].age == 35

            asyncio.run(run())

    def test_find_many_with_tuple_order_by(self) -> None:
        """Test find_many with tuple order_by."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice", age=30))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob", age=25))

                results = await engine.find_many(
                    JsonlUserModel,
                    filters=ComparisonFilter.eq("tenant_id", "t1"),
                    order_by=("age", "name"),
                )
                assert len(results) == 2
                assert results[0].age == 25

            asyncio.run(run())

    def test_find_many_no_filters(self) -> None:
        """Test find_many without filters returns all records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))
                await engine.create(JsonlUserModel(tenant_id="t2", user_id="u2", name="Bob"))

                results = await engine.find_many(JsonlUserModel)
                assert len(results) == 2

            asyncio.run(run())

    def test_create_many(self) -> None:
        """Test create_many operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                users = [
                    JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"),
                    JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob"),
                    JsonlUserModel(tenant_id="t1", user_id="u3", name="Charlie"),
                ]
                result = await engine.create_many(users)
                assert len(result) == 3

                # Verify all were created
                all_users = await engine.find_many(JsonlUserModel)
                assert len(all_users) == 3

            asyncio.run(run())

    def test_create_many_empty_list(self) -> None:
        """Test create_many with empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                result: list[JsonlUserModel] = await engine.create_many([])
                assert len(result) == 0

            asyncio.run(run())

    def test_delete(self) -> None:
        """Test delete operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                user = JsonlUserModel(tenant_id="t1", user_id="u1", name="ToDelete")
                await engine.create(user)

                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                count = await engine.delete(JsonlUserModel, filters=filters)
                assert count == 1

                loaded = await engine.find_first(JsonlUserModel, filters=filters)
                assert loaded is None

            asyncio.run(run())

    def test_delete_multiple(self) -> None:
        """Test delete operation with multiple records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob"))
                await engine.create(JsonlUserModel(tenant_id="t2", user_id="u3", name="Charlie"))

                # Delete all t1 users
                count = await engine.delete(
                    JsonlUserModel,
                    filters=ComparisonFilter.eq("tenant_id", "t1"),
                )
                assert count == 2

                # Verify only t2 user remains
                remaining = await engine.find_many(JsonlUserModel)
                assert len(remaining) == 1
                assert remaining[0].tenant_id == "t2"

            asyncio.run(run())

    def test_delete_nonexistent(self) -> None:
        """Test delete returns 0 for non-existent records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "nonexistent"),
                    ]
                )
                count = await engine.delete(JsonlUserModel, filters=filters)
                assert count == 0

            asyncio.run(run())

    def test_count(self) -> None:
        """Test count operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                filters = ComparisonFilter.eq("tenant_id", "t1")
                assert await engine.count(JsonlUserModel, filters=filters) == 0

                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))
                assert await engine.count(JsonlUserModel, filters=filters) == 1

                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u2", name="Bob"))
                assert await engine.count(JsonlUserModel, filters=filters) == 2

            asyncio.run(run())

    def test_count_no_filters(self) -> None:
        """Test count without filters returns total count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                await engine.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))
                await engine.create(JsonlUserModel(tenant_id="t2", user_id="u2", name="Bob"))

                count = await engine.count(JsonlUserModel)
                assert count == 2

            asyncio.run(run())

    def test_complex_model_with_json_fields(self) -> None:
        """Test with model containing JSON fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlSessionModel])
                session = JsonlSessionModel(user_id="u1", session_id="s1")
                session.context["key"] = "value"
                session.tags.append("tag1")
                await engine.create(session)

                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("user_id", "u1"),
                        ComparisonFilter.eq("session_id", "s1"),
                    ]
                )
                loaded = await engine.find_first(JsonlSessionModel, filters=filters)
                assert loaded is not None
                assert loaded.context["key"] == "value"
                assert "tag1" in loaded.tags

            asyncio.run(run())

    def test_load_existing_file(self) -> None:
        """Test loading data from existing JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine1 = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                # Create data with first engine
                await engine1.setup_models([JsonlUserModel])
                await engine1.create(JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice"))

                # Create new engine and load existing data
                engine2 = JSONLDatabaseEngine(base_path=tmpdir)
                await engine2.setup_models([JsonlUserModel])

                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                loaded = await engine2.find_first(JsonlUserModel, filters=filters)
                assert loaded is not None
                assert loaded.name == "Alice"

            asyncio.run(run())

    def test_load_corrupted_file(self) -> None:
        """Test loading file with corrupted lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])

                # Create file with corrupted data
                file_path = engine._get_file_path("jsonl_test_users")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w") as f:
                    f.write('{"tenant_id": "t1", "user_id": "u1", "name": "Alice"}\n')
                    f.write("corrupted line\n")
                    f.write('{"tenant_id": "t1", "user_id": "u2", "name": "Bob"}\n')

                # Should load valid lines and skip corrupted ones
                users = await engine.find_many(JsonlUserModel)
                assert len(users) == 2
                names = {u.name for u in users}
                assert names == {"Alice", "Bob"}

            asyncio.run(run())

    def test_multiple_tables(self) -> None:
        """Test multiple model types sharing the same engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel, JsonlSessionModel])

                user = JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice")
                await engine.create(user)

                session = JsonlSessionModel(user_id="u1", session_id="s1")
                await engine.create(session)

                # Both should be accessible
                user_filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                loaded_user = await engine.find_first(JsonlUserModel, filters=user_filters)

                session_filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("user_id", "u1"),
                        ComparisonFilter.eq("session_id", "s1"),
                    ]
                )
                loaded_session = await engine.find_first(JsonlSessionModel, filters=session_filters)

                assert loaded_user is not None
                assert loaded_session is not None
                assert loaded_user.name == "Alice"

            asyncio.run(run())

    def test_cache_behavior(self) -> None:
        """Test that cache is used after initial load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = JSONLDatabaseEngine(base_path=tmpdir)

            async def run() -> None:
                await engine.setup_models([JsonlUserModel])
                user = JsonlUserModel(tenant_id="t1", user_id="u1", name="Alice")
                await engine.create(user)

                # First load should populate cache
                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                loaded1 = await engine.find_first(JsonlUserModel, filters=filters)
                assert loaded1 is not None

                # Second load should use cache
                loaded2 = await engine.find_first(JsonlUserModel, filters=filters)
                assert loaded2 is not None
                assert loaded2.name == "Alice"

            asyncio.run(run())
