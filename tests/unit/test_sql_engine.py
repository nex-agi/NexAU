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

"""Unit tests for SQLDatabaseEngine."""

import asyncio

import pytest
from sqlmodel import Field, SQLModel

from nexau.archs.session.orm.filters import ComparisonFilter
from nexau.archs.session.orm.sql_engine import SQLDatabaseEngine


# Unique table name to avoid conflicts with other tests
class SqlEngineTestModel(SQLModel, table=True):
    """Test model for database tests."""

    __tablename__ = "sql_engine_test_items"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    value: int
    category: str | None = None


class TestSQLDatabaseEngineFromUrl:
    """Tests for SQLDatabaseEngine.from_url() factory method."""

    def test_from_url_sqlite(self):
        """Test creating engine from SQLite URL."""
        engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
        assert engine is not None
        assert engine._engine is not None

    def test_from_url_missing_async_driver_raises(self):
        """Test that URL without async driver raises ValueError."""
        with pytest.raises(ValueError, match="async driver"):
            SQLDatabaseEngine.from_url("sqlite:///test.db")

    def test_from_url_postgresql_missing_driver_raises(self):
        """Test PostgreSQL URL without async driver raises."""
        with pytest.raises(ValueError, match="async driver"):
            SQLDatabaseEngine.from_url("postgresql://localhost/test")

    def test_from_url_with_echo(self):
        """Test creating engine with echo enabled."""
        engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:", echo=True)
        assert engine is not None


class TestSQLDatabaseEngineCRUD:
    """Tests for CRUD operations."""

    def test_create_single_record(self):
        """Test creating a single record."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            model = SqlEngineTestModel(name="test1", value=100)
            created = await engine.create(model)

            assert created.id is not None
            assert created.name == "test1"
            assert created.value == 100

        asyncio.run(run())

    def test_create_many_records(self):
        """Test creating multiple records."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            models = [
                SqlEngineTestModel(name="item1", value=1),
                SqlEngineTestModel(name="item2", value=2),
                SqlEngineTestModel(name="item3", value=3),
            ]
            created = await engine.create_many(models)

            assert len(created) == 3
            assert all(m.id is not None for m in created)

        asyncio.run(run())

    def test_create_many_empty_list(self):
        """Test create_many with empty list returns empty list."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            result = await engine.create_many([])
            assert result == []

        asyncio.run(run())

    def test_find_first(self):
        """Test finding first record."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="find_me", value=42))
            await engine.create(SqlEngineTestModel(name="ignore_me", value=99))

            result = await engine.find_first(
                SqlEngineTestModel,
                filters=ComparisonFilter.eq("name", "find_me"),
            )

            assert result is not None
            assert result.name == "find_me"
            assert result.value == 42

        asyncio.run(run())

    def test_find_first_no_match(self):
        """Test find_first returns None when no match."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="exists", value=1))

            result = await engine.find_first(
                SqlEngineTestModel,
                filters=ComparisonFilter.eq("name", "not_exists"),
            )

            assert result is None

        asyncio.run(run())

    def test_find_many_all_records(self):
        """Test finding all records without filter."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="a", value=1))
            await engine.create(SqlEngineTestModel(name="b", value=2))
            await engine.create(SqlEngineTestModel(name="c", value=3))

            result = await engine.find_many(SqlEngineTestModel)

            assert len(result) == 3

        asyncio.run(run())

    def test_find_many_with_filter(self):
        """Test finding records with filter."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="a", value=10, category="cat1"))
            await engine.create(SqlEngineTestModel(name="b", value=20, category="cat2"))
            await engine.create(SqlEngineTestModel(name="c", value=30, category="cat1"))

            result = await engine.find_many(
                SqlEngineTestModel,
                filters=ComparisonFilter.eq("category", "cat1"),
            )

            assert len(result) == 2
            assert all(m.category == "cat1" for m in result)

        asyncio.run(run())

    def test_find_many_with_limit(self):
        """Test finding records with limit."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            for i in range(5):
                await engine.create(SqlEngineTestModel(name=f"item{i}", value=i))

            result = await engine.find_many(SqlEngineTestModel, limit=3)

            assert len(result) == 3

        asyncio.run(run())

    def test_find_many_with_offset(self):
        """Test finding records with offset."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            for i in range(5):
                await engine.create(SqlEngineTestModel(name=f"item{i}", value=i))

            result = await engine.find_many(SqlEngineTestModel, offset=2)

            assert len(result) == 3

        asyncio.run(run())

    def test_find_many_with_order_by_ascending(self):
        """Test finding records with ascending order."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="c", value=3))
            await engine.create(SqlEngineTestModel(name="a", value=1))
            await engine.create(SqlEngineTestModel(name="b", value=2))

            result = await engine.find_many(SqlEngineTestModel, order_by="value")

            assert len(result) == 3
            assert result[0].value == 1
            assert result[1].value == 2
            assert result[2].value == 3

        asyncio.run(run())

    def test_find_many_with_order_by_descending(self):
        """Test finding records with descending order."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="c", value=3))
            await engine.create(SqlEngineTestModel(name="a", value=1))
            await engine.create(SqlEngineTestModel(name="b", value=2))

            result = await engine.find_many(SqlEngineTestModel, order_by="-value")

            assert len(result) == 3
            assert result[0].value == 3
            assert result[1].value == 2
            assert result[2].value == 1

        asyncio.run(run())

    def test_find_many_with_multiple_order_by(self):
        """Test finding records with multiple order_by fields."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="a", value=1, category="cat1"))
            await engine.create(SqlEngineTestModel(name="b", value=2, category="cat1"))
            await engine.create(SqlEngineTestModel(name="c", value=1, category="cat2"))

            result = await engine.find_many(SqlEngineTestModel, order_by=("value", "name"))

            assert len(result) == 3
            # Should be ordered by value first, then name
            assert result[0].name == "a"  # value=1, name="a"
            assert result[1].name == "c"  # value=1, name="c"
            assert result[2].name == "b"  # value=2

        asyncio.run(run())

    def test_update_record(self):
        """Test updating a record."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            created = await engine.create(SqlEngineTestModel(name="original", value=100))

            created.name = "updated"
            created.value = 200
            updated = await engine.update(created)

            assert updated.name == "updated"
            assert updated.value == 200

        asyncio.run(run())

    def test_delete_records(self):
        """Test deleting records."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="keep", value=1, category="keeper"))
            await engine.create(SqlEngineTestModel(name="delete1", value=2, category="delete"))
            await engine.create(SqlEngineTestModel(name="delete2", value=3, category="delete"))

            deleted_count = await engine.delete(
                SqlEngineTestModel,
                filters=ComparisonFilter.eq("category", "delete"),
            )

            assert deleted_count == 2

            # Verify remaining records
            remaining = await engine.find_many(SqlEngineTestModel)
            assert len(remaining) == 1
            assert remaining[0].name == "keep"

        asyncio.run(run())

    def test_count_all_records(self):
        """Test counting all records."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            for i in range(5):
                await engine.create(SqlEngineTestModel(name=f"item{i}", value=i))

            count = await engine.count(SqlEngineTestModel)

            assert count == 5

        asyncio.run(run())

    def test_count_with_filter(self):
        """Test counting records with filter."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            await engine.create(SqlEngineTestModel(name="a", value=10, category="cat1"))
            await engine.create(SqlEngineTestModel(name="b", value=20, category="cat2"))
            await engine.create(SqlEngineTestModel(name="c", value=30, category="cat1"))

            count = await engine.count(
                SqlEngineTestModel,
                filters=ComparisonFilter.eq("category", "cat1"),
            )

            assert count == 2

        asyncio.run(run())

    def test_count_empty_table(self):
        """Test counting empty table returns 0."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            count = await engine.count(SqlEngineTestModel)
            assert count == 0

        asyncio.run(run())


class TestSQLDatabaseEngineInitModels:
    """Tests for setup_models method."""

    def test_setup_models_creates_tables(self):
        """Test that setup_models creates tables."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            # Should be able to create records after init
            created = await engine.create(SqlEngineTestModel(name="test", value=1))
            assert created.id is not None

        asyncio.run(run())

    def test_setup_models_tracks_initialized(self):
        """Test that setup_models tracks initialized models."""

        async def run():
            engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///:memory:")
            await engine.setup_models([SqlEngineTestModel])

            assert SqlEngineTestModel in engine._initialized_models

        asyncio.run(run())
