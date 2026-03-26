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

"""Coverage improvement tests for nexau/archs/session/orm/engine.py.

Targets uncovered paths in:
- get_table_name / get_pk_fields
- DatabaseEngine.upsert / get_or_create default implementations
- LoopSafeDatabaseEngine bridging paths
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Field, SQLModel

from nexau.archs.session.orm.engine import (
    DatabaseEngine,
    LoopSafeDatabaseEngine,
    get_pk_fields,
    get_table_name,
)
from nexau.archs.session.orm.filters import ComparisonFilter

# ---------------------------------------------------------------------------
# Helper model
# ---------------------------------------------------------------------------


class SampleModel(SQLModel, table=True):
    __tablename__ = "sample"
    id: int = Field(primary_key=True)
    name: str = ""


class CompositePKModel(SQLModel, table=True):
    __tablename__ = "composite"
    part_a: str = Field(primary_key=True)
    part_b: str = Field(primary_key=True)
    value: str = ""


# ---------------------------------------------------------------------------
# get_table_name / get_pk_fields
# ---------------------------------------------------------------------------


class TestEngineHelpers:
    def test_get_table_name(self):
        assert get_table_name(SampleModel) == "sample"

    def test_get_pk_fields_single(self):
        fields = get_pk_fields(SampleModel)
        assert fields == ["id"]

    def test_get_pk_fields_composite(self):
        fields = get_pk_fields(CompositePKModel)
        assert set(fields) == {"part_a", "part_b"}


# ---------------------------------------------------------------------------
# DatabaseEngine default upsert / get_or_create
# ---------------------------------------------------------------------------


class MockDatabaseEngine(DatabaseEngine):
    """Concrete subclass for testing abstract base class defaults."""

    def __init__(self):
        self.stored: dict[int, SampleModel] = {}

    async def setup_models(self, model_classes):
        pass

    async def find_first(self, model_class, *, filters):
        for item in self.stored.values():
            return item
        return None

    async def find_many(self, model_class, *, filters=None, limit=None, offset=None, order_by=None):
        return list(self.stored.values())

    async def create(self, model):
        self.stored[model.id] = model
        return model

    async def create_many(self, models):
        for m in models:
            self.stored[m.id] = m
        return models

    async def update(self, model):
        self.stored[model.id] = model
        return model

    async def delete(self, model_class, *, filters):
        count = len(self.stored)
        self.stored.clear()
        return count

    async def count(self, model_class, *, filters=None):
        return len(self.stored)


class TestDatabaseEngineDefaults:
    @pytest.mark.anyio
    async def test_upsert_creates_new_record(self):
        engine = MockDatabaseEngine()
        engine.stored = {}
        # Override find_first to return None
        engine.find_first = AsyncMock(return_value=None)

        model = SampleModel(id=1, name="Alice")
        result, created = await engine.upsert(model)
        assert created is True
        assert result.name == "Alice"

    @pytest.mark.anyio
    async def test_upsert_updates_existing_record(self):
        engine = MockDatabaseEngine()
        existing = SampleModel(id=1, name="Alice")
        engine.stored = {1: existing}

        updated_model = SampleModel(id=1, name="Bob")
        result, created = await engine.upsert(updated_model)
        assert created is False
        assert result.name == "Bob"

    @pytest.mark.anyio
    async def test_get_or_create_returns_existing(self):
        engine = MockDatabaseEngine()
        existing = SampleModel(id=1, name="Alice")
        engine.stored = {1: existing}

        model = SampleModel(id=1, name="Bob")
        result, created = await engine.get_or_create(model)
        assert created is False
        assert result.name == "Alice"  # existing, not new

    @pytest.mark.anyio
    async def test_get_or_create_creates_new(self):
        engine = MockDatabaseEngine()
        engine.stored = {}
        engine.find_first = AsyncMock(return_value=None)

        model = SampleModel(id=2, name="Carol")
        result, created = await engine.get_or_create(model)
        assert created is True
        assert result.name == "Carol"


# ---------------------------------------------------------------------------
# LoopSafeDatabaseEngine
# ---------------------------------------------------------------------------


class TestLoopSafeDatabaseEngine:
    @pytest.mark.anyio
    async def test_setup_models_captures_loop(self):
        inner = AsyncMock(spec=DatabaseEngine)
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])
        assert wrapper._owner_loop is not None
        inner.setup_models.assert_awaited_once_with([SampleModel])

    @pytest.mark.anyio
    async def test_find_first_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        inner.find_first.return_value = SampleModel(id=1, name="Alice")
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        filt = ComparisonFilter.eq("id", 1)
        result = await wrapper.find_first(SampleModel, filters=filt)
        assert result is not None

    @pytest.mark.anyio
    async def test_find_many_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        inner.find_many.return_value = []
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result = await wrapper.find_many(SampleModel)
        assert result == []

    @pytest.mark.anyio
    async def test_create_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        model = SampleModel(id=1, name="x")
        inner.create.return_value = model
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result = await wrapper.create(model)
        assert result.id == 1

    @pytest.mark.anyio
    async def test_create_many_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        inner.create_many.return_value = []
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result = await wrapper.create_many([])
        assert result == []

    @pytest.mark.anyio
    async def test_update_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        model = SampleModel(id=1, name="y")
        inner.update.return_value = model
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result = await wrapper.update(model)
        assert result.name == "y"

    @pytest.mark.anyio
    async def test_delete_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        inner.delete.return_value = 1
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        filt = ComparisonFilter.eq("id", 1)
        result = await wrapper.delete(SampleModel, filters=filt)
        assert result == 1

    @pytest.mark.anyio
    async def test_count_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        inner.count.return_value = 5
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result = await wrapper.count(SampleModel)
        assert result == 5

    @pytest.mark.anyio
    async def test_upsert_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        model = SampleModel(id=1, name="z")
        inner.upsert.return_value = (model, True)
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result, created = await wrapper.upsert(model)
        assert created is True

    @pytest.mark.anyio
    async def test_get_or_create_direct(self):
        inner = AsyncMock(spec=DatabaseEngine)
        model = SampleModel(id=1, name="w")
        inner.get_or_create.return_value = (model, False)
        wrapper = LoopSafeDatabaseEngine(inner)
        await wrapper.setup_models([SampleModel])

        result, created = await wrapper.get_or_create(model)
        assert created is False

    def test_get_bridge_loop_no_owner(self):
        inner = AsyncMock(spec=DatabaseEngine)
        wrapper = LoopSafeDatabaseEngine(inner)
        assert wrapper._get_bridge_loop() is None

    def test_get_bridge_loop_owner_not_running(self):
        inner = AsyncMock(spec=DatabaseEngine)
        wrapper = LoopSafeDatabaseEngine(inner)
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        wrapper._owner_loop = mock_loop
        assert wrapper._get_bridge_loop() is None
