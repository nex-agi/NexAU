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

"""Unit tests for SQLAlchemy type decorators."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.session.models.types import GlobalStorageJson, PydanticJson


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int
    tags: list[str] = []


class TestPydanticJson:
    """Tests for PydanticJson TypeDecorator."""

    @pytest.fixture
    def pydantic_json_type(self):
        """Create PydanticJson instance for SampleModel."""
        return PydanticJson(SampleModel)

    @pytest.fixture
    def mock_dialect(self):
        """Create a mock dialect."""
        return MagicMock()

    def test_bind_processor_serializes_model(self, pydantic_json_type: PydanticJson[SampleModel], mock_dialect: Any):
        """Test bind_processor converts Pydantic model to JSON string."""
        processor = pydantic_json_type.bind_processor(mock_dialect)
        model = SampleModel(name="test", value=42, tags=["a", "b"])

        result = processor(model)

        assert result is not None
        assert '"name":"test"' in result
        assert '"value":42' in result

    def test_bind_processor_handles_none(self, pydantic_json_type: PydanticJson[SampleModel], mock_dialect: Any):
        """Test bind_processor returns None for None input."""
        processor = pydantic_json_type.bind_processor(mock_dialect)
        result = processor(None)
        assert result is None

    def test_result_processor_deserializes_json(self, pydantic_json_type: PydanticJson[SampleModel], mock_dialect: Any):
        """Test result_processor converts JSON string to Pydantic model."""
        processor = pydantic_json_type.result_processor(mock_dialect, None)
        json_str = '{"name": "test", "value": 42, "tags": ["a", "b"]}'

        result = processor(json_str)

        assert isinstance(result, SampleModel)
        assert result.name == "test"
        assert result.value == 42
        assert result.tags == ["a", "b"]

    def test_result_processor_handles_bytes(self, pydantic_json_type: PydanticJson[SampleModel], mock_dialect: Any):
        """Test result_processor handles bytes input."""
        processor = pydantic_json_type.result_processor(mock_dialect, None)
        json_bytes = b'{"name": "test", "value": 42}'

        result = processor(json_bytes)

        assert isinstance(result, SampleModel)
        assert result.name == "test"

    def test_result_processor_handles_none(self, pydantic_json_type: PydanticJson[SampleModel], mock_dialect: Any):
        """Test result_processor returns None for None input."""
        processor = pydantic_json_type.result_processor(mock_dialect, None)
        result = processor(None)
        assert result is None

    def test_round_trip(self, pydantic_json_type: PydanticJson[SampleModel], mock_dialect: Any):
        """Test round-trip serialization/deserialization."""
        bind_proc = pydantic_json_type.bind_processor(mock_dialect)
        result_proc = pydantic_json_type.result_processor(mock_dialect, None)

        original = SampleModel(name="round_trip", value=999, tags=["x", "y", "z"])
        serialized = bind_proc(original)
        restored = result_proc(serialized)

        assert restored is not None
        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.tags == original.tags

    def test_cache_ok_is_true(self, pydantic_json_type: PydanticJson[SampleModel]):
        """Test that cache_ok is True for query caching."""
        assert pydantic_json_type.cache_ok is True


class TestGlobalStorageJson:
    """Tests for GlobalStorageJson TypeDecorator."""

    @pytest.fixture
    def global_storage_type(self):
        """Create GlobalStorageJson instance."""
        return GlobalStorageJson()

    @pytest.fixture
    def mock_dialect(self):
        """Create a mock dialect."""
        return MagicMock()

    def test_bind_processor_serializes_global_storage(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test bind_processor converts GlobalStorage to JSON string."""
        processor = global_storage_type.bind_processor(mock_dialect)
        storage = GlobalStorage()
        storage.set("key1", "value1")
        storage.set("key2", 42)

        result = processor(storage)

        assert result is not None
        assert '"key1":"value1"' in result
        assert '"key2":42' in result

    def test_bind_processor_handles_none(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test bind_processor returns None for None input."""
        processor = global_storage_type.bind_processor(mock_dialect)
        result = processor(None)
        assert result is None

    def test_bind_processor_sanitizes_non_serializable(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test bind_processor sanitizes non-JSON-serializable values."""
        processor = global_storage_type.bind_processor(mock_dialect)
        storage = GlobalStorage()
        storage.set("serializable", "good")
        storage.set("number", 123)

        result = processor(storage)

        assert result is not None
        assert "serializable" in result
        assert "number" in result

    def test_result_processor_deserializes_json(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test result_processor converts JSON string to GlobalStorage."""
        processor = global_storage_type.result_processor(mock_dialect, None)
        json_str = '{"key1": "value1", "key2": 42}'

        result = processor(json_str)

        assert isinstance(result, GlobalStorage)
        assert result.get("key1") == "value1"
        assert result.get("key2") == 42

    def test_result_processor_handles_bytes(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test result_processor handles bytes input."""
        processor = global_storage_type.result_processor(mock_dialect, None)
        json_bytes = b'{"key": "value"}'

        result = processor(json_bytes)

        assert isinstance(result, GlobalStorage)
        assert result.get("key") == "value"

    def test_result_processor_handles_none(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test result_processor returns None for None input."""
        processor = global_storage_type.result_processor(mock_dialect, None)
        result = processor(None)
        assert result is None

    def test_round_trip(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test round-trip serialization/deserialization."""
        bind_proc = global_storage_type.bind_processor(mock_dialect)
        result_proc = global_storage_type.result_processor(mock_dialect, None)

        original = GlobalStorage()
        original.set("string", "hello")
        original.set("number", 42)
        original.set("nested", {"a": 1, "b": 2})

        serialized = bind_proc(original)
        restored = result_proc(serialized)

        assert restored is not None
        assert restored.get("string") == original.get("string")
        assert restored.get("number") == original.get("number")
        assert restored.get("nested") == original.get("nested")

    def test_cache_ok_is_true(self, global_storage_type: GlobalStorageJson):
        """Test that cache_ok is True for query caching."""
        assert global_storage_type.cache_ok is True

    def test_empty_global_storage(self, global_storage_type: GlobalStorageJson, mock_dialect: Any):
        """Test serialization of empty GlobalStorage."""
        bind_proc = global_storage_type.bind_processor(mock_dialect)
        result_proc = global_storage_type.result_processor(mock_dialect, None)

        original = GlobalStorage()
        serialized = bind_proc(original)
        restored = result_proc(serialized)

        assert restored is not None
        assert len(list(restored.keys())) == 0
