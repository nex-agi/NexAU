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

"""Unit tests for serialization_utils."""

import uuid
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel

from nexau.archs.session.models.serialization_utils import sanitize_for_serialization


class SamplePydanticModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class LegacyDictModel:
    """Legacy model with dict() method."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def dict(self) -> dict[str, Any]:
        return self._data


class NonSerializable:
    """A class that cannot be serialized."""

    def __init__(self, value: str):
        self.value = value


class TestSanitizeForSerialization:
    """Tests for sanitize_for_serialization function."""

    def test_none_passthrough(self):
        """Test that None passes through unchanged."""
        assert sanitize_for_serialization(None) is None

    def test_primitives_passthrough(self):
        """Test that primitive types pass through unchanged."""
        assert sanitize_for_serialization("hello") == "hello"
        assert sanitize_for_serialization(42) == 42
        assert sanitize_for_serialization(3.14) == 3.14
        assert sanitize_for_serialization(True) is True
        assert sanitize_for_serialization(False) is False

    def test_pydantic_model_serialization(self):
        """Test Pydantic model serialization via model_dump."""
        model = SamplePydanticModel(name="test", value=123)
        result = sanitize_for_serialization(model)
        assert result == {"name": "test", "value": 123}

    def test_legacy_dict_method(self):
        """Test object with legacy dict() method."""
        obj = LegacyDictModel({"key": "value", "num": 42})
        result = sanitize_for_serialization(obj)
        assert result == {"key": "value", "num": 42}

    def test_dict_serialization(self):
        """Test dict serialization with nested values."""
        data = {"name": "test", "nested": {"inner": "value"}}
        result = sanitize_for_serialization(data)
        assert result == {"name": "test", "nested": {"inner": "value"}}

    def test_dict_with_non_serializable_values_skipped(self):
        """Test that non-serializable values in dict are skipped."""
        non_ser = NonSerializable("test")
        data = {"good": "value", "bad": non_ser}
        result = sanitize_for_serialization(data)
        # The "bad" key should be skipped
        assert result == {"good": "value"}

    def test_list_serialization(self):
        """Test list serialization."""
        data = [1, "two", 3.0, True]
        result = sanitize_for_serialization(data)
        assert result == [1, "two", 3.0, True]

    def test_tuple_serialization(self):
        """Test tuple serialization (converted to list)."""
        data = (1, "two", 3.0)
        result = sanitize_for_serialization(data)
        assert result == [1, "two", 3.0]

    def test_list_with_non_serializable_values_skipped(self):
        """Test that non-serializable values in list are skipped."""
        non_ser = NonSerializable("test")
        data = ["good", non_ser, "also_good"]
        result = sanitize_for_serialization(data)
        # The non-serializable item should be skipped
        assert result == ["good", "also_good"]

    def test_datetime_serialization(self):
        """Test datetime serialization to ISO format."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = sanitize_for_serialization(dt)
        assert result == "2024-01-15T10:30:45"

    def test_uuid_serialization(self):
        """Test UUID serialization to string."""
        test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = sanitize_for_serialization(test_uuid)
        assert result == "12345678-1234-5678-1234-567812345678"

    def test_json_serializable_object_passthrough(self):
        """Test that JSON-serializable objects pass through."""
        # A simple object that json.dumps can handle
        data = {"list": [1, 2, 3], "nested": {"a": 1}}
        result = sanitize_for_serialization(data)
        assert result == data

    def test_non_serializable_raises_value_error(self):
        """Test that non-serializable objects raise ValueError."""
        non_ser = NonSerializable("test")
        with pytest.raises(ValueError, match="not serializable"):
            sanitize_for_serialization(non_ser)

    def test_nested_pydantic_model(self):
        """Test nested Pydantic model serialization."""

        class OuterModel(BaseModel):
            inner: SamplePydanticModel
            extra: str

        outer = OuterModel(inner=SamplePydanticModel(name="nested", value=99), extra="data")
        result = sanitize_for_serialization(outer)
        assert result == {"inner": {"name": "nested", "value": 99}, "extra": "data"}

    def test_deeply_nested_dict(self):
        """Test deeply nested dict serialization."""
        data = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        result = sanitize_for_serialization(data)
        assert result == {"level1": {"level2": {"level3": {"value": "deep"}}}}

    def test_mixed_types_in_list(self):
        """Test list with mixed serializable types."""
        dt = datetime(2024, 1, 1)
        test_uuid = uuid.uuid4()
        model = SamplePydanticModel(name="test", value=1)

        data = [dt, test_uuid, model, "string", 42]
        result = sanitize_for_serialization(data)

        assert result[0] == dt.isoformat()
        assert result[1] == str(test_uuid)
        assert result[2] == {"name": "test", "value": 1}
        assert result[3] == "string"
        assert result[4] == 42

    def test_empty_dict(self):
        """Test empty dict serialization."""
        assert sanitize_for_serialization({}) == {}

    def test_empty_list(self):
        """Test empty list serialization."""
        assert sanitize_for_serialization([]) == []
