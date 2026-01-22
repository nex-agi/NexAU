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

"""Custom SQLAlchemy types for Pydantic models."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from sqlalchemy.types import JSON, TypeDecorator

from nexau.archs.main_sub.agent_context import GlobalStorage

from .serialization_utils import sanitize_for_serialization


class PydanticJson[T](TypeDecorator[T]):
    """SQLAlchemy TypeDecorator for Pydantic models.

    Handles serialization/deserialization of Pydantic models to/from JSON
    at the SQLAlchemy level using TypeAdapter.
    """

    impl = JSON()
    cache_ok = True

    def __init__(self, pydantic_type: type[T]) -> None:
        super().__init__()
        self._adapter: TypeAdapter[T] = TypeAdapter(pydantic_type)
        self.coerce_compared_value = self.impl.coerce_compared_value  # type: ignore[method-assign]

    def bind_processor(self, dialect: Any) -> Any:
        def process(value: T | None) -> str | None:
            return self._adapter.dump_json(value).decode("utf-8") if value is not None else None

        return process

    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        def process(value: str | bytes | None) -> T | None:
            return self._adapter.validate_json(value) if value is not None else None

        return process


class GlobalStorageJson(TypeDecorator[GlobalStorage]):
    """SQLAlchemy TypeDecorator for GlobalStorage objects.

    - On bind: accepts GlobalStorage or dict and serializes to JSON string
      after sanitizing non-serializable values.
    - On result: reconstructs a GlobalStorage populated with deserialized dict.
    """

    impl = JSON()
    cache_ok = True

    def __init__(self) -> None:
        super().__init__()
        self._dict_adapter: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
        self.coerce_compared_value = self.impl.coerce_compared_value  # type: ignore[method-assign]

    def bind_processor(self, dialect: Any) -> Any:
        def process(value: GlobalStorage | None) -> str | None:
            if value is None:
                return None
            raw: dict[str, Any] = value.to_dict()
            sanitized = sanitize_for_serialization(raw)
            return self._dict_adapter.dump_json(sanitized).decode("utf-8")

        return process

    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        def process(value: str | bytes | None) -> GlobalStorage | None:
            if value is None:
                return None
            data = self._dict_adapter.validate_json(value)
            gs = GlobalStorage()
            gs.update(data)
            return gs

        return process
