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

"""Database engine abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from sqlmodel import SQLModel

from .filters import AndFilter, ComparisonFilter, Filter

T = TypeVar("T", bound=SQLModel)


def get_table_name(model_class: type[SQLModel]) -> str:
    """Get table name from a SQLModel class."""
    return str(model_class.__tablename__)  # type: ignore[attr-defined]


def get_pk_fields(model_class: type[SQLModel]) -> list[str]:
    """Get primary key field names from a SQLModel class."""
    return [name for name, info in model_class.model_fields.items() if getattr(info, "primary_key", False) is True]


class DatabaseEngine(ABC):
    """Abstract database engine for multi-backend storage."""

    @abstractmethod
    async def setup_models(self, model_classes: list[type[SQLModel]]) -> None:
        """Setup storage for the given model classes."""

    @abstractmethod
    async def find_first(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> T | None:
        """Find the first record matching filters."""

    @abstractmethod
    async def find_many(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | tuple[str, ...] | None = None,
    ) -> list[T]:
        """Find all records matching filters."""

    @abstractmethod
    async def create(self, model: T) -> T:
        """Create a new record. Returns the created record."""

    @abstractmethod
    async def create_many(self, models: list[T]) -> list[T]:
        """Create multiple records. Returns the created records."""

    @abstractmethod
    async def update(self, model: T) -> T:
        """Update a record by primary key. Returns the updated record."""

    @abstractmethod
    async def delete(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> int:
        """Delete records matching filters. Returns count of deleted records."""

    @abstractmethod
    async def count(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
    ) -> int:
        """Count records matching filters."""

    async def upsert(self, model: T) -> tuple[T, bool]:
        """Create or update a record. Returns (model, created)."""
        model_class = type(model)
        pk_fields = [name for name, info in model_class.model_fields.items() if getattr(info, "primary_key", False) is True]

        # Build filter using Filter DSL
        pk_filters = [ComparisonFilter.eq(field, getattr(model, field)) for field in pk_fields]

        # Use AndFilter for multiple primary key fields, or single ComparisonFilter
        if len(pk_filters) == 1:
            filter_expr: Filter = pk_filters[0]
        else:
            filter_expr = AndFilter(filters=pk_filters)

        existing = await self.find_first(model_class, filters=filter_expr)
        if existing is not None:
            return await self.update(model), False
        return await self.create(model), True

    async def get_or_create(self, model: T) -> tuple[T, bool]:
        """Get or create a record. Returns (model, created)."""
        model_class = type(model)
        pk_fields = [name for name, info in model_class.model_fields.items() if getattr(info, "primary_key", False) is True]

        # Build filter using Filter DSL
        pk_filters = [ComparisonFilter.eq(field, getattr(model, field)) for field in pk_fields]

        # Use AndFilter for multiple primary key fields, or single ComparisonFilter
        if len(pk_filters) == 1:
            filter_expr: Filter = pk_filters[0]
        else:
            filter_expr = AndFilter(filters=pk_filters)

        existing = await self.find_first(model_class, filters=filter_expr)
        if existing is not None:
            return existing, False
        return await self.create(model), True
