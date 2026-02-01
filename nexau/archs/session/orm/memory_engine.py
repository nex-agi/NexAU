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

"""In-memory database engine implementation using pure Python objects.

This provides ephemeral storage with Filter DSL support using Python evaluation.
All data is lost when the Python process terminates.

Thread-safe implementation using asyncio.Lock for concurrent access protection.

For SQLite-based in-memory storage, use SQLDatabaseEngine with sqlite:///:memory:.
"""

from __future__ import annotations

import threading
from typing import TypeVar

from sqlmodel import SQLModel

from .engine import DatabaseEngine, get_pk_fields, get_table_name
from .filters import Filter, evaluate

T = TypeVar("T", bound=SQLModel)

# Global singleton instance for shared in-memory storage
_shared_instance: InMemoryDatabaseEngine | None = None


class InMemoryDatabaseEngine(DatabaseEngine):
    """Thread-safe in-memory storage engine using pure Python objects.

    This provides ephemeral storage with Filter DSL support using Python evaluation.
    All data is lost when the Python process terminates.

    Thread Safety:
        - Uses asyncio.Lock to protect all read/write operations
        - Safe for concurrent access from multiple asyncio tasks
        - Each engine instance is isolated (process-level storage)

    For SQLite-based in-memory storage with full SQL support, use:
        SQLDatabaseEngine("sqlite+aiosqlite:///:memory:")

    Example:
        >>> engine = InMemoryDatabaseEngine()
        >>> await engine.setup_models([SessionModel])
        >>> # Filter DSL
        >>> from nexau.archs.session.orm.filter_dsl import ComparisonFilter
        >>> result = await engine.find_first(SessionModel, filters=ComparisonFilter.eq("user_id", "alice"))
        >>>
        >>> # Combined filters using AndFilter
        >>> from nexau.archs.session.orm.filter_dsl import AndFilter
        >>> result = await engine.find_many(
        ...     SessionModel, filters=AndFilter(filters=[ComparisonFilter.eq("user_id", "alice"), ComparisonFilter.eq("session_id", "s1")])
        ... )
    """

    def __init__(self) -> None:
        """Initialize the in-memory database engine."""
        # Storage: {model_class_name: {pk_tuple: model_instance}}
        self._storage: dict[str, dict[tuple[object, ...], SQLModel]] = {}
        self._initialized_models: set[type[SQLModel]] = set()
        # Lock for thread-safe operations
        self._lock = threading.RLock()

    @staticmethod
    def get_shared_instance() -> InMemoryDatabaseEngine:
        """Get the global shared instance of InMemoryDatabaseEngine.

        This ensures all agents without an explicit session_manager share
        the same in-memory storage, allowing history to persist across
        agent instances within the same process.

        Returns:
            The shared InMemoryDatabaseEngine instance.
        """
        global _shared_instance
        if _shared_instance is None:
            _shared_instance = InMemoryDatabaseEngine()
        return _shared_instance

    async def setup_models(self, model_classes: list[type[SQLModel]]) -> None:
        """Initialize storage for model classes.

        Args:
            model_classes: List of SQLModel classes to initialize storage for.
        """
        with self._lock:
            for model_class in model_classes:
                table_name = get_table_name(model_class)
                if table_name not in self._storage:
                    self._storage[table_name] = {}
                self._initialized_models.add(model_class)

    def _get_pk_tuple(self, model: SQLModel) -> tuple[object, ...]:
        """Get primary key tuple from a model instance."""
        pk_fields = get_pk_fields(type(model))
        return tuple(getattr(model, f) for f in pk_fields)

    def _get_table(self, model_class: type[T]) -> dict[tuple[object, ...], SQLModel]:
        """Get storage table for a model class."""
        table_name = get_table_name(model_class)
        if table_name not in self._storage:
            self._storage[table_name] = {}
        return self._storage[table_name]

    async def find_first(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> T | None:
        """Find the first record matching filters.

        Args:
            model_class: The SQLModel class to query
            filters: Filter DSL expression

        Returns:
            The first matching record, or None if no match found
        """
        with self._lock:
            table = self._get_table(model_class)
            for model in table.values():
                if evaluate(filters, model):
                    return model  # type: ignore[return-value]
            return None

    async def find_many(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | tuple[str, ...] | None = None,
    ) -> list[T]:
        """Find all records matching filters.

        Args:
            model_class: The SQLModel class to query
            filters: Optional Filter DSL expression
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Field name(s) to order by. Prefix with "-" for descending.

        Returns:
            List of matching records
        """
        with self._lock:
            table = self._get_table(model_class)

            # Filter
            if filters is not None:
                results = [model for model in table.values() if evaluate(filters, model)]
            else:
                results = list(table.values())

            # Sort
            if order_by:
                fields = (order_by,) if isinstance(order_by, str) else order_by
                for field in reversed(fields):
                    reverse = field.startswith("-")
                    field_name = field[1:] if reverse else field
                    results.sort(
                        key=lambda m: getattr(m, field_name) or "",
                        reverse=reverse,
                    )

            # Offset and limit
            if offset:
                results = results[offset:]
            if limit:
                results = results[:limit]

            return results  # type: ignore[return-value]

    async def create(self, model: T) -> T:
        """Create a new record.

        Args:
            model: The SQLModel instance to persist

        Returns:
            The created model

        Raises:
            ValueError: If a record with the same primary key already exists
        """
        with self._lock:
            table = self._get_table(type(model))
            pk = self._get_pk_tuple(model)

            # Check for duplicate primary key
            if pk in table:
                pk_fields = get_pk_fields(type(model))
                pk_values = dict(zip(pk_fields, pk))
                raise ValueError(f"Duplicate primary key: {pk_values}")

            table[pk] = model
            return model

    async def create_many(self, models: list[T]) -> list[T]:
        """Create multiple records.

        Args:
            models: List of SQLModel instances to persist

        Returns:
            List of created models
        """
        with self._lock:
            for model in models:
                table = self._get_table(type(model))
                pk = self._get_pk_tuple(model)

                # Check for duplicate primary key
                if pk in table:
                    pk_fields = get_pk_fields(type(model))
                    pk_values = dict(zip(pk_fields, pk))
                    raise ValueError(f"Duplicate primary key: {pk_values}")

                table[pk] = model
        return models

    async def update(self, model: T) -> T:
        """Update a record by primary key.

        Args:
            model: The SQLModel instance with updated values

        Returns:
            The updated model
        """
        with self._lock:
            table = self._get_table(type(model))
            pk = self._get_pk_tuple(model)
            table[pk] = model
            return model

    async def delete(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> int:
        """Delete records matching filters.

        Args:
            model_class: The SQLModel class to delete from
            filters: Filter DSL expression

        Returns:
            Number of records deleted
        """
        with self._lock:
            table = self._get_table(model_class)
            to_delete = [pk for pk, model in table.items() if evaluate(filters, model)]
            for pk in to_delete:
                del table[pk]
            return len(to_delete)

    async def count(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
    ) -> int:
        """Count records matching filters.

        Args:
            model_class: The SQLModel class to count
            filters: Optional Filter DSL expression

        Returns:
            Number of matching records
        """
        with self._lock:
            table = self._get_table(model_class)
            if filters is None:
                return len(table)
            return sum(1 for model in table.values() if evaluate(filters, model))
