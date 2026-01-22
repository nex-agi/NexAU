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

"""SQL database engine using SQLModel/SQLAlchemy.

This engine accepts Filter DSL filters and converts them to SQLAlchemy
ColumnElement[bool] using to_sqlalchemy() before passing to where() clauses.

Requirements implemented:
- 5.4: THE SQLDatabaseEngine SHALL 使用 to_sqlalchemy() 方法转换过滤器为 SQL 表达式
"""

from __future__ import annotations

from typing import Any, TypeVar

from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel import SQLModel, select

from .engine import DatabaseEngine
from .filters import Filter, to_sqlalchemy

T = TypeVar("T", bound=SQLModel)


class SQLDatabaseEngine(DatabaseEngine):
    """Async SQL database engine supporting SQLite, PostgreSQL, MySQL."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._initialized_models: set[type[SQLModel]] = set()

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
        **kwargs: Any,
    ) -> SQLDatabaseEngine:
        """Create engine from database URL."""
        if not any(driver in url for driver in ["+asyncpg", "+aiosqlite", "+aiomysql"]):
            raise ValueError(f"URL must contain async driver (+asyncpg, +aiosqlite, or +aiomysql): {url}")

        if "sqlite" in url:
            # SQLite-specific configuration for better concurrency
            connect_args = kwargs.pop("connect_args", {})
            connect_args.setdefault("check_same_thread", False)
            connect_args.setdefault("timeout", 30)
            engine = create_async_engine(
                url,
                echo=echo,
                connect_args=connect_args,
                poolclass=kwargs.pop("poolclass", None),  # Use NullPool for SQLite by default
                **kwargs,
            )
        else:
            engine = create_async_engine(
                url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                echo=echo,
                **kwargs,
            )

        return cls(engine)

    async def setup_models(self, model_classes: list[type[SQLModel]]) -> None:
        """Create tables for all model classes."""
        async with self._engine.begin() as conn:
            if "sqlite" in str(self._engine.url.drivername):
                # Enable WAL mode for better concurrency
                await conn.execute(text("PRAGMA journal_mode=WAL"))
                # Set busy timeout to 30 seconds
                await conn.execute(text("PRAGMA busy_timeout=30000"))
                # Enable synchronous=NORMAL for better performance
                await conn.execute(text("PRAGMA synchronous=NORMAL"))

            await conn.run_sync(SQLModel.metadata.create_all)
        for model_class in model_classes:
            self._initialized_models.add(model_class)

    async def find_first(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> T | None:
        """Find the first record matching filters.

        Args:
            model_class: The SQLModel class to query
            filters: Filter DSL filter expression

        Returns:
            The first matching record, or None if no match found
        """
        # Convert Filter DSL to SQLAlchemy ColumnElement (Requirement 5.4)
        column_element = to_sqlalchemy(filters, model_class)
        stmt = select(model_class).where(column_element).limit(1)
        async with AsyncSession(self._engine) as session:
            result = await session.execute(stmt)
            return result.scalars().first()

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
            filters: Optional Filter DSL filter expression
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Field name(s) to order by. Prefix with "-" for descending.

        Returns:
            List of matching records
        """
        stmt = select(model_class)

        if filters is not None:
            # Convert Filter DSL to SQLAlchemy ColumnElement (Requirement 5.4)
            column_element = to_sqlalchemy(filters, model_class)
            stmt = stmt.where(column_element)

        if order_by:
            fields = (order_by,) if isinstance(order_by, str) else order_by
            for field in fields:
                if field.startswith("-"):
                    stmt = stmt.order_by(getattr(model_class, field[1:]).desc())
                else:
                    stmt = stmt.order_by(getattr(model_class, field))

        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)

        async with AsyncSession(self._engine) as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def create(self, model: T) -> T:
        """Create a new record.

        Note: This method flushes before committing to detect constraint violations early.
        """
        async with AsyncSession(self._engine) as session:
            session.add(model)
            try:
                # Flush first to detect constraint violations before commit
                await session.flush()
                await session.commit()
            except Exception:
                await session.rollback()
                raise

            # Refresh in a separate try-except to handle detached instances
            try:
                await session.refresh(model)
            except Exception as e:
                # If refresh fails, the object might be detached
                # This can happen in concurrent scenarios where the session closes
                # before refresh completes. Log for debugging but don't fail the create.
                import logging

                logging.getLogger(__name__).debug("Failed to refresh model after create (object may be detached): %s", e)

            return model

    async def create_many(self, models: list[T]) -> list[T]:
        """Create multiple records."""
        if not models:
            return []

        async with AsyncSession(self._engine) as session:
            for model in models:
                session.add(model)
            await session.commit()
            for model in models:
                await session.refresh(model)
            return models

    async def update(self, model: T) -> T:
        """Update a record by primary key."""
        async with AsyncSession(self._engine) as session:
            merged = await session.merge(model)
            await session.commit()
            await session.refresh(merged)
            return merged  # type: ignore[return-value]

    async def delete(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> int:
        """Delete records matching filters.

        Args:
            model_class: The SQLModel class to delete from
            filters: Filter DSL filter expression

        Returns:
            Number of records deleted
        """
        async with AsyncSession(self._engine) as session:
            # Convert Filter DSL to SQLAlchemy ColumnElement (Requirement 5.4)
            column_element = to_sqlalchemy(filters, model_class)
            stmt = select(model_class).where(column_element)
            result = await session.execute(stmt)
            models = list(result.scalars().all())
            count = len(models)
            for model in models:
                await session.delete(model)
            await session.commit()
            return count

    async def count(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
    ) -> int:
        """Count records matching filters.

        Args:
            model_class: The SQLModel class to count
            filters: Optional Filter DSL filter expression

        Returns:
            Number of matching records
        """
        stmt = select(func.count()).select_from(model_class)
        if filters is not None:
            # Convert Filter DSL to SQLAlchemy ColumnElement (Requirement 5.4)
            column_element = to_sqlalchemy(filters, model_class)
            stmt = stmt.where(column_element)

        async with AsyncSession(self._engine) as session:
            result = await session.execute(stmt)
            return result.scalar() or 0
