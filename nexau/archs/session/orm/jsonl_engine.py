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

"""JSONL file database engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, TypeVar

from sqlmodel import SQLModel

from .engine import DatabaseEngine, get_pk_fields, get_table_name
from .filters import Filter, evaluate

T = TypeVar("T", bound=SQLModel)


class JSONLDatabaseEngine(DatabaseEngine):
    """JSONL file database engine - one file per table."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        if base_path is None:
            self._base_path = Path.home() / ".nexau" / "data"
        else:
            self._base_path = Path(base_path).expanduser()

        # Cache: table -> {pk_str -> SQLModel instance}
        self._cache: dict[str, dict[str, SQLModel]] = {}
        self._loaded_tables: set[str] = set()
        self._model_classes: dict[str, type[SQLModel]] = {}

    def _get_file_path(self, table: str) -> Path:
        return self._base_path / f"{table}.jsonl"

    def _pk_to_str(self, primary_key: dict[str, Any]) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(primary_key.items()))

    def _get_pk_from_model(self, model: SQLModel) -> dict[str, Any]:
        pk_fields = get_pk_fields(type(model))
        return {f: model.__getattribute__(f) for f in pk_fields}

    async def _load_table(self, table: str, model_class: type[T]) -> None:
        if table in self._loaded_tables:
            return

        self._cache[table] = {}
        self._model_classes[table] = model_class
        file_path = self._get_file_path(table)

        if not file_path.exists():
            self._loaded_tables.add(table)
            return

        pk_fields = get_pk_fields(model_class)

        def _do_load() -> dict[str, SQLModel]:
            cache: dict[str, SQLModel] = {}
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        pk = {f: record.get(f) for f in pk_fields}
                        pk_str = self._pk_to_str(pk)
                        cache[pk_str] = model_class.model_validate(record)
                    except (json.JSONDecodeError, Exception):
                        continue
            return cache

        self._cache[table] = await asyncio.to_thread(_do_load)
        self._loaded_tables.add(table)

    async def _save_table(self, table: str) -> None:
        file_path = self._get_file_path(table)

        def _do_save() -> None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                for model in self._cache.get(table, {}).values():
                    f.write(model.model_dump_json() + "\n")

        await asyncio.to_thread(_do_save)

    async def setup_models(self, model_classes: list[type[SQLModel]]) -> None:
        for model_class in model_classes:
            table = get_table_name(model_class)
            self._model_classes[table] = model_class

    async def find_first(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> T | None:
        table = get_table_name(model_class)
        await self._load_table(table, model_class)

        for model in self._cache.get(table, {}).values():
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
        table = get_table_name(model_class)
        await self._load_table(table, model_class)

        results: list[T] = []
        for model in self._cache.get(table, {}).values():
            if filters is None or evaluate(filters, model):
                results.append(model)  # type: ignore[arg-type]

        if order_by:
            fields = (order_by,) if isinstance(order_by, str) else order_by

            def sort_key(m: SQLModel) -> tuple[Any, ...]:
                values: list[Any] = []
                for field in fields:
                    if field.startswith("-"):
                        # For descending, we'll handle it differently
                        values.append(m.__getattribute__(field[1:]))
                    else:
                        values.append(m.__getattribute__(field))
                return tuple(values)

            # Simple ascending sort for now
            results.sort(key=sort_key)

        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    async def create(self, model: T) -> T:
        table = get_table_name(type(model))
        await self._load_table(table, type(model))

        pk = self._get_pk_from_model(model)
        pk_str = self._pk_to_str(pk)

        if table not in self._cache:
            self._cache[table] = {}
        self._cache[table][pk_str] = model
        await self._save_table(table)
        return model

    async def create_many(self, models: list[T]) -> list[T]:
        if not models:
            return []

        tables_to_save: set[str] = set()
        for model in models:
            table = get_table_name(type(model))
            await self._load_table(table, type(model))

            pk = self._get_pk_from_model(model)
            pk_str = self._pk_to_str(pk)

            if table not in self._cache:
                self._cache[table] = {}
            self._cache[table][pk_str] = model
            tables_to_save.add(table)

        for table in tables_to_save:
            await self._save_table(table)

        return models

    async def update(self, model: T) -> T:
        # Same as create for JSONL (upsert behavior)
        return await self.create(model)

    async def delete(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> int:
        table = get_table_name(model_class)
        await self._load_table(table, model_class)

        to_delete: list[str] = []
        for pk_str, model in self._cache.get(table, {}).items():
            if evaluate(filters, model):
                to_delete.append(pk_str)

        for pk_str in to_delete:
            del self._cache[table][pk_str]

        if to_delete:
            await self._save_table(table)

        return len(to_delete)

    async def count(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
    ) -> int:
        table = get_table_name(model_class)
        await self._load_table(table, model_class)

        if filters is None:
            return len(self._cache.get(table, {}))

        count = 0
        for model in self._cache.get(table, {}).values():
            if evaluate(filters, model):
                count += 1
        return count
