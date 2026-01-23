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

"""Remote database engine that communicates with a Session Manager via HTTP.

This module provides a `RemoteDatabaseEngine` that implements the `DatabaseEngine`
interface by making HTTP requests to a remote Session Manager service.

Token Authentication:
    Pass `api_key` at construction time. For request-level tokens, create a new
    engine instance per request with the request's token.

Example:
    # Static token (for development)
    engine = RemoteDatabaseEngine(
        base_url="http://session-manager:8005/v1/sessions",
        api_key="dev-token",
    )

    # Per-request token (for production)
    async def handle_request(request_token: str):
        engine = RemoteDatabaseEngine(
            base_url="http://session-manager:8005/v1/sessions",
            api_key=request_token,
        )
        # Use engine for this request only
        ...
        await engine.close()
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from sqlmodel import SQLModel

from .engine import DatabaseEngine, get_table_name
from .filters import Filter

T = TypeVar("T", bound=SQLModel)


class RemoteDatabaseEngine(DatabaseEngine):
    """Database engine that communicates with Session Manager via HTTP.

    Uses httpx.AsyncClient for async HTTP operations.
    For request-level token authentication, create a new engine instance
    per request with the request's token as api_key.

    Attributes:
        _base_url: Base URL of the Session Manager API
        _api_key: API key for authentication (optional)
        _timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the remote database engine.

        Args:
            base_url: Base URL of the Session Manager API
            api_key: API key/token for authentication (optional)
            timeout: HTTP request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> RemoteDatabaseEngine:
        self._get_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _parse_model(self, model_class: type[T], data: Any) -> T:
        """Parse response data into a model instance."""
        if isinstance(data, str):
            return model_class.model_validate_json(data)
        return model_class.model_validate(data)

    def _parse_models(self, model_class: type[T], results: list[Any]) -> list[T]:
        """Parse a list of response data into model instances."""
        return [self._parse_model(model_class, r) for r in results]

    async def setup_models(self, model_classes: list[type[SQLModel]]) -> None:
        """Ensure tables exist for all model classes."""
        client = self._get_client()

        async def ensure_table(model_class: type[SQLModel]) -> None:
            table = get_table_name(model_class)
            pk_fields = [name for name, info in model_class.model_fields.items() if getattr(info, "primary_key", False) is True]
            index_fields = [name for name, info in model_class.model_fields.items() if getattr(info, "index", False) is True]
            field_types = {name: str(info.annotation) for name, info in model_class.model_fields.items() if info.annotation is not None}
            response = await client.post(
                f"/{table}/ensure",
                json={
                    "table": table,
                    "primary_key": pk_fields,
                    "indexes": index_fields,
                    "field_types": field_types,
                },
            )
            response.raise_for_status()

        import asyncio

        await asyncio.gather(*[ensure_table(mc) for mc in model_classes])

    async def find_first(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> T | None:
        """Find the first record matching the filters."""
        client = self._get_client()
        table = get_table_name(model_class)
        response = await client.post(
            f"/{table}/find_first",
            json={"filters": filters.model_dump(mode="json")},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json().get("data")
        if data is None:
            return None
        return self._parse_model(model_class, data)

    async def find_many(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | tuple[str, ...] | None = None,
    ) -> list[T]:
        """Find all records matching the filters."""
        client = self._get_client()
        table = get_table_name(model_class)
        order_by_data = None
        if order_by is not None:
            order_by_data = order_by if isinstance(order_by, str) else list(order_by)
        response = await client.post(
            f"/{table}/find_many",
            json={
                "filters": filters.model_dump(mode="json") if filters else None,
                "limit": limit,
                "offset": offset,
                "order_by": order_by_data,
            },
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return self._parse_models(model_class, results)

    async def create(self, model: T) -> T:
        """Create a new record."""
        client = self._get_client()
        table = get_table_name(type(model))
        response = await client.post(
            f"/{table}/create",
            json={"data": model.model_dump_json(exclude_none=True)},
        )
        response.raise_for_status()
        data = response.json().get("data")
        return self._parse_model(type(model), data)

    async def create_many(self, models: list[T]) -> list[T]:
        """Create multiple records."""
        if not models:
            return []

        client = self._get_client()
        table = get_table_name(type(models[0]))
        response = await client.post(
            f"/{table}/create_many",
            json={"models": [m.model_dump_json(exclude_none=True) for m in models]},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return self._parse_models(type(models[0]), results)

    async def update(self, model: T) -> T:
        """Update an existing record."""
        client = self._get_client()
        table = get_table_name(type(model))
        response = await client.post(
            f"/{table}/update",
            json={"data": model.model_dump_json(exclude_none=True)},
        )
        response.raise_for_status()
        data = response.json().get("data")
        return self._parse_model(type(model), data)

    async def delete(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> int:
        """Delete records matching the filters."""
        client = self._get_client()
        table = get_table_name(model_class)
        response = await client.post(
            f"/{table}/delete",
            json={"filters": filters.model_dump(mode="json")},
        )
        response.raise_for_status()
        return response.json().get("deleted_count", 0)

    async def count(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
    ) -> int:
        """Count records matching the filters."""
        client = self._get_client()
        table = get_table_name(model_class)
        response = await client.post(
            f"/{table}/count",
            json={"filters": filters.model_dump(mode="json") if filters else None},
        )
        response.raise_for_status()
        return response.json().get("count", 0)

    def __repr__(self) -> str:
        return f"RemoteDatabaseEngine(base_url={self._base_url!r})"
