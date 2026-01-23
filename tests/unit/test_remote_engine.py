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

"""Tests for RemoteDatabaseEngine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    RemoteDatabaseEngine,
)

# ============================================================================
# Test Models
# ============================================================================


class RemoteUserModel(SQLModel, table=True):
    """Test user model for remote engine."""

    __tablename__ = "remote_test_users"  # type: ignore[assignment]

    tenant_id: str = Field(primary_key=True)
    user_id: str = Field(primary_key=True)
    name: str = ""
    email: str = ""


class RemoteSessionModel(SQLModel, table=True):
    """Test session model for remote engine."""

    __tablename__ = "remote_test_sessions"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    context: dict = Field(default_factory=dict, sa_column=Column(JSON))


# ============================================================================
# RemoteDatabaseEngine Tests
# ============================================================================


class TestRemoteDatabaseEngine:
    """Tests for RemoteDatabaseEngine."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")
        assert engine._base_url == "http://localhost:8000"
        assert engine._api_key is None
        assert engine._timeout == 30.0

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        engine = RemoteDatabaseEngine(
            base_url="http://localhost:8000",
            api_key="test_key",
        )
        assert engine._api_key == "test_key"

    def test_init_with_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        engine = RemoteDatabaseEngine(
            base_url="http://localhost:8000",
            timeout=60.0,
        )
        assert engine._timeout == 60.0

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000/")
        assert engine._base_url == "http://localhost:8000"

    def test_get_client_creates_client(self) -> None:
        """Test that _get_client creates httpx.AsyncClient."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            client = engine._get_client()  # _get_client is sync now
            assert isinstance(client, httpx.AsyncClient)
            assert client is engine._client
            await engine.close()

        asyncio.run(run())

    def test_get_client_with_api_key(self) -> None:
        """Test that _get_client includes API key in headers."""
        engine = RemoteDatabaseEngine(
            base_url="http://localhost:8000",
            api_key="test_key",
        )

        async def run() -> None:
            client = engine._get_client()  # _get_client is sync now
            assert "Authorization" in client.headers
            assert client.headers["Authorization"] == "Bearer test_key"
            await engine.close()

        asyncio.run(run())

    def test_get_client_reuses_existing(self) -> None:
        """Test that _get_client reuses existing client."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            client1 = engine._get_client()  # _get_client is sync now
            client2 = engine._get_client()
            assert client1 is client2
            await engine.close()

        asyncio.run(run())

    def test_close(self) -> None:
        """Test close method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            engine._get_client()  # _get_client is sync now
            assert engine._client is not None
            await engine.close()
            assert engine._client is None

        asyncio.run(run())

    def test_context_manager(self) -> None:
        """Test async context manager protocol."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            async with engine as eng:
                assert eng is engine
                assert engine._client is not None
            assert engine._client is None

        asyncio.run(run())

    def test_parse_model_from_dict(self) -> None:
        """Test _parse_model with dict data."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")
        data = {"tenant_id": "t1", "user_id": "u1", "name": "Alice", "email": "alice@example.com"}
        model = engine._parse_model(RemoteUserModel, data)
        assert model.tenant_id == "t1"
        assert model.name == "Alice"

    def test_parse_model_from_json_string(self) -> None:
        """Test _parse_model with JSON string data."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")
        data = '{"tenant_id": "t1", "user_id": "u1", "name": "Bob", "email": "bob@example.com"}'
        model = engine._parse_model(RemoteUserModel, data)
        assert model.tenant_id == "t1"
        assert model.name == "Bob"

    def test_parse_models(self) -> None:
        """Test _parse_models with list of data."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")
        data = [
            {"tenant_id": "t1", "user_id": "u1", "name": "Alice", "email": "alice@example.com"},
            {"tenant_id": "t1", "user_id": "u2", "name": "Bob", "email": "bob@example.com"},
        ]
        models = engine._parse_models(RemoteUserModel, data)
        assert len(models) == 2
        assert models[0].name == "Alice"
        assert models[1].name == "Bob"

    def test_setup_models(self) -> None:
        """Test setup_models method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                await engine.setup_models([RemoteUserModel])

                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert "/remote_test_users/ensure" in call_args[0][0]

        asyncio.run(run())

    def test_find_first_success(self) -> None:
        """Test find_first returns model."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": {"tenant_id": "t1", "user_id": "u1", "name": "Alice", "email": "alice@example.com"}
                }
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = AndFilter(
                    filters=[
                        ComparisonFilter.eq("tenant_id", "t1"),
                        ComparisonFilter.eq("user_id", "u1"),
                    ]
                )
                result = await engine.find_first(RemoteUserModel, filters=filters)

                assert result is not None
                assert result.name == "Alice"

        asyncio.run(run())

    def test_find_first_not_found(self) -> None:
        """Test find_first returns None when not found."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = ComparisonFilter.eq("tenant_id", "t1")
                result = await engine.find_first(RemoteUserModel, filters=filters)

                assert result is None

        asyncio.run(run())

    def test_find_first_null_data(self) -> None:
        """Test find_first returns None when data is null."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": None}
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = ComparisonFilter.eq("tenant_id", "t1")
                result = await engine.find_first(RemoteUserModel, filters=filters)

                assert result is None

        asyncio.run(run())

    def test_find_many(self) -> None:
        """Test find_many returns list of models."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "results": [
                        {"tenant_id": "t1", "user_id": "u1", "name": "Alice", "email": "alice@example.com"},
                        {"tenant_id": "t1", "user_id": "u2", "name": "Bob", "email": "bob@example.com"},
                    ]
                }
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = ComparisonFilter.eq("tenant_id", "t1")
                results = await engine.find_many(RemoteUserModel, filters=filters)

                assert len(results) == 2
                assert results[0].name == "Alice"
                assert results[1].name == "Bob"

        asyncio.run(run())

    def test_find_many_with_options(self) -> None:
        """Test find_many with limit, offset, and order_by."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"results": []}
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = ComparisonFilter.eq("tenant_id", "t1")
                await engine.find_many(
                    RemoteUserModel,
                    filters=filters,
                    limit=10,
                    offset=5,
                    order_by="name",
                )

                call_args = mock_client.post.call_args
                json_data = call_args[1]["json"]
                assert json_data["limit"] == 10
                assert json_data["offset"] == 5
                assert json_data["order_by"] == "name"

        asyncio.run(run())

    def test_find_many_with_tuple_order_by(self) -> None:
        """Test find_many with tuple order_by."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"results": []}
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                await engine.find_many(
                    RemoteUserModel,
                    order_by=("name", "email"),
                )

                call_args = mock_client.post.call_args
                json_data = call_args[1]["json"]
                assert json_data["order_by"] == ["name", "email"]

        asyncio.run(run())

    def test_create(self) -> None:
        """Test create method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "data": {"tenant_id": "t1", "user_id": "u1", "name": "Alice", "email": "alice@example.com"}
                }
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                user = RemoteUserModel(tenant_id="t1", user_id="u1", name="Alice")
                result = await engine.create(user)

                assert result.name == "Alice"
                mock_client.post.assert_called_once()

        asyncio.run(run())

    def test_create_many(self) -> None:
        """Test create_many method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "results": [
                        {"tenant_id": "t1", "user_id": "u1", "name": "Alice", "email": "alice@example.com"},
                        {"tenant_id": "t1", "user_id": "u2", "name": "Bob", "email": "bob@example.com"},
                    ]
                }
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                users = [
                    RemoteUserModel(tenant_id="t1", user_id="u1", name="Alice"),
                    RemoteUserModel(tenant_id="t1", user_id="u2", name="Bob"),
                ]
                results = await engine.create_many(users)

                assert len(results) == 2
                assert results[0].name == "Alice"

        asyncio.run(run())

    def test_create_many_empty_list(self) -> None:
        """Test create_many with empty list."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            results: list[RemoteUserModel] = await engine.create_many([])
            assert len(results) == 0

        asyncio.run(run())

    def test_update(self) -> None:
        """Test update method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "data": {"tenant_id": "t1", "user_id": "u1", "name": "Updated", "email": "alice@example.com"}
                }
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                user = RemoteUserModel(tenant_id="t1", user_id="u1", name="Updated")
                result = await engine.update(user)

                assert result.name == "Updated"
                mock_client.post.assert_called_once()

        asyncio.run(run())

    def test_delete(self) -> None:
        """Test delete method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"deleted_count": 2}
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = ComparisonFilter.eq("tenant_id", "t1")
                count = await engine.delete(RemoteUserModel, filters=filters)

                assert count == 2

        asyncio.run(run())

    def test_count(self) -> None:
        """Test count method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"count": 5}
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                filters = ComparisonFilter.eq("tenant_id", "t1")
                count = await engine.count(RemoteUserModel, filters=filters)

                assert count == 5

        asyncio.run(run())

    def test_count_no_filters(self) -> None:
        """Test count without filters."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")

        async def run() -> None:
            with patch.object(engine, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"count": 10}
                mock_response.raise_for_status = MagicMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                count = await engine.count(RemoteUserModel)

                assert count == 10
                call_args = mock_client.post.call_args
                json_data = call_args[1]["json"]
                assert json_data["filters"] is None

        asyncio.run(run())

    def test_repr(self) -> None:
        """Test __repr__ method."""
        engine = RemoteDatabaseEngine(base_url="http://localhost:8000")
        repr_str = repr(engine)
        assert "RemoteDatabaseEngine" in repr_str
        assert "http://localhost:8000" in repr_str
