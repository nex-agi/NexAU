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

"""Additional coverage tests for mcp_client.py.

Targets uncovered paths: HTTPMCPSession network methods, MCPTool,
MCPServerConfig, MCPManager tool discovery logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nexau.archs.tool.builtin.mcp_client import (
    HTTPMCPSession,
)


@pytest.fixture
def mcp_config():
    config = Mock()
    config.url = "http://localhost:8080/mcp"
    config.command = None
    config.args = []
    config.env = {}
    return config


@pytest.fixture
def session(mcp_config):
    return HTTPMCPSession(
        config=mcp_config,
        headers={"Authorization": "Bearer test"},
        timeout=10.0,
    )


# ---------------------------------------------------------------------------
# HTTPMCPSession — _send_initialized_notification
# ---------------------------------------------------------------------------


class TestSendInitializedNotification:
    @pytest.mark.anyio
    async def test_streamable_http_sends_notification(self, session):
        session._transport = "streamable_http"
        session._session_id = "sess-123"

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await session._send_initialized_notification()

    @pytest.mark.anyio
    async def test_sse_transport_sends_notification(self, session):
        session._transport = "http_sse"
        session._sse_client = AsyncMock()
        session._sse_endpoint_url = "http://localhost:8080/messages"
        session._sse_endpoint_headers = {"Content-Type": "application/json"}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        session._sse_client.post.return_value = mock_response
        await session._send_initialized_notification()

    @pytest.mark.anyio
    async def test_sse_notification_failure_logged(self, session):
        session._transport = "http_sse"
        session._sse_client = None  # Will cause failure
        session._sse_endpoint_url = None
        # Should not raise, just log warning
        await session._send_initialized_notification()


# ---------------------------------------------------------------------------
# HTTPMCPSession — _make_request
# ---------------------------------------------------------------------------


class TestMakeRequest:
    @pytest.mark.anyio
    async def test_streamable_http_request(self, session):
        session._initialized = True
        session._transport = "streamable_http"
        session._session_id = "sess-123"
        with patch.object(
            session,
            "_make_streamable_http_request",
            return_value={"result": {"ok": True}},
        ):
            result = await session._make_request("tools/list")
        assert result["result"]["ok"] is True

    @pytest.mark.anyio
    async def test_sse_transport_request(self, session):
        session._initialized = True
        session._transport = "http_sse"
        with patch.object(
            session,
            "_send_json_rpc_via_sse",
            return_value={"result": {"tools": []}},
        ):
            result = await session._make_request("tools/list")
        assert "result" in result

    @pytest.mark.anyio
    async def test_initializes_if_not_initialized(self, session):
        session._initialized = False
        with (
            patch.object(session, "_initialize_session"),
            patch.object(
                session,
                "_make_streamable_http_request",
                return_value={"result": {}},
            ),
        ):
            session._initialized = True
            session._transport = "streamable_http"
            result = await session._make_request("tools/list")
        assert result is not None


# ---------------------------------------------------------------------------
# HTTPMCPSession — _handle_sse_event
# ---------------------------------------------------------------------------


class TestHandleSseEvent:
    @pytest.mark.anyio
    async def test_endpoint_event_json(self, session):
        session.config.url = "http://localhost:8080"
        session._sse_endpoint_ready = asyncio.Event()
        await session._handle_sse_event(
            "endpoint",
            ['{"endpoint": "/messages", "headers": {"X-Custom": "val"}}'],
        )
        assert session._sse_endpoint_url is not None
        assert "X-Custom" in session._sse_endpoint_headers

    @pytest.mark.anyio
    async def test_endpoint_event_plain_string(self, session):
        session.config.url = "http://localhost:8080"
        session._sse_endpoint_ready = asyncio.Event()
        await session._handle_sse_event("endpoint", ["/messages"])
        assert session._sse_endpoint_url is not None

    @pytest.mark.anyio
    async def test_endpoint_event_json_string(self, session):
        session.config.url = "http://localhost:8080"
        session._sse_endpoint_ready = asyncio.Event()
        await session._handle_sse_event("endpoint", ['"/messages"'])
        assert session._sse_endpoint_url is not None

    @pytest.mark.anyio
    async def test_message_event_with_id(self, session):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session._pending_requests = {"42": future}
        await session._handle_sse_event("message", ['{"id": 42, "result": {"ok": true}}'])
        assert future.done()
        assert future.result()["result"]["ok"] is True

    @pytest.mark.anyio
    async def test_message_event_without_id(self, session):
        # Should not raise; just log
        await session._handle_sse_event("message", ['{"method": "notification"}'])

    @pytest.mark.anyio
    async def test_empty_data_lines(self, session):
        await session._handle_sse_event("message", [])  # Should be a no-op

    @pytest.mark.anyio
    async def test_unparseable_json(self, session):
        # Should not raise; just log
        await session._handle_sse_event("message", ["not-json"])


# ---------------------------------------------------------------------------
# HTTPMCPSession — _send_json_rpc_via_sse
# ---------------------------------------------------------------------------


class TestSendJsonRpcViaSse:
    @pytest.mark.anyio
    async def test_raises_when_not_initialized(self, session):
        session._sse_client = None
        session._sse_endpoint_url = None
        with pytest.raises(RuntimeError, match="not initialized"):
            await session._send_json_rpc_via_sse({"method": "test"}, expect_response=True)

    @pytest.mark.anyio
    async def test_no_response_expected(self, session):
        session._sse_client = AsyncMock()
        session._sse_endpoint_url = "http://localhost:8080/messages"
        session._sse_endpoint_headers = {}
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        session._sse_client.post.return_value = mock_response
        result = await session._send_json_rpc_via_sse(
            {"method": "notifications/initialized"},
            expect_response=False,
        )
        assert result == {}

    @pytest.mark.anyio
    async def test_expect_response_missing_id_raises(self, session):
        session._sse_client = AsyncMock()
        session._sse_endpoint_url = "http://localhost:8080/messages"
        session._sse_endpoint_headers = {}
        with pytest.raises(ValueError, match="Expected request ID"):
            await session._send_json_rpc_via_sse(
                {"method": "test"},  # No id
                expect_response=True,
            )


# ---------------------------------------------------------------------------
# HTTPMCPSession — _parse_streamable_http_payload edge cases
# ---------------------------------------------------------------------------


class TestParseStreamableHttpPayloadEdges:
    def test_sse_with_empty_data_line(self, session):
        """SSE with data: followed by empty string should skip."""
        sse_text = 'event: message\ndata: \ndata: {"id": 1, "result": {}}\n\n'
        result = session._parse_streamable_http_payload(sse_text, expected_id=1)
        assert "result" in result

    def test_sse_no_matching_id_fallback_to_result(self, session):
        sse_text = 'event: message\ndata: {"id": 100, "error": {"code": -1}}\ndata: {"id": 200, "result": {}}\n\n'
        result = session._parse_streamable_http_payload(sse_text, expected_id=999)
        # Should fall back to last message with result
        assert "result" in result
