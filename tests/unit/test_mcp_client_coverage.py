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

"""Coverage improvement tests for nexau/archs/tool/builtin/mcp_client.py.

Targets uncovered paths: HTTPMCPSession helper methods, parsing helpers,
and validation logic.
"""

import json
from unittest.mock import Mock

import pytest

from nexau.archs.tool.builtin.mcp_client import (
    HTTPMCPSession,
)


@pytest.fixture
def mcp_config():
    """Minimal MCPServerConfig-like object."""
    config = Mock()
    config.url = "http://localhost:8080/mcp"
    config.command = None
    config.args = []
    config.env = {}
    return config


@pytest.fixture
def session(mcp_config):
    """HTTPMCPSession with test configuration."""
    return HTTPMCPSession(
        config=mcp_config,
        headers={"Authorization": "Bearer test"},
        timeout=10.0,
    )


class TestHTTPMCPSessionInit:
    def test_initial_state(self, session):
        assert session._request_id == 0
        assert session._session_id is None
        assert session._initialized is False
        assert session._transport is None


class TestGetNextId:
    def test_increments(self, session):
        assert session._get_next_id() == 1
        assert session._get_next_id() == 2
        assert session._get_next_id() == 3


class TestBuildInitializeRequest:
    def test_structure(self, session):
        req = session._build_initialize_request()
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "initialize"
        assert "params" in req
        assert req["params"]["clientInfo"]["name"] == "nexau-mcp-client"
        assert req["id"] == 1


class TestValidateInitializePayload:
    def test_none_payload_raises(self, session):
        with pytest.raises(Exception, match="No valid response"):
            session._validate_initialize_payload(None, raw_text="raw")

    def test_error_in_payload_raises(self, session):
        with pytest.raises(Exception, match="initialization error"):
            session._validate_initialize_payload(
                {"error": {"code": -1, "message": "fail"}},
            )

    def test_missing_result_raises(self, session):
        with pytest.raises(Exception, match="missing 'result'"):
            session._validate_initialize_payload(
                {"jsonrpc": "2.0", "id": 1},
            )

    def test_valid_payload_passes(self, session):
        # Should not raise
        session._validate_initialize_payload(
            {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "test"}}},
        )


class TestParseStreamableHttpPayload:
    def test_empty_response_raises(self, session):
        with pytest.raises(Exception, match="Empty response"):
            session._parse_streamable_http_payload("")

    def test_plain_json(self, session):
        payload = json.dumps({"result": {"ok": True}, "id": 1})
        result = session._parse_streamable_http_payload(payload)
        assert result["result"]["ok"] is True

    def test_sse_format_with_matching_id(self, session):
        sse_text = 'event: message\ndata: {"id": 42, "result": {"ok": true}}\n\n'
        result = session._parse_streamable_http_payload(sse_text, expected_id=42)
        assert result["result"]["ok"] is True

    def test_sse_format_without_matching_id_falls_back(self, session):
        sse_text = 'event: message\ndata: {"id": 99, "result": {"ok": true}}\n\n'
        result = session._parse_streamable_http_payload(sse_text, expected_id=1)
        # Falls back to last message with result
        assert result["result"]["ok"] is True

    def test_sse_format_no_expected_id(self, session):
        sse_text = 'event: message\ndata: {"result": {"ok": true}}\n\n'
        result = session._parse_streamable_http_payload(sse_text)
        assert result["result"]["ok"] is True

    def test_sse_format_no_parseable_data_raises(self, session):
        sse_text = "event: message\ndata: not-json\n\n"
        with pytest.raises(Exception, match="Could not parse SSE"):
            session._parse_streamable_http_payload(sse_text)


class TestFailPendingRequests:
    def test_fails_all_pending(self, session):
        import asyncio

        loop = asyncio.new_event_loop()
        future1 = loop.create_future()
        future2 = loop.create_future()
        session._pending_requests = {"1": future1, "2": future2}

        error = RuntimeError("stream closed")
        session._fail_pending_requests(error)

        assert future1.exception() is error
        assert future2.exception() is error
        assert session._pending_requests == {}
        loop.close()

    def test_no_op_when_empty(self, session):
        session._pending_requests = {}
        session._fail_pending_requests(RuntimeError("x"))  # Should not raise
