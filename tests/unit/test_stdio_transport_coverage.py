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

"""Coverage improvement tests for StdioTransport.

Targets uncovered paths in:
- nexau/archs/transports/stdio/stdio_transport.py
"""

import json
from unittest.mock import Mock

import pytest

from nexau.archs.transports.stdio.stdio_transport import (
    JsonRpcError,
    JsonRpcErrorResponse,
    JsonRpcEventFrame,
    JsonRpcRequest,
    JsonRpcSuccessResponse,
    StdioTransport,
)


class TestJsonRpcModels:
    def test_json_rpc_request_model(self):
        req = JsonRpcRequest(jsonrpc="2.0-stream", method="agent.query", id="1")
        assert req.method == "agent.query"
        assert req.params is None

    def test_json_rpc_request_with_params(self):
        req = JsonRpcRequest(
            jsonrpc="2.0-stream",
            method="agent.stream",
            params={"messages": "hello"},
            id="2",
        )
        assert req.params == {"messages": "hello"}

    def test_json_rpc_error(self):
        err = JsonRpcError(code=-32601, message="Method not found")
        assert err.code == -32601

    def test_json_rpc_success_response(self):
        resp = JsonRpcSuccessResponse(id="1", result={"status": "ok"})
        assert resp.result == {"status": "ok"}
        assert resp.jsonrpc == "2.0-stream"

    def test_json_rpc_error_response(self):
        resp = JsonRpcErrorResponse(
            id="1",
            error=JsonRpcError(code=-32000, message="error"),
        )
        data = json.loads(resp.model_dump_json())
        assert data["error"]["code"] == -32000

    def test_json_rpc_event_frame(self):
        frame = JsonRpcEventFrame(id="1", event={"type": "text", "text": "hello"})
        data = json.loads(frame.model_dump_json())
        assert data["event"]["type"] == "text"


class TestStdioTransportInit:
    def test_init(self):
        from nexau.archs.session.orm import InMemoryDatabaseEngine
        from nexau.archs.transports.stdio.config import StdioConfig

        engine = InMemoryDatabaseEngine()
        config = StdioConfig()
        agent_config = Mock()

        transport = StdioTransport(
            engine=engine,
            config=config,
            default_agent_config=agent_config,
        )
        assert transport._running is False
        assert transport._real_stdout is None

    def test_stop(self):
        from nexau.archs.session.orm import InMemoryDatabaseEngine
        from nexau.archs.transports.stdio.config import StdioConfig

        transport = StdioTransport(
            engine=InMemoryDatabaseEngine(),
            config=StdioConfig(),
            default_agent_config=Mock(),
        )
        transport._running = True
        transport.stop()
        assert transport._running is False


class TestStdioTransportWriteLine:
    def test_write_line_with_real_stdout(self):
        from nexau.archs.session.orm import InMemoryDatabaseEngine
        from nexau.archs.transports.stdio.config import StdioConfig

        transport = StdioTransport(
            engine=InMemoryDatabaseEngine(),
            config=StdioConfig(),
            default_agent_config=Mock(),
        )
        mock_stdout = Mock()
        transport._real_stdout = mock_stdout

        transport._write_line('{"test": true}')
        mock_stdout.write.assert_called_once_with('{"test": true}\n')
        mock_stdout.flush.assert_called_once()

    def test_write_line_without_stdout_fallback(self, capsys):
        from nexau.archs.session.orm import InMemoryDatabaseEngine
        from nexau.archs.transports.stdio.config import StdioConfig

        transport = StdioTransport(
            engine=InMemoryDatabaseEngine(),
            config=StdioConfig(),
            default_agent_config=Mock(),
        )
        transport._real_stdout = None
        transport._write_line('{"test": true}')
        captured = capsys.readouterr()
        assert '{"test": true}' in captured.out


class TestStdioTransportHandleLine:
    @pytest.mark.anyio
    async def test_handle_line_unknown_method(self):
        from nexau.archs.session.orm import InMemoryDatabaseEngine
        from nexau.archs.transports.stdio.config import StdioConfig

        transport = StdioTransport(
            engine=InMemoryDatabaseEngine(),
            config=StdioConfig(),
            default_agent_config=Mock(),
        )
        mock_stdout = Mock()
        transport._real_stdout = mock_stdout

        line = json.dumps(
            {
                "jsonrpc": "2.0-stream",
                "method": "unknown.method",
                "params": {"messages": "hello"},
                "id": "req-1",
            }
        )
        await transport._handle_line(line)
        mock_stdout.write.assert_called_once()
        written = mock_stdout.write.call_args[0][0]
        assert "Method not found" in written

    @pytest.mark.anyio
    async def test_handle_stop_request_missing_session_id(self):
        from nexau.archs.session.orm import InMemoryDatabaseEngine
        from nexau.archs.transports.stdio.config import StdioConfig

        transport = StdioTransport(
            engine=InMemoryDatabaseEngine(),
            config=StdioConfig(),
            default_agent_config=Mock(),
        )
        mock_stdout = Mock()
        transport._real_stdout = mock_stdout

        # Test _handle_stop_request_model directly (agent.stop without session_id)
        await transport._handle_stop_request_model({"user_id": "u1"}, "req-2")
        written = mock_stdout.write.call_args[0][0]
        assert "session_id is required" in written
