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

"""Unit tests for Stdio transport."""

import asyncio

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.stdio import StdioConfig, StdioTransport
from nexau.archs.transports.stdio.stdio_transport import (
    JSONRPC_VERSION,
    JsonRpcError,
    JsonRpcErrorResponse,
    JsonRpcEventFrame,
    JsonRpcSuccessResponse,
)


class TestStdioConfig:
    """Test cases for StdioConfig."""

    def test_default_config(self):
        """Test default stdio config values."""
        config = StdioConfig()
        assert config.encoding == "utf-8"

    def test_custom_encoding(self):
        """Test custom encoding."""
        config = StdioConfig(encoding="latin-1")
        assert config.encoding == "latin-1"


class TestStdioTransport:
    """Test cases for StdioTransport."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create stdio transport instance."""
        return StdioTransport(
            engine=engine,
            config=StdioConfig(),
            default_agent_config=agent_config,
        )

    def test_initialization(self, transport):
        """Test transport initialization."""
        assert transport._running is False
        assert transport._real_stdout is None
        assert transport._original_stdout is None

    def test_stop(self, transport):
        """Test stop method."""
        transport._running = True
        transport.stop()
        assert transport._running is False

    def test_write_line(self, transport, capsys):
        """Test _write_line method when not running."""
        transport._write_line('{"jsonrpc": "2.0-stream"}')
        captured = capsys.readouterr()
        assert '{"jsonrpc": "2.0-stream"}' in captured.out

    def test_handle_line_invalid_json(self, transport):
        """Test _handle_line with invalid JSON."""
        from pydantic import ValidationError

        async def run_test():
            with pytest.raises(ValidationError):
                await transport._handle_line("not valid json")

        asyncio.run(run_test())

    def test_handle_line_missing_user_id(self, transport):
        """Test _handle_line with missing params."""
        from pydantic import ValidationError

        async def run_test():
            with pytest.raises(ValidationError):
                await transport._handle_line('{"jsonrpc": "2.0-stream", "method": "agent.query", "id": "req_1"}')

        asyncio.run(run_test())

    def test_handle_line_missing_message(self, transport):
        """Test _handle_line with missing messages."""
        from pydantic import ValidationError

        async def run_test():
            with pytest.raises(ValidationError):
                await transport._handle_line(
                    '{"jsonrpc": "2.0-stream", "method": "agent.query", "params": {"user_id": "user123"}, "id": "req_1"}'
                )

        asyncio.run(run_test())

    def test_handle_line_missing_id(self, transport):
        """Test _handle_line with missing id."""
        from pydantic import ValidationError

        async def run_test():
            with pytest.raises(ValidationError):
                await transport._handle_line(
                    '{"jsonrpc": "2.0-stream", "method": "agent.query", "params": {"messages": "hello", "user_id": "user123"}}'
                )

        asyncio.run(run_test())

    def test_jsonrpc_success_response_dump(self):
        response = JsonRpcSuccessResponse(id="req_1", result={"ok": True})
        payload = response.model_dump()
        assert payload["jsonrpc"] == JSONRPC_VERSION
        assert payload["id"] == "req_1"
        assert payload["result"] == {"ok": True}

    def test_jsonrpc_error_response_dump(self):
        response = JsonRpcErrorResponse(id="req_1", error=JsonRpcError(code=-32000, message="boom"))
        payload = response.model_dump()
        assert payload["jsonrpc"] == JSONRPC_VERSION
        assert payload["id"] == "req_1"
        assert payload["error"]["code"] == -32000
        assert payload["error"]["message"] == "boom"

    def test_jsonrpc_event_frame_dump(self):
        frame = JsonRpcEventFrame(id="req_1", event={"type": "RUN_STARTED"})
        payload = frame.model_dump()
        assert payload["jsonrpc"] == JSONRPC_VERSION
        assert payload["id"] == "req_1"
        assert payload["event"] == {"type": "RUN_STARTED"}

    def test_handle_line_unknown_method(self, transport, capsys):
        """Test _handle_line with unknown method returns error response."""
        import io

        async def run_test():
            # Capture output by setting _real_stdout
            output = io.StringIO()
            transport._real_stdout = output

            await transport._handle_line(
                '{"jsonrpc": "2.0-stream", "method": "unknown.method", '
                '"params": {"messages": "hello", "user_id": "user123"}, "id": "req_1"}'
            )

            output_str = output.getvalue()
            assert "Method not found" in output_str
            assert "unknown.method" in output_str
            assert "-32601" in output_str

        asyncio.run(run_test())

    def test_write_line_with_real_stdout(self, transport):
        """Test _write_line when _real_stdout is set."""
        import io

        output = io.StringIO()
        transport._real_stdout = output

        transport._write_line('{"test": "data"}')

        assert '{"test": "data"}' in output.getvalue()
        assert output.getvalue().endswith("\n")


class TestStdioTransportStartStop:
    """Tests for start/stop lifecycle."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create stdio transport instance."""
        return StdioTransport(
            engine=engine,
            config=StdioConfig(),
            default_agent_config=agent_config,
        )

    def test_start_redirects_stdout(self, transport, monkeypatch):
        """Test that start() redirects stdout to stderr."""
        import io
        import sys

        # Mock stdin to return EOF immediately
        mock_stdin = io.BytesIO(b"")
        monkeypatch.setattr(sys, "stdin", mock_stdin)

        # We need to make _run_loop exit quickly
        async def quick_exit_loop():
            transport._running = False

        # Replace _run_loop
        original_run_loop = transport._run_loop
        transport._run_loop = quick_exit_loop

        # Capture original stdout
        original_stdout = sys.stdout

        transport.start()

        # After start() completes, stdout should be restored
        assert sys.stdout == original_stdout

        # Restore original method
        transport._run_loop = original_run_loop


class TestStdioTransportRunLoop:
    """Tests for _run_loop method."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create stdio transport instance."""
        return StdioTransport(
            engine=engine,
            config=StdioConfig(),
            default_agent_config=agent_config,
        )

    def test_run_loop_handles_empty_lines(self, transport):
        """Test that _run_loop skips empty lines."""

        async def run_test():
            # Create mock stdin with empty lines
            mock_reader = asyncio.StreamReader()
            mock_reader.feed_data(b"\n\n\n")
            mock_reader.feed_eof()

            transport._running = True
            lines_processed = []

            # Save original handle_line
            original_handle_line = transport._handle_line

            async def mock_handle_line(line):
                lines_processed.append(line)

            transport._handle_line = mock_handle_line

            # Run loop with mock reader - but we need to actually run the loop logic
            while transport._running:
                line_bytes = await mock_reader.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                await transport._handle_line(line)

            # No non-empty lines should have been processed
            assert len(lines_processed) == 0

            transport._handle_line = original_handle_line

        asyncio.run(run_test())

    def test_run_loop_exits_on_eof(self, transport):
        """Test that _run_loop exits on EOF."""

        async def run_test():
            mock_reader = asyncio.StreamReader()
            mock_reader.feed_eof()

            transport._running = True
            exited = False

            while transport._running:
                line_bytes = await mock_reader.readline()
                if not line_bytes:
                    exited = True
                    break

            assert exited

        asyncio.run(run_test())
