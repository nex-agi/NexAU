"""Stdio transport implementation for NexAU."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TextIO

from pydantic import BaseModel

from nexau.archs.llm.llm_aggregators.events import TransportErrorEvent
from nexau.archs.session.orm import DatabaseEngine
from nexau.archs.transports.base import TransportBase
from nexau.archs.transports.http.models import AgentRequest
from nexau.archs.transports.stdio.config import StdioConfig

if TYPE_CHECKING:
    from nexau.archs.main_sub.config import AgentConfig

logger = logging.getLogger(__name__)

JSONRPC_VERSION: Literal["2.0-stream"] = "2.0-stream"


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0-stream"]
    method: str
    params: dict[str, Any] | None = None
    id: str


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponseBase(BaseModel):
    jsonrpc: Literal["2.0-stream"] = JSONRPC_VERSION
    id: str


class JsonRpcSuccessResponse(JsonRpcResponseBase):
    result: Any | None


class JsonRpcErrorResponse(JsonRpcResponseBase):
    error: JsonRpcError


class JsonRpcEventFrame(JsonRpcResponseBase):
    event: dict[str, Any]


class StdioTransport(TransportBase[StdioConfig]):
    """Stdio transport for CLI and IPC communication.

    This transport reads JSON Lines from stdin and writes responses to stdout.
    Supports both streaming and synchronous modes.

    IMPORTANT: To prevent third-party libraries from polluting stdout with debug
    prints, this transport redirects sys.stdout to stderr during execution. Only
    the transport's controlled output goes to the real stdout.

    Example:
        >>> from nexau import AgentConfig
        >>> from nexau.archs.transports.stdio import StdioTransport, StdioConfig
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>>
        >>> config = AgentConfig.from_yaml("agent.yaml")
        >>> engine = InMemoryDatabaseEngine()
        >>> transport = StdioTransport(
        ...     engine=engine,
        ...     config=StdioConfig(),
        ...     default_agent_config=config,
        ... )
        >>> transport.start()  # Blocks and reads from stdin
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        config: StdioConfig = StdioConfig(),
        default_agent_config: AgentConfig,
    ):
        """Initialize stdio transport.

        Args:
            engine: DatabaseEngine for all model storage
            config: Stdio-specific configuration
            default_agent_config: Default agent configuration
        """
        super().__init__(
            engine=engine,
            config=config,
            default_agent_config=default_agent_config,
        )
        self._running = False
        self._real_stdout: TextIO | None = None
        self._original_stdout: TextIO | None = None

    def start(self) -> None:
        """Start the stdio transport (blocking).

        Reads from stdin in an event loop until EOF or stop() is called.

        This method redirects sys.stdout to stderr to prevent third-party
        library prints from polluting the JSON Lines output stream.
        """
        # Save original stdout for restoration
        self._original_stdout = sys.stdout
        self._real_stdout = sys.stdout

        self._running = True
        try:
            # Redirect sys.stdout to stderr inside try block to ensure restoration
            sys.stdout = sys.stderr
            asyncio.run(self._run_loop())
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down")
        finally:
            self._running = False
            # Always restore original stdout
            if self._original_stdout is not None:
                sys.stdout = self._original_stdout
                self._original_stdout = None

    def stop(self) -> None:
        """Stop the stdio transport."""
        self._running = False

    async def _run_loop(self) -> None:
        """Main event loop - read and process stdin lines."""
        # Create async stdin reader
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while self._running:
            try:
                # Read one line
                line_bytes = await reader.readline()
                if not line_bytes:  # EOF
                    break

                line = line_bytes.decode(self._config.encoding).strip()
                if not line:
                    continue

                await self._handle_line(line)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Don't send structured error for unexpected errors (e.g. input validation)
                # Just log it and continue loop

    async def _handle_line(self, line: str) -> None:
        """Parse and handle a single input line.

        Args:
            line: JSON string from stdin
        """
        rpc_request = JsonRpcRequest.model_validate_json(line)
        params = rpc_request.params or {}

        agent_request = AgentRequest.model_validate({**params})

        if rpc_request.method == "agent.stream":
            await self._handle_streaming_request_model(agent_request, rpc_request.id)
            return

        if rpc_request.method == "agent.query":
            await self._handle_sync_request_model(agent_request, rpc_request.id)
            return

        self._write_line(
            JsonRpcErrorResponse(
                id=rpc_request.id,
                error=JsonRpcError(code=-32601, message=f"Method not found: {rpc_request.method}"),
            ).model_dump_json()
        )

    async def _handle_streaming_request_model(self, request: AgentRequest, rpc_id: str) -> None:
        """Handle streaming request using AgentRequest model.

        Args:
            request: Validated AgentRequest
        """
        try:
            async for event in self.handle_streaming_request(
                message=request.messages,
                user_id=request.user_id,
                agent_config=None,  # Use default
                session_id=request.session_id,
                context=request.context,
            ):
                self._write_line(JsonRpcEventFrame(id=rpc_id, event=event.model_dump()).model_dump_json())

            self._write_line(JsonRpcSuccessResponse(id=rpc_id, result=None).model_dump_json())

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_event = TransportErrorEvent(
                message=str(e),
                timestamp=int(datetime.now().timestamp()),
            )
            self._write_line(
                JsonRpcErrorResponse(
                    id=rpc_id,
                    error=JsonRpcError(
                        code=-32000,
                        message=str(e),
                        data=error_event.model_dump(),
                    ),
                ).model_dump_json()
            )

    async def _handle_sync_request_model(self, request: AgentRequest, rpc_id: str) -> None:
        """Handle synchronous request using AgentRequest model.

        Args:
            request: Validated AgentRequest
        """
        try:
            result = await self.handle_request(
                message=request.messages,
                user_id=request.user_id,
                agent_config=None,  # Use default
                session_id=request.session_id,
                context=request.context,
            )

            self._write_line(JsonRpcSuccessResponse(id=rpc_id, result=result).model_dump_json())

        except Exception as e:
            logger.error(f"Request error: {e}")
            self._write_line(
                JsonRpcErrorResponse(
                    id=rpc_id,
                    error=JsonRpcError(code=-32000, message=str(e)),
                ).model_dump_json()
            )

    def _write_line(self, data: str) -> None:
        """Write a line to the real stdout (bypassing sys.stdout redirection).

        Args:
            data: String to write (without newline)
        """
        if self._real_stdout:
            self._real_stdout.write(data + "\n")
            self._real_stdout.flush()
        else:
            # Fallback if not running (shouldn't happen in normal operation)
            print(data, flush=True)
