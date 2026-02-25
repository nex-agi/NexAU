"""FastAPI Server-Sent Events (SSE) Transport for Nexau.

This module implements a FastAPI-based SSE server that wraps a Nexau agent using
the ORM Repository pattern with independent SessionModel and AgentModel storage.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from nexau import AgentConfig
from nexau.archs.llm.llm_aggregators.events import TransportErrorEvent
from nexau.archs.main_sub.context_value import ContextValue
from nexau.archs.session.orm import DatabaseEngine
from nexau.archs.transports.base import TransportBase
from nexau.archs.transports.http.config import HTTPConfig
from nexau.archs.transports.http.models import AgentRequest, AgentResponse, StopRequest, StopResponse
from nexau.core.messages import Message

if TYPE_CHECKING:
    from nexau.archs.transports.http.team_registry import TeamRegistry

logger = logging.getLogger(__name__)


class SSETransportServer(TransportBase[HTTPConfig]):
    """SSE transport server using FastAPI with ORM Repository pattern.

    This server exposes a Nexau agent via HTTP with SSE streaming support,
    using the ORM Repository pattern for independent SessionModel and AgentModel storage.

    Example:
        >>> from nexau import Agent
        >>> from nexau.archs.transports.http import SSETransportServer, HTTPConfig
        >>> from nexau.archs.session.orm import SQLDatabaseEngine
        >>> import uvicorn
        >>>
        >>> agent_config = AgentConfig.from_yaml("agent.yaml")
        >>>
        >>> # Create engine for all model storage
        >>> engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
        >>>
        >>> server = SSETransportServer(
        ...     engine=engine,
        ...     config=HTTPConfig(port=8000),
        ...     default_agent_config=agent_config,
        ... )
        >>> # Start the server (blocks)
        >>> uvicorn.run(server.app, host=server.host, port=server.port)
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        config: HTTPConfig = HTTPConfig(),
        default_agent_config: AgentConfig,
        on_stream_event: Callable[[str, str, dict[str, object]], None] | None = None,
        get_history: Callable[[str, str, int], list[dict[str, object]]] | None = None,
        count_events: Callable[[str, str], int] | None = None,
    ):
        """Initialize the SSE transport server.

        Args:
            engine: DatabaseEngine for all model storage
            config: HTTP-specific configuration (default: HTTPConfig())
            default_agent_config: Default agent configuration
            on_stream_event: Optional callback(user_id, session_id, envelope_dict) for team event persistence.
            get_history: Optional callback(user_id, session_id, after) returning stored envelopes for reconnection.
            count_events: Optional callback(user_id, session_id) returning total event count.
        """
        super().__init__(
            engine=engine,
            config=config,
            default_agent_config=default_agent_config,
        )

        # RFC-0002: Team registry for managing AgentTeam instances
        self._team_registry: TeamRegistry | None = None
        self._on_stream_event = on_stream_event
        self._get_history = get_history
        self._count_events = count_events

        # Create FastAPI app (_create_app sets self._team_registry)
        self.app = self._create_app()

        # Runtime context
        self._runtime_context = {
            "working_directory": os.getcwd(),
            "username": os.getenv("USER", "user"),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application.

        Returns:
            Configured FastAPI app
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan context manager for FastAPI."""
            self._is_running = True
            yield
            self._is_running = False

        app = FastAPI(
            title="Nexau SSE Server",
            description="FastAPI server for streaming Nexau agent responses via SSE",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._config.cors_origins,
            allow_credentials=self._config.cors_credentials,
            allow_methods=self._config.cors_methods,
            allow_headers=self._config.cors_headers,
        )

        # Add routes
        self._add_routes(app)

        # RFC-0002: Mount team endpoints with registry
        from nexau.archs.transports.http.team_registry import TeamRegistry
        from nexau.archs.transports.http.team_routes import create_team_router

        self._team_registry = TeamRegistry(
            engine=self._engine,
            session_manager=self._session_manager,
        )
        team_router = create_team_router(
            self._team_registry,
            on_stream_event=self._on_stream_event,
            get_history=self._get_history,
            count_events=self._count_events,
        )
        app.include_router(team_router)

        return app

    def _add_routes(self, app: FastAPI) -> None:
        """Add routes to the FastAPI app.

        Args:
            app: The FastAPI application
        """

        @app.get("/")
        async def root():  # pyright: ignore[reportUnusedFunction]
            """Root endpoint with service info."""
            return {
                "service": "Nexau SSE Server",
                "version": self.app.version,
                "status": "running" if self.is_running else "uninitialized",
                "endpoints": {
                    "health": self.health_url,
                    "info": self.info_url,
                    "stream": "/stream",
                    "query": "/query",
                    "stop": "/stop",
                },
            }

        @app.get("/health")
        async def health():  # pyright: ignore[reportUnusedFunction]
            """Health check endpoint."""
            return {
                "status": "healthy" if self.is_running else "unhealthy",
            }

        @app.post("/stream")
        async def stream_query(request: AgentRequest):  # pyright: ignore[reportUnusedFunction]
            """SSE streaming endpoint for agent queries."""
            return StreamingResponse(
                self._stream_agent_response(
                    message=request.messages,
                    user_id=request.user_id,
                    agent_config=self._default_agent_config,
                    session_id=request.session_id,
                    context=request.context,
                    variables=request.variables,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.post("/query")
        async def query(request: AgentRequest):  # pyright: ignore[reportUnusedFunction]
            """Synchronous query endpoint (non-streaming)."""
            try:
                response = await self.handle_request(
                    message=request.messages,
                    user_id=request.user_id,
                    agent_config=self._default_agent_config,
                    session_id=request.session_id,
                    context=request.context,
                    variables=request.variables,
                )
                return AgentResponse(status="success", response=response)
            except Exception as e:
                logger.error("POST /query failed (session_id: %s): %s", request.session_id, e)
                raise HTTPException(status_code=500, detail=str(e))

        # RFC-0001 Phase 4: stop 端点
        @app.post("/stop")
        async def stop_agent(request: StopRequest):  # pyright: ignore[reportUnusedFunction]
            """Stop a running agent and persist its state."""
            try:
                result = await self.handle_stop_request(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    agent_id=request.agent_id,
                    force=request.force,
                    timeout=request.timeout,
                )
                return StopResponse(
                    status="success",
                    stop_reason=result.stop_reason.name,
                    message_count=len(result.messages),
                )
            except ValueError as e:
                return StopResponse(
                    status="error",
                    error=str(e),
                )
            except Exception as e:
                logger.error("POST /stop failed (session_id: %s): %s", request.session_id, e)
                raise HTTPException(status_code=500, detail=str(e))

    async def _stream_agent_response(
        self,
        *,
        message: str | list[Message],
        user_id: str,
        agent_config: AgentConfig,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        variables: ContextValue | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream agent response as SSE events."""
        try:
            async for event in self.handle_streaming_request(
                message=message,
                user_id=user_id,
                agent_config=agent_config,
                session_id=session_id,
                context=context,
                variables=variables,
            ):
                yield f"data: {event.model_dump_json()}\n\n"
        except Exception as e:
            logger.error("Streaming error (session_id: %s): %s", session_id, e)
            # Yield error event for client-side handling
            error_event = TransportErrorEvent(
                message=str(e),
                timestamp=int(datetime.now().timestamp()),
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

    def start(self) -> None:
        """Start the SSE server.

        Note: This method returns immediately. To block, use uvicorn.run().
        """
        self._is_running = True

    def stop(self) -> None:
        """Stop the SSE server."""
        self._is_running = False

    @property
    def host(self) -> str:
        """Get the host address.

        Returns:
            The host address
        """
        return self._config.host

    @property
    def port(self) -> int:
        """Get the port number.

        Returns:
            The port number
        """
        return self._config.port

    @property
    def health_url(self) -> str:
        """Get the health check URL.

        Returns:
            The URL for health checks
        """
        return f"http://{self.host}:{self.port}/health"

    @property
    def info_url(self) -> str:
        """Get the info URL.

        Returns:
            The URL for server information
        """
        return f"http://{self.host}:{self.port}/"

    @property
    def is_running(self) -> bool:
        """Check if the server is running.

        Returns:
            True if the server is running, False otherwise
        """
        return getattr(self, "_is_running", False)

    @property
    def team_registry(self) -> TeamRegistry | None:
        """Get the team registry for config registration.

        RFC-0002: 获取 team 注册表

        Returns:
            TeamRegistry instance, or None if not initialized.
        """
        return self._team_registry

    def run(self) -> None:
        """Run the server (blocking).

        This is a convenience method that uses uvicorn to run the server.
        """
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self._config.log_level,
            loop="asyncio",
        )
