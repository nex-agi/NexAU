# Transport System Implementation Guide

This module contains transport implementations for exposing NexAU agents via various protocols.

## Architecture Overview

Transport system provides **multi-protocol communication** with stateful session management:

```
TransportBase (Abstract Base Class)
    ├── HTTP + SSE Transport
    ├── Stdio Transport
    ├── WebSocket Transport (planned)
    └── gRPC Transport (planned)
```

Each transport uses:
- **DatabaseEngine**: For session persistence (shared across all transports)
- **SessionManager**: For session and agent management
- **AgentEventsMiddleware**: For streaming event support

## Key Components

### TransportBase (`base.py`)

Abstract base class for all transport implementations.

**Architecture**:

```python
class TransportBase[TTransportConfig](ABC):
    """Base class for all transport servers.

    This base class provides common functionality for all transport
    implementations, using ORM Repository pattern for session and
    agent model management.
    """
```

**Key Methods**:

```python
class TransportBase[TTransportConfig](ABC):
    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        config: TTransportConfig,
        default_agent_config: AgentConfig,
        lock_ttl: float = 30.0,
        heartbeat_interval: float = 10.0,
    ):
        """Initialize transport base.

        Args:
            engine: DatabaseEngine for all model storage
            config: Transport-specific configuration
            default_agent_config: Default agent configuration
            lock_ttl: Lock time-to-live in seconds
            heartbeat_interval: Heartbeat interval in seconds
        """

    @abstractmethod
    def start(self) -> None:
        """Start transport server."""

    @abstractmethod
    def stop(self) -> None:
        """Stop transport server."""

    async def handle_request(
        self,
        *,
        message: str | list[Message],
        user_id: str,
        agent_config: AgentConfig | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Handle a single request (synchronous)."""

    async def handle_streaming_request(
        self,
        *,
        message: str | list[Message],
        user_id: str,
        agent_config: AgentConfig | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event]:
        """Handle a streaming request.

        Yields Event objects for streaming response.
        """
```

**Middleware Integration**:

```python
def _recursively_apply_middlewares(
    cfg: AgentConfig,
    *middlewares: Middleware,
    enable_stream: bool = False,
) -> AgentConfig:
    """Recursively apply multiple middlewares to agent config and all sub-agents.

    This method:
    1. Preserves stateful objects (tracers, middlewares)
    2. Deep copies config
    3. Assigns stateful objects to copy
    4. Adds new middlewares
    5. Enables streaming if requested
    6. Recursively applies to sub-agents
    """
```

### HTTP + SSE Transport (`http/`)

Server-Sent Events (SSE) for real-time streaming.

#### SSETransportServer (`http/sse_server.py`)

**Initialization**:

```python
from nexau.archs.transports.http import SSETransportServer, HTTPConfig

server = SSETransportServer(
    engine=engine,  # DatabaseEngine
    default_agent_config=agent_config,
    config=HTTPConfig(
        host="0.0.0.0",
        port=8000,
        cors_origins=["*"],
        log_level="info",
    ),
)
```

**Endpoints**:

| Endpoint | Method | Description |
| -------- | ------- | ----------- |
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/stream` | GET | SSE streaming query |
| `/query` | POST | Synchronous query |

**SSE Streaming Response**:

```python
# Event format for SSE
{
    "type": "event" | "complete" | "error",
    "event": {
        "type": "TEXT_MESSAGE_CONTENT",
        "content": "...",
        # ... other event fields
    },
    "session_id": "sess_123",
    "response": "..."  # For complete type
}
```

#### SSEClient (`http/sse_client.py`)

**Synchronous Query**:

```python
from nexau.archs.transports.http.sse_client import SSEClient

client = SSEClient("http://localhost:8000")

response = await client.query("What is AI?")
print(response)
```

**Streaming Query**:

```python
async for event in client.stream_events("What is AI?"):
    if event["type"] == "event":
        print(event["event"].get("content", ""), end="")
    elif event["type"] == "complete":
        print(f"\n\nSession ID: {event.get('session_id')}")
```

#### HTTPConfig (`http/config.py`)

**Configuration Options**:

```python
@dataclass
class HTTPConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    cors_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_headers: list[str] = field(default_factory=lambda: ["*"])
    log_level: str = "info"
```

### Stdio Transport (`stdio/`)

JSON-RPC 2.0-stream protocol for CLI interaction.

#### StdioTransportServer (`stdio/stdio_transport.py`)

**Initialization**:

```python
from nexau.archs.transports.stdio import StdioTransportServer, StdioConfig

server = StdioTransportServer(
    engine=engine,
    default_agent_config=agent_config,
    config=StdioConfig(
        log_level="info",
    ),
)
```

**Protocol**:

JSON-RPC 2.0-stream over stdin/stdout:

```json
// Request
{
    "jsonrpc": "2.0",
    "method": "run",
    "params": {
        "message": "Hello",
        "user_id": "user_123",
        "session_id": "sess_456",
        "context": {...}
    },
    "id": 1
}

// Response (streaming)
{"jsonrpc": "2.0", "result": {"type": "event", "event": {...}}, "id": 1}
{"jsonrpc": "2.0", "result": {"type": "event", "event": {...}}, "id": 1}
{"jsonrpc": "2.0", "result": {"type": "complete", "response": "...", "session_id": "..."}, "id": 1}
```

#### StdioConfig (`stdio/config.py`)

**Configuration Options**:

```python
@dataclass
class StdioConfig:
    log_level: str = "info"
```

## Key Patterns

### Custom Transport Pattern

Create a custom transport by extending `TransportBase`:

```python
from nexau.archs.transports.base import TransportBase
from nexau.archs.session.orm import DatabaseEngine
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexau.archs.main_sub.config import AgentConfig

class CustomTransport(TransportBase[dict]):
    """Custom transport implementation."""

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        config: dict,
        default_agent_config: AgentConfig,
    ):
        super().__init__(
            engine=engine,
            config=config,
            default_agent_config=default_agent_config,
        )

    def start(self) -> None:
        """Start transport server."""
        # Start listening for connections
        pass

    def stop(self) -> None:
        """Stop transport server."""
        # Clean up resources
        pass
```

### Streaming Request Pattern

Handle streaming requests with event emission:

```python
async def handle_streaming_request(self, message, user_id, session_id=None):
    """Handle streaming request with event emission."""

    # Create event queue
    event_queue = asyncio.Queue()

    # Create middleware with streaming enabled
    events_mw = AgentEventsMiddleware(
        session_id=session_id,
        on_event=event_queue.put_nowait,
    )

    # Apply middlewares with streaming
    config_with_middlewares = self._recursively_apply_middlewares(
        agent_config or self._default_agent_config,
        events_mw,
        enable_stream=True,  # ← Important
    )

    # Create agent
    agent = Agent(
        config=config_with_middlewares,
        session_manager=self._session_manager,
        user_id=user_id,
        session_id=session_id,
    )

    # Run agent in background
    async def run_agent():
        return await agent.run_async(message=message)

    agent_task = asyncio.create_task(run_agent())

    # Stream events from queue
    while not agent_task.done():
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
            yield event
        except TimeoutError:
            continue

    # Drain remaining events
    while not event_queue.empty():
        yield event_queue.get_nowait()

    # Wait for agent task to complete
    await agent_task
```

### Session Integration Pattern

All transports automatically handle:
- Session creation if `session_id` not provided
- Agent registration with `session_manager`
- History loading from previous runs
- History saving after each run

**No manual session management needed** in transport implementations.

### Middleware Application Pattern

Apply middlewares to all agents (including sub-agents):

```python
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)

events_mw = AgentEventsMiddleware(
    session_id=session_id,
    on_event=handle_event,
)

config_with_middlewares = self._recursively_apply_middlewares(
    agent_config,
    events_mw,
    enable_stream=True,
)
```

## Common Issues

### Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**: Use a different port:

```python
config = HTTPConfig(port=8080)  # Use different port
```

### CORS Errors

**Error**: Browser CORS errors when accessing from frontend

**Solution**: Configure CORS origins:

```python
config = HTTPConfig(
    cors_origins=["http://localhost:3000", "https://yourdomain.com"],
)
```

### Session Not Persisting

**Error**: Session data lost between requests

**Solution**: Ensure `DatabaseEngine` is properly initialized:

```python
# Incorrect
engine = None  # No persistence

# Correct
engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")
```

### Streaming Not Working

**Error**: No events emitted during streaming

**Solution**: Ensure `enable_stream=True` when applying middlewares:

```python
config_with_middlewares = self._recursively_apply_middlewares(
    agent_config,
    events_mw,
    enable_stream=True,  # ← Must be True
)
```

### Stdio Protocol Errors

**Error**: JSON-RPC 2.0 parse errors

**Solution**: Ensure proper JSON-RPC 2.0 format:

```json
// Request
{
    "jsonrpc": "2.0",
    "method": "run",
    "params": {...},
    "id": 1
}

// Response
{
    "jsonrpc": "2.0",
    "result": {...},
    "id": 1
}
```

## CLI Usage

### HTTP Server

```bash
# Start HTTP server with SSE streaming
uv run nexau serve http --config agent.yaml --port 8000

# HTTP query (synchronous)
uv run nexau serve http query --url http://localhost:8000 --message "Hello"

# HTTP stream (with events)
uv run nexau serve http stream --url http://localhost:8000 --message "Hello"
```

### Stdio Server

```bash
# Start stdio server with JSON-RPC 2.0-stream protocol
uv run nexau serve stdio --config agent.yaml --verbose
```

### Chat Command

```bash
# Interactive chat with agent
uv run nexau chat --config agent.yaml

# Non-interactive query
uv run nexau chat --config agent.yaml --query "Hello" --user-id "user123" --session-id "sess456" --verbose
```
