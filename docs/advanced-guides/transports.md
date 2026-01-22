# Transport System

> **⚠️ Experimental**: This feature is under active development. APIs may change.

The transport system allows you to expose NexAU agents as HTTP servers, CLI tools, or other protocols. It handles session management, streaming responses, and multi-user scenarios automatically.

**Available today**: HTTP + SSE, stdio (CLI/chat). **WebSocket** and **gRPC** are not yet implemented; they may be provided in future releases.

## Quick Start

### HTTP Server

Start an HTTP server with Server-Sent Events (SSE) for streaming:

```bash
# Start server
uv run nexau serve http --config agent.yaml --port 8000
```

The server provides:
- `/query` - Synchronous queries
- `/stream` - Streaming queries with SSE
- `/health` - Health check endpoint

### CLI Chat

Interactive chat interface:

```bash
# Interactive mode
uv run nexau chat --config agent.yaml

# Non-interactive query
uv run nexau chat \
    --config agent.yaml \
    --query "Hello" \
    --user-id "user123" \
    --session-id "sess456"
```

## HTTP + SSE Transport

### Starting the Server

```python
from nexau.archs.transports.http import SSETransportServer, HTTPConfig
from nexau.archs.session.orm import SQLDatabaseEngine

# Initialize storage
engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")
await engine.setup_models()

# Create server
server = SSETransportServer(
    engine=engine,
    default_agent_config=agent_config,
    config=HTTPConfig(
        host="0.0.0.0",
        port=8000,
        cors_origins=["*"],  # Allow all origins
        log_level="info",
    ),
)

# Start server
server.start()
```

### Configuration Options

```python
@dataclass
class HTTPConfig:
    host: str = "0.0.0.0"  # Bind address
    port: int = 8000  # Port number
    cors_origins: list[str] = ["*"]  # CORS allowed origins
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]
    log_level: str = "info"
```

### API Endpoints

#### Synchronous Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is AI?",
    "user_id": "user123",
    "session_id": "sess456"
  }'
```

**Response**:
```json
{
  "response": "AI is...",
  "session_id": "sess456"
}
```

#### Streaming Query (SSE)

```bash
curl -N http://localhost:8000/stream?message=Hello&user_id=user123&session_id=sess456
```

**Response** (Server-Sent Events):
```
data: {"type": "event", "event": {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}}
data: {"type": "event", "event": {"type": "TEXT_MESSAGE_CONTENT", "delta": " there"}}
data: {"type": "complete", "response": "Hello there", "session_id": "sess456"}
```

### Python Client

#### Synchronous Query

```python
from nexau.archs.transports.http.sse_client import SSEClient

client = SSEClient("http://localhost:8000")

response = await client.query(
    message="What is AI?",
    user_id="user_123",
    session_id="sess_456",
)
print(response)
```

#### Streaming Query

```python
async for event in client.stream_events(
    message="What is AI?",
    user_id="user_123",
    session_id="sess_456",
):
    if event["type"] == "event":
        # Print incremental text
        print(event["event"].get("delta", ""), end="", flush=True)
    elif event["type"] == "complete":
        print(f"\n\nComplete: {event.get('response')}")
        print(f"Session ID: {event.get('session_id')}")
```

### Frontend Integration

#### React Example

```typescript
async function streamResponse(message: string) {
  const response = await fetch(
    `http://localhost:8000/stream?message=${encodeURIComponent(message)}&user_id=user123&session_id=sess456`
  );

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.type === 'event' && data.event.type === 'TEXT_MESSAGE_CONTENT') {
          // Update UI with delta
          setResponse(prev => prev + data.event.delta);
        }
      }
    }
  }
}
```

#### JavaScript/TypeScript Client

```typescript
class SSEClient {
  constructor(private baseUrl: string) {}

  async *streamEvents(message: string, userId: string, sessionId: string) {
    const url = `${this.baseUrl}/stream?message=${encodeURIComponent(message)}&user_id=${userId}&session_id=${sessionId}`;
    const response = await fetch(url);
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          yield JSON.parse(line.slice(6));
        }
      }
    }
  }
}

// Usage
const client = new SSEClient('http://localhost:8000');
for await (const event of client.streamEvents('Hello', 'user123', 'sess456')) {
  if (event.type === 'event' && event.event.type === 'TEXT_MESSAGE_CONTENT') {
    console.log(event.event.delta);
  }
}
```

## Stdio Transport

JSON-RPC 2.0-stream protocol for CLI and process integration.

### Starting the Server

```bash
uv run nexau serve stdio --config agent.yaml
```

The server reads JSON-RPC 2.0 requests from stdin and writes responses to stdout.

### Protocol

**Request**:
```json
{
    "jsonrpc": "2.0",
    "method": "run",
    "params": {
        "message": "Hello",
        "user_id": "user_123",
        "session_id": "sess_456",
        "context": {}
    },
    "id": 1
}
```

**Response** (streaming):
```json
{"jsonrpc": "2.0", "result": {"type": "event", "event": {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}}, "id": 1}
{"jsonrpc": "2.0", "result": {"type": "event", "event": {"type": "TEXT_MESSAGE_CONTENT", "delta": " there"}}, "id": 1}
{"jsonrpc": "2.0", "result": {"type": "complete", "response": "Hello there", "session_id": "sess_456"}, "id": 1}
```

### Python Integration

```python
import subprocess
import json

process = subprocess.Popen(
    ["uv", "run", "nexau", "serve", "stdio", "--config", "agent.yaml"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

# Send request
request = {
    "jsonrpc": "2.0",
    "method": "run",
    "params": {
        "message": "Hello",
        "user_id": "user_123",
        "session_id": "sess_456",
    },
    "id": 1,
}
process.stdin.write(json.dumps(request) + "\n")
process.stdin.flush()

# Read responses
for line in process.stdout:
    response = json.loads(line)
    if response["result"]["type"] == "event":
        print(response["result"]["event"].get("delta", ""), end="")
    elif response["result"]["type"] == "complete":
        print(f"\n\nComplete: {response['result']['response']}")
        break
```

## Session Management

All transports automatically handle session management:

- **Session Creation**: If `session_id` is not provided, a new session is created
- **History Loading**: Previous conversation history is automatically loaded
- **History Persistence**: Messages are saved after each run
- **Multi-User Support**: Each `user_id` has isolated sessions

```python
# First request - creates new session
response1 = await client.query(
    message="My name is Alice",
    user_id="user_123",
    # session_id not provided - creates new session
)

# Second request - continues same session
response2 = await client.query(
    message="What's my name?",
    user_id="user_123",
    session_id=response1["session_id"],  # Use session from first request
)
```

## CLI Usage

### HTTP Server Commands

```bash
# Start server
uv run nexau serve http --config agent.yaml --port 8000

# Synchronous query
uv run nexau serve http query \
    --url http://localhost:8000 \
    --message "Hello" \
    --user-id "user123" \
    --session-id "sess456"

# Streaming query
uv run nexau serve http stream \
    --url http://localhost:8000 \
    --message "Hello" \
    --user-id "user123" \
    --session-id "sess456"
```

### Chat Command

```bash
# Interactive chat
uv run nexau chat --config agent.yaml

# Non-interactive
uv run nexau chat \
    --config agent.yaml \
    --query "Hello" \
    --user-id "user123" \
    --session-id "sess456" \
    --verbose
```

## Common Issues

### Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**: Use a different port:

```bash
uv run nexau serve http --config agent.yaml --port 8080
```

### CORS Errors

**Error**: Browser CORS errors when accessing from frontend

**Solution**: Configure CORS in server initialization:

```python
config = HTTPConfig(
    cors_origins=["http://localhost:3000", "https://yourdomain.com"],
)
```

### Session Not Persisting

**Error**: Session data lost between requests

**Solution**: Ensure database engine is properly initialized:

```python
engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")
await engine.setup_models()  # Required
```

### Streaming Not Working

**Error**: No events emitted during streaming

**Solution**: Ensure you're using the `/stream` endpoint, not `/query`:

```bash
# ❌ Wrong - synchronous endpoint
curl http://localhost:8000/query?message=Hello

# ✅ Correct - streaming endpoint
curl -N http://localhost:8000/stream?message=Hello
```

## Best Practices

1. **Use environment variables** for configuration:
   ```bash
   export NEXAU_DB_URL="sqlite:///sessions.db"
   export NEXAU_PORT=8000
   ```

2. **Configure CORS properly** for production:
   ```python
   config = HTTPConfig(
       cors_origins=["https://yourdomain.com"],  # Specific origins
       cors_credentials=True,
   )
   ```

3. **Handle errors gracefully** in clients:
   ```python
   try:
       response = await client.query(message="Hello")
   except Exception as e:
       print(f"Error: {e}")
   ```

4. **Use unique session IDs** for each conversation:
   ```python
   session_id = f"chat_{user_id}_{int(time.time())}"
   ```
