# Session Management

> **⚠️ Experimental**: This feature is under active development. APIs may change.

The session management system enables stateful conversations, allowing agents to remember previous interactions and maintain context across multiple runs. It supports multiple storage backends and automatic history persistence.

## Quick Start

The simplest way to use session management is to pass a `SessionManager` when creating an agent:

```python
from nexau.archs.main_sub.agent import Agent, AgentConfig
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import SQLDatabaseEngine

# Initialize storage backend
engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
await engine.setup_models()

# Create session manager
session_manager = SessionManager(engine=engine)

# Create agent with session management
agent = Agent(
    config=agent_config,
    session_manager=session_manager,
    user_id="user_123",
    session_id="sess_456",
)

# Agent automatically:
# - Creates/loads session if needed
# - Loads conversation history from previous runs
# - Saves history after each run
# - Maintains context across multiple interactions

response = await agent.run_async("Hello, how are you?")
# Next run will remember this conversation
response2 = await agent.run_async("What did I just ask?")
```

## Storage Backends

Choose a storage backend based on your needs:

### SQLite (Recommended for Development)

Simple file-based storage, perfect for development and small deployments:

```python
from nexau.archs.session.orm import SQLDatabaseEngine

engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
await engine.setup_models()

session_manager = SessionManager(engine=engine)
```

**Features**:
- No additional dependencies
- Persistent storage
- Easy to backup (just copy the file)

### PostgreSQL / MySQL

For production deployments requiring concurrent access:

```python
# PostgreSQL
engine = SQLDatabaseEngine.from_url(
    "postgresql+asyncpg://user:pass@localhost/db"
)

# MySQL
engine = SQLDatabaseEngine.from_url(
    "mysql+aiomysql://user:pass@localhost/db"
)

await engine.setup_models()
session_manager = SessionManager(engine=engine)
```

**Installation**:
```bash
# For PostgreSQL
uv pip install asyncpg

# For MySQL
uv pip install aiomysql
```

### In-Memory (Testing Only)

For testing or when persistence is not needed:

```python
from nexau.archs.session.orm import InMemoryDatabaseEngine

engine = InMemoryDatabaseEngine()
session_manager = SessionManager(engine=engine)
```

**Note**: Data is lost when the process exits. Use only for testing.

### JSONL (Simple Backup/Restore)

Human-readable line-delimited JSON storage:

```python
from nexau.archs.session.orm import JSONLDatabaseEngine

engine = JSONLDatabaseEngine(storage_path="/path/to/storage")
session_manager = SessionManager(engine=engine)
```

**Features**:
- Human-readable format
- Easy backup/restore
- No database dependency

## Session Configuration

### Session ID

Each conversation is identified by a `session_id`. If not provided, a new session is created:

```python
# Create new session
agent = Agent(
    config=agent_config,
    session_manager=session_manager,
    user_id="user_123",
    # session_id not provided - creates new session
)

# Continue existing session
agent = Agent(
    config=agent_config,
    session_manager=session_manager,
    user_id="user_123",
    session_id="sess_456",  # Loads existing session
)
```

### User ID

The `user_id` identifies the user across all sessions. This is useful for multi-user scenarios:

```python
agent = Agent(
    config=agent_config,
    session_manager=session_manager,
    user_id="alice@example.com",  # User identifier
    session_id="chat_001",
)
```

## Conversation History

History is automatically loaded and saved. The agent remembers previous messages in the conversation:

```python
agent = Agent(
    config=agent_config,
    session_manager=session_manager,
    user_id="user_123",
    session_id="sess_456",
)

# First message
response1 = await agent.run_async("My name is Alice")

# Second message - agent remembers the name
response2 = await agent.run_async("What's my name?")
# Agent will respond: "Your name is Alice"
```

## Concurrency Control

When multiple requests try to process the same session simultaneously, session management provides locking to prevent conflicts. This ensures only one agent run can modify a session's state at a time.

```python
session_manager = SessionManager(
    engine=engine,
    lock_ttl=30.0,  # Lock expires after 30 seconds
    heartbeat_interval=10.0,  # Renew lock every 10 seconds
)
```

**How it works**:
- Agents are stateless; only sessions maintain state
- When an agent run starts, it acquires a lock on the session
- Only one agent run can execute per session at a time
- Locks automatically expire to prevent deadlocks
- Heartbeat keeps locks alive during long-running operations
- If a lock is held, concurrent requests will wait or fail (configurable)

## Usage Examples

### Multi-User Chat Application

```python
from nexau.archs.main_sub.agent import Agent
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import SQLDatabaseEngine

# Initialize once
engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///chat.db")
await engine.setup_models()
session_manager = SessionManager(engine=engine)

async def handle_user_message(user_id: str, session_id: str, message: str):
    """Handle a message from a user."""
    agent = Agent(
        config=agent_config,
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
    )

    response = await agent.run_async(message)
    return response

# Usage
response = await handle_user_message(
    user_id="alice@example.com",
    session_id="chat_001",
    message="Hello!"
)
```

### Session Cleanup

To manage storage, you can query and delete old sessions using the engine's Filter DSL:

```python
from datetime import datetime, timedelta

from nexau.archs.session import SessionModel
from nexau.archs.session.orm import AndFilter, ComparisonFilter, SQLDatabaseEngine

# engine already created and setup_models() called
cutoff = datetime.now() - timedelta(days=30)
old_sessions = await engine.find_many(
    SessionModel,
    filters=ComparisonFilter.lt("updated_at", cutoff),
)

for session in old_sessions:
    await engine.delete(
        SessionModel,
        filters=AndFilter(filters=[
            ComparisonFilter.eq("user_id", session.user_id),
            ComparisonFilter.eq("session_id", session.session_id),
        ]),
    )
```

## Common Issues

### Table Not Found

**Error**: `TableNotFoundError: Table 'sessions' does not exist`

**Solution**: Initialize tables before use:

```python
engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
await engine.setup_models()  # ← Required
```

### History Not Persisting

**Error**: Conversation history lost between runs

**Solution**: Ensure `SessionManager` is provided:

```python
# ❌ Incorrect - no session manager
agent = Agent(config=agent_config)

# ✅ Correct - with session manager
agent = Agent(
    config=agent_config,
    session_manager=session_manager,  # Required
    user_id="user_123",
    session_id="sess_456",
)
```

### Lock Expired

**Error**: `AgentLockExpiredError: Lock expired`

**Solution**: Increase lock TTL for longer operations:

```python
session_manager = SessionManager(
    engine=engine,
    lock_ttl=60.0,  # Increase from default 30 seconds
    heartbeat_interval=20.0,
)
```

### Database Connection Errors

**Error**: Connection errors with PostgreSQL/MySQL

**Solution**: Ensure database is running and credentials are correct:

```python
# Test connection
engine = SQLDatabaseEngine.from_url("postgresql://...")
await engine.setup_models()  # Will fail if connection is bad
```

## Best Practices

1. **Initialize tables once** at application startup:
   ```python
   engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
   await engine.setup_models()  # Do this once
   ```

2. **Use unique session IDs** for each conversation:
   ```python
   session_id = f"chat_{user_id}_{timestamp}"  # Unique per conversation
   ```

3. **SessionManager is stateless** — it does not hold session data. Persistence is handled by the storage engine (SQLite, PostgreSQL, etc.). You can create a `SessionManager` per request or share one; both work. **Exception**: with the in-memory engine, data is lost when the process exits, so use it only for testing.

4. **Clean up old sessions** periodically to manage storage. Use `engine.find_many` with `ComparisonFilter.lt("updated_at", cutoff)` and `engine.delete` as in [Session Cleanup](#session-cleanup) above.

---

## See also

- [Transport System](./transports.md) — Transports use `user_id` and `session_id` for session management; sessions are created and persisted automatically.
