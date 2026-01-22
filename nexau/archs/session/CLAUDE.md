# Session Management Implementation Guide

This module contains ORM, repositories, and services for stateful session management in NexAU.

## Architecture Overview

The session system uses an **ORM Repository pattern** with pluggable backends:

```
Repository[T] (Generic ORM Repository Pattern)
    ├── Backend (Abstract Storage)
    │   ├── MemoryBackend: Single-process dev/testing
    │   ├── SQLBackend: SQLite/PostgreSQL/MySQL
    │   ├── JSONLBackend: Line-delimited JSON storage
    │   └── RemoteBackend: Remote HTTP API calls
    │
    ├── StorageModel (Base Model with Meta)
    │   ├── SessionModel: Session state (context, storage, agent_ids)
    │   ├── AgentModel: Agent metadata (agent_id, agent_name, parent)
    │   ├── AgentLockModel: Concurrency control (lock_id, ttl)
    │   └── AgentRunActionModel: Message history actions
    │
    └── Services
        ├── SessionManager: Unified session/agent/action management
        ├── AgentLockService: Distributed locking
        └── AgentRunActionService: History persistence
```

## Key Components

### Repository (`orm/engine.py`)

Generic repository for any StorageModel with pluggable backend.

**Initialization**:

```python
from nexau.archs.session.orm import Repository, SQLDatabaseEngine

# Create backend
engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")

# Create repository for a model
session_repo = Repository[SessionModel](engine)
```

**Key Methods**:

```python
class Repository[T: StorageModel]:
    async def acquire(
        self,
        **kwargs
    ) -> AsyncGenerator[tuple[T, SaveFn], None]:
        """Acquire model by primary key.

        Yields (model, save_fn) and auto-saves on exit.
        """

    async def get(self, **kwargs) -> T | None:
        """Get model by primary key."""

    async def save(self, model: T) -> None:
        """Save or update model."""

    async def find(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        order_by: list[str] | None = None,
        **filters,
    ) -> list[T]:
        """Find models matching filters.

        Supports filter expressions with property paths and operators.
        """
```

**Filtering Support**:

```python
# Simple equality
await agent_repo.find(user_id="user1", session_id="sess_123")

# Comparison operators
await agent_repo.find(created_at__gt=datetime(...))

# Nested property paths
await action_repo.find(
    action_message__content__icontains="hello",
)
```

### Backend Types

#### InMemoryDatabaseEngine (`orm/memory_engine.py`)

Single-process in-memory storage for development/testing.

```python
from nexau.archs.session.orm import InMemoryDatabaseEngine

engine = InMemoryDatabaseEngine()
```

**Features**:
- Thread-safe with asyncio locks
- No persistence
- Fast for testing

#### SQLDatabaseEngine (`orm/sql_engine.py`)

SQL-based storage with support for SQLite, PostgreSQL, MySQL.

```python
from nexau.archs.session.orm import SQLDatabaseEngine

# SQLite (default)
engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")

# PostgreSQL
engine = SQLDatabaseEngine.from_url(
    "postgresql+aiosqlite://user:pass@localhost/db"
)

# MySQL
engine = SQLDatabaseEngine.from_url(
    "mysql+aiomysql://user:pass@localhost/db"
)
```

**Features**:
- Persistent storage
- Automatic table creation
- Thread-safe with connection pooling

#### JSONLDatabaseEngine (`orm/jsonl_engine.py`)

Line-delimited JSON storage.

```python
from nexau.archs.session.orm import JSONLDatabaseEngine

engine = JSONLDatabaseEngine(storage_path="/path/to/storage")
```

**Features**:
- Human-readable storage
- Simple backup/restore
- No database dependency

#### RemoteDatabaseEngine (`orm/remote_engine.py`)

Remote HTTP API calls for session storage.

```python
from nexau.archs.session.orm import RemoteDatabaseEngine

engine = RemoteDatabaseEngine(
    base_url="https://api.example.com/sessions",
    api_key="your-api-key",
)
```

**Features**:
- Centralized storage
- Multi-service sharing
- Requires remote service

### Storage Models

#### SessionModel (`models/session.py`)

Session state with context, storage, and agent IDs.

**Schema**:

```python
class SessionModel(StorageModel):
    user_id: str  # (primary key)
    session_id: str  # (primary key)
    storage: dict[str, Any]  # Shared agent storage
    context: dict[str, Any]  # Runtime context
    agent_ids: list[str]  # Only IDs, not full objects
    root_agent_id: str | None
    created_at: datetime
    updated_at: datetime
```

**Meta Configuration**:

```python
class SessionModel(StorageModel):
    class Meta:
        table_name = "sessions"
        primary_key = ["user_id", "session_id"]
```

#### AgentModel (`models/agent.py`)

Agent metadata for tracking agent instances.

**Schema**:

```python
class AgentModel(StorageModel):
    user_id: str  # (primary key)
    session_id: str  # (primary key)
    agent_id: str  # (primary key)
    agent_name: str
    parent_agent_id: str | None
    created_at: datetime
    last_updated: datetime
```

#### AgentRunActionModel (`models/agent_run_action_model.py`)

Run-level message history actions (APPEND/UNDO/REPLACE).

**Schema**:

```python
class AgentRunActionModel(StorageModel):
    user_id: str  # (primary key)
    session_id: str  # (primary key)
    agent_id: str  # (primary key)
    run_id: str  # (primary key)
    action_type: RunActionType  # APPEND | UNDO | REPLACE
    messages: list[Message]
    agent_name: str
    parent_run_id: str | None
    created_at: datetime
```

**Action Types**:

```python
class RunActionType(str, Enum):
    APPEND = "APPEND"  # Add new messages
    UNDO = "UNDO"  # Revert to previous state
    REPLACE = "REPLACE"  # Replace entire history
```

#### AgentLockModel (`models/agent_lock.py`)

Concurrency control with time-to-live (TTL).

**Schema**:

```python
class AgentLockModel(StorageModel):
    user_id: str  # (primary key)
    session_id: str  # (primary key)
    agent_id: str  # (primary key)
    lock_id: str  # Unique per acquisition
    ttl: datetime  # Lock expiration
    created_at: datetime
```

### SessionManager (`session_manager.py`)

Unified session, agent, and action management.

**Initialization**:

```python
from nexau.archs.session import SessionManager, SQLDatabaseEngine

engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")

session_manager = SessionManager(
    engine=engine,
    lock_ttl=30.0,  # Lock TTL in seconds
    heartbeat_interval=10.0,  # Heartbeat interval in seconds
)
```

**Key Methods**:

```python
class SessionManager:
    @asynccontextmanager
    async def acquire_session(
        self,
        user_id: str,
        session_id: str,
    ) -> AsyncGenerator[tuple[SessionModel, SaveFn], None]:
        """Acquire session with auto-save on exit."""

    @asynccontextmanager
    async def acquire_agent(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        agent_name: str,
        parent_agent_id: str | None = None,
    ) -> AsyncGenerator[tuple[AgentModel, SaveFn], None]:
        """Acquire or create agent with auto-save on exit."""

    def get_history_key(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
    ) -> AgentRunActionKey:
        """Get history key for action service."""
```

### AgentLockService (`agent_lock_service.py`)

Distributed locking with automatic expiration and heartbeats.

**Usage**:

```python
from nexau.archs.session import AgentLockService

lock_service = AgentLockService(
    agent_repo=agent_repo,
    lock_repo=lock_repo,
    ttl=30.0,  # Lock expires after 30 seconds
    heartbeat_interval=10.0,  # Renew lock every 10 seconds
)

async with lock_service.acquire(
    user_id="user1",
    session_id="sess_123",
    agent_id="agent_a",
) as lock_acquired:
    if lock_acquired:
        # Critical section - only one agent instance can run here
        await agent.run_async(message)
    else:
        # Another agent instance is holding the lock
        raise AgentLockError("Agent is locked by another instance")
```

**Lock Behavior**:

- Automatic expiration after TTL
- Heartbeat renews lock periodically
- Concurrent acquisitions wait or fail immediately (configurable)

### AgentRunActionService (`agent_run_action_service.py`)

Run-level history persistence with action tracking.

**Usage**:

```python
from nexau.archs.session import AgentRunActionService

action_service = AgentRunActionService(action_repo=action_repo)

key = AgentRunActionKey(
    user_id="user1",
    session_id="sess_123",
    agent_id="agent_a",
)

# Append new messages
await action_service.persist_append(
    key=key,
    run_id="run_456",
    root_run_id="root_123",
    parent_run_id="parent_789",
    agent_name="my_agent",
    messages=[new_msg1, new_msg2],
)

# Replace entire history
await action_service.persist_replace(
    key=key,
    run_id="run_456",
    root_run_id="root_123",
    messages=[msg1, msg2, msg3],
    parent_run_id="parent_789",
    agent_name="my_agent",
)
```

### AgentRunActionKey

Unique identifier for agent run actions.

```python
from nexau.archs.session import AgentRunActionKey

key = AgentRunActionKey(
    user_id="user1",
    session_id="sess_123",
    agent_id="agent_a",
)
```

## Key Patterns

### Repository Pattern

```python
from nexau.archs.session.orm import Repository, SQLDatabaseEngine, SessionModel

# Create engine and repository
engine = SQLDatabaseEngine.from_url("sqlite:///sessions.db")
session_repo = Repository[SessionModel](engine)

# Acquire and modify
async with session_repo.acquire(user_id="user1", session_id="sess_123") as (session, save):
    session.context["key"] = "value"
    session.storage["data"] = {"example": 123}
    # Auto-saves on exit
```

### Session Management Pattern

```python
from nexau.archs.session import SessionManager

session_manager = SessionManager(engine=engine)

# Acquire session
async with session_manager.acquire_session(
    user_id="user1",
    session_id="sess_123",
) as (session, save):
    # Work with session
    session.context["user_data"] = {...}

# Acquire agent
async with session_manager.acquire_agent(
    user_id="user1",
    session_id="sess_123",
    agent_id="agent_a",
    agent_name="My Agent",
) as (agent, save):
    # Work with agent
    agent.last_updated = datetime.now()
```

### Locking Pattern

```python
from nexau.archs.session import AgentLockService

lock_service = AgentLockService(
    agent_repo=agent_repo,
    lock_repo=lock_repo,
)

# Try to acquire lock
async with lock_service.acquire(
    user_id="user1",
    session_id="sess_123",
    agent_id="agent_a",
) as acquired:
    if not acquired:
        raise Exception("Agent is locked")
    # Run agent
    await agent.run_async(message)
```

### History Persistence Pattern

```python
from nexau.archs.session import AgentRunActionService, AgentRunActionKey

action_service = AgentRunActionService(action_repo=action_repo)
key = AgentRunActionKey(
    user_id="user1",
    session_id="sess_123",
    agent_id="agent_a",
)

# Persist new messages
await action_service.persist_append(
    key=key,
    run_id=run_id,
    root_run_id=root_run_id,
    messages=[msg1, msg2],
    parent_run_id=parent_run_id,
    agent_name="My Agent",
)
```

## Common Issues

### Table Not Found

**Error**: `TableNotFoundError: Table 'sessions' does not exist`

**Solution**: Ensure `DatabaseEngine.setup_models()` is called on startup. Agent handles this automatically when using `session_manager`.

### Lock Expired

**Error**: `AgentLockExpiredError: Lock expired`

**Solution**: The lock TTL was reached without heartbeat. Increase `lock_ttl` or ensure heartbeat is running.

### Backend Not Compatible

**Error**: Import error when using specific backend

**Solution**: Ensure required dependencies are installed:

```bash
# For SQLDatabaseEngine
uv pip install aiosqlite  # or aiomysql, aiopgsql

# For JSONLDatabaseEngine
# No extra dependencies

# For RemoteDatabaseEngine
uv pip install httpx
```

### Filter Not Working

**Error**: Filter not returning expected results

**Solution**: Use property paths for nested fields:

```python
# Incorrect
await action_repo.find(action_message="...")

# Correct (property path)
await action_repo.find(action_message__content="...")
```
