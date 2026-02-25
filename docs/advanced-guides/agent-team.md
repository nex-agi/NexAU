# Agent Team

> **⚠️ Experimental**: This feature is under active development. APIs may change.

Agent Team lets a **leader agent** coordinate multiple **teammate agents** in parallel. The leader spawns teammates on demand, creates tasks on a shared task board, and assigns work. Teammates execute independently and communicate through a message bus. All agents stream their output through a single SSE connection.

## Concepts

- **Leader**: The coordinator. Spawns teammates, creates and assigns tasks, monitors progress, and calls `finish_team` when done.
- **Teammates**: Workers spawned from pre-configured role templates (`candidates`). Each runs in a forever-loop, waiting for messages or task assignments.
- **Task Board**: A shared, DB-backed list of tasks with status (`pending → in_progress → completed`), priorities, and dependency tracking.
- **Message Bus**: Persistent point-to-point and broadcast messaging between agents.
- **TeamSSEMultiplexer**: Aggregates all agent streams into one SSE connection, with each event tagged by `agent_id`.

## Quick Start

### 1. Define Agent Configs

Create YAML configs for the leader and each candidate role. The leader gets team tools injected automatically — you only need to define its domain tools.

**leader_agent.yaml**

```yaml
type: agent
name: team_leader
system_prompt: ./systemprompt_leader.md
max_iterations: 200
llm_config:
  model: ${env.LLM_MODEL}
  api_key: ${env.LLM_API_KEY}
  stream: true
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
  # ... other domain tools
stop_tools: [ask_user]
```

**builder_agent.yaml** (a candidate role)

```yaml
type: agent
name: builder
system_prompt: ./systemprompt_builder.md
max_iterations: 80
llm_config:
  model: ${env.LLM_MODEL}
  api_key: ${env.LLM_API_KEY}
  stream: true
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
  - name: write_file
    yaml_path: ./tools/write_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:write_file
  # ... other domain tools
```

### 2. Start the Server

Register the team config with `SSETransportServer` via its `team_registry`:

```python
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.transports.http import HTTPConfig, SSETransportServer

leader_config = AgentConfig.from_yaml("leader_agent.yaml")
rfc_writer_config = AgentConfig.from_yaml("rfc_writer_agent.yaml")
builder_config = AgentConfig.from_yaml("builder_agent.yaml")

engine = InMemoryDatabaseEngine()
server = SSETransportServer(
    engine=engine,
    config=HTTPConfig(host="0.0.0.0", port=8000),
    default_agent_config=leader_config,
)

# Register the team: one leader + named candidate roles
registry = server.team_registry
if registry is not None:
    registry.register_config(
        "default",
        leader_config=leader_config,
        candidates={
            "rfc_writer": rfc_writer_config,
            "builder": builder_config,
        },
    )

server.run()
```

For production, swap `InMemoryDatabaseEngine` with `SQLDatabaseEngine`:

```python
from nexau.archs.session.orm import SQLDatabaseEngine

engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///team.db")
```

### 3. Send a Request

```bash
curl -X POST http://localhost:8000/team/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "session_id": "s1", "message": "Build a TODO app"}'
```

The response is an SSE stream where each event is a `TeamStreamEnvelope`:

```
data: {"team_id":"team_abc","agent_id":"leader-001","role_name":"leader","event":{"type":"TEXT_MESSAGE_CONTENT","delta":"Let me plan this..."}}

data: {"team_id":"team_abc","agent_id":"builder-1","role_name":"builder","event":{"type":"TEXT_MESSAGE_CONTENT","delta":"Writing the API..."}}
```

## Direct Usage (Without HTTP)

You can use `AgentTeam` directly in Python without starting an HTTP server. This is useful for scripts, notebooks, testing, or embedding team coordination into your own application.

### Non-Streaming

```python
import asyncio

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.team.agent_team import AgentTeam
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.session.session_manager import SessionManager

leader_config = AgentConfig.from_yaml("leader_agent.yaml")
builder_config = AgentConfig.from_yaml("builder_agent.yaml")
rfc_writer_config = AgentConfig.from_yaml("rfc_writer_agent.yaml")

engine = InMemoryDatabaseEngine()
session_manager = SessionManager(engine=engine)

team = AgentTeam(
    leader_config=leader_config,
    candidates={
        "rfc_writer": rfc_writer_config,
        "builder": builder_config,
    },
    engine=engine,
    session_manager=session_manager,
    user_id="user_1",
    session_id="session_1",
)

result = asyncio.run(team.run(message="Build a TODO app"))
print(result)
```

`run()` calls `initialize()` internally, so you don't need to call it yourself. The method blocks until the leader calls `finish_team`, then returns the leader's final response as a string.

### Streaming

`run_streaming()` returns an async generator of `TeamStreamEnvelope` objects — the same envelope format used by the HTTP SSE endpoint:

```python
async def main() -> None:
    # ... same setup as above ...

    async for envelope in team.run_streaming(message="Build a TODO app"):
        print(f"[{envelope.role_name}:{envelope.agent_id}] {envelope.event}")

asyncio.run(main())
```

You can also pass an `on_envelope` callback for side-effects like persistence:

```python
envelopes: list[TeamStreamEnvelope] = []

async for envelope in team.run_streaming(
    message="Build a TODO app",
    on_envelope=envelopes.append,
):
    pass  # or process in real-time

# envelopes now contains the full event history
```

### Programmatic Control

`AgentTeam` exposes several methods for runtime inspection and control:

| Method / Property | Description |
|---|---|
| `team.task_board` | Access the shared `TaskBoard` (after `initialize()`) |
| `team.message_bus` | Access the `TeamMessageBus` |
| `team.get_teammate_info()` | List all teammates with status (`idle` / `running` / `error`) |
| `team.enqueue_user_message(agent_id, text)` | Inject a user message to a specific agent mid-run |
| `team.stop_all()` | Force-stop the leader and all teammates immediately |
| `team.is_running` | Whether the team is currently executing |

### Using SQL-Backed Storage

For persistence across restarts, swap `InMemoryDatabaseEngine` with `SQLDatabaseEngine`:

```python
from nexau.archs.session.orm import SQLDatabaseEngine

engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///team.db")
```

Team state (tasks, messages, members) will be persisted and automatically restored on the next `run()` call with the same `user_id` + `session_id`.

## Team Tools

Team tools are injected automatically into the leader and teammates. You don't define them in YAML.

### Leader tools

| Tool | Description |
|------|-------------|
| `spawn_teammate` | Instantiate a new teammate from a candidate role |
| `remove_teammate` | Remove an idle teammate |
| `create_task` | Add a task to the shared task board |
| `claim_task` | Assign a task to a specific teammate |
| `update_task_status` | Update task status (`pending → in_progress → completed`) |
| `release_task` | Unassign a task |
| `list_tasks` | List tasks with optional status filter |
| `list_teammates` | List all teammates and their status |
| `message` | Send a point-to-point message |
| `broadcast` | Send a message to all teammates |
| `finish_team` | End the team run and return a summary |

### Teammate tools

Teammates get a subset: `message`, `broadcast`, `list_teammates`, `list_tasks`, `claim_task`, `update_task_status`, `release_task`.

Teammates cannot spawn other teammates, create tasks, or finish the team.

## Typical Workflow

```
User → POST /team/stream
         │
         ▼
Leader analyzes the request
  ├─ spawn_teammate("builder")     → builder-1
  ├─ spawn_teammate("rfc_writer")  → rfc_writer-1
  │
  ├─ create_task("Write RFC", priority=1)          → T-001
  ├─ create_task("Implement feature", deps=["T-001"]) → T-002
  │
  ├─ claim_task("T-001", assignee="rfc_writer-1")
  │     └─ rfc_writer-1 receives message, starts working
  │
  ├─ [rfc_writer-1 completes T-001]
  │     └─ update_task_status("T-001", "completed")
  │
  ├─ [Watchdog detects all-idle → notifies leader]
  │
  ├─ claim_task("T-002", assignee="builder-1")
  │     └─ builder-1 receives message, starts working
  │
  └─ [All tasks done] → finish_team("Summary...")
```

## Task Board

Tasks have a simple state machine: `pending → in_progress → completed`.

- **Dependencies**: A task with `dependencies=["T-001"]` is blocked until T-001 is completed. Attempting to claim a blocked task returns a `ToolError(code="conflict")`.
- **Single-task constraint**: Each agent can only hold one `in_progress` task at a time.
- **Concurrent claim safety**: Claims use a short-lived DB-backed TTL lock (5s). If two agents race to claim the same task, one gets a `ToolError(code="conflict")` and should retry with a different task.

### Deliverable paths

Each task gets a `deliverable_path` (e.g. `.nexau/tasks/T-001-write-rfc.md`) where the assigned agent should write its output. The leader can read these files to review results.

## Idle Detection

The built-in `TeammateWatchdog` runs as a background task. When all spawned teammates are idle (waiting for messages), it sends the leader a notification:

```
[All Idle] All agents are idle. Review task board status and decide next steps — assign new tasks, check results.
```

The leader can then assign more tasks, review deliverables, or call `finish_team`.

## HTTP Endpoints

All team endpoints are under `/team`:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/team/stream` | Run team with SSE streaming |
| `POST` | `/team/query` | Run team synchronously |
| `GET` | `/team/tasks` | List tasks (`?status=pending\|in_progress\|completed`) |
| `GET` | `/team/teammates` | List teammates |
| `POST` | `/team/message` | Send a message to an agent |
| `POST` | `/team/user-message` | Inject a user message during streaming |
| `POST` | `/team/stop` | Force-stop all agents |
| `GET` | `/team/status` | Query team run status |
| `GET` | `/team/subscribe` | Re-subscribe to SSE after reconnect |

All endpoints require `user_id` and `session_id` to scope the team.

## Session Isolation

Each agent in a team has its own isolated session:

```
team_session_id  →  used only for team-level data (TaskBoard, MessageBus, TeamModel)

leader session   →  "{team_session_id}:leader"
teammate session →  "{team_session_id}:{agent_id}"   e.g. "sess-abc:builder-1"
```

Agents don't share history or global storage. All inter-agent communication goes through the message bus.

## Writing Effective System Prompts

### Leader prompt

The leader prompt should instruct the agent to:

1. Analyze the user request before spawning teammates
2. Create tasks with clear titles, descriptions, and dependencies
3. Assign tasks explicitly — teammates won't self-start without a message or assignment
4. Check `list_tasks` after receiving an all-idle notification
5. Call `finish_team` with a summary once all tasks are completed

```markdown
You are the team leader. Your job is to coordinate teammates to complete the user's request.

Workflow:
1. Analyze the request and decide which roles you need
2. Spawn teammates using spawn_teammate(role_name)
3. Create tasks with create_task, set dependencies where needed
4. Assign tasks to teammates with claim_task(task_id, assignee_agent_id=...)
5. When notified that all agents are idle, review task results and assign next tasks
6. When all work is done, call finish_team with a summary
```

### Teammate prompt

Teammate prompts should instruct the agent to:

1. Wait for task assignments or messages from the leader
2. Use `list_tasks` to find their assigned task
3. Write deliverables to the `deliverable_path` from the task
4. Call `update_task_status("T-xxx", "completed", result_summary="...")` when done
5. Message the leader if blocked or needing clarification

```markdown
You are a builder agent. You implement features based on task assignments from the leader.

When you receive a task assignment:
1. Call list_tasks() to see the task details and deliverable_path
2. Implement the feature and write output to deliverable_path
3. Call update_task_status(task_id, "completed", result_summary="...") when done
4. If blocked, message the leader explaining the issue
```

## Full Example

See [examples/nexau_building_team/](../../examples/nexau_building_team/) for a complete working example with:

- A leader that coordinates RFC writing and implementation
- An `rfc_writer` role that produces design documents
- A `builder` role that implements agent components
- A React frontend that renders per-agent streams in separate panels
- A file browser API for viewing generated files
