# Async / Sync Usage Guide

NexAU provides full async/sync dual-path support. **Core principle: `await` everything in async contexts, use `asyncio.run()` or `to_thread` to bridge in sync contexts — never mix the two paths.**

This guide is for framework users and extension developers, covering the following scenarios:

- Running agents from scripts / CLI (sync entry point)
- Running agents from web services / Transports (async entry point)
- Writing custom tools (sync / async)
- Making sub-agent calls
- Thread-safe coroutine scheduling

---

## Quick Reference

| Scenario | Create Agent | Run Agent |
|---|---|---|
| CLI / script (no event loop) | `Agent(config=...)` | `agent.run(message=...)` |
| async handler (Transport / Team) | `await Agent.create(config=...)` | `await agent.run_async(message=...)` |

| Scenario | Execute Tool |
|---|---|
| sync caller | `tool.execute(**params)` |
| async caller (inside executor) | `await tool.execute_async(**params)` |

---

## 1. Sync Entry Point: CLI and Scripts

In a pure sync environment with no running event loop (e.g. CLI scripts, `__main__` entry points), use the sync API directly:

```python
from nexau import Agent, AgentConfig, LLMConfig

config = AgentConfig(
    name="my_agent",
    llm_config=LLMConfig(model="gpt-4o"),
    system_prompt="You are a helpful assistant.",
)

# Sync construction + sync run
agent = Agent(config=config)
response = agent.run(message="Hello!")
print(response)
```

### How It Works

- `Agent.__init__()` completes session initialization synchronously on the current thread (DB model creation, agent registration, storage restoration).
- `Agent.run()` internally calls `asyncio.run(self.run_async(...))` to create a temporary event loop that drives the async execution chain.
- If a running event loop is detected on the current thread, `run()` raises `RuntimeError` instead of silently nesting — use `await agent.run_async(...)` in that case.

### ⚠️ Important Notes

- **Do not call `agent.run()` from inside an async function** — it raises `RuntimeError`.
- **Do not call `agent.run()` from within an `asyncio.run()` callback** — it triggers a nested event loop error.

---

## 2. Async Entry Point: Transports and Web Services

In an async environment with a running event loop (e.g. FastAPI handlers, Transports, AgentTeam), use the async factory + async run:

```python
from nexau import Agent, AgentConfig, LLMConfig

config = AgentConfig(
    name="my_agent",
    llm_config=LLMConfig(model="gpt-4o"),
    system_prompt="You are a helpful assistant.",
)

# Async construction + async run
agent = await Agent.create(config=config, session_manager=sm)
response = await agent.run_async(message="Hello!")
```

### Why `Agent.create()`?

Agent initialization involves async I/O (database writes, MCP tool discovery, storage restoration). In an async context:

- ❌ `Agent(config=...)` — sync `__init__` cannot `await`; if async session operations are needed, it may attempt to create a nested event loop.
- ✅ `await Agent.create(config=...)` — all initialization runs natively on the current event loop without creating extra threads or temporary loops.

`Agent.create()` accepts the same parameters as `Agent.__init__()`.

### Usage in Transports

NexAU's built-in `TransportBase` already uses `Agent.create()`:

```python
# nexau/archs/transports/base.py (simplified)
async def handle_request(self, message: str, context: dict | None = None) -> str:
    agent = await Agent.create(
        config=self.config,
        session_manager=self.session_manager,
        user_id=user_id,
        session_id=session_id,
    )
    response = await agent.run_async(message=message, context=context)
    return response
```

### Usage in AgentTeam

```python
# nexau/archs/main_sub/team/agent_team.py (simplified)
async def spawn_teammate(self, config: AgentConfig) -> Agent:
    agent = await Agent.create(
        config=config,
        session_manager=self.session_manager,
        global_storage=self.global_storage,
        is_root=False,
    )
    return agent
```

---

## 3. Writing Custom Tools

### 3.1 Sync Tools (Recommended for Most Cases)

Most tools only need a sync function. The executor automatically dispatches sync tools to a thread pool via `asyncio.to_thread()` on the async path, so they never block the event loop:

```python
from nexau import Tool

def my_search(query: str) -> dict:
    """Sync tool — the executor runs it in a thread pool automatically."""
    import requests
    result = requests.get(f"https://api.example.com/search?q={query}")
    return {"results": result.json()}

tool = Tool(
    name="my_search",
    description="Search the web",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    implementation=my_search,
)
```

### 3.2 Async Tools

For scenarios requiring native async I/O (e.g. WebSocket, streaming HTTP), define the implementation as `async def`:

```python
import httpx

async def my_async_search(query: str) -> dict:
    """Native async tool — the executor awaits it directly, no thread pool."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/search?q={query}")
        return {"results": resp.json()}

tool = Tool(
    name="my_async_search",
    description="Async search",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    implementation=my_async_search,
)
```

### 3.3 How the Executor Dispatches Tools

The executor's async path (`execute_async()`) automatically selects the optimal dispatch strategy based on tool type:

| Tool Type | Dispatch Strategy |
|---|---|
| sync implementation | `await asyncio.to_thread(tool.execute, ...)` |
| async implementation | `await tool.execute_async(...)` → directly `await impl(...)` |
| `has_native_async_execute = True` (e.g. MCPTool) | `await tool.execute_async(...)` |

**You don't need to manage dispatch manually** — just define your sync/async implementation correctly and the executor handles the rest.

### 3.4 `execute()` vs `execute_async()` Behavior

| Method | Calling Context | Behavior with async impl |
|---|---|---|
| `tool.execute()` | sync context (CLI / thread pool worker) | `asyncio.run(impl(...))` — creates temporary loop |
| `tool.execute_async()` | async context (executor async path) | `await impl(...)` — reuses the main loop |

> ⚠️ Calling the sync `tool.execute()` from an async context when the tool implementation is async will raise `RuntimeError` instead of silently nesting an event loop.

### 3.5 Implementing a Native Async Tool Subclass (Advanced)

If you subclass `Tool` and need a fully independent async execution path (e.g. MCP protocol RPC calls), you need to:

1. Set `self._has_native_async_execute = True` in `__init__`
2. Override the `execute_async()` method

```python
from nexau.archs.tool.tool import Tool

class MyProtocolTool(Tool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_native_async_execute = True

    async def execute_async(self, **params) -> dict:
        # Native async implementation, runs directly on the main event loop
        result = await self._protocol_client.call(params)
        return {"result": result}
```

This way the executor will directly `await tool.execute_async(...)` instead of going through the `to_thread → sync execute` indirection.

---

## 4. Sub-agent Calls

### 4.1 Async Path (Recommended)

In the executor's async execution chain, sub-agents are invoked via `call_sub_agent_async()`, reusing the main event loop:

```python
# SubAgentManager internals (simplified)
async def call_sub_agent_async(self, sub_agent_name: str, message: str) -> str:
    sub_agent = await Agent.create(
        config=self.sub_agents[sub_agent_name],
        global_storage=self.global_storage,
        session_manager=self.session_manager,
        is_root=False,
    )
    result = await sub_agent.run_async(message=message)
    return result
```

This is much more efficient than the old sync path — it no longer creates two temporary event loops per sub-agent invocation.

### 4.2 Sync Path (Backward Compatible)

The sync `call_sub_agent()` is still available for the sync executor path. It uses `Agent()` + `agent.run()` internally, which creates a temporary loop via `asyncio.run()`.

---

## 5. Utility Functions

### `run_async_function_sync()`

A safe bridge for running async functions from a sync context:

```python
from nexau.core.utils import run_async_function_sync

# ✅ Use in sync context (no running loop)
result = run_async_function_sync(lambda: some_async_function())

# ❌ Using in async context — raises RuntimeError
# Use `await some_async_function()` instead
```

**Behavior:**
- No running loop → executes via `asyncio.run()`
- Running loop detected → raises `RuntimeError`, guiding the user to use `await`

### `schedule_coroutine_from_sync()`

Thread-safe fire-and-forget coroutine scheduling from sync context:

```python
from nexau.core.utils import schedule_coroutine_from_sync

# Schedule a coroutine on a target event loop (does not wait for the result)
schedule_coroutine_from_sync(some_coro(), target_loop)
```

**Dispatch strategy:**
- `target_loop` exists and is running → `run_coroutine_threadsafe(coro, loop)`
- `target_loop` is `None` or closed → `asyncio.run(coro)` as fallback

---

## 6. Middleware Hooks and Async

NexAU's middleware hooks (`before_model`, `after_tool`, etc.) are currently a **sync API**. On the executor's async path, middlewares are run in a thread pool via `asyncio.to_thread()` so they don't block the main event loop:

```python
from nexau.archs.main_sub.execution.hooks import Middleware, HookResult

class MyMiddleware(Middleware):
    def before_model(self, hook_input):
        # Sync implementation — executor's async path bridges via to_thread automatically
        return HookResult.no_changes()
```

> ℹ️ Middleware hooks should not (and need not) be defined as `async def`. The framework handles the async/sync bridging automatically.

---

## 7. Common Anti-patterns and Fixes

### ❌ Anti-pattern 1: Using `Agent()` in an Async Context

```python
# ❌ Wrong — sync construction in an async handler may trigger nested loop
async def handle(request):
    agent = Agent(config=config)  # sync session init → may trigger asyncio.run() nesting
    ...
```

```python
# ✅ Correct — use the async factory
async def handle(request):
    agent = await Agent.create(config=config)
    response = await agent.run_async(message=request.message)
    ...
```

### ❌ Anti-pattern 2: Calling `agent.run()` in an Async Context

```python
# ❌ Wrong — asyncio.run() cannot be called inside a running loop
async def handle(request):
    agent = await Agent.create(config=config)
    response = agent.run(message="hello")  # RuntimeError!
```

```python
# ✅ Correct — use await run_async()
async def handle(request):
    agent = await Agent.create(config=config)
    response = await agent.run_async(message="hello")
```

### ❌ Anti-pattern 3: `nest_asyncio.apply()`

```python
# ❌ Wrong — monkey-patches the running loop, incompatible with uvloop/TaskGroup
import nest_asyncio
nest_asyncio.apply()
asyncio.run(inner())  # nested asyncio.run() inside a running loop
```

```python
# ✅ Correct — await all the way in async contexts
async def outer():
    result = await inner()  # direct await, no nesting
```

### ❌ Anti-pattern 4: Manually Creating Event Loops

```python
# ❌ Wrong — manual loop lifecycle management, prone to leaks
loop = asyncio.new_event_loop()
try:
    result = loop.run_until_complete(some_coro())
finally:
    loop.close()
```

```python
# ✅ Correct — use asyncio.run() in sync contexts
result = asyncio.run(some_coro())

# ✅ Correct — await directly in async contexts
result = await some_coro()
```

### ❌ Anti-pattern 5: `syncify()` / `asyncify()`

```python
# ❌ Wrong — extra thread + BlockingPortal overhead
from asyncer import syncify
result = syncify(async_function)(args)
```

```python
# ✅ Correct — use the standard library
# Sync context:
result = asyncio.run(async_function(args))

# Async context:
result = await async_function(args)

# Sync function in async context:
result = await asyncio.to_thread(sync_function, args)
```

---

## 8. Decision Flowchart

When you're unsure whether to use the sync or async API, follow this decision tree:

```
Is your code running in an async context?
(i.e. inside an async def, or a running event loop exists)
│
├── YES ──→ Create Agent: await Agent.create(...)
│           Run Agent:    await agent.run_async(...)
│           Execute Tool: await tool.execute_async(...)
│
└── NO  ──→ Create Agent: Agent(...)
            Run Agent:    agent.run(...)
            Execute Tool: tool.execute(...)
```

To detect whether you're in an async context:

```python
from nexau.core.utils import get_running_loop_or_none

if get_running_loop_or_none() is not None:
    # async context — use async APIs
    ...
else:
    # sync context — use sync APIs
    ...
```

---

## 9. Summary

| Rule | Description |
|---|---|
| **One loop, await all the way** | All I/O operations in an async context should be `await`ed — never create extra event loops |
| **One-way sync ↔ async bridging** | sync → async via `asyncio.run()`; async → sync via `asyncio.to_thread()` |
| **Never mix the two paths** | Don't call sync APIs (e.g. `agent.run()`) from async code, and don't `await` from sync code |
| **Prefer `Agent.create()`** | In async contexts, always use `await Agent.create()` instead of `Agent()` |
| **Let the executor handle tool dispatch** | Just define your sync/async implementation correctly — the executor picks the optimal path |
| **No `nest_asyncio`** | No monkey-patching, no nested event loops |
| **No `syncify` / `asyncify`** | No third-party async/sync bridging libraries — use the standard `asyncio` module |
