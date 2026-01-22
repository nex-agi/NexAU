# Agent System Implementation Guide

This module contains the core agent orchestration logic for NexAU framework.

## Architecture Overview

The agent system follows a **container pattern**:

```
Agent (Lightweight Container)
    ├── AgentConfig (Configuration)
    ├── Executor (Heavy-lift Orchestrator)
    │   ├── AgentState (Per-execution state)
    │   ├── Middleware Pipeline
    │   ├── Tool Executor
    │   └── LLM Caller
    └── GlobalStorage (Shared state across process)
```

## Key Components

### Agent (`agent.py`)

Lightweight container that delegates to Executor.

**Key Methods**:

```python
class Agent:
    def __init__(
        self,
        config: AgentConfig,
        session_manager: SessionManager | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        global_storage: GlobalStorage | None = None,
    ):
        """Initialize agent with optional session management."""

    def run(
        self,
        message: str | list[Message],
        context: dict[str, Any] | None = None,
        event_handlers: list[Callable[[Event], None]] | None = None,
    ) -> str:
        """Run agent synchronously."""

    async def run_async(
        self,
        message: str | list[Message],
        context: dict[str, Any] | None = None,
        event_handlers: list[Callable[[Event], None]] | None = None,
    ) -> str:
        """Run agent asynchronously."""
```

**Session Management**:

When `session_manager` is provided, Agent automatically:
1. Creates session if `session_id` not provided
2. Registers itself with `session_manager`
3. Loads history from previous runs
4. Saves history after each run

### Executor (`execution/executor.py`)

Heavy-lift orchestrator managing the agent execution loop.

**Execution Flow**:

```python
# 1. Pre-execution middleware
before_agent_hook(hook_input)

# 2. Iterative loop (max_iterations)
for iteration in range(max_iterations):
    # Token budget check
    if exceed_token_limit:
        context_compaction_middleware.compact()

    # LLM call
    response = llm_caller.call(messages, llm_config)

    # Parse response
    tool_calls = response_parser.parse_tool_calls(response)

    # Parallel tool execution
    tool_results = tool_executor.execute_parallel(tool_calls)

    # Update history
    history.extend([assistant_message, tool_result_messages])

# 3. Post-execution middleware
after_agent_hook(hook_input)
```

**Key Methods**:

```python
class Executor:
    def execute(
        self,
        config: AgentConfig,
        messages: HistoryList,
        agent_state: AgentState,
        global_storage: GlobalStorage,
    ) -> ExecutorOutput:
        """Execute agent and return output."""
```

### AgentState (`agent_state.py`)

Per-execution state container with context, history, and global storage access.

**Key Attributes**:

```python
class AgentState:
    # Identification
    agent_id: str
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    agent_name: str

    # State
    current_iteration: int
    max_iterations: int

    # References
    history: HistoryList
    global_storage: GlobalStorage
```

**Context Propagation**:

- Sub-agents inherit `parent_run_id` and `root_run_id`
- `global_storage` is shared across entire agent hierarchy
- `history` is per-agent (not shared)

### Middleware System (`execution/middleware/`)

Pluggable pipeline for cross-cutting concerns.

**Hook Points** (defined in `execution/hooks.py`):

```python
class HookInput:
    agent_state: AgentState
    messages: list[Message]
    global_storage: GlobalStorage

# Available hooks
def before_agent(hook_input: BeforeAgentHookInput) -> HookResult:
    """Called before agent execution starts."""

def after_agent(hook_input: AfterAgentHookInput) -> HookResult:
    """Called after agent execution finishes."""

def before_model(hook_input: BeforeModelHookInput) -> HookResult:
    """Called before LLM API call."""

def after_model(hook_input: AfterModelHookInput) -> HookResult:
    """Called after LLM API call returns."""

def before_tool(hook_input: BeforeToolHookInput) -> HookResult:
    """Called before tool execution."""

def after_tool(hook_input: AfterToolHookInput) -> HookResult:
    """Called after tool execution finishes."""

def wrap_model_call(
    hook_input: ModelCallParams,
    func: Callable,
) -> ChatCompletionChunk | ResponseStreamEvent | MessageStreamEvent:
    """Wraps LLM call for streaming support."""

def stream_chunk(
    chunk: ChatCompletionChunk | ResponseStreamEvent | MessageStreamEvent,
    params: ModelCallParams,
) -> ChatCompletionChunk | ResponseStreamEvent | MessageStreamEvent:
    """Called for each streaming chunk."""
```

### AgentEventsMiddleware (`execution/middleware/agent_events_middleware.py`)

Bridges LLM aggregators with agent execution.

**Purpose**:

- Connects `llm_aggregators` layer (raw stream chunks) to agent execution layer (unified Event objects)
- Provides `on_event` callback that receives unified Event objects

**Usage**:

```python
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)
from nexau.archs.llm.llm_aggregators import Event

def handle_event(event: Event):
    """Handle unified streaming events"""
    if event.type == "TEXT_MESSAGE_CONTENT":
        print(event.delta, end="")

middleware = AgentEventsMiddleware(
    session_id="sess_123",
    on_event=handle_event,
)
```

**Event Types**:

The middleware emits events from `nexau.archs.llm.llm_aggregators.events`:
- `TextMessageStartEvent`, `TextMessageContentEvent`, `TextMessageEndEvent`
- `ToolCallStartEvent`, `ToolCallArgsEvent`, `ToolCallEndEvent`
- `ToolCallResultEvent`
- `ThinkingTextMessageStartEvent`, `ThinkingTextMessageContentEvent`, `ThinkingTextMessageEndEvent`
- `ImageMessageStartEvent`, `ImageMessageContentEvent`, `ImageMessageEndEvent`
- `RunStartedEvent`, `RunFinishedEvent`, `RunErrorEvent`

### ContextCompactionMiddleware (`execution/middleware/context_compaction/`)

Manages conversation context when token limits approached.

**Strategies**:

```python
from nexau.archs.main_sub.execution.middleware.context_compaction.config import (
    CompactionStrategy,
)

class CompactionStrategy(str, Enum):
    SLIDING_WINDOW = "sliding_window"  # Keep last N rounds
    TOOL_RESULT_COMPACT = "tool_result_compaction"  # Summarize tool results
    CUSTOM = "custom"  # Use custom function
```

**Configuration**:

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.context_compaction:ContextCompactionMiddleware
    params:
      max_context_tokens: 200000
      auto_compact: true
      threshold: 0.75
      compaction_strategy: "sliding_window"
      window_size: 2
```

### Sub-Agent System (`execution/subagent_manager.py`)

Hierarchical delegation: Agents can call sub-agents forming a task tree.

**Features**:

- Parallel execution via ThreadPoolExecutor (`max_running_subagents`)
- Context inheritance (parent → child)
- Shared global storage
- Traced as SUB_AGENT span
- Runtime registration support
- **Streaming support**: Sub-agents emit text/tool_call events

**Calling Sub-Agents**:

```python
# In agent code or tools
from nexau.archs.main_sub.execution.subagent_manager import call_sub_agent

result = await call_sub_agent(
    agent_name="specialist_agent",
    message="Task to delegate",
    agent_state=agent_state,
    global_storage=global_storage,
)
```

**Sub-Agent Registration**:

```yaml
# In parent agent config
sub_agents:
  specialist_agent:
    name: "specialist"
    system_prompt: "You are a specialist..."
    tools: [...]
```

### HistoryList (`history_list.py`)

A list that automatically persists modifications to SessionManager.

**Purpose**:

- Transparent persistence for agent history
- Run-level action tracking (APPEND/UNDO/REPLACE)
- Maintains backward compatibility with `list[Message]`

**Usage**:

```python
from nexau.archs.main_sub.history_list import HistoryList

# Create history with persistence
history = HistoryList(
    messages=[msg1, msg2],
    session_manager=session_manager,
    history_key=action_key,
    run_id=run_id,
    root_run_id=root_run_id,
    agent_name="my_agent",
)

# Normal list operations persist automatically
history.append(new_message)

# Update context for new run
history.update_context(run_id=new_run_id, root_run_id=new_root_id)

# Flush at end of run
history.flush()
```

**Important Notes**:

- `history.append()` and `history.extend()` persist automatically
- `history[index] = value` only updates locally (no persistence)
- Use `history.replace_all(new_messages)` for true replacement operations
- `history.flush()` should be called at end of each run

## Key Patterns

### Agent Initialization Pattern

```python
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.session import SessionManager

# Create global storage
global_storage = GlobalStorage()

# Create agent
agent = Agent(
    config=agent_config,
    session_manager=session_manager,
    user_id="user_123",
    session_id="sess_456",
    global_storage=global_storage,
)
```

### Middleware Registration Pattern

```python
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)

# Add middleware to agent config
agent_config.middlewares = [
    AgentEventsMiddleware(
        session_id="sess_123",
        on_event=handle_event,
    ),
    ContextCompactionMiddleware(
        max_context_tokens=200000,
        auto_compact=True,
    ),
]
```

### State Access Pattern

```python
from nexau.archs.main_sub.agent_context import get_context

def my_tool(param1: str, agent_state: AgentState):
    # Access current execution state
    current_iteration = agent_state.current_iteration
    global_storage = agent_state.global_storage
    run_id = agent_state.run_id

    # Access context via contextvar
    agent_context = get_context()
    print(f"Agent ID: {agent_context.agent_id}")
```

## Common Issues

### Agent Not Found

**Error**: `SubAgentNotFoundError: sub_agent 'xxx' not found`

**Solution**: Verify agent name matches config key (not the agent's `name` field):

```yaml
# Correct: Use config key
sub_agents:
  research_assistant:  # ← This is the key to use
    name: "Research Agent"  # ← This is just a display name
    ...
```

### History Not Persisting

**Error**: History changes not saved to session

**Solution**: Ensure `history.flush()` is called after each run. Agent.run() handles this automatically when `session_manager` is provided.

### Sub-Agent Streaming Not Working

**Error**: Sub-agent events not emitted

**Solution**: Ensure `AgentEventsMiddleware` is added to agent config with streaming enabled:

```python
middleware = AgentEventsMiddleware(
    session_id=session_id,
    on_event=event_handler,
)
config_with_middlewares = TransportBase._recursively_apply_middlewares(
    agent_config,
    middleware,
    enable_stream=True,  # ← Important
)
```
