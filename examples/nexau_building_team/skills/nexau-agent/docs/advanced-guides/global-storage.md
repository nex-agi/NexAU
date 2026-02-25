
### Global Storage System

The NexAU framework provides a thread-safe GlobalStorage system that allows tools and hooks to share data across the entire agent hierarchy. This is particularly useful for maintaining state across multiple tool calls and sub-agents.

#### Using GlobalStorage in Tools

Tools can optionally receive a `global_storage` parameter by including it in their function signature:

```python
from nexau.archs.main_sub.agent_context import GlobalStorage

def my_tool_with_storage(param1: str, global_storage: GlobalStorage) -> dict:
    """Tool that uses global storage for persistent data."""

    # Get values from global storage
    user_count = global_storage.get("user_count", 0)
    session_data = global_storage.get("session_data", {})

    # Update values
    global_storage.set("user_count", user_count + 1)
    global_storage.update({
        "last_tool_used": "my_tool_with_storage",
        "last_param": param1
    })

    # Use key-specific locking for concurrent access
    with global_storage.lock_key("counter"):
        current = global_storage.get("counter", 0)
        global_storage.set("counter", current + 1)

    # Lock multiple keys at once
    with global_storage.lock_multiple("key1", "key2"):
        # Safely access key1 and key2 exclusively
        val1 = global_storage.get("key1", 0)
        val2 = global_storage.get("key2", 0)
        global_storage.set("key1", val1 + 1)
        global_storage.set("key2", val2 + 1)

    return {"result": f"Processed {param1}, total users: {user_count + 1}"}

def my_tool_without_storage(param1: str) -> dict:
    """Tool that doesn't use global storage - framework handles this automatically."""
    return {"result": f"Processed {param1}"}
```

**Important**: Tools that don't need global storage don't need to include it in their function signature. The framework automatically filters out the `global_storage` parameter for tools that don't expect it.

#### Using GlobalStorage in Hooks

Hooks can access global storage through the agent context:

```python
from nexau.archs.main_sub.execution.hooks import AfterModelHook, AfterModelHookResult, AfterModelHookInput
from nexau.archs.main_sub.agent_context import get_context

def create_storage_hook() -> AfterModelHook:
    def storage_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        # Access global storage through context
        ctx = get_context()
        if ctx and hasattr(ctx, 'global_storage'):
            storage = ctx.global_storage

            # Track hook executions
            hook_count = storage.get("hook_executions", 0)
            storage.set("hook_executions", hook_count + 1)

            # Store recent tool calls for analysis
            if hook_input.parsed_response and hook_input.parsed_response.tool_calls:
                recent_tools = storage.get("recent_tool_calls", [])
                recent_tools.extend([call.tool_name for call in hook_input.parsed_response.tool_calls])
                # Keep only last 10 tool calls
                storage.set("recent_tool_calls", recent_tools[-10:])

                # Add context message with tool usage stats
                modified_messages = hook_input.messages.copy()
                total_tools = len(recent_tools)
                unique_tools = len(set(recent_tools))
                modified_messages.append({
                    "role": "system",
                    "content": f"[STATS] Hook #{hook_count + 1}: {total_tools} total tools used, {unique_tools} unique"
                })

                return AfterModelHookResult.with_modifications(messages=modified_messages)

        return AfterModelHookResult.no_changes()

    return storage_hook
```

#### GlobalStorage API

The GlobalStorage class provides thread-safe operations:

```python
# Basic operations
storage.set(key: str, value: Any)                    # Set a value
storage.get(key: str, default: Any = None) -> Any    # Get a value
storage.update(updates: Dict[str, Any])              # Update multiple values
storage.delete(key: str) -> bool                     # Delete a key
storage.keys() -> List[str]                          # Get all keys
storage.items() -> List[Tuple[str, Any]]             # Get all items
storage.clear()                                      # Clear all data

# Thread-safe locking
with storage.lock_key("my_key"):                     # Lock single key
    # Exclusive access to "my_key"
    value = storage.get("my_key", 0)
    storage.set("my_key", value + 1)

with storage.lock_multiple("key1", "key2", "key3"):  # Lock multiple keys
    # Exclusive access to all specified keys
    # Keys are sorted to prevent deadlocks
    pass
```

#### Practical Example: Session Analytics Tool

```python
from nexau.archs.main_sub.agent_context import GlobalStorage
from datetime import datetime

def session_analytics(action: str, data: dict = None, global_storage: GlobalStorage = None) -> dict:
    """Advanced session analytics using global storage."""

    if action == "start_session":
        session_id = f"session_{datetime.now().timestamp()}"

        # Use locking for session initialization
        with global_storage.lock_key("session_counter"):
            session_num = global_storage.get("session_counter", 0) + 1
            global_storage.set("session_counter", session_num)

        global_storage.update({
            "current_session": session_id,
            "session_start": datetime.now().isoformat(),
            "session_number": session_num,
            f"{session_id}_events": [],
            f"{session_id}_tools_used": {},
            f"{session_id}_errors": []
        })

        return {"session_id": session_id, "session_number": session_num}

    elif action == "log_event":
        session_id = global_storage.get("current_session")
        if not session_id:
            return {"error": "No active session"}

        # Thread-safe event logging
        with global_storage.lock_key(f"{session_id}_events"):
            events = global_storage.get(f"{session_id}_events", [])
            events.append({
                "timestamp": datetime.now().isoformat(),
                "event_type": data.get("type"),
                "details": data.get("details", {})
            })
            global_storage.set(f"{session_id}_events", events)

        return {"result": f"Logged event for session {session_id}"}

    elif action == "get_analytics":
        session_id = global_storage.get("current_session", "no-session")

        # Safely read analytics data
        with global_storage.lock_multiple(
            f"{session_id}_events",
            f"{session_id}_tools_used",
            f"{session_id}_errors"
        ):
            events = global_storage.get(f"{session_id}_events", [])
            tools = global_storage.get(f"{session_id}_tools_used", {})
            errors = global_storage.get(f"{session_id}_errors", [])

        return {
            "session_id": session_id,
            "events_count": len(events),
            "unique_tools": len(tools),
            "total_tool_calls": sum(tools.values()),
            "errors_count": len(errors),
            "session_start": global_storage.get("session_start", "unknown")
        }

    return {"error": f"Unknown action: {action}"}
```

#### Best Practices

1. **Thread Safety**: Always use locking when modifying shared data in concurrent scenarios
2. **Key Naming**: Use descriptive, hierarchical key names (e.g., `"session_123_events"`)
3. **Cleanup**: Clear unused data to prevent memory leaks in long-running agents
4. **Optional Usage**: Tools work automatically whether they use global storage or not
5. **Error Handling**: Check for storage availability in hooks before accessing

The global storage system enables powerful coordination between tools and provides persistent state management across the entire agent execution lifecycle.
