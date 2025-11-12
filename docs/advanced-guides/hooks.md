## Middleware Hooks

NexAU no longer exposes separate `before_model_hooks`, `after_model_hooks`, or `after_tool_hooks`. Instead, the agent runtime is driven entirely by **middlewares**—Python objects that can plug into every phase of the loop.

### Middleware Interface

A middleware can implement any of these optional methods:

- `before_model(hook_input)` – inspect/modify the message list before the LLM call.
- `after_model(hook_input)` – inspect/modify parsed responses and conversation state after the LLM call.
- `before_tool(hook_input)` – adjust tool inputs (or cancel calls) right before execution.
- `after_tool(hook_input)` – inspect/modify tool outputs before they are fed back into the loop.
- `wrap_model_call(call_next)` – wrap the low-level LLM invocation (for custom providers, retries, tracing, etc.).
- `wrap_tool_call(call_next)` – wrap each tool execution.

Execution order is deterministic:

- `before_model` / `before_tool`: first → last
- `after_model` / `after_tool`: last → first
- `wrap_*`: nested, so the first middleware wraps everything else (outermost wins)

### Minimal Example

```python
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware

class AuditMiddleware(Middleware):
    def after_model(self, hook_input):
        print("Model emitted", len(hook_input.parsed_response.tool_calls or []), "tool calls")
        return HookResult.no_changes()

    def after_tool(self, hook_input):
        print("Tool", hook_input.tool_name, "returned", hook_input.tool_output)
        return HookResult.no_changes()

    def wrap_model_call(self, call_next):
        def wrapped(params):
            print("Calling LLM with", len(params.messages), "messages")
            return call_next(params)
        return wrapped
```

### What Can a Middleware Change?

A middleware returns a `HookResult` describing any modifications. Common patterns:

- **Conversation** – supply `messages=[...]` to rewrite the next prompt (add reminders, system notes, or scratchpad content).
- **Parsed response** – supply `parsed_response=...` to add/remove tool calls, toggle parallelism flags, or set `force_continue=True` to keep iterating without new calls.
- **Tool input** – via `before_tool`, return `tool_input=...` to tweak parameters (add defaults, redact secrets) before the tool runs.
- **Tool output** – supply `tool_output=...` to redact/reshape a tool result before it flows back into the conversation.
- **Agent state** – `hook_input.agent_state` is mutable; you can stash counters, feature flags, or tracing IDs for later iterations.

Returning a `HookResult` makes the intent explicit and lets later middlewares build on your changes without guessing what mutated in-place.

### Examples

```python
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware

class PrefixMiddleware(Middleware):
    def before_model(self, hook_input):
        updated = hook_input.messages + [{
            "role": "system",
            "content": "Reminder: stay within budget.",
        }]
        return HookResult.with_modifications(messages=updated)

class ToolFilter(Middleware):
    def after_model(self, hook_input):
        parsed = hook_input.parsed_response
        if not parsed:
            return HookResult.no_changes()
        parsed.tool_calls = [call for call in parsed.tool_calls if call.tool_name != "system_command"]
        return HookResult.with_modifications(parsed_response=parsed)

class JsonToolNormalizer(Middleware):
    def after_tool(self, hook_input):
        if isinstance(hook_input.tool_output, str):
            return HookResult.with_modifications(tool_output={"text": hook_input.tool_output})
        return HookResult.no_changes()

class ClampInputMiddleware(Middleware):
    def before_tool(self, hook_input):
        updated = dict(hook_input.tool_input)
        updated.setdefault("timeout", 30)
        return HookResult.with_modifications(tool_input=updated)
```

```

### Working with Agent State

`hook_input.agent_state` exposes the live `AgentState` instance. You can read/write custom fields (e.g. `agent_state.context.storage['metrics'] = ...`) to persist values across iterations. Because agent state is shared by every middleware, prefer namespaced keys or dataclasses to avoid collisions.

### Wiring Middlewares

Middlewares are registered through the `middlewares` field on your agent configuration (YAML or code). Example YAML snippet:

```yaml
middlewares:
  - import: my_project.middleware:AuditMiddleware
    params:
      log_file: "/tmp/audit.log"
```

When building agents programmatically, pass actual middleware instances to `create_agent(..., middlewares=[...])`.


## Customizing LLM Calls via Middleware

The former `custom_llm_generator` hook chain has been retired. To customize how NexAU talks to an LLM (swap providers, add caching, manipulate parameters, etc.) you now implement the `wrap_model_call` method on a middleware.

### Why Middleware?

- Works alongside other `before_model` / `after_model` logic.
- Fully nested: the first middleware in your list can wrap every downstream call.
- No special config keys or bespoke plumbing.

### Example: Provider Switch + Metrics

```python
from nexau.archs.main_sub.execution.hooks import Middleware, ModelCallParams
from nexau.archs.main_sub.execution.model_response import ModelResponse

class ProviderSwitchMiddleware(Middleware):
    def __init__(self, fallback_client):
        self.fallback_client = fallback_client

    def wrap_model_call(self, call_next):
        def wrapped(params: ModelCallParams) -> ModelResponse | None:
            # Try the default client first
            try:
                return call_next(params)
            except Exception as primary_error:
                print("Primary client failed, falling back:", primary_error)

            # Fallback path – call a completely different provider
            response = self._call_fallback(params)
            print("Fallback response preview:", (response.content or "")[:200])
            return response

        return wrapped

    def _call_fallback(self, params: ModelCallParams) -> ModelResponse:
        raw = self.fallback_client.chat.completions.create(
            model="custom-fallback",
            messages=params.messages,
            max_tokens=params.max_tokens,
        )
        return ModelResponse.from_openai_message(raw.choices[0].message)
```

Register the middleware as usual:

```yaml
middlewares:
  - import: my_project.middlewares:ProviderSwitchMiddleware
    params:
      fallback_client: !python/object:my_project.clients:FallbackClient {}
```


### Built-in Middleware

- `LoggingMiddleware`: replaces the old logging hooks and supports both after-model/after-tool logging as well as wrapping model calls to trace custom generators.

You can combine built-in middleware with your own; the manager guarantees the ordering rules described above.
