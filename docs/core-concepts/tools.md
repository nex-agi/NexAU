# 🛠️ Core Concepts: Tools

Tools are the functions that an agent can execute to interact with the outside world, such as searching the web, reading files, or running code.

## Built-in Tools

NexAU comes with a variety of pre-built tools for common tasks.

#### File Tools (`nexau.archs.tool.builtin.file_tools`)
- **read_file**: Read text files (with pagination). For images/videos, use `read_visual_file`.
- **read_visual_file**: Read image and video files for multimodal LLMs. Requires a model with vision support.
- **write_file**, **replace**, **apply_patch**: Write and edit files.
- **glob**, **list_directory**, **read_many_files**, **search_file_content**: Find and search files.

#### Web Tools (`nexau.archs.tool.builtin.web_tools`)
- **google_web_search**: Search the web.
- **web_fetch**: Fetch and parse content from a URL.

#### Shell Tools (`nexau.archs.tool.builtin.shell_tools`)
- **run_shell_command**: Execute shell commands.

#### Session Tools (`nexau.archs.tool.builtin.session_tools`)
- **write_todos**, **complete_task**, **save_memory**, **ask_user**: Task management, persistence, user interaction.

#### Tool Search (`nexau.archs.tool.builtin.tool_search`)
- **ToolSearch**: Search and inject deferred tools on demand. Automatically registered when any tool has `defer_loading: true`.

### How to use Use these built-in tools
If you use python to create Agent:

```python
from nexau import Agent, AgentConfig, Tool
from nexau.archs.tool.builtin.session_tools import write_todos
from nexau.archs.tool.builtin.web_tools import google_web_search, web_fetch

# Create the tool instance
# Note: you need to create the yaml files for tool config, you may refer to examples/deep_research/tools for examples

web_search_tool = Tool.from_yaml(
    str(script_dir / 'tools/WebSearch.yaml'),
    binding=google_web_search,
)
web_read_tool = Tool.from_yaml(
    str(script_dir / 'tools/WebRead.yaml'),
    binding=web_fetch,
)
todo_write_tool = Tool.from_yaml(
    str(script_dir / 'tools/TodoWrite.tool.yaml'),
    binding=write_todos,
)

# Add it to your agent's tool list
agent_config = AgentConfig(
    name="web_agent",
    tools=[web_search_tool, web_read_tool, todo_write_tool],
    # ... other agent config (llm_config, system_prompt, etc.)
)
agent = Agent(config=agent_config)
```

If you use agent yaml config, you can add these lines to the config file:

```yaml
tools:
  - name: web_search
    yaml_path: ./tools/WebSearch.yaml
    binding: nexau.archs.tool.builtin.web_tools:google_web_search
  - name: web_read
    yaml_path: ./tools/WebRead.yaml
    binding: nexau.archs.tool.builtin.web_tools:web_fetch
  - name: todo_write
    yaml_path: ./tools/TodoWrite.tool.yaml
    binding: nexau.archs.tool.builtin.session_tools:write_todos
```

## Creating Custom Tools

You can easily extend an agent's capabilities by creating your own custom tools.

### extra_kwargs (preset parameters)
- Use `Tool.from_yaml(..., extra_kwargs=...)` or a YAML `extra_kwargs` block to preset fixed arguments (e.g., `base_url`/`api_key`/`model`) so callers can omit them.
- Call-time arguments with the same name override preset values. Reserved keys `agent_state` and `global_storage` are not allowed in `extra_kwargs`.
- Extra fields are not blocked by schema validation and are passed to the tool function; if the function signature does not accept them, a `TypeError` will be raised. To reject unknown fields up front, add `additionalProperties: false` in the tool’s `input_schema`.

**Step 1: Define the Tool's Python Function**

Create a standard Python function with type hints. The docstring will be used to inform the agent how the tool works.

```python
# my_tools/calculator.py

def simple_calculator(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.
    Supports addition (+), subtraction (-), multiplication (*), and division (/).

    Args:
        expression: The mathematical expression to evaluate (e.g., "10 + 5*2").

    Returns:
        The result of the calculation as a string, or an error message.
    """
    try:
        # Note: Using eval() is unsafe for untrusted input. This is for demonstration.
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

**Step 2: Create a YAML Configuration**

The YAML file defines the tool's name, description, and input schema for the agent.

**File: `tools/SimpleCalculator.tool.yaml`**
```yaml
type: tool
name: SimpleCalculator
description: >-
  A tool to evaluate simple mathematical expressions like "10 + 5*2".
  It supports addition, subtraction, multiplication, and division.

input_schema:
  type: object
  properties:
    expression:
      type: string
      description: The mathematical string to evaluate.
  required:
    - expression
  additionalProperties: false
  $schema: [http://json-schema.org/draft-07/schema#](http://json-schema.org/draft-07/schema#)
```

**Step 3: Use the Tool in Your Agent**

Load the tool from its YAML file and bind it to the Python function you created.

```python
from nexau import Agent, AgentConfig, Tool
from my_tools.calculator import simple_calculator

# Create the tool instance
calculator_tool = Tool.from_yaml(
    "tools/SimpleCalculator.tool.yaml",
    binding=simple_calculator
)

# Add it to your agent's tool list
agent_config = AgentConfig(
    name="calculator_agent",
    tools=[calculator_tool],
    # ... other agent config (llm_config, system_prompt, etc.)
)
agent = Agent(config=agent_config)
```

If you use agent yaml config, you can add these lines to the config file:

```yaml
tools:
  - name: simple_calculator
    yaml_path: ./tools/SimpleCalculator.tool.yaml
    binding: my_tools.calculator.simple_calculator:simple_calculator
```

## External Tools (Caller-Executed)

External tools let the **caller**, not the agent, execute tool implementations. NexAU
only registers the tool schema with the LLM; when the model calls an external tool,
the agent loop pauses and hands the pending call back to the caller. The caller runs
the tool (in a plugin, IDE, sandbox, or another process) and feeds the result back to
resume the loop.

Typical use cases:

- **Remote tool execution** — IDE plugins or desktop apps that own tool implementations.
- **Cross-language integration** — tool implementations live in Rust/TypeScript/etc.
- **Security isolation** — file-system or code-execution tools that must run inside the
  caller's sandbox.
- **LLM API pass-through** — mirror OpenAI/Anthropic-style tool-use where the client
  executes tool calls.

See [RFC-0018](../rfcs/0018-external-tool.md) for the full design.

### Declaring an External Tool

Declare a tool with `kind: external` and **no** `binding`:

```yaml
type: tool
name: read_file
kind: external
description: Reads the content of a text file.
input_schema:
  type: object
  properties:
    file_path:
      type: string
  required: [file_path]
  additionalProperties: false
```

Python API:

```python
from nexau import Tool

# From YAML (no binding argument)
tool = Tool.from_yaml("tools/read_file.tool.yaml")

# Or inline
tool = Tool(
    name="read_file",
    description="Reads the content of a text file.",
    input_schema={...},
    implementation=None,
    kind="external",
)

assert tool.is_external
```

Passing a `binding` to an external tool raises `ConfigError`.

### Wiring into an Agent

In agent YAML, list the tool like any other — just omit `binding` (the `kind: external`
declaration lives in the tool YAML itself):

```yaml
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    # no binding — tool YAML declares kind: external

  - name: write_file
    yaml_path: ./tools/write_file.tool.yaml
```

If a `binding` is provided for a tool declared `kind: external`, loading fails with
`ConfigError`.

In Python, build the `Tool` without a binding and pass it to `AgentConfig.tools`
alongside regular tools:

```python
from nexau import Agent, AgentConfig, Tool

tools = [
    Tool.from_yaml("tools/read_file.tool.yaml"),   # external (kind: external in YAML)
    Tool.from_yaml("tools/search.tool.yaml", binding=my_search),  # normal local tool
]

agent = Agent(config=AgentConfig(name="mixed_agent", tools=tools, ...))
```

A single agent can mix local and external tools freely — the executor runs local ones
in-loop and pauses only for external ones.

### Pause / Resume Flow

When the LLM calls an external tool, `agent.run_async()` returns a **tuple** instead
of a bare string:

```python
result = await agent.run_async(message="read /tmp/foo.txt")

if isinstance(result, str):
    final_answer = result  # Normal completion
else:
    response_text, meta = result
    # meta = {
    #   "stop_reason": "EXTERNAL_TOOL_CALL",
    #   "pending_tool_calls": [
    #     {"id": "call_abc", "name": "read_file", "input": {"file_path": "/tmp/foo.txt"}},
    #   ],
    #   "trace_id": "…",
    # }
```

Mixed turns (local + external calls) execute local tools first, persist their results
to history, then pause for the external ones.

To resume, send a `Role.TOOL` message containing one `ToolResultBlock` per pending
call (use the same `tool_use_id`):

```python
from nexau.core.messages import Message, Role, ToolResultBlock

resume = await agent.run_async(
    message=[
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call_abc",
                    content="file contents here",
                    is_error=False,
                ),
            ],
        ),
    ],
)
```

The same `session_id` carries history and `trace_id` across the pause — no extra state
to thread through. Loop until `run_async()` returns a bare string.

### Over HTTP

`AgentResponse` gains three optional fields used only when the agent is paused:

| Field | Type | Meaning |
|-------|------|---------|
| `stop_reason` | `str \| None` | `"EXTERNAL_TOOL_CALL"` when paused |
| `pending_tool_calls` | `list[dict] \| None` | Each item has `id`, `name`, `input` |
| `trace_id` | `str \| None` | Session-level trace id (read-only observability) |

Resume through the **same `/query` endpoint** — no new endpoint. Reuse the same
`user_id` and `session_id`; put the `ToolResultBlock` into `messages`:

```json
{
  "messages": [{
    "role": "tool",
    "content": [{
      "type": "tool_result",
      "tool_use_id": "call_abc",
      "content": "file contents here",
      "is_error": false
    }]
  }],
  "user_id": "user123",
  "session_id": "sess_abc"
}
```

### Full Example

A working end-to-end example (every tool declared as external, driven by a pause/resume
loop in the host script) lives at
[`examples/code_agent_external_tool/`](../../examples/code_agent_external_tool/).

## Tool Return Values: `returnDisplay`

When a tool returns a dictionary, you can include a `returnDisplay` field to provide a short, human-readable summary for the frontend UI. The framework automatically strips `returnDisplay` before sending the result to the LLM, so it won't consume extra tokens.

```python
def my_search_tool(query: str, agent_state=None) -> dict:
    results = perform_search(query)
    return {
        "content": json.dumps(results),                    # Sent to LLM
        "returnDisplay": f"Found {len(results)} results",  # Shown in frontend only
    }
```

**How it works:**
- `content` — the full tool output, forwarded to the LLM as the tool result.
- `returnDisplay` — a concise summary streamed to the frontend for display, then stripped before the LLM sees it.

This is the same pattern used by all NexAU built-in tools (file tools, web tools, shell tools, etc.). Use it in your custom tools whenever the full output is verbose but you want a clean one-liner in the UI.

## Tool Output Formatters

Besides `returnDisplay`, tools can also control the **model-facing** result via a formatter.

- The tool implementation returns the raw structured value.
- The formatter converts that raw value into the payload the LLM will actually see.
- `returnDisplay` remains frontend-only and is stripped before the model sees the result.

If no formatter is configured, NexAU uses the built-in `xml` formatter.

Typical reasons to add a custom formatter:

- flatten verbose runtime metadata into cleaner prompt text
- preserve raw structured output for middleware or tracing while giving the model a simpler summary
- produce tool-specific output formats, such as shell-style command summaries

You can configure a formatter in YAML:

```yaml
type: tool
name: my_tool
description: Example tool
formatter: xml
input_schema:
  type: object
  properties: {}
```

Or use an import path for a custom formatter:

```yaml
formatter: my_project.tool_formatters:format_my_tool_output
```

See [Tool Output Formatters](../advanced-guides/tool-formatters.md) for the execution order, middleware interaction, and custom formatter examples.

## Tool Output Truncation

Tools like file search or code analysis can produce very large outputs that waste LLM context tokens. NexAU provides two complementary truncation mechanisms:

### 1. Sandbox-level truncation (bash commands)

The `execute_bash` command automatically redirects stdout/stderr to temporary files and truncates the output when it exceeds a configurable threshold. This is handled at the sandbox level — see [Sandbox — Bash Output Truncation](../advanced-guides/sandbox.md#bash-output-truncation) for configuration details.

### 2. Middleware-level truncation (all tools)

For tools that don't handle their own truncation, `LongToolOutputMiddleware` can be added to the agent's middleware stack. It truncates any tool output exceeding `max_output_chars`, saves the full content to a temp file, and provides a hint so the model can read the full output if needed.

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.long_tool_output:LongToolOutputMiddleware
    params:
      max_output_chars: 10000
      head_lines: 50
      tail_lines: 30
      bypass_tool_names:
        - execute_bash  # Already truncated at sandbox level
```

See [Middleware Hooks — LongToolOutputMiddleware](../advanced-guides/hooks.md#longtooloutputmiddleware) for full configuration reference.

## Deferred Tool Loading

When an agent has many tools, sending all tool schemas to the LLM every turn wastes context tokens. **Deferred loading** (`defer_loading: true`) keeps a tool's schema out of the LLM context until the LLM explicitly searches for it via the built-in `ToolSearch` tool.

### How it works

1. Tools with `defer_loading: true` are registered but **not** included in the LLM's tool list
2. A built-in `ToolSearch` tool is automatically added to the agent
3. When the LLM calls `ToolSearch`, matched tools are **injected** into the tool list
4. From the next turn onwards, the LLM can call the injected tools directly

### YAML Configuration

```yaml
type: tool
name: SlackSendMessage
description: "Send a message to a Slack channel"
defer_loading: true        # Not sent to LLM until searched
search_hint: "slack chat"  # Optional: improves search relevance

input_schema:
  type: object
  properties:
    channel:
      type: string
    message:
      type: string
  required: [channel, message]
  additionalProperties: false
```

### Agent YAML Configuration

```yaml
tools:
  - name: read_file
    yaml_path: ./tools/ReadFile.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
    # No defer_loading → always available (eager)

  - name: slack_send
    yaml_path: ./tools/SlackSendMessage.yaml
    binding: my_tools.slack:send_message
    defer_loading: true   # Only available after ToolSearch
    search_hint: "slack chat messaging"
```

### When to use

- **Many optional tools** — MCP integrations, specialized APIs that the LLM rarely needs
- **Large tool schemas** — Tools with complex input schemas that consume many tokens
- **Conditional capabilities** — Tools that only make sense in certain conversation contexts

### `defer_loading` vs `lazy` vs `as_skill`

| Attribute | What it defers | When loaded |
|-----------|---------------|-------------|
| `defer_loading` | Tool **schema** from LLM context | When LLM calls `ToolSearch` |
| `lazy` | Python **import** of the binding | On first tool execution |
| `as_skill` | Tool from direct LLM access | When LLM calls the Skill tool |

## Lazy loading long tool descriptions via Skills

When a tool is marked with `as_skill: true`, NexAU exposes it progressively:
- In `xml` mode, the prompt only includes the brief `skill_description`, and the agent uses `LoadSkill` to fetch the full detail.
- In `structured` mode, the model gets the brief `skill_description` plus the JSON Schema up front, then calls `LoadSkill` only if it needs the full description.

Refer to [Skills](../advanced-guides/skills.md) for details about the Skill mechanism.
