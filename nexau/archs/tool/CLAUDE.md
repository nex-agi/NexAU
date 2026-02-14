# Tool System Implementation Guide

This module contains tool definition, execution, and binding logic for NexAU framework.

## Architecture Overview

The Tool system uses a **YAML-first** approach with JSON Schema validation:

```
Tool Definition (YAML + JSON Schema)
    ├── Tool Binding (Python function)
    ├── Tool Executor (Execution with hooks)
    └── Tool Adapter (Protocol conversion)
```

## Key Components

### Tool (`tool.py`)

Main tool class that combines YAML definition with Python binding.

**Initialization**:

```python
from nexau.archs.tool.tool import Tool

tool = Tool.from_yaml(
    yaml_path="tools/MyTool.yaml",
    binding=my_tool_function,
)
```

**Key Methods**:

```python
class Tool:
    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        binding: Callable,
    ) -> Tool:
        """Load tool from YAML file and bind to function."""

    @staticmethod
    def from_dict(
        tool_dict: dict[str, Any],
        binding: Callable,
    ) -> Tool:
        """Create tool from dictionary."""

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""

    def to_anthropic(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce arguments using JSON Schema."""
```

### Tool Definition Format

**YAML Structure**:

```yaml
type: tool
name: tool_name
description: >-
  Multi-line description of what the tool does.

skill_description: "Optional context for skills"

use_cache: false          # Enable result caching (diskcache backend)
disable_parallel: false   # Prevent parallel execution
lazy: false               # Defer import until first use

input_schema:
  type: object
  properties:
    param1:
      type: string
      description: Description of param1
      additionalProperties: false
    param2:
      type: integer
      description: Description of param2
  required:
    - param1
  additionalProperties: false
  $schema: http://json-schema.org/draft-07/schema#
```

### Tool Binding

**Function Signature**:

```python
def my_tool(
    param1: str,
    param2: int,
    agent_state: AgentState,  # Optional: injected by framework
    global_storage: GlobalStorage,  # Optional: injected by framework
) -> str:
    """Tool implementation.

    The docstring is used to inform the LLM about the tool.

    Args:
        param1: Description of param1
        param2: Description of param2
        agent_state: Current execution state (injected automatically)
        global_storage: Global storage (injected automatically)

    Returns:
        Tool result as string or serializable object
    """
    # Tool implementation...
    return "result"
```

**AgentState Injection**:

The framework automatically injects `agent_state` when the function signature requests it:

```python
def tool_with_state(param: str, agent_state: AgentState):
    """Tool that needs access to execution state."""
    current_iteration = agent_state.current_iteration
    history = agent_state.history
    # Use agent state...
```

**Reserved Parameters**:

- `agent_state`: Injected by framework (not allowed in `extra_kwargs`)
- `global_storage`: Injected by framework (not allowed in `extra_kwargs`)

### Tool Executor (`builtin/` directory)

Handles tool execution with before/after hooks and parameter binding.

**Execution Flow**:

```
1. Validate arguments using JSON Schema (type coercion)
2. Apply extra_kwargs (preset parameters)
3. Call before_tool hook
4. Execute tool function
5. Call after_tool hook
6. Return result
```

**Error Handling**:

- Validation errors raise `ToolValidationError`
- Execution errors are caught and returned as `is_error=True`
- Tool results are always JSON-serializable

### Built-in Tools

Located in `nexau/archs/tool/builtin/`:

#### File Tools (`file_tools/`)

- `read_file`, `write_file`, `replace`: Read, write, edit files
- `glob`, `list_directory`, `read_many_files`, `search_file_content`: Find and search files

#### Web Tools (`web_tools/`)

- `google_web_search`: Search web
- `web_fetch`: Fetch and parse content from a URL

#### Shell Tools (`shell_tools/`)

- `run_shell_command`: Execute shell commands

#### Session Tools (`session_tools/`)

- `write_todos`, `complete_task`, `save_memory`, `ask_user`: Task management, persistence, user interaction

#### Other Tools

- `multiedit_tool`: Apply multiple edits to a single file
- `run_code_tool`: Execute code in notebook environment

#### MCP Integration (`mcp_client.py`)

- Connect to external services via Model Context Protocol
- Auto-discover and register MCP tools

#### Other Tools

- `recall_sub_agent_tool`: Recall sub-agent history
- `llm_friendly`: Format output for LLM consumption
- `feishu`: Feishu integration tools

## Key Patterns

### Tool Creation Pattern

```python
# Step 1: Define function
def my_tool(param1: str, param2: int) -> str:
    """Tool implementation."""
    return f"Result: {param1} x {param2}"

# Step 2: Create YAML
# tools/MyTool.yaml
# type: tool
# name: MyTool
# description: "A tool that..."
# input_schema: {...}

# Step 3: Bind and use
from nexau.archs.tool import Tool

tool = Tool.from_yaml("tools/MyTool.yaml", binding=my_tool)
agent_config = AgentConfig(tools=[tool], ...)
```

### Extra Kwargs Pattern

Preset fixed arguments (e.g., `base_url`, `api_key`) so callers can omit them:

```python
# In YAML
tools:
  - name: my_api_tool
    yaml_path: ./tools/MyApiTool.yaml
    binding: mypkg.tools:my_api_tool
    extra_kwargs:
      base_url: https://api.example.com
      api_key: ${env.API_KEY}  # Loaded from environment
```

**Note**: Call-time arguments with the same name override preset values.

### Tool Caching Pattern

Enable result caching with `use_cache: true` in YAML:

```yaml
type: tool
name: expensive_operation
description: "An expensive operation"
use_cache: true  # ← Enable caching
input_schema: {...}
```

Caching uses diskcache backend with TTL (time-to-live).

### Parallel Execution Control

Prevent parallel execution with `disable_parallel: true`:

```yaml
type: tool
name: sensitive_operation
description: "A sensitive operation"
disable_parallel: true  # ← Disable parallel execution
input_schema: {...}
```

This ensures the tool runs sequentially, one at a time.

### Lazy Loading Pattern

Defer import until first use:

```yaml
type: tool
name: heavy_dependency_tool
description: "Tool with heavy dependencies"
lazy: true  # ← Defer import
input_schema: {...}
```

Useful for tools with heavy dependencies to speed up initialization.

### Tool Error Handling Pattern

```python
def my_tool(param: str) -> str:
    """Tool with error handling."""
    try:
        # Tool logic
        result = process(param)
        return result
    except Exception as e:
        # Return error message (will be marked as is_error=True)
        return f"Error: {str(e)}"
```

The framework automatically catches exceptions and marks them as errors.

## Common Issues

### Tool Not Found

**Error**: `ToolNotFoundError: tool 'xxx' not found`

**Solution**: Verify tool name matches tool's `name` field in YAML (not the binding function name):

```yaml
# Correct: Use YAML name
tools:
  my_tool:  # ← This is the name to use in agent code
    name: "My Tool Display Name"  # ← This is just display
    ...
```

### Parameter Validation Failing

**Error**: `ToolValidationError` when calling tool

**Solution**: Check JSON Schema in tool YAML matches function signature:

```yaml
# YAML
input_schema:
  type: object
  properties:
    count:  # ← Must match function parameter name
      type: integer
  required:
    - count
```

```python
# Function
def my_tool(count: int) -> str:  # ← Must match YAML
    ...
```

### Extra Kwargs Not Working

**Error**: Extra kwargs not applied to tool calls

**Solution**: Verify extra kwargs don't use reserved names (`agent_state`, `global_storage`):

```yaml
# Incorrect
extra_kwargs:
  agent_state: "invalid"  # ← Reserved name

# Correct
extra_kwargs:
  base_url: "https://api.example.com"  # ← OK
```

### Tool Not Injecting AgentState

**Error**: `agent_state` parameter is None

**Solution**: Ensure the function signature includes `agent_state: AgentState`:

```python
# Correct
def my_tool(param: str, agent_state: AgentState):
    # agent_state is injected automatically
    pass

# Incorrect
def my_tool(param: str):
    # agent_state is not available
    pass
```
