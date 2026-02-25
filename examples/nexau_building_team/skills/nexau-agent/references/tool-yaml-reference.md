# Tool YAML Definition Reference

Complete reference for NexAU tool YAML definition files.

## Format

```yaml
type: tool                           # Required. Must be "tool"
name: tool_name                      # Required. Tool identifier
description: >-                      # Required. Detailed description of what the tool does
  Multi-line description of the tool's
  purpose, behavior, and return values.
input_schema:                        # Required. JSON Schema for tool parameters
  type: object
  properties:
    param_name:
      type: string                   # "string" | "number" | "boolean" | "array" | "object"
      description: What this parameter does
    optional_param:
      type: string
      description: An optional parameter
  required:                          # List of required parameter names
    - param_name
  additionalProperties: false        # Always set to false
  $schema: http://json-schema.org/draft-07/schema#
```

## Skill-Enabled Tools

To expose a tool as a discoverable skill:

```yaml
type: tool
name: tool_name
description: Tool description
as_skill: true
skill_description: Brief skill description for the registry
input_schema:
  # ... schema as above
```

## Parameter Types

### String

```yaml
param_name:
  type: string
  description: A text parameter
```

### Number

```yaml
param_name:
  type: number
  description: A numeric parameter
```

### Boolean

```yaml
param_name:
  type: boolean
  description: A true/false parameter
```

### Array

```yaml
param_name:
  type: array
  items:
    type: string
  description: A list of strings
```

### Enum

```yaml
param_name:
  type: string
  enum: [option_a, option_b, option_c]
  description: One of the allowed values
```

### Optional Parameters

Simply omit the parameter name from the `required` list. Add "(OPTIONAL)" to the description for clarity:

```yaml
properties:
  required_param:
    type: string
    description: This is required
  optional_param:
    type: string
    description: >-
      (OPTIONAL) This is optional. If not provided, defaults to X.
required:
  - required_param
```

## Tool Binding

In the agent YAML, tools are bound to Python functions:

```yaml
tools:
  - name: tool_name
    yaml_path: ./tools/tool_name.tool.yaml
    binding: module.path:function_name
```

The binding format is `python.module.path:function_name`. For builtin tools:

```
nexau.archs.tool.builtin.file_tools:read_file
nexau.archs.tool.builtin.web_tools:google_web_search
nexau.archs.tool.builtin.shell_tools:run_shell_command
nexau.archs.tool.builtin.session_tools:write_todos
```

For custom tools in the project:

```
custom_tools.my_module:my_function
```

## Custom Tool Implementation

The Python function signature must match the `input_schema` properties:

```python
def my_tool(param_name: str, optional_param: str = "default") -> str:
    """Tool description.

    Args:
        param_name: What this parameter does.
        optional_param: Optional parameter description.

    Returns:
        Result as a string.
    """
    # Implementation
    return result
```

### Accessing Agent State

To access the sandbox (file system, working directory), add `agent_state: AgentState`:

```python
from nexau.archs.main_sub.agent_state import AgentState

def my_tool(param_name: str, agent_state: AgentState) -> str:
    sandbox = agent_state.sandbox
    work_dir = sandbox.work_dir
    # ... use sandbox for file operations
```

Note: `agent_state` is injected automatically — do NOT include it in the tool's `input_schema`.

## Extra Kwargs (Preset Parameters)

To preset fixed arguments that callers don't need to provide:

```yaml
tools:
  - name: my_tool
    yaml_path: ./tools/my_tool.tool.yaml
    binding: my_module:my_function
    extra_kwargs:
      api_key: ${env.MY_API_KEY}
      base_url: https://api.example.com
```

Call-time arguments with the same name override preset values.
