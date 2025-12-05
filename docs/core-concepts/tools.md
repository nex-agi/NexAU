# 🛠️ Core Concepts: Tools

Tools are the functions that an agent can execute to interact with the outside world, such as searching the web, reading files, or running code.

## Built-in Tools

NexAU comes with a variety of pre-built tools for common tasks.

#### File Tools
- **file_read_tool**: Read file contents.
- **file_write_tool**: Write or create files.
- **file_edit_tool**: Edit existing files.
- **grep_tool**: Search for patterns within files.
- **glob_tool**: Find files using patterns.
- **ls_tool**: List directory contents.
- **multiedit_tool**: Apply multiple edits to a single file.

#### Web Tools
- **web_search**: Search the web for information (uses Serper).
- **web_read**: Read and parse content from a URL.

#### System Tools
- **bash_tool**: Execute bash commands.
- **todo_write**: Manage a simple todo list.

### How to use Use these built-in tools
If you use python to create Agent:

```python
from nexau.archs.tool import Tool
from nexau.archs.tool.builtin.todo_write import todo_write
from nexau.archs.tool.builtin.web_tool import web_read
from nexau.archs.tool.builtin.web_tool import web_search

# Create the tool instance
# Note: you need to create the yaml files for tool config, you may refer to examples/deep_research/tools for examples

web_search_tool = Tool.from_yaml(
    str(script_dir / 'tools/WebSearch.yaml'),
    binding=web_search,
)
web_read_tool = Tool.from_yaml(
    str(script_dir / 'tools/WebRead.yaml'),
    binding=web_read,
)
todo_write_tool = Tool.from_yaml(
    str(script_dir / 'tools/TodoWrite.tool.yaml'),
    binding=todo_write,
)

# Add it to your agent's tool list
agent = create_agent(
    name="web_agent",
    tools=[web_search_tool, web_read_tool, todo_write_tool],
    # ... other agent config
)
```

If you use agent yaml config, you can add these lines to the config file:

```yaml
tools:
  - name: web_search
    yaml_path: ./tools/WebSearch.yaml
    binding: nexau.archs.tool.builtin.web_tool:web_search
  - name: web_read
    yaml_path: ./tools/WebRead.yaml
    binding: nexau.archs.tool.builtin.web_tool:web_read
  - name: todo_write
    yaml_path: ./tools/TodoWrite.tool.yaml
    binding: nexau.archs.tool.builtin.todo_write:todo_write
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
from nexau.archs.tool import Tool
from my_tools.calculator import simple_calculator

# Create the tool instance
calculator_tool = Tool.from_yaml(
    "tools/SimpleCalculator.tool.yaml",
    binding=simple_calculator
)

# Add it to your agent's tool list
agent = create_agent(
    name="calculator_agent",
    tools=[calculator_tool],
    # ... other agent config
)
```

If you use agent yaml config, you can add these lines to the config file:

```yaml
tools:
  - name: simple_calculator
    yaml_path: ./tools/SimpleCalculator.tool.yaml
    binding: my_tools.calculator.simple_calculator:simple_calculator
```
