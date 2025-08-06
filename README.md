# Northau Framework

A general-purpose agent framework inspired by Claude Code's architecture for building intelligent agents with tool capabilities.

## Features

- **Modular Tool System**: Easy-to-configure tools with YAML-based configuration
- **Agent Architecture**: Create specialized agents with different capabilities
- **Built-in Tools**: File operations, web search, bash execution, and more
- **Todo Management**: Built-in task tracking and management
- **LLM Integration**: Support for various LLM providers (OpenAI, Claude, etc.)
- **YAML Configuration**: Define agents and tools declaratively

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd northau

# Install dependencies using uv
pip install uv
uv sync
```

## Quick Start

### Environment Setup

First, set up your environment variables:

```bash
export LLM_MODEL="gpt-4"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="your-api-key-here"
```

### Example 1: Basic Agent with Code Tools

```python
#!/usr/bin/env python3
"""Basic example showing how to create an agent with programmatic configuration."""

import os
from datetime import datetime
from northau.archs.main_sub import create_agent
from northau.archs.tool import Tool
from northau.archs.tool.builtin.bash_tool import bash
from northau.archs.tool.builtin.file_tool import file_edit, file_search
from northau.archs.tool.builtin.todo_write import todo_write
from northau.archs.tool.builtin.web_tool import web_search, web_read
from northau.archs.llm import LLMConfig

def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    # Create tools with YAML configurations
    web_search_tool = Tool.from_yaml("tools/WebSearch.yaml", binding=web_search)
    web_read_tool = Tool.from_yaml("tools/WebRead.yaml", binding=web_read)
    
    # Configure LLM
    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY")
    )
    
    # Create the agent
    research_agent = create_agent(
        name="research_agent",
        tools=[web_search_tool, web_read_tool],
        llm_config=llm_config,
        system_prompt="You are a research agent. Use web_search and web_read tools to find information.",
    )
    
    # Run the agent
    response = research_agent.run(
        "What's the latest news about AI developments?",
        context={"date": get_date()}
    )
    print(response)

if __name__ == "__main__":
    main()
```

### Example 2: YAML-Based Agent Configuration

Create an agent configuration file `agents/my_agent.yaml`:

```yaml
name: my_research_agent
max_context: 100000
system_prompt: |
  Date: {{date}}. You are a research agent specialized in finding and analyzing information.
  Use web_search to find relevant information, then web_read to get detailed content.
system_prompt_type: string
llm_config:
  temperature: 0.7
  max_tokens: 4096
tools:
  - name: web_search
    yaml_path: ../tools/WebSearch.yaml
    binding: northau.archs.tool.builtin.web_tool:web_search
  - name: web_read
    yaml_path: ../tools/WebRead.yaml
    binding: northau.archs.tool.builtin.web_tool:web_read
sub_agents: []
```

Then load and use the agent:

```python
#!/usr/bin/env python3
"""YAML-based agent configuration example."""

import os
from datetime import datetime
from northau.archs.config.config_loader import load_agent_config

def main():
    # Load agent from YAML with environment overrides
    config_overrides = {
        "llm_config": {
            "model": os.getenv("LLM_MODEL"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "api_key": os.getenv("LLM_API_KEY"),
            "temperature": 0.7
        }
    }
    
    agent = load_agent_config(
        "agents/my_agent.yaml",
        overrides=config_overrides
    )
    
    # Use the agent
    response = agent.run(
        "Research the latest developments in quantum computing",
        context={"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )
    print(response)

if __name__ == "__main__":
    main()
```

### Example 3: Claude-code-like Agent with All Tools

```python
#!/usr/bin/env python3
"""Comprehensive example with all available tools."""

import os
from datetime import datetime
from northau.archs.main_sub import create_agent
from northau.archs.tool import Tool
from northau.archs.tool.builtin.file_tools.file_edit_tool import file_edit_tool
from northau.archs.tool.builtin.file_tools.file_read_tool import file_read_tool
from northau.archs.tool.builtin.file_tools.file_write_tool import file_write_tool
from northau.archs.tool.builtin.file_tools.grep_tool import grep_tool
from northau.archs.tool.builtin.file_tools.glob_tool import glob_tool
from northau.archs.tool.builtin.todo_write import todo_write
from northau.archs.tool.builtin.web_tool import web_search, web_read
from northau.archs.tool.builtin.bash_tool import bash_tool
from northau.archs.tool.builtin.ls_tool import ls_tool
from northau.archs.tool.builtin.multiedit_tool import multiedit_tool
from northau.archs.llm import LLMConfig

def main():
    # Create all tools with YAML configurations
    tools = [
        Tool.from_yaml("tools/claude_code/Grep.tool.yaml", binding=grep_tool),
        Tool.from_yaml("tools/claude_code/Glob.tool.yaml", binding=glob_tool),
        Tool.from_yaml("tools/claude_code/TodoWrite.tool.yaml", binding=todo_write),
        Tool.from_yaml("tools/claude_code/WebSearch.tool.yaml", binding=web_search),
        Tool.from_yaml("tools/claude_code/WebFetch.tool.yaml", binding=web_read),
        Tool.from_yaml("tools/claude_code/Read.tool.yaml", binding=file_read_tool),
        Tool.from_yaml("tools/claude_code/Write.tool.yaml", binding=file_write_tool),
        Tool.from_yaml("tools/claude_code/Edit.tool.yaml", binding=file_edit_tool),
        Tool.from_yaml("tools/claude_code/Bash.tool.yaml", binding=bash_tool),
        Tool.from_yaml("tools/claude_code/Ls.tool.yaml", binding=ls_tool),
        Tool.from_yaml("tools/claude_code/MultiEdit.tool.yaml", binding=multiedit_tool)
    ]
    
    # Configure LLM
    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )
    
    # Create a comprehensive coding assistant
    coding_agent = create_agent(
        name="coding_assistant",
        tools=tools,
        llm_config=llm_config,
        system_prompt=open("agents/claude_code/system-workflow.md").read(),
    )
    
    # Use the agent for a complex task
    response = coding_agent.run(
        "Create a simple Python web server that serves static files",
        context={
            "env_content": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": os.getenv("USER"),
            }
        }
    )
    print(response)

if __name__ == "__main__":
    main()
```

## Tool Configuration

### Built-in Tools

Northau comes with several built-in tools:

#### File Tools
- **file_read_tool**: Read file contents
- **file_write_tool**: Write/create files
- **file_edit_tool**: Edit existing files
- **grep_tool**: Search within files
- **glob_tool**: Find files by pattern
- **ls_tool**: List directory contents
- **multiedit_tool**: Make multiple edits to a file

#### Web Tools
- **web_search**: Search the web for information
- **web_read**: Read content from URLs

#### System Tools
- **bash_tool**: Execute bash commands
- **todo_write**: Manage todo lists and tasks

### Creating Custom Tools

1. **Define the tool function:**

```python
def my_custom_tool(param1: str, param2: int = 10) -> str:
    """My custom tool description."""
    # Tool implementation
    return f"Result: {param1} with {param2}"
```

2. **Create a YAML configuration:**

```yaml
# tools/MyCustomTool.yaml
name: MyCustomTool
description: >-
  Description of what this tool does.
  
  Usage guidelines and examples.

input_schema:
  type: object
  properties:
    param1:
      type: string
      description: Description of param1
    param2:
      type: integer
      default: 10
      description: Description of param2
  required:
    - param1
  additionalProperties: false
  $schema: http://json-schema.org/draft-07/schema#
```

3. **Use the tool in your agent:**

```python
from northau.archs.tool import Tool

my_tool = Tool.from_yaml("tools/MyCustomTool.yaml", binding=my_custom_tool)
```

## LLM Configuration

### Supported Providers

```python
# OpenAI
llm_config = LLMConfig(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-openai-key",
    temperature=0.7,
    max_tokens=4096
)

# Anthropic Claude
llm_config = LLMConfig(
    model="claude-3-sonnet-20240229",
    base_url="https://api.anthropic.com",
    api_key="your-anthropic-key",
    temperature=0.7,
    max_tokens=4096
)

# Local/Custom endpoint
llm_config = LLMConfig(
    model="custom-model",
    base_url="http://localhost:8000/v1",
    api_key="local-key",
    temperature=0.7,
    max_tokens=4096
)
```

## Running Examples

Use `uv run` to execute the examples:

```bash
# Run the basic quickstart
uv run quickstart.py

# Run the YAML-based example
uv run quickstart_yaml.py

# Run with specific LLM configuration
LLM_MODEL="gpt-4" LLM_BASE_URL="https://api.openai.com/v1" LLM_API_KEY="your-key" uv run quickstart.py
```

## Advanced Usage

### Context and Templating

Agents support Jinja2 templating in system prompts:

```python
system_prompt = """
Date: {{date}}
User: {{username}}
Task: {{task_description}}

You are an assistant with access to the following context:
{% for key, value in env_content.items() %}
- {{key}}: {{value}}
{% endfor %}
"""

response = agent.run(
    "Your message here",
    context={
        "date": "2024-01-01",
        "username": "user",
        "task_description": "Complete the project",
        "env_content": {
            "working_dir": "/path/to/project",
            "python_version": "3.12"
        }
    }
)
```

### Error Handling

```python
try:
    response = agent.run(message, context=context)
    print(f"Success: {response}")
except Exception as e:
    print(f"Error: {e}")
    # Handle specific error types as needed
```

## Project Structure

```
northau/
   agents/                 # Agent configurations
      *.yaml
   tools/                  # Tool configurations  
      *.yaml             # Basic tool configs
      claude_code/       # Claude Code compatible tools
          *.tool.yaml
   northau/               # Main package
      archs/
         main_sub/      # Agent creation and management
         tool/          # Tool system
            builtin/   # Built-in tools
         llm/           # LLM configuration
         config/        # Configuration loading
   tests/                 # Test files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Your License Here]