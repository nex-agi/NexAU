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
git clone git@github.com:/northau.git
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
Refer to `fake_claude_code.py` for a full example

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

Build an .env file
```
SERPER_API_KEY=xxx
BP_HTML_PARSER_URL="http://***REMOVED***/url2md"
BP_HTML_PARSER_API_KEY="xxx"
BP_HTML_PARSER_SECRET="xxx"
LLM_MODEL="glm-4.5"
LLM_BASE_URL="https://***REMOVED***/v1/"
LLM_API_KEY="sk-xxxx"
```

There are some exampels in `examples`
```bash
# Run the basic quickstart
dotenv run python examples.deep_research.quickstart.py

# Run the YAML-based example
dotenv run python examples.deep_research.quickstart_yaml.py

dotenv run python examples.fake_claude_code.fake_claude_code.py

dotenv run python -m examples.mcp.mcp_amap_example

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

### Modifying State and Config in Tools using AgentContext

Tools can access and modify the agent's state and configuration during execution using the AgentContext system. This allows tools to persist information across tool calls and dynamically adjust agent behavior.

#### Accessing Current State and Config

```python
from northau.archs.main_sub.agent_context import get_state, get_config, get_context

def my_stateful_tool(param1: str) -> dict:
    """Example tool that accesses and modifies agent state."""
    
    # Get current agent state and config
    current_state = get_state()  # Returns Dict[str, Any]
    current_config = get_config()  # Returns Dict[str, Any]
    
    # Access specific values with defaults
    from northau.archs.main_sub.agent_context import get_state_value, get_config_value
    user_name = get_state_value("user_name", "anonymous")
    debug_mode = get_config_value("debug", False)
    
    return {"result": f"Processing {param1} for user {user_name}"}
```

#### Modifying State and Config from Tools

```python
from northau.archs.main_sub.agent_context import (
    update_state, update_config, 
    set_state_value, set_config_value
)

def learning_tool(new_info: str) -> dict:
    """Tool that learns and updates agent state."""
    
    # Update multiple state values at once
    update_state(
        last_learned=new_info,
        learning_count=get_state_value("learning_count", 0) + 1
    )
    
    # Update individual values
    set_state_value("last_update_time", datetime.now().isoformat())
    
    # Modify config (affects agent behavior)
    if get_state_value("learning_count", 0) > 10:
        set_config_value("expert_mode", True)
    
    return {"result": f"Learned: {new_info}"}

```

#### Using Full Context Object

```python
from northau.archs.main_sub.agent_context import get_context

def context_aware_tool(action: str) -> dict:
    """Tool that uses the full context object for advanced operations."""
    
    # Get the full context object
    ctx = get_context()
    if ctx is None:
        return {"error": "No agent context available"}
    
    # Access context data
    all_state = ctx.state
    all_config = ctx.config
    
    # Check if context was recently modified
    if ctx.is_modified():
        # Context was changed by another tool
        pass
    
    # Add a callback for when context changes
    def on_context_change():
        print("Context was modified!")
    
    ctx.add_modification_callback(on_context_change)
    
    # Modify context
    ctx.update_state({"tool_action": action})
    ctx.set_config_value("last_tool", "context_aware_tool")
    
    return {"result": f"Executed {action} with full context awareness"}
```

#### Practical Example: Session Management Tool

```python
from datetime import datetime
from northau.archs.main_sub.agent_context import (
    get_state_value, set_state_value, update_state, get_config_value
)

def session_manager(action: str, data: dict = None) -> dict:
    """Manage user session state across tool calls."""
    
    if action == "start_session":
        session_id = f"session_{datetime.now().timestamp()}"
        update_state(
            session_id=session_id,
            session_start=datetime.now().isoformat(),
            user_actions=[],
            session_data=data or {}
        )
        return {"result": f"Started session {session_id}"}
    
    elif action == "log_action":
        actions = get_state_value("user_actions", [])
        actions.append({
            "action": data.get("action"),
            "timestamp": datetime.now().isoformat(),
            "details": data.get("details", {})
        })
        set_state_value("user_actions", actions)
        
        # Adjust agent behavior based on session length
        if len(actions) > 20:
            # Long session - make agent more concise
            from northau.archs.main_sub.agent_context import update_config
            update_config(max_tokens=1000)
            
        return {"result": f"Logged action: {data.get('action')}"}
    
    elif action == "get_session_info":
        session_id = get_state_value("session_id", "no-session")
        actions_count = len(get_state_value("user_actions", []))
        session_start = get_state_value("session_start", "unknown")
        
        return {
            "session_id": session_id,
            "actions_count": actions_count,
            "session_start": session_start,
            "current_config": get_config_value("max_tokens", "default")
        }
    
    return {"error": f"Unknown session action: {action}"}
```

**Note**: State and config modifications persist for the duration of the agent's execution context and automatically trigger system prompt refresh when the context is modified, ensuring the agent's behavior adapts to the new state.

### MCP
Define the mcp servers and create agent by setting `mcp_servers` as follows:

```python
llm_config = LLMConfig(
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

mcp_servers = [
    {
        "name": "amap-maps-streamableHTTP",
        "type": "http",
        "url": "https://mcp.amap.com/mcp?key=xxx",
        "headers": {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        },
        "timeout": 10
    }
]

agent = create_agent(
    name="amap_agent",
    system_prompt="""You are an AI agent with access to Amap Maps services through MCP.
    
You can use Amap Maps tools to:
- Search for locations and points of interest
- Get directions and navigation information
- Calculate distances and travel times
- Find nearby businesses and services
- Access real-time traffic information
- And other location-based services

When using map tools, always provide clear and helpful information to users.
Explain what you're doing and provide context for the results.""",
    mcp_servers=mcp_servers,
    llm_config=llm_config,
)

response = agent.run("现在从漕河泾现代服务园A6到上南路 4265弄要多久？")
print(response)
```