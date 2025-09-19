# Northau Framework

A general-purpose agent framework inspired by Claude Code's architecture for building intelligent agents with tool capabilities.

## Features

- **Modular Tool System**: Easy-to-configure tools with YAML-based configuration
- **Agent Architecture**: Create specialized agents with different capabilities
- **Built-in Tools**: File operations, web search, bash execution, and more
- **LLM Integration**: Support for various LLM providers (OpenAI, Claude, etc.)
- **Custom LLM Generators**: Customize LLM behavior with custom preprocessing, caching, logging, and provider switching
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

NorthAU supports tracing with Langfuse. If you need tracing, please set the following environment variables:

```bash
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key-here"
export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key-here"
export LANGFUSE_HOST="your-langfuse-host-here"
```

You can also use .env file to set the environment variables and run the following command to start the agent:

```.env
LLM_MODEL=glm-4.5
LLM_BASE_URL=https://***REMOVED***/v1/
LLM_API_KEY=sk-xxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_HOST=***REMOVED***
```

```bash
dotenv run uv run examples/deep_research/quickstart.py
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

## Custom LLM Generators

Northau supports custom LLM generators, allowing you to customize how LLM requests are processed. This is useful for:
- Adding custom preprocessing/postprocessing
- Integrating with different LLM providers
- Implementing custom caching or logging
- Adjusting parameters based on context

### Creating a Custom LLM Generator

A custom LLM generator is a function that takes the OpenAI client and request parameters, and returns a response:

```python
from typing import Any, Dict

def my_custom_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """
    Custom LLM generator function.

    Args:
        openai_client: The OpenAI client instance
        kwargs: Parameters that would be passed to openai_client.chat.completions.create()

    Returns:
        Response object with same structure as OpenAI's response
    """
    # Add custom logic here
    print(f"🔧 Processing request with {len(kwargs.get('messages', []))} messages")

    # Modify parameters if needed
    modified_kwargs = kwargs.copy()
    if modified_kwargs.get('temperature', 0.7) > 0.5:
        modified_kwargs['temperature'] = 0.3  # Lower temperature for more focused responses
        print("🎯 Adjusted temperature for better focus")

    # Call the LLM (you can use any provider here)
    response = openai_client.chat.completions.create(**modified_kwargs)

    # Add custom post-processing
    if response and hasattr(response, 'choices') and response.choices:
        content = response.choices[0].message.content
        print(f"📊 Generated response: {len(content)} characters")

    return response
```

### Using Custom LLM Generators

#### 1. Programmatic Usage

```python
from northau.archs.main_sub import create_agent
from northau.archs.llm import LLMConfig

# Create agent with custom LLM generator
agent = create_agent(
    name="custom_agent",
    system_prompt="You are a helpful assistant.",
    llm_config=LLMConfig(model="gpt-4", temperature=0.7),
    custom_llm_generator=my_custom_generator,  # Your custom function
    tools=[]
)

# Use the agent normally - custom generator will be used automatically
response = agent.run("Hello, how can you help me?")
```

#### 2. YAML Configuration

You can configure custom LLM generators directly in YAML files:

**Simple Configuration:**
```yaml
name: my_agent
system_prompt: "You are a helpful assistant"
llm_config:
  model: gpt-4
  temperature: 0.7
custom_llm_generator: "my_module.generators:my_custom_generator"
tools: []
```

**Parameterized Configuration:**
```yaml
name: research_agent
system_prompt: "You are a research assistant"
llm_config:
  model: gpt-4
  temperature: 0.7
custom_llm_generator:
  import: "my_module.generators:parameterized_generator"
  params:
    min_temperature: 0.1
    max_temperature: 0.4
    add_context: true
    log_requests: true
tools: []
```

For parameterized generators, create a function that accepts the parameters:

```python
def parameterized_generator(openai_client: Any, kwargs: Dict[str, Any],
                          min_temperature: float = 0.2,
                          max_temperature: float = 0.8,
                          add_context: bool = False,
                          log_requests: bool = False) -> Any:
    """Parameterized custom LLM generator."""
    if log_requests:
        print(f"🔍 LLM Request: {kwargs.get('model', 'unknown')} model")

    # Clamp temperature within specified range
    current_temp = kwargs.get('temperature', 0.7)
    modified_kwargs = kwargs.copy()
    modified_kwargs['temperature'] = max(min_temperature, min(max_temperature, current_temp))

    if add_context:
        # Add custom context to system message
        messages = modified_kwargs.get('messages', [])
        if messages and messages[0].get('role') == 'system':
            enhanced_content = f"{messages[0]['content']}\n\nAdditional Context: Focus on providing detailed, accurate responses."
            modified_kwargs['messages'] = messages.copy()
            modified_kwargs['messages'][0] = {'role': 'system', 'content': enhanced_content}

    return openai_client.chat.completions.create(**modified_kwargs)
```

### Example Use Cases

#### 1. Research-Optimized Generator

```python
def research_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """Generator optimized for research tasks."""
    # Lower temperature for more focused responses
    modified_kwargs = kwargs.copy()
    modified_kwargs['temperature'] = min(kwargs.get('temperature', 0.7), 0.3)

    # Add research context
    messages = kwargs.get('messages', [])
    if messages and messages[0].get('role') == 'system':
        research_prompt = f"{messages[0]['content']}\n\nIMPORTANT: Provide accurate, well-researched responses with citations when possible."
        modified_kwargs['messages'] = messages.copy()
        modified_kwargs['messages'][0] = {'role': 'system', 'content': research_prompt}

    return openai_client.chat.completions.create(**modified_kwargs)
```

#### 2. Multi-Provider Generator

```python
def multi_provider_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """Generator that can switch between different LLM providers."""
    model = kwargs.get('model', 'gpt-4')

    if model.startswith('claude'):
        # Use Anthropic client
        # (You would implement Anthropic client logic here)
        pass
    elif model.startswith('gpt'):
        # Use OpenAI client
        return openai_client.chat.completions.create(**kwargs)
    else:
        # Use custom provider
        # (You would implement custom provider logic here)
        pass
```

#### 3. Caching Generator

```python
import hashlib
import json

cache = {}

def caching_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """Generator with response caching."""
    # Create cache key from request
    cache_key = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()

    # Check cache first
    if cache_key in cache:
        print("💾 Using cached response")
        return cache[cache_key]

    # Generate new response
    response = openai_client.chat.completions.create(**kwargs)

    # Cache the response
    cache[cache_key] = response
    print("🆕 Generated and cached new response")

    return response
```

### Built-in Custom Generators

Northau includes some built-in custom generators for common use cases:

#### Bypass Generator

A simple pass-through generator that adds logging but doesn't modify the LLM behavior:

```yaml
custom_llm_generator: "northau.archs.main_sub.execution.response_generator:bypass_llm_generator"
```

This generator:
- Logs the number of messages being processed
- Calls the standard OpenAI API without modifications
- Useful for debugging and monitoring LLM calls

### Loading Agents with Custom Generators

```python
from northau.archs.config.config_loader import load_agent_config

# Load agent from YAML with custom LLM generator
agent = load_agent_config('path/to/config.yaml')

# The custom generator will be automatically used for all LLM calls
response = agent.run("Your message here")
```

### Best Practices

1. **Maintain Compatibility**: Ensure your custom generator returns a response object with the same structure as OpenAI's response (with `.choices[0].message.content` attribute).

2. **Error Handling**: Implement proper error handling in your custom generator:
   ```python
   def robust_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
       try:
           return openai_client.chat.completions.create(**kwargs)
       except Exception as e:
           print(f"❌ LLM call failed: {e}")
           # Implement fallback logic or re-raise
           raise
   ```

3. **Parameter Validation**: Validate and sanitize parameters before making LLM calls.

4. **Logging**: Add appropriate logging for debugging and monitoring.

5. **Performance**: Consider the performance impact of your custom logic, especially for high-frequency use cases.

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

#### Using Full Context Object

```python
from northau.archs.main_sub.agent_context import get_context

def context_aware_tool(action: str) -> dict:
    """Tool that uses the full context object for advanced operations."""

    # Get the full context object
    ctx = get_context()
    if ctx is None:
        return {"error": "No agent context available"}

    # Check if context was recently modified
    if ctx.is_modified():
        # Context was changed by another tool
        pass

    # Add a callback for when context changes
    def on_context_change():
        print("Context was modified!")

    ctx.add_modification_callback(on_context_change)

    return {"result": f"Executed {action} with full context awareness"}
```

### Dump trace to file

In addition to langfuse trace, you can also dump trace to files by setting `dump_trace_path` in `agent.run`, the input and output of each round of main agent is dump to `dump_trace_path` and the trace of sub-agent will be saved in to a subfolder with the same name of dump_trace_path (without .json extention).

```python
response = agent.run(input_message, dump_trace_path="test_log.json")
```

```
|- test_log.json
|- test_log
    |- sub_agent_1.json
    |- sub_agent_2.json
```


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

### Hooks

The Northau framework provides a powerful hook system that allows you to intercept and modify agent behavior after the LLM generates a response but before tool/agent execution. This enables custom logic, safety measures, conversation management, and much more.

#### Hook System Overview

Hooks receive an `AfterModelHookInput` containing:
- `original_response`: The raw LLM response
- `parsed_response`: Parsed tool/agent calls
- `messages`: Current conversation history
- `max_iterations`: Maximum allowed iterations
- `current_iteration`: Current iteration number

Hooks return a `AfterModelHookResult` that can modify:
- `parsed_response`: Filter or modify tool/agent calls
- `messages`: Add context or modify conversation history

#### Basic Hook Usage

```python
from northau.archs.main_sub.execution.hooks import AfterModelHook, AfterModelHookResult, AfterModelHookInput

def create_context_hook() -> AfterModelHook:
    def context_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        if hook_input.parsed_response and hook_input.parsed_response.tool_calls:
            # Add context message before tool execution
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": f"[HOOK] About to execute {len(hook_input.parsed_response.tool_calls)} tool(s)"
            })
            return AfterModelHookResult.with_modifications(messages=modified_messages)

        return AfterModelHookResult.no_changes()

    return context_hook

# Use with agent
agent = create_agent(
    name="my_agent",
    tools=[...],
    after_model_hooks=[create_context_hook()],
    llm_config=llm_config
)
```

#### Built-in Hook Factories

##### Logging Hook
```python
from northau.archs.main_sub.execution.hooks import create_logging_hook

# Logs detailed information about each hook execution
logging_hook = create_logging_hook("my_logger")
```

##### Filter Hook
```python
from northau.archs.main_sub.execution.hooks import create_filter_hook

# Only allow specific tools and agents
filter_hook = create_filter_hook(
    allowed_tools={'web_search', 'file_read'},
    allowed_agents={'research_agent', 'data_agent'}
)
```

##### Remaining Iterations Hook
```python
from northau.archs.main_sub.execution.hooks import create_remaining_reminder_hook

# Adds iteration count reminders to conversation
reminder_hook = create_remaining_reminder_hook()
```
#### Tool Approve Hook
Each time the tool with `tool_name` is called, the CLI prompts whether to approve (y/n). If no, the agent stops.

```python
from northau.archs.main_sub.execution.hooks import create_tool_after_approve_hook

tool_after_approve_hook = create_tool_after_approve_hook(
    tool_name='WebSearch'
)
```


#### Advanced Hook Examples

##### Safety and Compliance Hook
```python
def create_safety_hook() -> AfterModelHook:
    def safety_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        if not hook_input.parsed_response:
            return AfterModelHookResult.no_changes()

        # Filter out potentially dangerous tools
        dangerous_tools = {'system_command', 'file_delete', 'network_access'}
        safe_calls = [
            call for call in hook_input.parsed_response.tool_calls
            if call.tool_name not in dangerous_tools
        ]

        if len(safe_calls) != len(hook_input.parsed_response.tool_calls):
            # Create modified parsed response
            from northau.archs.main_sub.execution.parse_structures import ParsedResponse
            modified_parsed = ParsedResponse(
                original_response=hook_input.parsed_response.original_response,
                tool_calls=safe_calls,
                sub_agent_calls=hook_input.parsed_response.sub_agent_calls,
                batch_agent_calls=hook_input.parsed_response.batch_agent_calls,
                is_parallel_tools=hook_input.parsed_response.is_parallel_tools,
                is_parallel_sub_agents=hook_input.parsed_response.is_parallel_sub_agents
            )

            # Add safety message
            modified_messages = hook_input.messages.copy()
            filtered_count = len(hook_input.parsed_response.tool_calls) - len(safe_calls)
            modified_messages.append({
                "role": "system",
                "content": f"[SAFETY] Blocked {filtered_count} potentially dangerous tool calls"
            })

            return AfterModelHookResult.with_modifications(
                parsed_response=modified_parsed,
                messages=modified_messages
            )

        return AfterModelHookResult.no_changes()

    return safety_hook
```

##### Conversation Management Hook
```python
def create_conversation_manager_hook() -> AfterModelHook:
    def conversation_manager_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        # Add iteration warnings when approaching limit
        remaining = hook_input.max_iterations - hook_input.current_iteration

        if remaining <= 2:
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": f"[WARNING] Only {remaining} iterations remaining. Please provide a conclusive response."
            })
            return AfterModelHookResult.with_modifications(messages=modified_messages)

        # Add conversation length management
        if len(hook_input.messages) > 20:
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": "[INFO] Long conversation detected. Consider summarizing key points."
            })
            return AfterModelHookResult.with_modifications(messages=modified_messages)

        return AfterModelHookResult.no_changes()

    return conversation_manager_hook
```

##### Custom Business Logic Hook
```python
def create_business_logic_hook(user_permissions: set[str]) -> AfterModelHook:
    def business_logic_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        if not hook_input.parsed_response:
            return AfterModelHookResult.no_changes()

        # Apply business rules based on user permissions
        allowed_calls = []
        for call in hook_input.parsed_response.tool_calls:
            if call.tool_name == 'database_query' and 'db_read' not in user_permissions:
                continue  # Skip unauthorized database calls
            elif call.tool_name == 'file_write' and 'file_modify' not in user_permissions:
                continue  # Skip unauthorized file operations
            else:
                allowed_calls.append(call)

        if len(allowed_calls) != len(hook_input.parsed_response.tool_calls):
            # Create modified response with business logic applied
            from northau.archs.main_sub.execution.parse_structures import ParsedResponse
            modified_parsed = ParsedResponse(
                original_response=hook_input.parsed_response.original_response,
                tool_calls=allowed_calls,
                sub_agent_calls=hook_input.parsed_response.sub_agent_calls,
                batch_agent_calls=hook_input.parsed_response.batch_agent_calls,
                is_parallel_tools=hook_input.parsed_response.is_parallel_tools,
                is_parallel_sub_agents=hook_input.parsed_response.is_parallel_sub_agents
            )

            return AfterModelHookResult.with_modifications(parsed_response=modified_parsed)

        return AfterModelHookResult.no_changes()

    return business_logic_hook
```

#### Using Multiple Hooks

```python
# Combine multiple hooks for comprehensive control
agent = create_agent(
    name="secure_agent",
    tools=[web_search_tool, file_tools, database_tools],
    after_model_hooks=[
        create_logging_hook("agent_logger"),
        create_safety_hook(),
        create_business_logic_hook({'db_read', 'file_modify'}),
        create_conversation_manager_hook(),
        create_filter_hook(allowed_tools={'web_search', 'file_read'})
    ],
    llm_config=llm_config
)
```

#### Hook Execution Order

Hooks are executed in the order they are provided:
1. Each hook receives the current state (after previous hooks)
2. If a hook modifies the parsed response or messages, subsequent hooks see those changes
3. The final result is used for tool/agent execution

#### Hook Result Methods

```python
# No modifications
return AfterModelHookResult.no_changes()

# Only modify parsed response
return AfterModelHookResult.with_modifications(parsed_response=modified_parsed)

# Only modify messages
return AfterModelHookResult.with_modifications(messages=modified_messages)

# Modify both
return AfterModelHookResult.with_modifications(
    parsed_response=modified_parsed,
    messages=modified_messages
)

# Direct construction (also valid)
return AfterModelHookResult(parsed_response=modified_parsed, messages=modified_messages)
```

#### Use Cases

- **Safety & Compliance**: Filter dangerous tools, enforce business rules
- **Debugging & Monitoring**: Log execution patterns, track tool usage
- **Conversation Management**: Add context, manage conversation length
- **Custom Business Logic**: Apply domain-specific rules and permissions
- **Quality Control**: Validate tool parameters, ensure proper usage
- **User Experience**: Add helpful hints, progress indicators, warnings

The hook system provides powerful capabilities for customizing agent behavior while maintaining clean separation of concerns and type safety.

### Tool Hooks

In addition to model hooks, Northau provides `after_tool_hooks` that intercept and modify tool execution results. These hooks are called after each tool is executed but before the result is processed by the agent.

#### Tool Hook System Overview

Tool hooks receive an `AfterToolHookInput` containing:
- `tool_name`: Name of the executed tool
- `tool_input`: Parameters passed to the tool
- `tool_output`: Result returned by the tool
- `global_storage`: Shared storage for cross-tool state

Tool hooks return an `AfterToolHookResult` that can modify:
- `tool_output`: Transform or filter the tool's output

#### Basic Tool Hook Usage

```python
from northau.archs.main_sub.execution.hooks import AfterToolHook, AfterToolHookInput, AfterToolHookResult

def create_tool_output_logger() -> AfterToolHook:
    def tool_logger_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        print(f"🔧 Tool '{hook_input.tool_name}' executed")
        print(f"   Input: {hook_input.tool_input}")
        print(f"   Output: {hook_input.tool_output}")

        # Return no modifications - just log
        return AfterToolHookResult.no_changes()

    return tool_logger_hook

# Use with agent
agent = create_agent(
    name="my_agent",
    tools=[web_search_tool, file_tools],
    after_tool_hooks=[create_tool_output_logger()],
    llm_config=llm_config
)
```

#### Built-in Tool Hook Factories

##### Tool Logging Hook
```python
from northau.archs.main_sub.execution.hooks import create_tool_logging_hook

# Logs detailed tool execution information
tool_logging_hook = create_tool_logging_hook("tool_debug_logger")
```

#### YAML Configuration for Tool Hooks

Tool hooks can be configured directly in YAML files:

```yaml
name: research_agent
system_prompt: "You are a research assistant with web access."
llm_config:
  model: gpt-4
  temperature: 0.7
tools:
  - name: web_search
    yaml_path: ./tools/WebSearch.yaml
    binding: northau.archs.tool.builtin.web_tool:web_search
  - name: web_read
    yaml_path: ./tools/WebRead.yaml
    binding: northau.archs.tool.builtin.web_tool:web_read
after_tool_hooks:
  # Simple logging hook
  - import: northau.archs.main_sub.execution.hooks:create_tool_logging_hook
    params:
      logger_name: "research_tool_debug"

  # Custom transformer
  - import: my_module.hooks:create_custom_tool_transformer
    params:
      add_timestamps: true
      format_output: true
```

#### Advanced Tool Hook Examples

##### Data Validation Hook
```python
def create_tool_validation_hook() -> AfterToolHook:
    def validation_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        tool_output = hook_input.tool_output

        # Validate web search results
        if hook_input.tool_name == "web_search":
            if isinstance(tool_output, dict) and "results" in tool_output:
                # Filter out invalid results
                valid_results = [
                    r for r in tool_output["results"]
                    if r.get("title") and r.get("link")
                ]

                if len(valid_results) != len(tool_output["results"]):
                    modified_output = tool_output.copy()
                    modified_output["results"] = valid_results
                    modified_output["filtered_count"] = len(tool_output["results"]) - len(valid_results)

                    return AfterToolHookResult.with_modifications(tool_output=modified_output)

        return AfterToolHookResult.no_changes()

    return validation_hook
```


#### Using Multiple Tool Hooks

```python
# Combine multiple tool hooks for comprehensive tool management
agent = create_agent(
    name="production_agent",
    tools=[web_search_tool, file_tools, database_tools],
    after_tool_hooks=[
        create_tool_logging_hook("production_tools"),
        create_tool_validation_hook(),
        create_performance_monitor_hook(),
        create_tool_output_filter_hook({"password", "secret", "api_key"}),
        create_error_recovery_hook()
    ],
    llm_config=llm_config
)
```

#### Tool Hook Result Methods

```python
# No modifications
return AfterToolHookResult.no_changes()

# Modify tool output
return AfterToolHookResult.with_modifications(tool_output=modified_output)

# Direct construction (also valid)
return AfterToolHookResult(tool_output=modified_output)
```

#### Tool Hook Use Cases

- **Debugging & Monitoring**: Log tool inputs/outputs, track performance metrics
- **Data Validation**: Validate and sanitize tool outputs before processing
- **Security & Privacy**: Filter sensitive information from tool results
- **Error Handling**: Transform error responses into user-friendly messages
- **Data Transformation**: Format, enhance, or standardize tool outputs
- **Analytics**: Collect usage statistics and performance data
- **Caching**: Implement custom caching strategies for expensive tools
- **Rate Limiting**: Monitor and control tool usage patterns

#### Tool Hook Execution Flow

1. Tool is executed with provided parameters
2. Raw tool output is generated
3. Each tool hook processes the output in sequence
4. Modified output (if any) is passed to the next hook
5. Final output is returned to the agent for processing

Tool hooks provide fine-grained control over tool behavior and enable powerful tool output processing pipelines while maintaining clean separation of concerns.

### Global Storage System

The Northau framework provides a thread-safe GlobalStorage system that allows tools and hooks to share data across the entire agent hierarchy. This is particularly useful for maintaining state across multiple tool calls and sub-agents.

#### Using GlobalStorage in Tools

Tools can optionally receive a `global_storage` parameter by including it in their function signature:

```python
from northau.archs.main_sub.agent_context import GlobalStorage

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
from northau.archs.main_sub.execution.hooks import AfterModelHook, AfterModelHookResult, AfterModelHookInput
from northau.archs.main_sub.agent_context import get_context

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
from northau.archs.main_sub.agent_context import GlobalStorage
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
