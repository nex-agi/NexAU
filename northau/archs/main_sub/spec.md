# Northau Framework Specification

## Overview

Northau is a general-purpose agent framework inspired by Claude Code's architecture, supporting main agents with specialized sub-agents for efficient task delegation and execution.

## Quick Start Example

```python
from northau.archs.main_sub import create_agent
from northau.archs.tool import Tool
from northau.archs.tool.builtin.bash_tool import bash
from northau.archs.tool.builtin.file_tool import file_edit, file_search
from northau.archs.tool.builtin.web_tool import web_search, web_read



bash_tool = Tool.from_yaml("Bash.tool.yaml", binding=bash)
edit_tool = Tool.from_yaml("Edit.tool.yaml", binding=file_edit)
file_search_tool = Tool.from_yaml("Grep.tool.yaml", binding=file_search)
web_search_tool = Tool.from_yaml("WebSearch.tool.yaml",binding=web_search)
web_read_tool = Tool.from_yaml("WebRead.tool.yaml",binding=web_read)

file_search_agent = create_agent(
  tools = [file_search_tool]
)
deep_research_agent = create_agent(
  tools = [web_search_tool, web_read_tool]
)

main_agent = create_agent(
  tools = [bash_tool, edit_tool],
  sub_agents = [file_search_agent, deep_research_agent]
)

main_agent.run(message="implement a code assistant project")
```

## Architecture

### Core Principles

1. **Main-Sub Agent Pattern**: Primary agent handles general tasks, delegates specialized work to sub-agents
2. **Tool Modularity**: Tools are defined declaratively via YAML and bound to Python implementations
3. **Domain Agnostic**: Framework can be customized for any task domain, not just coding
4. **Context Efficiency**: Long-context consuming tasks are delegated to specialized sub-agents
5. **Composability**: Agents and tools can be composed into complex workflows

### Agent Hierarchy

```
Main Agent
├── General Tools (bash, edit, etc.)
├── Sub-Agent: File Operations
│   └── Tools: file_search, file_read, file_write
├── Sub-Agent: Web Research
│   └── Tools: web_search, web_read, web_scrape
└── Sub-Agent: Domain Specific
    └── Tools: custom domain tools
```

## Core Components

### 1. LLM Configuration System

The framework uses a dedicated `LLMConfig` class to handle all LLM-related parameters in an extensible way:

#### LLMConfig Class

```python
from northau.archs.llm import LLMConfig

# Basic configuration
config = LLMConfig(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=4096
)

# Advanced configuration with custom parameters
config = LLMConfig(
    model="***REMOVED***",
    base_url="https://***REMOVED***/v1/",
    api_key="***REMOVED***",
    temperature=0.6,
    max_tokens=8192,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    timeout=30.0,
    max_retries=3,
    # Custom parameters for specific providers
    custom_param="value",
    provider_specific_setting=True
)

# Dynamic parameter management
config.set_param("new_param", "value")
value = config.get_param("custom_param", "default")
config.update(temperature=0.8, max_tokens=2048)
```

#### Supported Parameters

**Standard OpenAI Parameters:**
- `model`: Model identifier
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Top-p sampling parameter
- `frequency_penalty`: Frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Presence penalty (-2.0 to 2.0)

**Connection Parameters:**
- `base_url`: API endpoint URL
- `api_key`: Authentication key
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum retry attempts

**Extensible Parameters:**
- Any additional parameters via `**kwargs`
- Dynamic parameter setting with `set_param()`
- Provider-specific configurations

#### Popular Provider Examples

```python
# OpenAI Official
LLMConfig(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="sk-your-key"
)

# Ollama (Local)
LLMConfig(
    model="llama2",
    base_url="http://localhost:11434/v1"
)

# Together AI
LLMConfig(
    model="meta-llama/Llama-2-70b-chat-hf",
    base_url="https://api.together.xyz/v1",
    api_key="your-together-key"
)

# Groq
LLMConfig(
    model="mixtral-8x7b-32768",
    base_url="https://api.groq.com/openai/v1",
    api_key="your-groq-key"
)

# Custom Proxy
LLMConfig(
    model="***REMOVED***",
    base_url="https://***REMOVED***/v1/",
    api_key="***REMOVED***",
    temperature=0.6,
    max_tokens=8192
)
```

### 2. Agent System

#### Agent Creation

```python
from northau.archs.main_sub import create_agent
from northau.archs.llm import LLMConfig

# Method 1: Using LLMConfig object (recommended)
llm_config = LLMConfig(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=4096
)

agent = create_agent(
    name="my_agent",                    # Optional: agent identifier
    tools=[tool1, tool2],              # List of Tool objects
    sub_agents=[sub1, sub2],           # List of sub-agents for delegation
    system_prompt="Custom prompt",      # Optional: system behavior (string)
    system_prompt_type="string",        # Optional: string|file|jinja
    llm_config=llm_config,             # LLM configuration
    max_context=100000                 # Optional: context window limit
)

# Method 2: Using dictionary
agent = create_agent(
    name="my_agent",
    llm_config={
        "model": "***REMOVED***",
        "base_url": "https://***REMOVED***/v1/",
        "api_key": "***REMOVED***",
        "temperature": 0.6,
        "max_tokens": 8192
    }
)

# Method 3: Backward compatibility (deprecated)
agent = create_agent(
    name="my_agent",
    model="gpt-4",                     # Deprecated: use llm_config
    model_base_url="https://api.openai.com/v1",  # Deprecated: use llm_config
    temperature=0.7                    # Additional LLM params
)
```

#### Agent Methods

```python
# Run a single message
response = agent.run(message="Your task here")

# Run with conversation history
response = agent.run(
    message="Follow up task",
    history=[{"role": "user", "content": "Previous message"}]
)

# Stream responses
for chunk in agent.stream(message="Your task"):
    print(chunk, end="")

# Add tools and sub-agents dynamically
agent.add_tool(new_tool)
agent.add_sub_agent("name", new_sub_agent)

# Call sub-agents directly
result = agent.call_sub_agent("sub_agent_name", "task message")

# Different system prompt types
# String prompt (default)
string_agent = create_agent(
    system_prompt="You are a helpful coding assistant.",
    system_prompt_type="string"
)

# File-based prompt (markdown)
file_agent = create_agent(
    system_prompt="prompts/specialist.md",
    system_prompt_type="file"
)

# Jinja template prompt
template_agent = create_agent(
    system_prompt="prompts/dynamic.jinja",
    system_prompt_type="jinja"
)
```

### 2. Tool System

#### Tool Definition (YAML)

Tools are defined using YAML files with JSON Schema validation:

```yaml
name: Edit
description: >-
  Performs exact string replacements in files.

  Usage:
  - You must use your `Read` tool at least once in the conversation before
  editing. This tool will error if you attempt an edit without reading the file.
  
  - When editing text from Read tool output, ensure you preserve the exact
  indentation (tabs/spaces) as it appears AFTER the line number prefix. The line
  number prefix format is: spaces + line number + tab. Everything after that tab
  is the actual file content to match. Never include any part of the line number
  prefix in the old_string or new_string.
  
  - ALWAYS prefer editing existing files in the codebase. NEVER write new files
  unless explicitly required.
  
  - Only use emojis if the user explicitly requests it. Avoid adding emojis to
  files unless asked.
  
  - The edit will FAIL if `old_string` is not unique in the file. Either provide
  a larger string with more surrounding context to make it unique or use
  `replace_all` to change every instance of `old_string`.
  
  - Use `replace_all` for replacing and renaming strings across the file. This
  parameter is useful if you want to rename a variable for instance.
input_schema:
  type: object
  properties:
    file_path:
      type: string
      description: The absolute path to the file to modify
    old_string:
      type: string
      description: The text to replace
    new_string:
      type: string
      description: The text to replace it with (must be different from old_string)
    replace_all:
      type: boolean
      default: false
      description: Replace all occurences of old_string (default false)
  required:
    - file_path
    - old_string
    - new_string
  additionalProperties: false
  $schema: http://json-schema.org/draft-07/schema#
```

#### Tool Implementation and Binding

```python
from northau.archs.tool import Tool

# Define the Python implementation
def file_edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False):
    """Implementation of file editing logic"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        if content.count(old_string) != 1:
            raise ValueError("old_string must be unique in file")
        new_content = content.replace(old_string, new_string, 1)
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    return {"status": "success", "message": f"File {file_path} updated"}

# Bind YAML definition to Python implementation
edit_tool = Tool.from_yaml("Edit.tool.yaml", binding=file_edit)

# Alternative: Create tool programmatically
edit_tool = Tool(
    name="Edit",
    description="Performs exact string replacements in files.",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to file"},
            "old_string": {"type": "string", "description": "Text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "default": False}
        },
        "required": ["file_path", "old_string", "new_string"]
    },
    implementation=file_edit
)
```

### 3. Built-in Tools

#### File Operations
- **Read**: Read file contents with line numbers
- **Edit**: String replacement in files
- **Write**: Create or overwrite files
- **Grep**: Search for patterns in files
- **Glob**: Find files matching patterns
- **LS**: List directory contents

#### Web Tools
- **WebSearch**: Search the web for information
- **WebFetch**: Fetch and parse web page content
- **WebScrape**: Extract structured data from websites

#### System Tools
- **Bash**: Execute shell commands
- **Process**: Manage system processes
- **Environment**: Access environment variables

#### Utility Tools
- **JSON**: Parse and manipulate JSON data
- **XML**: Parse and manipulate XML data
- **CSV**: Handle CSV file operations
- **Image**: Basic image processing operations

### 4. LLM Integration with XML-based Tool and Sub-Agent Calls

The framework integrates with OpenAI's API and uses XML-based syntax for tool calls and sub-agent delegation. This provides a clear, structured way for the LLM to invoke capabilities:

#### Tool Calls
The LLM can use tools by including XML blocks in its response:

```xml
<tool_use>
  <tool_name>Bash</tool_name>
  <parameters>
    <command>ls -la</command>
    <timeout>30</timeout>
  </parameters>
</tool_use>
```

#### Sub-Agent Calls
Similarly, sub-agents are called using XML syntax:

```xml
<sub_agent>
  <agent_name>file_search</agent_name>
  <message>find all Python files in the project</message>
</sub_agent>
```

#### Implementation
```python
# Agent automatically processes XML calls in LLM responses
main_agent = create_agent(
    tools=[bash_tool, edit_tool],
    sub_agents=[
        ("file_search", file_search_agent),
        ("web_research", web_research_agent)
    ],
    model="gpt-4",  # Uses OpenAI API
    model_base_url="https://api.openai.com/v1"  # Can use other OpenAI-compatible endpoints
)

# Examples of other compatible endpoints:
# Ollama (local): model_base_url="http://localhost:11434/v1"
# LM Studio (local): model_base_url="http://localhost:1234/v1"  
# Together AI: model_base_url="https://api.together.xyz/v1"
# Groq: model_base_url="https://api.groq.com/openai/v1"

# The LLM receives documentation about available tools and sub-agents
# When it responds with XML blocks, they are automatically executed
response = main_agent.run("List files and search for Python files")

# Manual delegation is also available:
result = main_agent.call_sub_agent("file_search", "find all TODO comments")
```

## Usage Patterns

### Code Assistant

```python
# Specialized for software development
bash_tool = Tool.from_yaml("tools/Bash.yaml", binding=bash)
edit_tool = Tool.from_yaml("tools/Edit.yaml", binding=file_edit)
read_tool = Tool.from_yaml("tools/Read.yaml", binding=file_read)
grep_tool = Tool.from_yaml("tools/Grep.yaml", binding=file_search)

file_agent = create_agent(tools=[read_tool, grep_tool])
code_agent = create_agent(
    tools=[bash_tool, edit_tool],
    sub_agents=[("file_ops", file_agent)],
    system_prompt="You are a software engineering assistant."
)

code_agent.run("Add error handling to the login function")
```

### Research Assistant

```python
# Specialized for information gathering and analysis
web_search_tool = Tool.from_yaml("tools/WebSearch.yaml", binding=web_search)
web_fetch_tool = Tool.from_yaml("tools/WebFetch.yaml", binding=web_fetch)
json_tool = Tool.from_yaml("tools/JSON.yaml", binding=json_parse)

web_agent = create_agent(tools=[web_search_tool, web_fetch_tool])
research_agent = create_agent(
    tools=[json_tool],
    sub_agents=[("web_research", web_agent)],
    system_prompt="You are a research assistant that gathers and analyzes information."
)

research_agent.run("Research the latest developments in quantum computing")
```

### Data Analysis Assistant

```python
# Specialized for data processing and analysis
csv_tool = Tool.from_yaml("tools/CSV.yaml", binding=csv_operations)
plot_tool = Tool.from_yaml("tools/Plot.yaml", binding=plotting)
stats_tool = Tool.from_yaml("tools/Statistics.yaml", binding=statistics)

data_agent = create_agent(
    tools=[csv_tool, plot_tool, stats_tool],
    system_prompt="You are a data analyst. Always visualize results when possible."
)

data_agent.run("Analyze the sales data in quarterly_report.csv")
```

### Content Creation Assistant

```python
# Specialized for writing and content generation
grammar_tool = Tool.from_yaml("tools/Grammar.yaml", binding=grammar_check)
style_tool = Tool.from_yaml("tools/StyleCheck.yaml", binding=style_check)
image_tool = Tool.from_yaml("tools/ImageGen.yaml", binding=image_generation)

writing_agent = create_agent(
    tools=[grammar_tool, style_tool, image_tool],
    system_prompt="You are a professional content creator and editor."
)

writing_agent.run("Write a blog post about sustainable living with accompanying images")
```

## Advanced Configuration

### Multi-Level Agent Hierarchies

```python
# Three-tier hierarchy: Main -> Specialist -> Expert
expert_file_agent = create_agent(
    tools=[advanced_grep_tool, ast_parser_tool],
    system_prompt="Expert in code analysis and AST manipulation"
)

specialist_code_agent = create_agent(
    tools=[edit_tool, bash_tool],
    sub_agents=[("expert_analysis", expert_file_agent)],
    system_prompt="Specialist in software development tasks"
)

main_agent = create_agent(
    tools=[general_tools],
    sub_agents=[
        ("code_specialist", specialist_code_agent),
        ("research_specialist", research_agent),
        ("data_specialist", data_agent)
    ],
    system_prompt="General assistant that delegates complex tasks to specialists"
)
```

### Custom Error Handling

```python
def custom_error_handler(error, agent, context):
    if isinstance(error, ToolNotFoundError):
        return "I don't have the right tool for this task. Let me try a different approach."
    elif isinstance(error, SubAgentTimeoutError):
        return "The task is taking longer than expected. Let me try with a simpler approach."
    else:
        return f"An error occurred: {error}. Let me try to resolve this."

agent = create_agent(
    tools=[...],
    error_handler=custom_error_handler,
    retry_attempts=5,
    timeout=300
)
```


## Extension Guide

### Creating Custom Tools

1. **Define the YAML schema**:

```yaml
name: CustomTool
description: Description of what this tool does
input_schema:
  type: object
  properties:
    param1:
      type: string
      description: Description of parameter
    param2:
      type: integer
      default: 42
  required: [param1]
  additionalProperties: false
  $schema: http://json-schema.org/draft-07/schema#
```

2. **Implement the Python function**:

```python
def custom_tool_implementation(param1: str, param2: int = 42):
    """
    Implement your tool logic here.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Dict with result data
    """
    # Your implementation
    result = do_something(param1, param2)
    return {"result": result, "status": "success"}
```

3. **Create and register the tool**:

```python
from northau.archs.tool import Tool

custom_tool = Tool.from_yaml("CustomTool.yaml", binding=custom_tool_implementation)

# Use in agent
agent = create_agent(tools=[custom_tool])
```

### Creating Domain-Specific Agents

```python
def create_domain_agent(domain_config):
    """Factory function for domain-specific agents"""
    
    # Load domain-specific tools
    tools = []
    for tool_config in domain_config['tools']:
        tool = Tool.from_yaml(
            tool_config['yaml_path'], 
            binding=tool_config['implementation']
        )
        tools.append(tool)
    
    # Create sub-agents if needed
    sub_agents = []
    for sub_config in domain_config.get('sub_agents', []):
        sub_agent = create_domain_agent(sub_config)
        sub_agents.append((sub_config['name'], sub_agent))
    
    return create_agent(
        name=domain_config['name'],
        tools=tools,
        sub_agents=sub_agents,
        system_prompt=domain_config['system_prompt'],
        **domain_config.get('agent_params', {})
    )

# Usage
marketing_config = {
    'name': 'marketing_assistant',
    'tools': [
        {'yaml_path': 'tools/SocialMedia.yaml', 'implementation': social_media_api},
        {'yaml_path': 'tools/Analytics.yaml', 'implementation': analytics_api}
    ],
    'system_prompt': 'You are a marketing specialist...',
    'agent_params': {'model': 'claude-3-opus'}
}

marketing_agent = create_domain_agent(marketing_config)
```

## API Reference

### Core Functions

#### `create_agent()`

```python
def create_agent(
    name: Optional[str] = None,
    tools: List[Tool] = None,
    sub_agents: List[Tuple[str, Agent]] = None,
    system_prompt: Optional[str] = None,
    system_prompt_type: str = "string",
    model: str = "claude-3-sonnet",
    model_base_url: Optional[str] = None,
    max_context: int = 100000,
    error_handler: Optional[Callable] = None,
    retry_attempts: int = 5,
    timeout: int = 300
) -> Agent:
    """
    Create a new agent with specified configuration.
    
    Args:
        name: Optional identifier for the agent
        tools: List of Tool objects available to this agent
        sub_agents: List of (name, agent) tuples for delegation
        system_prompt: System behavior instructions (content or file path)
        system_prompt_type: Type of system prompt (string|file|jinja)
        model: LLM model to use (gpt-4, gpt-3.5-turbo, etc.)
        model_base_url: Base URL for OpenAI-compatible API endpoints
        max_context: Maximum context window size
        error_handler: Custom error handling function
        retry_attempts: Number of retries on failure
        timeout: Task timeout in seconds
    
    Returns:
        Configured Agent instance
    """
```

### Tool Class

```python
class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict,
        implementation: Callable,
        cache_results: bool = False,
        timeout: Optional[int] = None
    ):
        """Initialize a tool with schema and implementation."""
        
    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        binding: Callable,
        **kwargs
    ) -> 'Tool':
        """Load tool definition from YAML file and bind to implementation."""
        
    def execute(self, **params) -> Dict:
        """Execute the tool with given parameters."""
        
    def validate_params(self, params: Dict) -> bool:
        """Validate parameters against schema."""
```

### Agent Class

```python
class Agent:
    def run(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> str:
        """Run agent with a message and return response."""
        
    def stream(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> Iterator[str]:
        """Stream agent response in chunks."""
        
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        
    def add_sub_agent(self, name: str, agent: 'Agent') -> None:
        """Add a sub-agent for delegation."""
        
    def delegate_task(
        self,
        task: str,
        sub_agent_name: str,
        context: Optional[Dict] = None
    ) -> str:
        """Explicitly delegate a task to a sub-agent."""
        
    def call_sub_agent(
        self,
        sub_agent_name: str,
        message: str,
        context: Optional[Dict] = None
    ) -> str:
        """Call a sub-agent like a tool call."""
```

## Configuration Files

### Agent Configuration (YAML)

```yaml
# config/agents/code_assistant.yaml
name: code_assistant
max_context: 100000
system_prompt: |
  You are a software engineering assistant. You help with coding tasks,
  debugging, and project management.
system_prompt_type: string  # string|file|jinja

# New LLM configuration format (recommended)
llm_config:
  model: gpt-4
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}  # Environment variable
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.9

# Alternative: Custom proxy configuration
# llm_config:
#   model: "***REMOVED***"
#   base_url: "https://***REMOVED***/v1/"
#   api_key: "***REMOVED***"
#   temperature: 0.6
#   max_tokens: 8192

# Backward compatibility (deprecated)
# model: gpt-4
# model_base_url: https://api.openai.com/v1

tools:
  - name: bash
    yaml_path: tools/Bash.yaml
    binding: northau.archs.tool.builtin.bash_tool:bash
  - name: edit
    yaml_path: tools/Edit.yaml
    binding: northau.archs.tool.builtin.file_tool:file_edit

sub_agents:
  - name: file_search
    config_path: config/agents/file_search_agent.yaml
  - name: web_research
    config_path: config/agents/web_research_agent.yaml

# Sub-agents are automatically used by LLM when appropriate
# No manual delegation rules needed

```

### System Prompt Types

The framework supports three types of system prompts:

#### 1. String Type (Default)
```yaml
system_prompt: |
  You are a helpful assistant specialized in software development.
  Always write clean, well-documented code.
system_prompt_type: string
```

#### 2. File Type (Markdown)
```yaml
system_prompt: prompts/code_assistant.md
system_prompt_type: file
```

Where `prompts/code_assistant.md` contains:
```markdown
# Code Assistant System Prompt

You are a software engineering assistant with expertise in:

- **Code Review**: Analyze code for bugs, performance, and best practices
- **Documentation**: Generate clear, comprehensive documentation
- **Testing**: Write unit tests and integration tests
- **Refactoring**: Improve code structure and maintainability

## Guidelines
- Always prioritize code quality and readability
- Follow established coding conventions
- Provide explanations for complex solutions
```

#### 3. Jinja Template Type
```yaml
system_prompt: prompts/dynamic_assistant.jinja
system_prompt_type: jinja
```

Where `prompts/dynamic_assistant.jinja` contains:
```jinja
You are a {{ domain }} assistant specialized in {{ specialty }}.

{% if tools %}
Available tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
{% endif %}

{% if sub_agents %}
You can delegate tasks to these specialized agents:
{% for name, agent in sub_agents %}
- {{ name }}: {{ agent.description }}
{% endfor %}
{% endif %}

Current context: {{ context.task_type | default("general") }}
User expertise level: {{ context.user_level | default("intermediate") }}
```

### Loading from Configuration

```python
from northau.archs.config import load_agent_config

# Load agent from configuration file
agent = load_agent_config("config/agents/code_assistant.yaml")

# Override specific parameters
agent = load_agent_config(
    "config/agents/code_assistant.yaml",
    overrides={"model": "claude-3-opus", "max_context": 200000}
)

# Load with jinja template context
agent = load_agent_config(
    "config/agents/dynamic_assistant.yaml",
    template_context={
        "domain": "data science",
        "specialty": "machine learning",
        "context": {"task_type": "analysis", "user_level": "expert"}
    }
)
```

This comprehensive specification provides a complete foundation for building a flexible, extensible agent framework that can be customized for any domain while maintaining the efficient main-sub agent architecture inspired by Claude Code.