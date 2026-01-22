# Simple Sub-Agent Example

This example demonstrates how to create an agent with sub-agents in NexAU.

## Overview

- **main_agent.yaml**: The main agent that can delegate research tasks
- **research_agent.yaml**: A sub-agent specialized in web research

## Architecture

```
Main Agent
├─ Tools: (none - focuses on coordination)
└─ Sub-Agent: research_assistant
   └─ Tools: web_search, web_read
```

## How It Works

1. The main agent receives user requests
2. When a research task is needed, it calls the `research_assistant` sub-agent
3. The research assistant uses web search and web read tools to gather information
4. The research assistant returns synthesized results to the main agent
5. The main agent delivers the final response to the user

## Usage

### Running the Agent

```bash
# Make sure you have set the required environment variables
export LLM_MODEL="your-model"
export LLM_BASE_URL="your-api-base-url"
export LLM_API_KEY="your-api-key"
export SERPER_API_KEY="your-serper-api-key"  # For web search

# Run the agent
./run-agent examples/with-sub-agent/main_agent.yaml
```

### Example Interaction

```
User: Can you research the latest developments in quantum computing?

Main Agent: I'll delegate this research task to my research assistant.
           [Calls research_assistant sub-agent]

Research Assistant:
           [Uses web_search to find sources]
           [Uses web_read to read content]
           [Returns synthesized information]

Main Agent: Based on my research assistant's findings, here's what I learned...
```

## Design Principles

### Safety First

The main agent intentionally has **no tools** - it focuses purely on coordination and delegation. This is a safe design pattern because:

- The main agent can't execute dangerous commands
- All tool usage is isolated to specialized sub-agents
- You have fine-grained control over what each sub-agent can do

If you need to give the main agent tools, prefer safe ones like `todo_write`, `file_read`, or domain-specific tools.

## Key Concepts

### Sub-Agent Definition

In `main_agent.yaml`:

```yaml
sub_agents:
  - name: research_assistant
    config_path: ./research_agent.yaml
```

### Sub-Agent Description

In `research_agent.yaml`, the `description` field helps the parent agent understand when to use this sub-agent:

```yaml
description: >-
  A specialized research agent that can search the web for information.

  Usage:
  - Assign a research question and get back synthesized information
  - Can search the web and read web pages
  - Good for finding recent information, facts, and data
```

### Calling a Sub-Agent

The framework automatically creates a `call_sub_agent` tool for each defined sub-agent. The main agent can call it like:

```
call_sub_agent(
  agent_name="research_assistant",
  message="Research the latest quantum computing breakthroughs"
)
```

## Customization

You can extend this example by:

1. Adding more sub-agents (e.g., a code generation agent, a data analysis agent)
2. Giving the main agent more tools for direct tasks
3. Creating hierarchical sub-agents (sub-agents that have their own sub-agents)
4. Adding middleware for logging, tracing, or context management

## See Also

- [Deep Research Example](../deep_research/) - A more complex research agent with nested sub-agents
- [Code Agent Example](../code_agent/) - A comprehensive coding assistant
