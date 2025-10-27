# Welcome to the Northau Framework

Northau is a general-purpose agent framework inspired by Claude Code's architecture for building intelligent agents with tool capabilities.

## Features

- **Modular Tool System**: Easy-to-configure tools with YAML-based configuration.
- **Agent Architecture**: Create specialized agents with different capabilities.
- **Built-in Tools**: File operations, web search, bash execution, and more.
- **LLM Integration**: Support for various LLM providers (OpenAI, Claude, etc.).
- **Custom LLM Generators**: Customize LLM behavior with custom preprocessing, caching, logging, and provider switching.
- **YAML Configuration**: Define agents and tools declaratively.

## Explore the Documentation

* **[🚀 Getting Started](./getting-started.md)**: Installation, environment setup, and running your first agent.

* **Core Concepts**
    * **[🤖 Agents](./core-concepts/agents.md)**: Learn how to create and configure agents.
    * **[🛠️ Tools](./core-concepts/tools.md)**: Use built-in tools and create your own.
    * **[🧠 LLMs](./core-concepts/llms.md)**: Configure LLM providers and custom generators.

* **Advanced Guides**
    * **[Skills](./advanced-guides/hooks.md)**: Skills (compatible with Claude Skill format) to dynamically ingest skill context (support both tool and file).
    * **[Hooks](./advanced-guides/hooks.md)**: Intercept and modify agent behavior.
    * **[Global Storage](./advanced-guides/global-storage.md)**: Share state across tools and agents.
    * **[Templating](./advanced-guides/templating.md)**: Use Jinja2 for dynamic system prompts.
    * **[Tracing](./advanced-guides/tracing.md)**: Dump execution traces to a file.
    * **[MCP Integration](./advanced-guides/mcp.md)**: Connect to external services via MCP.
    * **[Custom LLM Generator](./advanced-guides/custom-llm-generator.md)**: Custom logics for calling LLMs.

## Todos
- [ ] Currently, sub-agent context is destoryed after sub-agent finishes. Sometimes, deliverables may be done by sub-agents and users want to interact with these sub-agents. We need to support persistent sub-agents.
- [ ] Currently, event streaming requires developers to use hooks to implement custom event streaming logics. A better design is a unified hook class with on_xxx events to provide a unified interface for event streaming.
- [ ] Support GeneralAgent class instead of just ReAct-style agent, which supports mixture of workflow (pythonic defined) and NorthAU Agent.
- [ ] Support both xml and tool_call format
