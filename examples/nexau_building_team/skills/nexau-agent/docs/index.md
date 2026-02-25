# Welcome to the NexAU Framework

NexAU is a general-purpose agent framework for building intelligent agents with tool capabilities.

## Features

- **Modular Tool System**: Easy-to-configure tools with YAML-based configuration.
- **Agent Architecture**: Create specialized agents with different capabilities.
- **Built-in Tools**: File operations, web search, bash execution, and more.
- **LLM Integration**: Support for various LLM providers (OpenAI, Claude, etc.).
- **Session Management**: Stateful session persistence with pluggable storage backends (SQL, JSONL, Memory, Remote).
- **Transport System**: Multi-protocol communication (HTTP/SSE, stdio today; WebSocket, gRPC may be added later).
- **Middleware Hooks**: Customize LLM behavior with preprocessing, caching, logging, and provider switching.
- **YAML Configuration**: Define agents and tools declaratively.

## Explore the Documentation

* **[🚀 Getting Started](./getting-started.md)**: Installation, environment setup, and running your first agent.

* **Core Concepts**
    * **[🤖 Agents](./core-concepts/agents.md)**: Learn how to create and configure agents.
    * **[🛠️ Tools](./core-concepts/tools.md)**: Use built-in tools and create your own.
    * **[🧠 LLMs](./core-concepts/llms.md)**: Configure LLM providers and middleware-based extensions.

* **Advanced Guides**
    * **[Skills](./advanced-guides/hooks.md)**: Skills (compatible with Claude Skill format) to dynamically ingest skill context (support both tool and file).
    * **[Hooks/Middleware](./advanced-guides/hooks.md)**: Intercept and modify agent behavior.
    * **[Global Storage](./advanced-guides/global-storage.md)**: Share state across tools and agents.
    * **[Templating](./advanced-guides/templating.md)**: Use Jinja2 for dynamic system prompts.
    * **[Tracer Integration](./advanced-guides/tracer.md)**: Configure and share tracers across agents.
    * **[MCP Integration](./advanced-guides/mcp.md)**: Connect to external services via MCP.
    * **[Context Compaction](./advanced-guides/context_compaction.md)**: Compact long context windows.
    * **[Image Handling](./advanced-guides/image.md)**: Handle images in messages and tools.
    * **[Streaming Events](./advanced-guides/streaming-events.md)** ⚠️ *Experimental*: Incremental, provider-agnostic events (text deltas, tool calls, etc.) for streaming UIs.
    * **[Session Management](./advanced-guides/session-management.md)** ⚠️ *Experimental*: Stateful session persistence, agent tracking, and concurrency control.
    * **[Transport System](./advanced-guides/transports.md)** ⚠️ *Experimental*: Multi-protocol communication (HTTP/SSE, stdio today; WebSocket, gRPC may be added later).
