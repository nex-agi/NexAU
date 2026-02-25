### Tracer Integration

NexAU ships with a pluggable tracing interface so you can forward execution data to Langfuse, your own observability backend, or a simple in-memory recorder. Agents accept a `tracers` list that automatically propagates the composed tracer instance to the agent's `global_storage`, making it immediately available to the main agent, sub-agents, hooks, and tools.

#### Configure tracers in YAML

Add a `tracers` list to your agent configuration. Each entry mirrors hook definitions, so you can reference an import path (string) or use the `{import, params}` form for constructors that need keyword arguments. If you provide multiple entries, NexAU automatically wraps them in a `CompositeTracer` so every span fans out to all backends.

```yaml
# agent.yaml
name: traced_agent
llm_config:
  model: gpt-4o-mini

tracers:
  - import: nexau.archs.tracer.adapters.langfuse:LangfuseTracer
    params:
      public_key: ${env.LANGFUSE_PUBLIC_KEY}
      secret_key: ${env.LANGFUSE_SECRET_KEY}
      host: https://cloud.langfuse.com
  - import: mypkg.telemetry:ConsoleTracer

tools: []
```

When `Agent.from_yaml` builds the agent, each tracer is instantiated, chained (if needed), and the resulting tracer is injected into the shared `global_storage["tracer"]`. Hooks, tool executors, and the LLM caller read from the same slot, so no additional wiring is required.

#### Provide a tracer when constructing `AgentConfig`

Python callers can also pass concrete tracer instances. This is helpful for tests or custom runners that don't rely on YAML files:

```python
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer
from nexau.archs.main_sub.agent_context import GlobalStorage

tracer = LangfuseTracer(debug=True)

agent_config = AgentConfig(
    name="code_agent",
    llm_config={"model": "gpt-4o-mini"},
    tools=[],
    tracers=[tracer],
)

agent = Agent(config=agent_config, global_storage=GlobalStorage())
# agent.global_storage.get("tracer") is the composed tracer instance
```

Because the agent constructor writes the tracer into the root `GlobalStorage`, any sub-agent created through delegation automatically observes the same tracer.

#### Accessing the tracer

Anywhere you can access `global_storage` you can retrieve the tracer:

```python
from nexau.archs.tracer.core import SpanType

tracer = global_storage.get("tracer")
if tracer:
    span = tracer.start_span("my_tool", SpanType.TOOL)
    # ...
    tracer.end_span(span, outputs={"result": "done"})
```

Hooks can retrieve it via `get_context().global_storage`, and custom execution components can look it up on demand. If no tracer is configured, the storage slot is simply absent.

#### Common patterns

1. **Per-environment overrides** – change the `tracers` via `AgentConfig(tracers=[tracer])` before calling `Agent(config = agent_config)` to swap tracer implementations without editing the base YAML.
2. **Multi-backend tracing** – list several entries under `tracers` (or construct your own `CompositeTracer`) so every span fan-outs to your preferred sinks.
3. **Test instrumentation** – inject fake tracers via `AgentConfig(tracers=[RecordingTracer()])` to assert span sequences in unit tests.

This workflow removes the need to manually mutate `global_storage` whenever you enable tracing, ensuring every agent hierarchy stays in sync with the configured tracer.
