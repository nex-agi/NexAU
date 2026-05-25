# Workflows

> Experimental: Workflow APIs and YAML fields are still evolving.

NexAU workflows are durable YAML graphs executed by `WorkflowExecutor`. They are useful when you need deterministic orchestration around agents and tools: structured outputs, branches, loops, human checkpoints, reusable subgraphs, and bounded parallel map execution.

The workflow runtime persists an append-only event log at node boundaries. A run can stop at a human checkpoint, survive process restart, reload from the run definition snapshot, and continue from the next safe boundary.

## When to Use Workflows

Use a workflow when the control flow matters as much as the model behavior:

| Use case | Workflow fit |
| --- | --- |
| Generate data with one agent, validate it, then pass it to another agent | Agent nodes with `output_schema` |
| Pause for approval or missing information | `human` nodes and `resume_async(...)` |
| Reuse a review, normalization, or execution sequence | External `subgraph` nodes |
| Process many independent items with controlled concurrency | `parallel_map` |
| Recover from worker restart or uncertain side effects | Durable event log, leases, and `reconcile_async(...)` |

Use `AgentTeam` instead when you want open-ended multi-agent collaboration through a task board and message bus. A workflow can call an agent that internally uses team behavior, but workflow edges remain deterministic.

## Minimal Workflow

A workflow file is a JSON-compatible YAML document. Each graph must have exactly one `start` node and an acyclic `edges` map.

```yaml
type: workflow
version: "1"
name: approval_flow

inputs:
  title:
    type: string

nodes:
  start:
    type: start
    output:
      title: "{{ inputs.title }}"

  review:
    type: human
    prompt: Review this title.
    input:
      title: "{{ nodes.start.output.title }}"
    output_schema:
      type: object
      properties:
        approved:
          type: boolean
        title:
          type: string
      required: [approved, title]

  finish:
    type: end
    output:
      approved: "{{ nodes.review.output.approved }}"
      title: "{{ nodes.review.output.title }}"

edges:
  start: review
  review: finish
```

Use an `end` node when you want explicit root workflow output. Subgraphs may also use a graph-level `output`, but an `end` node is the clearest output boundary for user-facing workflows.

## Loading and Running

The direct Python API is the simplest way to embed workflows in scripts or tests.

```python
import asyncio

from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.workflow import WorkflowConfig, WorkflowExecutor, WorkflowStore


async def main() -> None:
    engine = InMemoryDatabaseEngine()
    session_manager = SessionManager(engine=engine)
    store = WorkflowStore(engine)
    workflow = WorkflowConfig.from_yaml("workflow.yaml")

    executor = WorkflowExecutor(workflow=workflow, store=store)
    result = await executor.run_async(
        inputs={"title": "Release checklist"},
        run_id="wf_release_checklist",
        session_manager=session_manager,
    )

    if result.status.value == "waiting" and result.checkpoint_id is not None:
        result = await executor.resume_async(
            run_id=result.run_id,
            checkpoint_id=result.checkpoint_id,
            output={"approved": True, "title": "Release checklist"},
            session_manager=session_manager,
        )

    print(result.status.value)
    print(result.output)


asyncio.run(main())
```

`WorkflowConfig.from_yaml(...)` resolves relative `includes.graphs` immediately and stores an expanded definition snapshot when the run starts. Recovery prefers the snapshot, so changing YAML on disk does not change an already-started run.

For persistent runs, pass a SQL-backed engine to `WorkflowStore` and `SessionManager` instead of `InMemoryDatabaseEngine`.

## Runtime Dependencies

Agent, tool, and MCP nodes can resolve dependencies from constructor arguments or from `includes`.

```yaml
includes:
  agents:
    planner: ./agents/planner.yaml
  tools:
    normalize: ./tools/normalize.tool.yaml
```

```python
executor = WorkflowExecutor(
    workflow=workflow,
    store=store,
    agents={"planner": planner_config},
    tools={"normalize": normalize_tool},
    tool_registry=tool_registry,
    mcp_tools={"github.search": github_search_tool},
)
```

Resolution order:

| Node kind | Runtime lookup |
| --- | --- |
| `agent` | `agents` dict, then `includes.agents` |
| `tool` | `tools` dict, then `tool_registry`, then `includes.tools` |
| `mcp` | `mcp_tools` by `server.tool`, `server:tool`, or `tool`, then normal tool resolution |

## Expressions and Data Flow

Workflow templates use `{{ ... }}` expressions. Expressions are evaluated with the side-effect-free CEL interpreter. If a string is exactly one template expression, the original JSON type is preserved. If it is embedded in other text, the value is stringified.

Available expression context:

| Name | Meaning |
| --- | --- |
| `inputs` | Inputs passed to the current graph frame |
| `vars` | Variables from the current graph YAML |
| `state` | Durable state for the current graph or item scope |
| `nodes` | Completed node status and output visible inside the current graph frame |
| `run.id` | Workflow run id |
| `node.id`, `node.scope_path`, `node.graph_id`, `node.depth` | Current node metadata |

`parallel_map` body nodes also get the configured item and index names, plus `item_key`.

```yaml
input:
  title: "{{ nodes.start.output.title }}"
  count: "{{ state.items.length }}"
  rest: "{{ state.items[1:] }}"
```

The runtime adds compatibility for `.length` and simple slice syntax such as `items[1:]`.

## Node Types

| Type | Required fields | Output behavior |
| --- | --- | --- |
| `start` | none | Renders `output` or returns graph inputs |
| `agent` | `agent` | Runs a NexAU agent; with no schema returns `{"result": text}` |
| `tool` | `tool` | Calls a `Tool` with rendered `input` |
| `mcp` | `server`, `tool` | Calls an MCP-backed tool |
| `human` | `prompt` | Creates a durable checkpoint and waits for resume output |
| `if_else` | `branches` or `else` | Returns `{"next": "<node id>"}` and routes to that node |
| `while` | `condition`, `max_iterations`, `body` | Runs one body node repeatedly with durable scopes |
| `parallel_map` | `items`, `body` | Runs one body node once per item with bounded concurrency |
| `subgraph` | `graph` | Executes an included workflow file inline |
| `transform` | none | Renders `output` or returns rendered `input` |
| `set_state` | none | Renders `update`, patches durable state, and returns the patch |
| `end` | none | Renders final node output |
| `note` | none | No-op annotation node |

Common node fields:

| Field | Applies to | Meaning |
| --- | --- | --- |
| `input` | agent, tool, mcp, human, subgraph, transform | Rendered JSON object passed to the node |
| `output` | start, transform, end | Rendered output value |
| `update` | agent, tool, mcp, set_state | State patch rendered after node output is available |
| `output_schema` | agent, human, subgraph, parallel_map, workflow graph | JSON Schema validation for structured output |
| `retry_policy` | executable nodes | Overrides `durable.default_retry_policy` |
| `side_effect` | executable nodes | `read_only`, `local_write`, `idempotent_write`, or `external_write` |
| `idempotency_key` | write nodes | Stable key rendered before execution |

## Structured Agent Output

An `agent` node with `output_schema` asks the agent to return a JSON object that validates against the schema.

```yaml
draft_report:
  type: agent
  agent: writer
  input:
    topic: "{{ inputs.topic }}"
  output_mode: complete_task
  output_retries: 2
  output_schema:
    type: object
    additionalProperties: false
    properties:
      title:
        type: string
      markdown:
        type: string
    required: [title, markdown]
```

Supported `output_mode` values are `auto`, `native`, `complete_task`, and `json_block`. The portable runtime path currently treats `auto`, `native`, and `complete_task` as the dynamic `complete_task` stop-tool strategy. `json_block` asks the model for exactly one fenced JSON block. `output_retries` controls schema-repair attempts, and `output_name` changes the friendly name used by the dynamic tool.

Without `output_schema`, the agent runs normally and the workflow wraps the final text as `{"result": "..."}`.

## Durable Execution

The only durable mode today is `node_boundary`.

```yaml
durable:
  mode: node_boundary
  lease_timeout_seconds: 60
  default_parallelism: 2
  max_parallelism: 8
  max_subgraph_depth: 5
  default_retry_policy:
    max_attempts: 2
    backoff: exponential
    on_uncertain: human_review
```

The event log is the canonical source of truth. Materialized run, node, checkpoint, lease, and state rows are maintained for fast lookup.

Important recovery behavior:

| Situation | Runtime behavior |
| --- | --- |
| Completed node exists in the event log | Skipped on recovery |
| Human checkpoint is open | Run status is `waiting` |
| Non-external running node lease expires | Node is scheduled for retry |
| `external_write` running node lease expires | Node becomes `uncertain` |
| Run definition changed after start | Existing run uses its saved definition snapshot |

Write nodes with `side_effect: idempotent_write` or `side_effect: external_write` must provide either an `idempotency_key` or an `on_uncertain` policy. For `parallel_map`, a side-effecting body also requires an explicit `item_key`.

Use `reconcile_async(...)` for uncertain nodes:

```python
await executor.reconcile_async(
    run_id="wf_release_checklist",
    node_id="send_notification",
    scope_path="",
    decision="completed",
    output={"message_id": "msg_123"},
)
```

`decision` must be `completed`, `failed`, or `retry`.

## Human Checkpoints

A `human` node creates a checkpoint and returns a `WorkflowRunResult` with `status == waiting`.

```yaml
review:
  type: human
  prompt: "{{ nodes.start.output.reviewer_hint }}"
  input:
    cases: "{{ nodes.generate_cases.output.cases }}"
  output_schema:
    type: object
    properties:
      approved:
        type: boolean
      cases:
        type: array
    required: [approved, cases]
```

Resume output is validated against `output_schema`, persisted as the human node output, and execution continues from the successor node.

When multiple checkpoints are open, such as a `parallel_map` over a subgraph containing a human node, the result includes both `checkpoint_id` and `checkpoint_ids`. Resume each checkpoint independently.

## Subgraphs

Each `*.workflow.yaml` file is one graph. A parent graph references child graph files through `includes.graphs`, then invokes one with a `subgraph` node.

```yaml
includes:
  graphs:
    human_case_review: ./graphs/human_case_review.workflow.yaml

nodes:
  review_cases:
    type: subgraph
    graph: human_case_review
    input:
      cases: "{{ nodes.generate_cases.output.cases }}"
      reviewer_hint: Review the generated QA cases before execution.
    output_schema:
      type: object
      properties:
        approved:
          type: boolean
        cases:
          type: array
      required: [approved, cases]
```

The child file uses the same workflow format:

```yaml
type: workflow
version: "1"
name: human_case_review

inputs:
  cases:
    type: array
  reviewer_hint:
    type: string

nodes:
  start:
    type: start
    output:
      cases: "{{ inputs.cases }}"
      reviewer_hint: "{{ inputs.reviewer_hint }}"

  review:
    type: human
    prompt: "{{ nodes.start.output.reviewer_hint }}"
    input:
      cases: "{{ nodes.start.output.cases }}"

  finish:
    type: end
    output:
      approved: "{{ nodes.review.output.approved }}"
      cases: "{{ nodes.review.output.cases }}"

edges:
  start: review
  review: finish
```

Subgraph inputs, state, and node outputs are isolated from the parent. The parent reads only `nodes.<subgraph_node>.output`. The runtime rejects recursive graph includes and enforces `durable.max_subgraph_depth`.

Optional `state_in` copies parent values into the child frame state before the subgraph runs. Optional `state_out` patches parent state after the child frame completes.

```yaml
review_cases:
  type: subgraph
  graph: human_case_review
  input:
    cases: "{{ nodes.generate_cases.output.cases }}"
  state_in:
    reviewer: "{{ inputs.reviewer }}"
  state_out:
    reviewed_count: "{{ nodes.review_cases.output.cases.size() }}"
```

Subgraph events include `graph_id`, `parent_node_id`, `subgraph`, `depth`, and `scope_path` metadata. A child human checkpoint pauses the same workflow run with a scoped path such as `review_cases/review`, then `resume_async(...)` continues inside the child graph.

## Parallel Map

`parallel_map` maps a JSON array to one body node, executes item scopes with bounded concurrency, then collects ordered results.

```yaml
nodes:
  run_cases:
    type: parallel_map
    items: "{{ nodes.review_cases.output.cases }}"
    item_name: qa_case
    index_name: qa_case_index
    item_key: "{{ qa_case.id }}"
    max_concurrency: 2
    result_order: input
    failure_policy: collect_errors
    body: run_one_case
    collect:
      output:
        results: "{{ results }}"
        errors: "{{ errors }}"
        stats: "{{ stats }}"
    output_schema:
      type: object
      properties:
        results:
          type: array
        errors:
          type: array
        stats:
          type: object
      required: [results, errors, stats]

  run_one_case:
    type: agent
    agent: qa_runner
    side_effect: external_write
    idempotency_key: "{{ run.id }}:{{ node.scope_path }}"
    input:
      case: "{{ qa_case }}"
      case_index: "{{ qa_case_index }}"

edges:
  review_cases: run_cases
```

The body node is not part of normal `edges`; it is invoked only by `parallel_map.body`. Supported body node types are `agent`, `tool`, `mcp`, `transform`, `set_state`, and `subgraph`.

Default output shape:

```json
{
  "results": [
    {
      "index": 0,
      "key": "C001",
      "output": {"case_id": "C001", "status": "passed"}
    }
  ],
  "errors": [],
  "stats": {
    "total": 1,
    "completed": 1,
    "failed": 0,
    "waiting": 0,
    "uncertain": 0
  }
}
```

`collect.output` can project this into a different JSON object. In the collect context, `items`, `results`, `errors`, and `stats` are available.

Failure policies:

| Policy | Behavior |
| --- | --- |
| `fail_fast` | Stops once a failed item is observed |
| `fail_after_all` | Runs all items, then fails if any item failed |
| `collect_errors` | Completes the map and places item failures in `errors` |

`result_order` defaults to `input`, which is stable across retries and independent of completion order. `completion` orders completed items by finish order.

Parallel map item events include `parallel_node_id`, `item_index`, `item_key`, `item_scope_path`, and `body_node_id`. These fields are also emitted through workflow HTTP events.

## HTTP Routes

`create_workflow_router(...)` exposes workflow runs through FastAPI.

```python
from fastapi import FastAPI

from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.http.workflow_routes import WorkflowRegistry, create_workflow_router
from nexau.archs.workflow import WorkflowConfig, WorkflowStore

engine = InMemoryDatabaseEngine()
store = WorkflowStore(engine)
workflow = WorkflowConfig.from_yaml("workflow.yaml")

registry = WorkflowRegistry(store)
registry.register(workflow)

app = FastAPI()
app.include_router(create_workflow_router(registry))
```

Available routes:

| Method and path | Purpose |
| --- | --- |
| `POST /workflows/{workflow_name}/runs` | Start or recover a run |
| `POST /workflows/{workflow_name}/runs/stream` | Start or recover a run and stream workflow plus nested agent events |
| `GET /workflow-runs/{run_id}` | Read materialized run status |
| `GET /workflow-runs/{run_id}/events` | Stream stored workflow events as SSE |
| `POST /workflow-runs/{run_id}/resume` | Resume a human checkpoint |
| `POST /workflow-runs/{run_id}/resume/stream` | Resume a checkpoint and stream workflow plus nested agent events |
| `POST /workflow-runs/{run_id}/cancel` | Cancel a run or one open checkpoint |
| `POST /workflow-runs/{run_id}/reconcile` | Resolve an uncertain node |

Start request:

```json
{
  "run_id": "wf_release_checklist",
  "inputs": {"title": "Release checklist"},
  "user_id": "user_1",
  "session_id": "session_1"
}
```

Resume request:

```json
{
  "checkpoint_id": "wf_ckpt_...",
  "output": {"approved": true, "title": "Release checklist"}
}
```

Responses include `run_id`, `status`, `output`, `state`, `checkpoint_id`, and `waiting_checkpoint_ids`.

## Live Streaming

`GET /workflow-runs/{run_id}/events` is a persisted workflow event replay API. It only returns durable orchestration events such as `node_started`, `checkpoint_created`, and `parallel_item_completed`. It does not include LLM token deltas or Agent tool events.

Use the live stream routes when you want the complete runtime stream for a UI:

```bash
curl -N \
  -H 'Content-Type: application/json' \
  -d '{
    "run_id": "wf_release_checklist",
    "inputs": {"title": "Release checklist"},
    "stream": {
      "include_workflow_events": true,
      "include_agent_events": true,
      "include_text_deltas": true,
      "include_tool_events": true,
      "include_usage_events": true
    }
  }' \
  http://localhost:8000/workflows/approval_flow/runs/stream
```

The SSE `event:` name is the envelope type:

```text
event: workflow_event
data: {"type":"workflow_event","workflow_event":{"event_type":"node_started", ...}}

event: agent_event
data: {"type":"agent_event","workflow":{"node_id":"agent_node", ...},"agent":{"event":{"type":"TEXT_MESSAGE_CONTENT", ...}}}

event: complete
data: {"type":"complete","status":"completed", ...}
```

`workflow_event` envelopes are persisted first and can be replayed later through `/events`. `agent_event` envelopes are live-only by default; they are intended for rendering Agent internals inside the current workflow node, subgraph, or parallel map item. Thinking events remain off unless `stream.include_thinking_events` is set to `true`.

## Tracing

Pass a `BaseTracer` to `WorkflowExecutor` or `WorkflowRegistry.register(...)` to get workflow spans.

```python
executor = WorkflowExecutor(
    workflow=workflow,
    store=store,
    tracer=tracer,
)
```

The runtime emits `WORKFLOW`, `WORKFLOW_NODE`, `WORKFLOW_SUBGRAPH`, and `WORKFLOW_PARALLEL_ITEM` spans. Agent nodes keep their normal Agent, LLM, and tool spans under the workflow node span.

Useful span attributes include:

| Attribute | Meaning |
| --- | --- |
| `workflow.run_id` | Durable workflow run id |
| `workflow.graph_id` | Current graph name |
| `workflow.node_id` | Current node id |
| `workflow.node_type` | Current node type |
| `workflow.scope_path` | Durable scope path |
| `workflow.subgraph` | Included graph key, when inside a subgraph |
| `workflow.parallel_node_id` | Parent map node, when inside a parallel item |
| `workflow.item_key` | Stable parallel item key |

## Full Example

See `examples/workflows/qa_release_check/` for a complete workflow with:

| Feature | File |
| --- | --- |
| Parent graph with agent, subgraph, branch, parallel map, and end nodes | `qa_release.workflow.yaml` |
| Reusable human review subgraph | `graphs/human_case_review.workflow.yaml` |
| Structured agent configs | `agents/qa_planner.yaml`, `agents/qa_runner.yaml` |
| Direct Python runner with mock-agent and live-LLM modes | `run.py` |

Validate without calling an LLM:

```bash
uv run python examples/workflows/qa_release_check/run.py --validate-only
```

Run a deterministic smoke test:

```bash
uv run python examples/workflows/qa_release_check/run.py --mock-agents
```

Run with live LLM credentials:

```bash
uv run python examples/workflows/qa_release_check/run.py \
  --requirement "Checkout retry should show a clear error after three failed payment attempts."
```

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Referencing a `parallel_map` body from `edges` | Only reference the map node in `edges`; the body is called by `body` |
| Using a side-effecting parallel body without `item_key` | Add a stable, unique `item_key` |
| Expecting a subgraph to read parent `nodes` or `state` directly | Pass values through `input` or `state_in` |
| Changing YAML and expecting an existing run to use it | Start a new run id; existing runs use the saved snapshot |
| Returning free text from a structured agent node | Configure `output_schema` and let the runtime validate JSON |
| Forgetting `max_iterations` on `while` | Set a hard upper bound for every loop |
