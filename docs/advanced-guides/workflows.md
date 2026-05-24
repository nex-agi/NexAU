# Workflows

NexAU workflows are YAML graphs executed by `WorkflowExecutor` with durable node-boundary persistence. A run stores an append-only event log, node outputs, checkpoints, and a definition snapshot so a waiting run can resume even if YAML files change later.

## Subgraphs

RFC-0028 adds external workflow subgraphs. Each `*.workflow.yaml` file is one graph. A parent graph references child graph files through `includes.graphs`, then invokes one by using a `type: subgraph` node:

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
        approved: { type: boolean }
        cases: { type: array }
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

Subgraph inputs, state, and node outputs are isolated from the parent. The parent can only read the returned object through `nodes.<subgraph_node>.output`. A child human checkpoint pauses the same workflow run, with a scoped checkpoint such as `review_cases/review`; `resume_async(...)` then continues inside the child graph before completing the parent subgraph node.

Events include `graph_id`, `parent_node_id`, `subgraph`, `depth`, and `scope_path` metadata. The optional `subgraph_started`, `subgraph_waiting`, `subgraph_completed`, and `subgraph_failed` events are emitted for UI and tracing views.

See `examples/workflows/qa_release_check/` for a real LLM-backed parent workflow that calls `graphs/human_case_review.workflow.yaml`.
