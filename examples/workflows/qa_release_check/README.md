# QA Release Check Workflow

这个例子演示如何把多个 Agent、一个可复用子图和并行 map 串成真实 Workflow：父图中的 Planner Agent 先根据需求生成 QA 测试用例，然后通过 `type: subgraph` 调用 `graphs/human_case_review.workflow.yaml`。子图内部暂停到人工审核节点，脚本模拟人工批准后 resume，父图再用 `type: parallel_map` 并行调用 Runner Agent 执行测试用例并汇总结果。

## 文件结构

- `qa_release.workflow.yaml`: 父 Workflow 定义，包含 `start`、`agent`、`subgraph`、`if_else`、`parallel_map`、`transform`、`end` 节点。
- `graphs/human_case_review.workflow.yaml`: 独立子图文件，包含 human checkpoint 和明确的 output contract。
- `agents/qa_planner.yaml`: 生成测试用例的 Agent。
- `agents/qa_runner.yaml`: 执行单个测试用例并返回结构化结果的 Agent。
- `run.py`: 从 YAML 加载父图和子图，启动运行，读取子图内部 human checkpoint，再用审核结果恢复执行，并打印 live workflow SSE 事件流。

## 准备环境

从仓库根目录执行：

```bash
export LLM_MODEL="your-model"
export LLM_BASE_URL="https://your-api-base-url/v1"
export LLM_API_KEY="your-api-key"
export LLM_API_TYPE="openai_chat_completion"
```

也可以把这些变量放在仓库根目录的 `.env` 文件里。

如果要把 Workflow trace 发到 Langfuse，再配置：

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

## 运行

先只校验配置，不调用 LLM：

```bash
uv run python examples/workflows/qa_release_check/run.py --validate-only
```

不用 LLM 跑完整控制流 smoke test：

```bash
uv run python examples/workflows/qa_release_check/run.py --mock-agents
```

不用 LLM，但把 Workflow/node/mock Agent span 发到 Langfuse：

```bash
uv run python examples/workflows/qa_release_check/run.py \
  --mock-agents \
  --langfuse \
  --langfuse-session-id qa-release-workflow-local
```

运行完整 Workflow：

```bash
uv run python examples/workflows/qa_release_check/run.py \
  --requirement "Checkout retry should show a clear error after three failed payment attempts."
```

运行完整 Workflow 并发送 Langfuse trace：

```bash
uv run python examples/workflows/qa_release_check/run.py \
  --langfuse \
  --langfuse-session-id qa-release-workflow-live \
  --requirement "Checkout retry should show a clear error after three failed payment attempts."
```

默认会自动批准 Planner 生成的用例。要演示人工拒绝分支：

```bash
uv run python examples/workflows/qa_release_check/run.py --reject
```

脚本运行时会打印 live workflow SSE envelope。`workflow_event` 来自 durable workflow event log；真实 LLM 运行时还会在 Agent 节点内部打印 `agent_event`，包括 text delta、tool call args、tool result、usage 和 run lifecycle。`--mock-agents` 走本地 mock runner，不会产生 LLM/Agent middleware 事件。

```text
event: workflow_event
data: {"type":"workflow_event","workflow_event":{"event_type":"node_started", ...}}

event: agent_event
data: {"type":"agent_event","workflow":{"node_id":"generate_cases", ...},"agent":{"event":{"type":"TEXT_MESSAGE_CONTENT", ...}}}
```

## 关键点

Workflow 的 Agent 节点通过 `includes.agents` 引用普通 Agent YAML：

```yaml
includes:
  agents:
    qa_planner: ./agents/qa_planner.yaml
    qa_runner: ./agents/qa_runner.yaml
```

父图通过 `includes.graphs` 引用独立子图文件，再用 `type: subgraph` 把它当作一个节点调用：

```yaml
includes:
  graphs:
    human_case_review: ./graphs/human_case_review.workflow.yaml

nodes:
  review_cases:
    type: subgraph
    graph: human_case_review
    input:
      requirement: "{{ nodes.start.output.requirement }}"
      cases: "{{ nodes.generate_cases.output.cases }}"
```

子图内部的 Human 节点会把整个 run 状态变成 `waiting`，checkpoint 的 `scope_path` 类似 `review_cases/review`。业务系统保存这个 ID，用户审核后调用 `resume_async(...)` 继续执行：

```python
completed = await executor.resume_async(
    run_id=waiting.run_id,
    checkpoint_id=waiting.checkpoint_id,
    output={"approved": True, "cases": reviewed_cases, "review_note": "Approved."},
)
```

子图的输出会作为父图 `review_cases` 节点的输出暴露，后续节点只读取 `nodes.review_cases.output`，不能隐式读取子图内部的 `nodes.review`。

`parallel_map` 节点把审核后的 case 列表 fan out 到同一个 Runner Agent body。每个 item 都有稳定的 `item_key`，因此 side-effecting Agent body 可以用 scoped idempotency key 安全重试；collect 阶段再把所有 item 输出按输入顺序汇总：

```yaml
run_cases:
  type: parallel_map
  items: "{{ nodes.review_cases.output.cases }}"
  item_name: qa_case
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
```

`run_one_case` 作为 `parallel_map` body 不会出现在普通 `edges` 中。它通过 `qa_case` / `qa_case_index` 读取当前 item，而不是读取父图 state：

```yaml
run_one_case:
  type: agent
  agent: qa_runner
  side_effect: external_write
  idempotency_key: "{{ run.id }}:{{ node.scope_path }}"
  input:
    requirement: "{{ inputs.requirement }}"
    case: "{{ qa_case }}"
    case_index: "{{ qa_case_index }}"
```

这个例子使用 `common-expression-language` 执行 CEL 表达式；NexAU 只在模板层补充了 `.length` 和简单 slice 的兼容转换。

## Tracing

Workflow 支持把同一个 tracer 传给 `WorkflowExecutor`。运行时会为每次驱动创建一个 `WORKFLOW` span，并为每个实际执行的节点创建 `WORKFLOW_NODE` span；Agent 节点内部仍然使用原来的 Agent/LLM/tool span 结构，并挂在对应 node span 下面。

```python
from nexau.archs.tracer.adapters import LangfuseTracer
from nexau.archs.workflow import WorkflowExecutor

tracer = LangfuseTracer(session_id="qa-release-workflow-live")
executor = WorkflowExecutor(workflow=workflow, store=store, tracer=tracer)
```

如果通过 HTTP registry 暴露 Workflow，也可以在注册时传入同一个 tracer：

```python
registry.register(workflow, tracer=tracer)
```
