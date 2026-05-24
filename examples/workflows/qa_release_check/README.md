# QA Release Check Workflow

这个例子演示如何把多个 Agent 串成一个真实 Workflow：Planner Agent 先根据需求生成 QA 测试用例，Workflow 暂停到人工审核节点，脚本模拟人工批准后 resume，Runner Agent 再逐条执行测试用例并汇总结果。

## 文件结构

- `qa_release.workflow.yaml`: Workflow 定义，包含 `start`、`agent`、`human`、`if_else`、`while`、`transform`、`end` 节点。
- `agents/qa_planner.yaml`: 生成测试用例的 Agent。
- `agents/qa_runner.yaml`: 执行单个测试用例并返回结构化结果的 Agent。
- `run.py`: 从 YAML 加载 Workflow，启动运行，读取 human checkpoint，再用审核结果恢复执行。

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

## 关键点

Workflow 的 Agent 节点通过 `includes.agents` 引用普通 Agent YAML：

```yaml
includes:
  agents:
    qa_planner: ./agents/qa_planner.yaml
    qa_runner: ./agents/qa_runner.yaml
```

Human 节点会把 run 状态变成 `waiting`，并返回 `checkpoint_id`。业务系统保存这个 ID，用户审核后调用 `resume_async(...)` 继续执行：

```python
completed = await executor.resume_async(
    run_id=waiting.run_id,
    checkpoint_id=waiting.checkpoint_id,
    output={"approved": True, "cases": reviewed_cases},
)
```

`while` 节点用 CEL 表达式控制循环，用 `update` 把每次 Agent 输出追加到 durable state：

```yaml
condition: "state.remaining_cases.size() > 0"
update:
  remaining_cases: "{{ state.remaining_cases[1:] }}"
  results: "{{ state.results + [nodes.run_one_case.output] }}"
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
