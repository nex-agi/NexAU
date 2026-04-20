# RFC-0013: 子 Agent 委派与上下文传播

- **状态**: implemented
- **优先级**: P1
- **标签**: `architecture`, `subagent`, `delegation`, `context`, `concurrency`
- **影响服务**: `nexau/archs/main_sub/execution/subagent_manager.py`, `nexau/archs/main_sub/execution/executor.py`, `nexau/archs/main_sub/execution/batch_processor.py`, `nexau/archs/main_sub/execution/parse_structures.py`, `nexau/archs/main_sub/execution/response_parser.py`, `nexau/archs/main_sub/config/config.py`, `nexau/archs/tool/builtin/recall_sub_agent_tool.py`, `nexau/archs/main_sub/agent.py`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 允许一个父 Agent 配置若干"子 Agent"，并在推理过程中把具体任务分派给它们。本 RFC 固化子 Agent 委派的单一入口 `SubAgentManager.call_sub_agent`（subagent_manager.py:68），描述其 **spawn / recall 双路径**、**上下文显式覆盖优先**的传播规则、**parent_agent_state 父链回溯**、**结果信封语义**、**shutdown 级联**、以及 `recall_sub_agent` 工具的自动注册机制。本 RFC 覆盖**单个父 Agent 下的单层委派**；多 Agent 团队级编排参见 RFC-0002。

## 动机

在 ReAct 循环里，LLM 会通过 `<sub_agent>...</sub_agent>` 标签发起子 Agent 调用，类似调用工具。框架需要回答：

1. **spawn 还是继续？** LLM 既可以创建新 sub-agent（首次委派），也可以沿用之前同一任务内已生成的 sub-agent（继续对话）。两条路径必须在同一入口里统一处理，否则调用方要管理生命周期就太复杂。
2. **上下文怎么传？** 父 Agent 有 `AgentContext`（线程局部栈）与 `GlobalStorage`（跨 Agent 共享）（参见 RFC-0012）；显式传的 `context` dict 与线程栈里的 `AgentContext` 谁优先？
3. **追踪父链怎么接？** observability/tracer 要求子 Agent run 的 `root_run_id`/`parent_run_id` 与父 Agent 对齐；并行 sub-agent 执行还要共享 `parallel_execution_id` 做聚合。
4. **停机怎么处理？** 父 Agent cleanup 时必须停掉所有在跑的 sub-agent，否则会产生孤儿线程 + 未持久化的状态。
5. **怎么告诉 LLM "这个 sub-agent 还活着"？** 没有一个明确的结果约定，LLM 就无法 recall——必须在返回里嵌入 `sub_agent_name + sub_agent_id` 的回讯线索。

这些语义必须显式成文档：`call_sub_agent` 签名一旦被第三方集成使用，上述隐式规则再改就是 breaking change。

## 设计

### 概述

```
                ┌──────────────────────────────────────┐
                │         Executor (per Agent)         │
                │  - response_parser → SubAgentCall    │
                │  - _execute_sub_agent_call_safe      │
                │  - 并发：sub_agent_futures           │
                └──────────────┬───────────────────────┘
                               │ call_sub_agent(...)
                               ▼
               ┌───────────────────────────────────────┐
               │       SubAgentManager (per Agent)     │
               │  - sub_agents: dict[name, AgentConfig]│
               │  - running_sub_agents: dict[id, Agent]│
               │  - _shutdown_event: threading.Event   │
               └──────┬─────────────────────────┬──────┘
                      │ spawn                   │ recall
                      │ (sub_agent_id is None)  │ (sub_agent_id 给了)
                      ▼                         ▼
        ┌────────────────────┐       ┌────────────────────────┐
        │ Agent(is_root=False│       │ Agent(is_root=False,   │
        │  共享 global_storage,       │  agent_id=<已有>,       │
        │  session 串接)     │       │  history 自动从       │
        │                    │       │  agent_repo 恢复)      │
        └──────────┬─────────┘       └───────────┬────────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
                       sub_agent.run(message,
                           context=effective_context,
                           parent_agent_state,
                           custom_llm_client_provider)
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │  "<sub-agent output>\n         │
                  │   Sub-agent finished          │
                  │   (sub_agent_name: X,         │
                  │    sub_agent_id: Y. Recall    │
                  │    this agent if needed)."    │
                  └───────────────────────────────┘
```

单层委派的 **三大传播规则** 是本 RFC 的核心不变式：

| 状态容器 | 传播方式 | 依据 |
|----------|----------|------|
| **GlobalStorage** | by-reference 穿透，父子共享同一实例 | subagent_manager.py:60, 123, 133 |
| **AgentContext（线程栈）** | 每次调用 `dict.copy()` 一份新字典，显式 `context` 参数优先于隐式 `get_context()` | subagent_manager.py:143-151 |
| **parent_agent_state（父 Agent 的 AgentState）** | by-reference 传入，用于追踪父链 + 写 parallel_execution_id | subagent_manager.py:154-160 |

详见 RFC-0012 对 3 个状态容器语义的定义；本 RFC 描述它们**在子 Agent 调用点上的具体传播行为**。

### 详细设计

#### 1. `SubAgentManager` 的生命周期

`SubAgentManager` 与父 Agent 的 `Executor` 一对一绑定，在 `Executor.__init__` 里实例化（executor.py:162）：

```python
self.subagent_manager = SubAgentManager(
    agent_name,
    sub_agents,            # dict[name, AgentConfig]，来自 AgentConfig.sub_agents
    global_storage,        # 与父 Agent 共享
    session_manager=session_manager,
    user_id=user_id,
    session_id=session_id,
)
```

`SubAgentManager` 拥有两份跨调用的状态（subagent_manager.py:65-66）：

- `_shutdown_event: threading.Event`：父 Agent cleanup 时由 `Executor.cleanup()`（executor.py:1311）调用 `subagent_manager.shutdown()` 置位，之后任何 `call_sub_agent` 立即抛 `RuntimeError`。
- `running_sub_agents: dict[agent_id, Agent]`：当前在跑的所有 sub-agent 实例，`call_sub_agent` 进入时登记（subagent_manager.py:140），`finally` 块里移除（subagent_manager.py:179）；`shutdown()` 遍历该 dict 对每个 running sub-agent 调用 `sync_cleanup()`（subagent_manager.py:184-186）。

#### 2. `call_sub_agent`：单一委派入口

完整签名（subagent_manager.py:68-77）：

```python
def call_sub_agent(
    self,
    sub_agent_name: str,
    message: str,
    sub_agent_id: str | None = None,
    context: dict[str, Any] | None = None,
    parent_agent_state: AgentState | None = None,
    custom_llm_client_provider: Callable[[str], Any] | None = None,
    parallel_execution_id: str | None = None,
) -> str
```

**必填**：`sub_agent_name`（必须存在于 `self.sub_agents`，否则 `ValueError`）、`message`。

**分支判定**（subagent_manager.py:113-138）：

| `sub_agent_id` | 路径 | 行为 |
|----------------|------|------|
| `None` | **spawn** | 新实例 `Agent(config=..., is_root=False, ...)`；`Agent.__init__` 自动分配新 `agent_id` 并在 `agent_repo` 登记（agent.py:84/89/286） |
| 非 `None` | **recall** | 新实例 `Agent(agent_id=sub_agent_id, is_root=False, ...)`；`Agent.run()` 会自动从 `agent_repo` 按 `agent_id` 恢复历史 |

两条路径都共享同一份 `global_storage`、同一份 `session_manager`、同一对 `user_id/session_id`（subagent_manager.py:120-138），因此 sub-agent 读到的 GlobalStorage 与父 Agent 完全一致——RFC-0012 规定的 by-reference 穿透在此落地。

#### 3. 上下文传播规则（重要）

代码（subagent_manager.py:143-151）：

```python
effective_context = None
if context:
    effective_context = context
else:
    current_context = get_context()
    if current_context:
        effective_context = current_context.context.copy()
```

**规则**：

1. **显式 > 隐式**：调用方显式传的 `context` 参数优先；未传时才回落到父 Agent 当前线程栈顶的 `AgentContext`。
2. **copy 语义**：回落时用 `dict.copy()` 做浅拷贝，sub-agent 在自己的 run 里修改 `context` 不会反噬父 Agent 的 `AgentContext.context`。
3. **空也允许**：若 `context` 未传且 `get_context()` 返回 `None`（当前线程没有活动 AgentContext），`effective_context=None` 传给 `sub_agent.run`，sub-agent 不继承任何 AgentContext 起点字段。

与 **GlobalStorage** 形成对照——GlobalStorage 不 copy，完全 by-ref（参见 RFC-0012 的 `GlobalStorage` 章节）。这份不对称是**有意设计**：GlobalStorage 需要父子任意时刻读到同一份 `skill_registry`/`tracer`；AgentContext 只是"起点快照"，各自 run 内部的修改不应互相干扰。

#### 4. `parent_agent_state` 父链

`parent_agent_state` 是父 Agent 的 `AgentState` 实例（`AgentState.from_agent(...)`），通过 kwarg 传入 sub-agent run（subagent_manager.py:160）。它承载两种能力：

- **追踪父链**（agent.py:826-827）：sub-agent `Agent.run()` 从 `parent_agent_state.root_run_id / run_id` 派生自身的 `root_run_id` 与 `parent_run_id`，形成可回溯的执行树。
- **并行分组**（subagent_manager.py:154-155）：若调用方传了 `parallel_execution_id`，先写回父 AgentState：
  ```python
  if parallel_execution_id and parent_agent_state:
      parent_agent_state.set_global_value("parallel_execution_id", parallel_execution_id)
  ```
  `set_global_value`（agent_state.py:116）是 `GlobalStorage.set_value` 的 facade，因此该 id 对同一轮并行里的所有 sibling sub-agent 立即可见，用于日志/metric 聚合。

#### 5. 结果信封（recall 线索）

`sub_agent.run()` 的原始返回会被 wrapper 一层（subagent_manager.py:163-167）：

```python
result = (
    f"{result}\n"
    f"Sub-agent finished (sub_agent_name: {sub_agent.agent_name}, "
    f"sub_agent_id: {sub_agent.agent_id}. Recall this agent if needed)."
)
```

**为什么**：LLM 在下一步思考时会看到这个完整字符串，如果决定继续与同一 sub-agent 对话，就要通过 `recall_sub_agent` 工具显式传 `sub_agent_id`——这个 id **必须从结果文本里解析出来**。这段 envelope 是**面向 LLM 的 UI**，不是面向代码调用的。

#### 6. `recall_sub_agent` 工具自动注册

父 Agent 只要在 `AgentConfig.sub_agents` 里配置了至少一个 sub-agent，`recall_sub_agent` 工具就会在 `AgentConfig._finalize`（config.py:232-244）里被自动注入到父 Agent 的 `tools` 列表：

```python
if self.sub_agents:
    recall_subagent_tool = Tool.from_yaml(
        str(nexau_package_path / "archs" / "tool" / "builtin" / "description" / "recall_sub_agent_tool.yaml"),
        binding=recall_sub_agent,
    )
    self.tools.append(recall_subagent_tool)
```

工具实现（recall_sub_agent_tool.py:11-68）通过 `AgentState._executor` 反向拿到 `Executor`（recall_sub_agent_tool.py:33），再取 `subagent_manager`（recall_sub_agent_tool.py:40），最终调用 `call_sub_agent(sub_agent_name, message, sub_agent_id, parent_agent_state=agent_state)`（recall_sub_agent_tool.py:48-53）。

**defensive getattr 是有意的**：`AgentState._executor` 不是公开 API，它的存在以 contract-style 由 framework 保证；工具层做 `getattr` 以便在极端情况下返回 `{"status": "error", ...}` 而不是崩溃（recall_sub_agent_tool.py:34-45）。未来若把 `_executor` 提升为公开字段，工具实现可以简化。

#### 7. 并行：`SubAgentCall` × 线程池

`response_parser` 可以把 LLM 一轮输出里的多个 `<sub_agent>` 解析成多个 `SubAgentCall`（response_parser.py:365-420；parse_structures.py:53-66）。Executor 在 parallel 模式下把它们丢进 thread pool：

- `SubAgentCall` 携带 `parallel_execution_id`（parse_structures.py:62）——由解析器分配，同一轮里平行的调用共享同一个 id。
- Executor 内部 `sub_agent_futures` dict 追踪所有 in-flight（executor.py:1045）。
- 每个 future 最终路由到 `_execute_sub_agent_call_safe`（executor.py:1258）→ `subagent_manager.call_sub_agent(...)`（executor.py:1268），`parent_agent_state` 被传进去。
- Sibling sub-agent 并行跑时**共享同一个 `GlobalStorage`**（线程安全由 `GlobalStorage` 内部的 per-key RLock 保证，参见 RFC-0012）；**各自 context 是 copy**，互不干扰。

#### 8. Batch：`BatchProcessor` 把 data-driven 拆成多次 `call_sub_agent`

`BatchProcessor`（batch_processor.py:43）在构造时持有 `subagent_manager` 引用（batch_processor.py:50），处理 `<batch_agent>` 时按 data 展开成 N 次 sub-agent 调用（batch_processor.py:307-332），每次都走 `self.subagent_manager.call_sub_agent(agent_name, message)`（batch_processor.py:327）——**没有**传 `parent_agent_state` 和 `context`，因此 batch item 之间是**完全独立的 sub-agent 调用**，不做父链 trace 也不共享 parallel id。这是 batch 语义的简化约束：把 batch 视为"对同一个 sub-agent 配多次 fire-and-collect"，而不是"fork-join 子任务"。

### 示例

**父 Agent 配置 sub-agent**（YAML，简化版）：

```yaml
agent_name: main
model: gpt-4.1
sub_agents:
  researcher:
    agent_name: researcher
    model: gpt-4.1
    system_prompt: "You research topics deeply..."
    tools: [...]
```

`AgentConfig._finalize` 检测到 `sub_agents` 非空，自动注入 `recall_sub_agent` 工具（config.py:239-244）。LLM 在输出里看到 `recall_sub_agent` 这个可用工具。

**首次委派（spawn）**——LLM 输出：

```xml
<sub_agent>
  <name>researcher</name>
  <message>Research the history of NexAU</message>
</sub_agent>
```

`response_parser` 解析（response_parser.py:79-90）→ `SubAgentCall(agent_name="researcher", message="...", sub_agent_call_id=auto)` → `executor._execute_sub_agent_call_safe`（executor.py:1258）→ `subagent_manager.call_sub_agent("researcher", "Research the history of NexAU", parent_agent_state=agent_state)`。`sub_agent_id=None` → **spawn 路径**，新建 `Agent(is_root=False)`（subagent_manager.py:131-138），拿到 `agent_id=researcher-<uuid>`。

返回给 LLM 的文本：

```
<researcher 的 final 回答>
Sub-agent finished (sub_agent_name: researcher, sub_agent_id: researcher-abc123. Recall this agent if needed).
```

**继续（recall）**——LLM 下一步输出：

```xml
<recall_sub_agent>
  <sub_agent_name>researcher</sub_agent_name>
  <sub_agent_id>researcher-abc123</sub_agent_id>
  <message>Now summarize into 3 bullets</message>
</recall_sub_agent>
```

`recall_sub_agent` 工具（recall_sub_agent_tool.py:11-68）→ `call_sub_agent("researcher", "Now summarize into 3 bullets", sub_agent_id="researcher-abc123", parent_agent_state=agent_state)`。`sub_agent_id` 非 `None` → **recall 路径**（subagent_manager.py:113-128），`Agent.run()` 从 `agent_repo` 自动恢复历史，继续对话。

## 权衡取舍

### 考虑过的替代方案

1. **共享同一 history（父子 interleave）**：让父 Agent 直接在自己的 history 里插入 sub-agent 对话。拒绝原因：污染父 Agent 的 context 长度且破坏 ReAct turn 边界；LLM 无法分辨"自己在想"和"sub-agent 在答"。
2. **不支持 recall，每次都新 spawn**：sub-agent 永远无状态，每次都重启。拒绝原因：无法做"先调查、再总结、最后写"这种跨 turn 的连贯任务，把 continuation 责任推给 parent prompt 反而更复杂。
3. **context by-reference 而非 copy**：sub-agent 修改 context 立即反噬父 Agent。拒绝原因：sub-agent 的内部迭代（`set_context_value`）往往是临时的，反噬会让父 Agent 读到意料外的值；对比 GlobalStorage 的 by-ref 行为，by-copy 在"临时 vs 共享"之间画了清晰的线。
4. **递归多层 sub-agent 嵌套**：sub-agent 自己也能有 sub-agent。当前行为：`AgentConfig.sub_agents` 是平的，多层嵌套要显式在 sub-agent 的 config 里再配。没有全局深度限制。这块留给 RFC-0002 `AgentTeam` 描述。

### 缺点

1. **`recall_sub_agent_tool` 依赖私有属性 `AgentState._executor`**：框架内部契约而非公开 API，外部工具难以干净地复现同类行为。长期应把 `_executor` 提升为 public 或提供显式的 "subagent facade"。
2. **`sub_agent_id` 暴露在 LLM 的 prompt 里**：id 的格式稳定性变成 LLM-observable ABI，改 id scheme 可能破坏模型习惯的 tool call shape。
3. **BatchProcessor 的 fire-and-collect 不共享父链 trace**：批处理的可观测性比串行/并行 sub-agent 弱；如果 batch 里某一项崩了，看不到完整 trace chain。
4. **shutdown 用 `threading.Event` + `sync_cleanup`**：同步等待每个 running sub-agent 清理；若某 sub-agent 在 LLM 长调用中卡住，cleanup 会阻塞父 Agent 关闭。目前依赖 LLM 客户端自身的超时保护。

## 实现计划

### 阶段划分

- [x] Phase 1: `SubAgentManager` 双路径 + 上下文传播 + shutdown（subagent_manager.py 全部）
- [x] Phase 2: `recall_sub_agent` 工具自动注册（config.py:236-244）与实现（recall_sub_agent_tool.py）
- [x] Phase 3: Executor 里 spawn/recall/batch 的接线（executor.py:1258-1289, 1311）
- [x] Phase 4: BatchProcessor 引用 SubAgentManager（batch_processor.py:43, 327）
- [x] Phase 5: `SubAgentCall` 数据结构与 response parser 解析（parse_structures.py:53, response_parser.py:365）
- [ ] Phase 6（未来）：把 `AgentState._executor` 从私有属性提升为 framework-public，以移除 `recall_sub_agent_tool` 的 defensive getattr
- [ ] Phase 7（未来）：给 BatchProcessor 加 `parent_agent_state` 与 `parallel_execution_id`，使 batch 跑也能贡献父链 trace

### 相关文件

- `nexau/archs/main_sub/execution/subagent_manager.py` - 委派核心（class + call_sub_agent + shutdown）
- `nexau/archs/main_sub/execution/executor.py` - SubAgentManager/BatchProcessor 接线、_execute_sub_agent_call_safe、cleanup
- `nexau/archs/main_sub/execution/batch_processor.py` - 批量 data-driven 委派
- `nexau/archs/main_sub/execution/parse_structures.py` - `SubAgentCall` / `ExecutableCall` dataclass
- `nexau/archs/main_sub/execution/response_parser.py` - `<sub_agent>` XML 解析为 `SubAgentCall`
- `nexau/archs/main_sub/config/config.py` - `recall_sub_agent` 自动注入 tools 的逻辑
- `nexau/archs/tool/builtin/recall_sub_agent_tool.py` - recall 工具实现
- `nexau/archs/tool/builtin/description/recall_sub_agent_tool.yaml` - 工具 schema（LLM 可见）
- `nexau/archs/main_sub/agent.py` - `Agent.__init__(is_root)` 与 `Agent.run()` 父链接线
- `nexau/archs/main_sub/agent_state.py` - `set_global_value` facade

## 测试方案

### 单元测试

- **spawn 路径**：mock `Agent`，验证 `call_sub_agent("x", "msg")` 创建新 `Agent(is_root=False)` 且 `sub_agents["x"]` 被传入 config。
- **recall 路径**：`call_sub_agent("x", "msg", sub_agent_id="x-123")` 创建 `Agent(agent_id="x-123", is_root=False)`；验证 `agent_id` 原样传递。
- **上下文优先级**：
  - 显式 `context={"a":1}` → effective = `{"a":1}`。
  - 不传 `context`，`AgentContext({"b":2})` 在线程栈顶 → effective = `{"b":2}` 且为独立对象。
  - 都没有 → effective = `None`。
- **shutdown**：`shutdown()` 后再调 `call_sub_agent` 抛 `RuntimeError`；`running_sub_agents` 里的每个都被 `sync_cleanup()` 调用（可用 mock 验证）。
- **parallel_execution_id 写入父 state**：传了 `parallel_execution_id` 且 `parent_agent_state` 非 `None` → `set_global_value` 被调用一次；否则不调用。
- **结果信封**：sub_agent.run 返回 `"RESULT"` → 最终返回字符串以 `"RESULT\nSub-agent finished (sub_agent_name: ..., sub_agent_id: ...)"` 结尾。
- **finally 清理**：即便 sub_agent.run 抛异常，`running_sub_agents` 里的 entry 必须被 pop（try/finally 覆盖）。
- **`recall_sub_agent` tool**：
  - `agent_state is None` → `{"status":"error", "error": "Agent state not available"}`。
  - `agent_state` 有但 `_executor is None` → error envelope。
  - 正常路径成功 → dict 含 `status=success, sub_agent, sub_agent_id, result`。

### 集成测试

- **端到端 spawn → recall**：配一个带 sub-agent 的 agent → 跑 1 轮让它 spawn → 从结果里 parse `sub_agent_id` → 手动调 `recall_sub_agent` → 验证 history 被恢复（sub-agent 能引用前一轮的内容）。
- **并行 sub-agent**：LLM 输出两个 sibling `<sub_agent>`，Executor 并发跑；验证两个都返回成功，且各自 `AgentContext` 修改不互相污染；验证 `parallel_execution_id` 在父 GlobalStorage 里被写入（last-write-wins）。
- **GlobalStorage 共享**：父 Agent `set_global_value("k", "v")` → sub-agent 内部 `get_global_value("k")` 读到 `"v"`；sub-agent 修改后父 Agent 也能读到新值（验证 by-ref）。
- **cleanup 级联**：父 Agent `Executor.cleanup()` → `subagent_manager.shutdown()` → running sub-agent 收到 `sync_cleanup` 信号并退出。

### 手动验证

1. 跑 `examples/code_agent/` 下任一含 sub-agent 的配置（观察 log `🤖➡️🤖` 与 `✅ Sub-agent 'X' returned`）。
2. 在 LLM 输出里手动发一个带错误 `sub_agent_name` 的 `<sub_agent>` → 框架应回 `ValueError`，Executor 捕获后作为错误结果回传给 LLM 而非崩溃。
3. 长跑一个 sub-agent，期间 Ctrl+C 中断主程序 → 验证父 Agent cleanup 能在合理时间内完成。

## 未解决的问题

1. **多层嵌套深度无限制**：sub-agent 自己可以再配 sub-agent，理论上无限递归。目前没有全局 max-depth gate；极端 prompt 可能诱发深嵌套递归爆栈。需不需要在 `SubAgentManager` 里维护 depth counter？
2. **agent_id 跨 session 索引**：当前 `agent_repo` 按 `agent_id` 恢复；但 `agent_id` 的唯一性范围是 session 内还是全局？跨 session recall 一个旧 `sub_agent_id` 应该被拒绝还是复用？
3. **`parent_agent_state` 与 `context` 的职责重叠**：两者都是"父 Agent 传递给子 Agent 的上下文信息"，但一个 by-ref（state）、一个 by-copy（context）；第三方工具编写者不易分清。考虑合并为单一 `DelegationContext` dataclass，内部分字段再写清楚 by-ref vs by-copy。
4. **BatchProcessor 的父链缺失**：batch 调用不传 `parent_agent_state`（batch_processor.py:327），batch item 在 tracer 里是孤立的。未来 RFC-0019（图像令牌化与缓存，T13）要做 metric 聚合时，这一块会暴露。
5. **shutdown 的阻塞语义**：`sync_cleanup` 同步调用，若 sub-agent 卡在 LLM 长请求上，父 Agent cleanup 可能长时间不返回。是否应改为"尽力而为 + 超时后 kill thread"？

## 参考资料

- `nexau/archs/main_sub/execution/subagent_manager.py:34` — `SubAgentManager` class
- `nexau/archs/main_sub/execution/subagent_manager.py:37` — `__init__` 签名
- `nexau/archs/main_sub/execution/subagent_manager.py:65` — `_shutdown_event`
- `nexau/archs/main_sub/execution/subagent_manager.py:66` — `running_sub_agents`
- `nexau/archs/main_sub/execution/subagent_manager.py:68` — `call_sub_agent` entry
- `nexau/archs/main_sub/execution/subagent_manager.py:96` — 关机检查
- `nexau/archs/main_sub/execution/subagent_manager.py:113` — recall 路径分支
- `nexau/archs/main_sub/execution/subagent_manager.py:129` — spawn 路径分支
- `nexau/archs/main_sub/execution/subagent_manager.py:140` — `running_sub_agents` 登记
- `nexau/archs/main_sub/execution/subagent_manager.py:143-151` — 上下文传播（显式 > 隐式 + copy）
- `nexau/archs/main_sub/execution/subagent_manager.py:154-155` — `parallel_execution_id` 写入父 state
- `nexau/archs/main_sub/execution/subagent_manager.py:157` — `sub_agent.run` 调用
- `nexau/archs/main_sub/execution/subagent_manager.py:163-167` — 结果信封
- `nexau/archs/main_sub/execution/subagent_manager.py:178-179` — finally 里清理
- `nexau/archs/main_sub/execution/subagent_manager.py:181` — `shutdown`
- `nexau/archs/main_sub/execution/subagent_manager.py:192` — `add_sub_agent`
- `nexau/archs/main_sub/execution/executor.py:162` — 把 `SubAgentManager` 装到 Executor
- `nexau/archs/main_sub/execution/executor.py:170` — `BatchProcessor` 初始化
- `nexau/archs/main_sub/execution/executor.py:1045` — 并行 `sub_agent_futures` dict
- `nexau/archs/main_sub/execution/executor.py:1258` — `_execute_sub_agent_call_safe` 私有方法
- `nexau/archs/main_sub/execution/executor.py:1268` — Executor → `call_sub_agent`
- `nexau/archs/main_sub/execution/executor.py:1311` — `subagent_manager.shutdown()` in cleanup
- `nexau/archs/main_sub/execution/executor.py:1353` — `add_sub_agent` 公开接线
- `nexau/archs/main_sub/execution/batch_processor.py:27` — 导入 `SubAgentManager`
- `nexau/archs/main_sub/execution/batch_processor.py:43` — `BatchProcessor.__init__`
- `nexau/archs/main_sub/execution/batch_processor.py:50` — 持有 `subagent_manager`
- `nexau/archs/main_sub/execution/batch_processor.py:307` — `_execute_batch_item_safe`
- `nexau/archs/main_sub/execution/batch_processor.py:327` — batch 内 `call_sub_agent`
- `nexau/archs/main_sub/execution/parse_structures.py:53-66` — `SubAgentCall` dataclass
- `nexau/archs/main_sub/execution/parse_structures.py:94` — `ExecutableCall` 联合类型
- `nexau/archs/main_sub/execution/response_parser.py:79` — `SubAgentCall` 构造
- `nexau/archs/main_sub/execution/response_parser.py:365` — `_parse_sub_agent_call`
- `nexau/archs/main_sub/agent.py:84` — `agent_id` 参数
- `nexau/archs/main_sub/agent.py:89` — `is_root` 参数
- `nexau/archs/main_sub/agent.py:286` — 子 Agent 内部继承 `is_root`
- `nexau/archs/main_sub/agent.py:826-827` — `parent_agent_state.root_run_id / run_id` 父链
- `nexau/archs/main_sub/agent.py:970` — `Agent.run()` 入口
- `nexau/archs/main_sub/agent_state.py:116` — `set_global_value` facade
- `nexau/archs/main_sub/config/config.py:236` — 导入 `recall_sub_agent`
- `nexau/archs/main_sub/config/config.py:239-244` — 有 sub_agents 时自动注入工具
- `nexau/archs/tool/builtin/recall_sub_agent_tool.py:11` — `recall_sub_agent` 函数
- `nexau/archs/tool/builtin/recall_sub_agent_tool.py:33` — defensive `getattr(agent_state, "_executor")`
- `nexau/archs/tool/builtin/recall_sub_agent_tool.py:40` — defensive `getattr(executor, "subagent_manager")`
- `nexau/archs/tool/builtin/recall_sub_agent_tool.py:48-53` — 调用 `call_sub_agent`
- `nexau/archs/tool/builtin/recall_sub_agent_tool.py:54-60` — success envelope
- `nexau/archs/tool/builtin/description/recall_sub_agent_tool.yaml` — 工具 schema

### 相关 RFC

- `rfcs/0002-agent-team.md` — 多 Agent 团队级编排（本 RFC 覆盖的"单父单层"是它的子集）
- `rfcs/0008-session-persistence-and-history.md` — sub-agent history 依赖的 session/agent_repo 层
- `rfcs/0012-global-storage-and-session-mutations.md` — GlobalStorage / AgentContext / AgentState 3-tier 语义（本 RFC 是其在子 Agent 调用点的应用）
- `rfcs/0006-rfc-catalog-completion-master-plan.md` — 本 RFC 的任务 T7
