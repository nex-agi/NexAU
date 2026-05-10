# RFC-0024: ToolResultBlock raw_output + RunStartExtra.trace_id（LLM/UI 双视图持久化）

- **状态**: draft
- **优先级**: P1
- **标签**: `architecture`, `persistence`, `llm`, `observability`
- **影响服务**: nexau core (`messages.py` schema, `tool_executor.py`, `agent.py` run lifecycle)、所有 nexau 下游持久化消费方（NAC、coder、workbench、OSS）
- **创建日期**: 2026-05-07
- **更新日期**: 2026-05-07

## 摘要

nexau 当前在工具结果与 run 生命周期的两个关键路径上存在 **event channel 富 / persisted channel 穷** 的信息不对等，使下游 SSOT 消费者（如 NAC RFC-0088 v2）无法从持久化的 `nexau_agent_run_actions` 复现 live event 的 UI 视图。本 RFC 提议：

1. **`ToolResultBlock.raw_output: dict | None`**：工具返回 dict 时持久化原始结构，`returnDisplay` 不再被 `tool_executor` strip 而是迁入 `raw_output`。`ToolCallResultEvent` 改为从 `raw_output` 派生。
2. **`RunStartExtra.trace_id` 落地 populate**：`agent.run()` 入口从当前 OTel span context 取 W3C trace id 写入 `RUN_START.extra.trace_id`，`RUN_STARTED` event 同时携带。

整改后，下游消费者可以 100% 从持久化数据复现 live UI；当前各下游各自维护的临时方案（NAC gateway 的 `_trace_id` stamp 等）可下线。

## 一、动机

### 1.1 当前的信息不对等

nexau 在 chat 流式路径上发送 `ToolCallResultEvent` 给消费方（`{tool_use_id, raw, returnDisplay, ...}`），但 `tool_executor.py` 在持久化前会做：

```python
# tool_executor.py:368-373
raw_output.pop("returnDisplay", None)
llm_tool_output = self._strip_display_only_from_llm_output(llm_tool_output)
return ToolExecutionResult(raw_output=raw_output, llm_tool_output=llm_tool_output)
```

之后 `llm_tool_output` 经过 RFC-0017 默认 XML formatter 压缩成扁平文本，再写入 `ToolResultBlock.content`。最终落到 `nexau_agent_run_actions.append_messages[*].content[*]` 的 tool_result 块只有压缩文本——`returnDisplay`、原始 dict 字段（`output_dir`、`stdout_file`、`exit_code` 等 meta）一并丢失。

### 1.2 SSOT 消费者受冲击

当下游服务（如 NAC playground）尝试用 `nexau_agent_run_actions` 作为 UI 单一真相源：
- 历史会话 replay 时，`returnDisplay` 不存在 → 工具结果摘要消失
- `output_dir` / `stdout_file` 路径不存在 → "打开输出目录"等 UI affordance 失效
- 自定义 formatter（RFC-0017 §3.4）的输出对 UI 完全 opaque → 无法做任何 typed 渲染

这违反了"persisted ⊇ event"的 SSOT 不变式，导致 live 路径和 replay 路径的渲染输出不一致——用户开 chat 看到漂亮的 UI hint，刷新页面后变成裸文本。

### 1.3 trace_id 同类问题

`RunStartExtra` schema 已经在 RFC-0022 Phase 1 里预留了 `trace_id: str | None` 字段，但 `agent.run()` 调用 `create_run_start()` 时没有 populate（实测生产 DB 200+ run_start 行 `extra->>'trace_id'` 全 NULL）。

`RUN_STARTED` event 也没有 `trace_id` 字段。下游想做 "View Trace" 链接需要外部补全（例如 NAC 在 gateway 处从 traceparent 提取 stamp 进 live event 的 `_trace_id` 字段——纯 in-memory，进程重启就丢，刷新页面失效）。

### 1.4 为什么不只在下游补

每个 nexau 下游消费方都自己 reverse-engineer 这套语义，重复造轮子且容易漂移：
- NAC：gateway tap stamp `_trace_id`、前端 XML envelope unwrap → JSON
- 假设 coder / workbench / OSS 各自实现自己的 returnDisplay 取回路径

这把 LLM/UI 分离的语义边界推到下游，违反 nexau 作为 library 的"封装语义、暴露契约"定位。`returnDisplay` 这个字段本就是 nexau 设计来做 LLM/UI 分离的，但目前只在 ephemeral event channel 起作用，没贯彻到持久化层。

## 二、设计

### 2.1 数据模型变更

#### 2.1.1 `ToolResultBlock.raw_output` 新增字段

```python
# nexau/core/messages.py
class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    is_error: bool = False
    content: str | list[TextBlock | ImageBlock]  # ← 不变, formatter 输出, LLM 视图

    # 新增：tool 经过 ToolExecutor.finalize_tool_execution 标准化后的原始返回值
    # （dict / list / None）。Schema 限定 dict | list | None 以匹配 Pydantic
    # 持久化约束。UI 消费方从这里取所有结构化字段：
    #   - raw_output["returnDisplay"]   旧 returnDisplay 的接班人
    #   - raw_output["duration_ms"]     工具自带的 meta
    #   - raw_output["output_dir"]      文件路径等 affordance
    #   - 其他工具自定义字段
    # 注：返回字符串的工具会被 finalize_tool_execution 包成
    # ``{"result": "..."}``，框架不剥离这层 trivial wrapping —— 保持字段
    # 名 "raw" 的契约，UI 想隐藏 trivial wrapping 自己 filter。
    raw_output: dict[str, Any] | list[Any] | None = None
```

序列化策略：`raw_output is None` 时 `model_dump(exclude_none=True)` 不写入 JSON，避免对老下游产生 schema 噪音。

#### 2.1.2 `RunStartExtra.trace_id` 落地 populate

schema 已存在（RFC-0022 Phase 1 落地），仅需在 call site 填值：

```python
# nexau/archs/main_sub/agent.py 或 history_list.py 的 persist_run_start 调用处
import opentelemetry.trace as ot_trace

def _current_trace_id() -> str | None:
    """Extract W3C trace id (32-hex) from current OTel span context, or None."""
    span = ot_trace.get_current_span()
    if span is None:
        return None
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")

# create_run_start 调用：
extra = RunStartExtra(
    user_message_blocks=user_message_blocks,
    fresh_context=fresh_context,
    trace_id=_current_trace_id(),  # ← 新增
)
```

`RUN_STARTED` event 同步加 `trace_id` 字段：

```python
class RunStartedEvent(BaseEvent):
    type: Literal["RUN_STARTED"] = "RUN_STARTED"
    run_id: str
    root_run_id: str
    agent_id: str
    thread_id: str
    parent_run_id: str | None = None
    trace_id: str | None = None  # ← 新增
    # 其他既有字段...
```

### 2.2 行为变更

#### 2.2.1 `tool_executor.py` 不再 strip returnDisplay

当前：

```python
# 当前 tool_executor.py:368-373
raw_output.pop("returnDisplay", None)
llm_tool_output = self._strip_display_only_from_llm_output(llm_tool_output)
```

改为：

```python
# llm_tool_output 仍然 strip — LLM 看不到 returnDisplay（防止 token 浪费）
llm_tool_output = self._strip_display_only_from_llm_output(llm_tool_output)

# raw_output 不再 strip — 完整传递给 ToolResultBlock.raw_output
# returnDisplay 在 raw_output 里保留，UI 消费方可读
# 下游 ToolResultBlock 构造处：
#   ToolResultBlock(
#       tool_use_id=tool_use_id,
#       content=llm_tool_output,          # formatter 输出（XML / 自定义）
#       raw_output=raw_output if isinstance(raw_output, dict) else None,
#       is_error=...,
#   )
```

#### 2.2.2 `ToolCallResultEvent` 改为从 raw_output 派生

```python
# AgentEventsMiddleware.after_tool（或类似 emit point）
ToolCallResultEvent(
    tool_call_id=tool_use_id,
    raw=raw_output,                                    # 同 ToolResultBlock.raw_output
    returnDisplay=raw_output.get("returnDisplay") if isinstance(raw_output, dict) else None,
    ...
)
```

以前 `ToolCallResultEvent.returnDisplay` 是独立字段，现在它就是 `raw.get("returnDisplay")`，**单一数据源**。

### 2.3 向前兼容

新字段 `raw_output` 是 optional + default None，老 reader 反序列化时忽略未知字段（pydantic `extra='allow'` 已配置）→ 下游可以渐进升级，不需要 schema 同步打通。

老数据迁移：
- 无 `raw_output` 的历史 ToolResultBlock 行：consumer 走 fallback（NAC 已实现 XML envelope unwrap 兜底）
- 选项：写一个一次性 backfill 脚本，对 `content` 是 XML envelope 的行做 unwrap 反推 `raw_output`。可选，不强制。

### 2.4 跨语言契约

nexau 是 library，下游有 Rust（NAC）、TypeScript（NAC frontend）、Python（OSS / coder / workbench）。schema 演进必须维持反序列化兼容：

- pydantic `extra='allow'` 已配置 → Python 老版本读新数据 OK
- Rust serde：`#[serde(default)]` + `Option<T>` 处理新字段 → 老版本读新数据 OK
- TypeScript：`raw_output?: Record<string, unknown>` → tsc 不报错

新字段添加后，nexau bump minor 版本，下游按需升级。

## 三、权衡取舍

### 3.1 已废弃方案

| 方案 | 理由 |
|---|---|
| **保持现状，下游各自补全** | 重复造轮子，下游漂移；自定义 formatter 完全无解；NAC 的 in-memory `_trace_id` map 无法跨进程/跨 tab 持久 |
| **改 `content` 字段为 dict（一并装下 LLM 文本和 raw）** | 破坏 RFC-0017 设计（content 必须是 LLM-ready 文本，截断和 token 计算依赖此），影响面更大 |
| **加 `metadata: dict` 模糊字段** | schema 不清晰；下游 consumer 不知道里面有什么；本 RFC 选 typed `raw_output`（语义明确） |
| **不动 nexau，改 NAC 写自己的 UI 投影表** | 退化回 RFC-0088 v2 §1.1 已废弃的双写架构 |
| **trace_id 走 event channel transitional** | 当前 NAC 已在做，但进程重启 / 多 tab 跨实例不持久；本 RFC 落地后下线 |

### 3.2 缺点

1. **存储成本上升**：tool_result 行平均增加 ~30% 体积（dict 字段未压缩）。但工具输出本就比 message body 小，绝对增量在 KB 级别，月度增长可控。可以选择性对超大 raw_output（>16KB）压缩或 truncate。
2. **schema 演进风险**：所有持久化 schema 加新字段都有兼容性风险。`Optional + default None` 已最大化降低，但下游必须合理处理。
3. **returnDisplay 语义迁移**：现有工具 `return {"content": ..., "returnDisplay": ...}` 写法不变；下游消费 `event.returnDisplay` 的代码也兼容（仍由 raw 派生）。短期无破坏。

### 3.3 与 RFC-0017 的关系

RFC-0017 解决了 **LLM 输入** 的扁平化问题（XML formatter 替代 `str(dict)`）。本 RFC 解决 **UI 持久化** 的结构丢失问题（raw_output 替代 strip-after-event）。两者正交：

- RFC-0017：`content` 字段如何为 LLM 优化
- RFC-0024：`raw_output` 字段如何为 UI 保留结构

formatter 该咋优化继续咋优化，UI 不再依赖 `content` 反向 parse。

## 四、实现计划

### Phase 1: schema + populate（无破坏，可独立合）

- [ ] `nexau/core/messages.py`：`ToolResultBlock` 加 `raw_output` 字段
- [ ] `nexau/archs/main_sub/execution/tool_executor.py`：构造 `ToolResultBlock` 时把 raw_output dict 传入；不再 strip raw_output 的 returnDisplay
- [ ] `nexau/archs/main_sub/agent.py`：`create_run_start` 调用从 OTel span 取 trace_id 填入 `RunStartExtra.trace_id`
- [ ] `nexau/archs/llm/llm_aggregators/events.py`：`RunStartedEvent` 加 `trace_id` 字段，发射处从 RunStartExtra 取
- [ ] `nexau/archs/llm/llm_aggregators/events.py`：`ToolCallResultEvent.raw` 字段类型 align 到 `raw_output`，emit 处从 ToolResultBlock 取
- [ ] 单测：roundtrip 测试覆盖 raw_output 持久化与读取；trace_id populate 覆盖 OTel context 有/无两种情况
- [ ] CHANGELOG / 升级文档

### Phase 2: deprecate strip 行为（行为变更，下游 grace period）

- [ ] `tool_executor.py` 增加 deprecation log：仍 emit `returnDisplay` 在 event 顶层，但同时存进 raw_output；下游 N 个版本后下游切换到 raw_output.returnDisplay
- [ ] `ToolCallResultEvent.returnDisplay` 标 `# deprecated, read raw_output["returnDisplay"]`

### Phase 3: 下游切换 + 下线临时方案

- [ ] NAC：前端 `_trace_id` map 改读 `RunStartExtra.trace_id`，gateway tap stamp 下线
- [ ] NAC：前端 returnDisplay 改读 `tool_result.raw_output.returnDisplay`，XML envelope unwrap 改成 fallback only

### Phase 4: 老数据 backfill（可选）

视生产数据形态决定。NAC 实测旧数据大头是 `json` 字符串形态（RFC-0017 之前），unwrap 简单；XML 形态可反向 parse。只在确实需要历史 session 完整 UI 的时候做。

## 4.A trace_id 契约（API surface）

`trace_id` 是 nexau 持久化和广播的标识符，但 nexau library **不做任何来源推断**。完整契约：

### 4.A.1 类型与语义

- 类型：`str | None`，建议是 W3C 32-hex 格式（与 OTel / Langfuse v3 / Jaeger 现代默认对齐）
- nexau 不做格式校验（保留 opaque）；非 W3C 字符串也存，但跨系统 trace 链可能断
- None 表示"未知/未提供"——所有观测后端独立工作但无法关联

### 4.A.2 nexau library 不做的事

- ❌ 不读 `opentelemetry.trace.get_current_span()`
- ❌ 不读任何全局 / 线程局部 / asyncio ContextVar
- ❌ 不读环境变量
- ❌ 不解析 `traceparent` header（nexau 不感知 HTTP 边界）

**理由**：消费者的观测栈差异大：

| 消费者 | trace_id 来源 | 隐式推断风险 |
|---|---|---|
| **NAC** | gateway 解析 `traceparent` header → OTel global → 显式参数 | 低 |
| **xiaobei** | `ext.trace.trace_id` body 字段 → `LangfuseTracer` (isolated SdkTracerProvider) | **高**——OTel global 拿到的可能是 httpx auto-instrumentation 的随机 span，跟 Langfuse trace 完全无关 |
| OSS / 其他 | 各种各样 | 不可知 |

任何隐式 fallback 在某些环境里会写入**错误**的 trace_id（看起来"有 trace 链接"但点开是无关 trace）——比 None 还糟。

### 4.A.3 caller 责任

- 显式传 `agent.run_async(trace_id=...)` 或 `agent.run(trace_id=...)`
- 确保自己的所有观测后端（Jaeger / Langfuse / 日志）以同一个 W3C trace_id 为根

### 4.A.4 nexau 内部的 trace_id 单一来源

caller 传入后，nexau 把 trace_id 存进 `AgentState.trace_id`，从此**所有派生位置**都从这个字段读：

- `RunStartExtra.trace_id`（DB 持久化）
- `RunStartedEvent.trace_id`（live SSE event，via `AgentEventsMiddleware.before_agent`）
- 子 agent 通过 `parent_agent_state.trace_id` 自动继承

保证 live 和 replay 一致、root 和 sub-agent 一致。

### 4.A.5 消费者迁移清单

| 调用方 | 当前 | 升级（opt-in） | 行为不变？ |
|---|---|---|---|
| NAC（agent-runtime） | 已显式传 | 无需改 | 是 |
| xiaobei（server.py） | 不传 → DB NULL | `current_agent.run(..., trace_id=ext.get("trace", {}).get("trace_id"))` | 不升级则 NULL（同今天）；升级则 DB 有值 |
| OSS / 其他 | 不传 → DB NULL | 同上 | 不升级则 NULL（同今天） |

**默认行为完全等价于 RFC-0024 之前**——零回归。

## 五、未解决的问题

1. **超大 raw_output 处理**：是否对 >16KB / >64KB 做截断或外存？目前留空——大多数工具输出在 KB 级别，不优化。
2. **自定义 formatter 的 raw_output 形态**：自定义 formatter 工具是否必须保证返回结构化 dict？还是允许工具直接返回 str（raw_output=None）？建议：尊重工具自由，str 工具就是 None，UI 兜底渲染原文。
3. **跨 process trace_id**：sub-agent 跑在另一个 thread/task，OTel context 是否自动传递？需要验证 `opentelemetry.context.attach()` 在 ThreadPoolExecutor / asyncio.Task 边界的行为。
4. **是否同步把 `model_call_id` / `usage` 也确认进 SSOT**：调研发现 `message_metadata.usage` 已落库，OK；`MODEL_CALL_FINISHED.model_name` 没有持久化对照——本 RFC 范围外但可以一并审计。

## 六、参考资料

- [RFC-0017: 工具输出扁平化](./0017-flatten-tool-output.md) — `content` 字段的 LLM-视图优化
- [RFC-0022: Agent Run Action Lifecycle](./0022-agent-run-action-lifecycle-and-typed-blocks.md) — `RunStartExtra.trace_id` schema 起源
- NAC RFC-0088 v2: Run Actions 持久化 — SSOT 消费者侧需求来源（`docs/rfcs/0088-run-actions-merge.md` §3.5）
- 历史调研：north-coder 本地 `nexau.db`，2026-03-16 ~ 2026-05-05，2446 个 tool_result 行——证实 `returnDisplay` 在 100% 行里都被 strip，从未持久化
