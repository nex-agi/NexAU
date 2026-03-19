# RFC-0011: 统一 Token Usage 核算体系

- **状态**: draft
- **优先级**: P1
- **标签**: `architecture`, `observability`, `dx`
- **影响服务**: `nexau/archs/main_sub/execution/`, `nexau/archs/llm/llm_aggregators/`, `nexau/archs/session/`, `nexau/archs/tracer/`, `nexau/core/`
- **创建日期**: 2026-03-16
- **更新日期**: 2026-03-16

## 摘要

当前 NexAU 的 token usage 以 `JsonDict | None`（`dict[str, Any]`）松散传递，缺少类型化数据模型、会话级聚合与 usage 事件流。本 RFC 提出三层统一核算体系：

1. **规范数据模型** — `TokenUsage` 不可变 dataclass，替代裸 dict；
2. **Provider 归一化** — 在 `_normalize_usage()` 出口处直接产出 `TokenUsage`，覆盖 OpenAI / Anthropic / Gemini / 兼容网关；
3. **层级聚合器** — `UsageAccumulator` 沿 Agent → Sub-Agent 树自底向上汇总，支持 token budget 限额；
4. **事件与持久化** — 新增 `UsageEvent`，经 `AgentEventsMiddleware` 推送；`SessionManager` 持久化 run-level 汇总。

目标是让每一次 LLM 调用的 token 消耗可类型化追踪、可聚合核算、可对外暴露，同时不破坏现有 `ModelResponse` / middleware / tracer 链路。

> **Scope 说明**：费用计算（cost estimation / pricing table / dollar budget）不在本 RFC 范围内，后续可作为独立 RFC 在本体系之上扩展。

---

## 动机

### 1) Usage 以裸 dict 传递，缺少类型安全

当前 `ModelResponse.usage` 类型为 `JsonDict | None`（即 `dict[str, Any] | None`），下游消费者只能靠 `.get("input_tokens", 0)` 猜键名。这直接违反项目 "Zero `Any`" 的类型安全方针，也导致以下问题：

- IDE 无法推断 usage 字段，无法自动补全；
- 没有编译期保护：拼错键名（如 `"imput_tokens"`）只有运行时才暴露；
- `_normalize_usage()` 返回 `dict[str, Any] | None`，标准化后仍然是松散 dict。

### 2) 无会话级 / Agent 树级 usage 聚合

当前 usage 只存在于单次 `ModelResponse` 中：

- 一个 run 跨多轮迭代（iteration）的 token 消耗没有汇总；
- Sub-agent 树中各节点的 usage 没有向 parent 冒泡；
- `SessionManager` 持久化 history 时不保存 usage 汇总。

OpenHands 用 `Metrics` 在 agent controller 级别做累加；OpenCode 用 session-level `getUsage()` 做结算。NexAU 需要类似机制。

### 3) 无 usage 事件流

`AgentEventsMiddleware` 已经桥接了 text / tool_call / thinking 等流式事件，但没有 usage 事件。用户（尤其是 transport 层和 UI）无法实时获知 token 消耗进度，只能在 run 结束后手动从 `ModelResponse` 中提取。

---

## 设计

### 概述

```
┌─────────────────────────────────────────────────────────┐
│                   Layer 4: 持久化 & 事件                  │
│  UsageEvent → AgentEventsMiddleware → Transport/UI      │
│  Message.metadata["usage"]  (随 history 自动落盘)        │
└────────────────────────┬────────────────────────────────┘
                         │ emits / persists
┌────────────────────────▼────────────────────────────────┐
│              Layer 3: 层级聚合器                          │
│  UsageAccumulator (per Agent)                           │
│    .record(token_usage)                                 │
│    .merge_child(child_accumulator)                      │
│    .check_budget() → over_token / ok                    │
└────────────────────────┬────────────────────────────────┘
                         │ consumes
┌────────────────────────▼────────────────────────────────┐
│              Layer 2: Provider 归一化                     │
│  _normalize_usage() → TokenUsage (not dict)             │
└────────────────────────┬────────────────────────────────┘
                         │ normalizes
┌────────────────────────▼────────────────────────────────┐
│              Layer 1: 规范数据模型                        │
│  TokenUsage    — 不可变 token 计数                       │
│  UsageSummary  — run-level 聚合快照                      │
└─────────────────────────────────────────────────────────┘
```

### 详细设计

#### Layer 1: 规范数据模型

新增模块 `nexau/core/usage.py`，定义 frozen dataclass。

##### `TokenUsage`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Immutable, type-safe token count snapshot.

    RFC-0011: 规范化 token 计数，替代 JsonDict | None。

    All fields default to 0 so callers never need null-checks.
    Provider-specific extras (e.g. Anthropic cache tokens) are stored
    as explicit fields to allow precise accumulation via __add__.
    """

    input_tokens: int = 0
    """Prompt / input tokens (including cache contributions when applicable)."""

    completion_tokens: int = 0
    """Output / completion tokens."""

    reasoning_tokens: int = 0
    """Tokens consumed by chain-of-thought / extended thinking (0 if not applicable)."""

    total_tokens: int = 0
    """Grand total. If the provider supplies it, use theirs; otherwise sum of above."""

    cache_creation_tokens: int = 0
    """Anthropic: tokens written to prompt cache in this request."""

    cache_read_tokens: int = 0
    """Anthropic: tokens read from prompt cache in this request."""

    input_tokens_uncached: int = 0
    """Base input tokens excluding cache contributions (Anthropic accounting)."""

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Element-wise addition for accumulation."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            input_tokens_uncached=self.input_tokens_uncached + other.input_tokens_uncached,
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize to plain dict for JSON / tracer compatibility."""
        ...
```

**设计决策 — 为何用 frozen dataclass 而非 Pydantic**：

- `TokenUsage` 是内部数据传输对象，不涉及 API validation；
- frozen 保证不可变，避免共享引用时被意外篡改；
- slots=True 减少内存占用（高频创建场景）；
- 与现有 `ModelToolCall`、`ModelResponse` 等 dataclass 风格一致。

**设计决策 — 为何显式列出 cache 字段而非用 extras dict**：

- Anthropic cache tokens 已经在 `_normalize_usage()` 中显式处理，说明它们是已知的、常用的维度；
- 显式字段允许 `__add__` 精确累加，extras dict 累加语义不明；
- 如未来新增 provider 有全新维度（如 audio tokens），再添加字段或引入 extras。

##### `UsageSummary`

```python
@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Run-level usage summary, persisted to session storage.

    RFC-0011: 每次 run 结束时由 UsageAccumulator 生成的快照。
    """

    run_id: str
    agent_name: str
    total_usage: TokenUsage
    llm_call_count: int = 0
    child_summaries: tuple[UsageSummary, ...] = ()
    """Sub-agent summaries, forming a recursive tree mirroring the agent hierarchy."""
```

##### Provider 兼容表

| Provider          | input_tokens | completion_tokens | reasoning_tokens | cache_creation | cache_read | total_tokens |
|-------------------|:------------:|:-----------------:|:----------------:|:--------------:|:----------:|:------------:|
| OpenAI Chat       | prompt_tokens | completion_tokens | completion_tokens_details.reasoning_tokens | — | — | total_tokens |
| OpenAI Responses  | input_tokens  | output_tokens     | reasoning_tokens  | — | — | total_tokens |
| Anthropic         | input_tokens  | output_tokens     | — (thinking tokens billed as output) | cache_creation_input_tokens | cache_read_input_tokens | input + output |
| Gemini REST       | promptTokenCount | candidatesTokenCount | thoughtsTokenCount | — | — | totalTokenCount |
| OpenAI-compatible | prompt_tokens | completion_tokens | varies | — | — | total_tokens |

> **注意**：Anthropic 的 extended thinking tokens 计入 `output_tokens`，不单独报告 reasoning_tokens。`_normalize_usage()` 当前从 `completion_tokens_details` 或 `output_tokens_details` 提取 reasoning_tokens，对 Anthropic 该值为 0。

#### Layer 2: Provider 归一化

修改 `nexau/archs/main_sub/execution/model_response.py`：

##### `_normalize_usage()` 返回 `TokenUsage`

```python
from nexau.core.usage import TokenUsage


def _normalize_usage(usage: dict[str, object] | None) -> TokenUsage:
    """Normalize provider-specific usage dict into canonical TokenUsage.

    RFC-0011: 归一化入口，所有 from_openai_message / from_anthropic_message /
    from_openai_response / from_gemini_rest 统一在此收敛。

    Returns TokenUsage (never None) — 无 usage 时返回零值 TokenUsage()。
    """
    if usage is None:
        return TokenUsage()

    # ... 现有归一化逻辑保持不变，最后构造 TokenUsage 而非 dict ...
    return TokenUsage(
        input_tokens=direct_input,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        total_tokens=total,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
        input_tokens_uncached=uncached,
    )
```

**向后兼容**：`ModelResponse.usage` 类型从 `JsonDict | None` 改为 `TokenUsage`。由于 `TokenUsage` 提供 `to_dict()`，且下游主要消费者（tracer adapters、middleware）目前只调用 `.get()`，需在过渡期提供 `__getitem__` / `get` 兼容方法或一次性迁移下游。推荐一次性迁移（文件数量有限）。

#### Layer 3: 层级聚合器

新增 `nexau/core/usage.py` 中的 `UsageAccumulator` 类。

```python
import threading
from enum import Enum


class BudgetStatus(Enum):
    OK = "ok"
    OVER_TOKEN = "over_token"


class UsageAccumulator:
    """Thread-safe, per-agent token accumulator.

    RFC-0011: 层级聚合器，每个 Agent/Executor 实例持有一个。

    # 1. Executor 每次 LLM 调用后 record()
    # 2. Sub-agent 完成后 merge_child()
    # 3. Run 结束时 snapshot() 生成 UsageSummary
    """

    def __init__(
        self,
        agent_name: str,
        run_id: str,
        token_budget: int | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._agent_name = agent_name
        self._run_id = run_id
        self._token_budget = token_budget

        self._total_usage = TokenUsage()
        self._llm_call_count = 0
        self._children: list[UsageSummary] = []

    def record(self, usage: TokenUsage) -> None:
        """Record a single LLM call's usage.

        RFC-0011: 由 Executor 在每次 after_model hook 之后调用。
        线程安全：sub-agent 可能并行执行。
        """
        with self._lock:
            self._total_usage = self._total_usage + usage
            self._llm_call_count += 1

    def merge_child(self, child_summary: UsageSummary) -> None:
        """Merge a completed sub-agent's summary into this accumulator.

        RFC-0011: 由 SubAgentManager 在子 agent 完成后调用。
        子 usage 累加到 parent total，同时保留子 summary 用于树形展示。
        """
        with self._lock:
            self._total_usage = self._total_usage + child_summary.total_usage
            self._children.append(child_summary)

    def check_budget(self) -> BudgetStatus:
        """Check whether token budget is exceeded.

        RFC-0011: 由 Executor 在每轮迭代前调用。
        返回枚举值而非抛异常，让调用方决定策略（warning / stop / compact）。
        """
        with self._lock:
            if self._token_budget is not None and self._total_usage.total_tokens >= self._token_budget:
                return BudgetStatus.OVER_TOKEN
            return BudgetStatus.OK

    @property
    def total_usage(self) -> TokenUsage:
        """Current cumulative usage (read-only snapshot)."""
        with self._lock:
            return self._total_usage

    def snapshot(self) -> UsageSummary:
        """Create an immutable summary of current state.

        RFC-0011: 由 Executor 在 run 结束时调用，传给 SessionManager 持久化。
        """
        with self._lock:
            return UsageSummary(
                run_id=self._run_id,
                agent_name=self._agent_name,
                total_usage=self._total_usage,
                llm_call_count=self._llm_call_count,
                child_summaries=tuple(self._children),
            )
```

**与 ContextCompactionMiddleware 的关系**：

`ContextCompactionMiddleware` 管理的是**单次 run 内的上下文窗口**（"还剩多少 token 可以填进 prompt"）。`UsageAccumulator` 管理的是**累计消耗**（"总共用了多少 token"）。两者正交：

- compaction 看 `max_context_tokens` vs 当前 history token 数；
- budget 看 `token_budget` vs 累计 `total_tokens`。

#### Layer 4: 事件与持久化

##### 4a. Usage 事件

新增 `UsageUpdateEvent` 到 `nexau/archs/llm/llm_aggregators/events.py`：

```python
@dataclass
class UsageUpdateEvent(Event):
    """Emitted after each LLM call with token usage data.

    RFC-0011: 由 AgentEventsMiddleware 在 after_model hook 中发射。
    Transport 层可将此事件转发给 UI，实现实时 token 消耗展示。
    """

    type: str = field(default="USAGE_UPDATE", init=False)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cumulative_usage: TokenUsage = field(default_factory=TokenUsage)
    iteration: int = 0
```

新增 `UsageSummaryEvent` 在 run 结束时发射：

```python
@dataclass
class UsageSummaryEvent(Event):
    """Emitted when a run finishes, carrying the complete UsageSummary tree.

    RFC-0011: 由 AgentEventsMiddleware 在 RunFinishedEvent 之前发射。
    """

    type: str = field(default="USAGE_SUMMARY", init=False)
    summary: UsageSummary = field(default_factory=lambda: UsageSummary(run_id="", agent_name="", total_usage=TokenUsage()))
```

##### 4b. History 持久化：`Message.metadata["usage"]`

每条 assistant 消息在构造时将该次 LLM 调用的 `TokenUsage` 写入 `metadata`。`Message.metadata` 已经承载了 `response_items`、`thought_signature` 等字段，usage 自然归入同一层。

```python
# executor.py — 构造 assistant message 时写入 usage
assistant_msg = response.to_ump_message()
assistant_msg.metadata["usage"] = response.usage.to_dict()
history.append(assistant_msg)
```

好处：

- **零 schema 变更**：`Message.metadata` 是现有的 `dict[str, Any]`，`HistoryList.flush()` → `AgentRunActionModel.append_messages` 自动序列化 metadata，不需要改 ORM 或加字段；
- **history replay 自带 usage**：每条消息自带 usage，不需要额外查询；
- **run/session 级汇总按需计算**：遍历某个 run_id 下所有 assistant messages 的 `metadata["usage"]` 累加即可得到 run 级汇总，不需要冗余存储；
- 与 `response_items`、`thought_signature` 等现有 metadata 字段模式一致。

###### 按需查询接口

```python
class SessionManager:
    async def get_run_usage(self, session_id: str, run_id: str) -> TokenUsage:
        """Compute run-level usage by summing assistant message metadata.

        RFC-0011: 从该 run_id 的 AgentRunActionModel 中提取所有
        assistant messages，累加 metadata["usage"]。
        """
        ...

    async def get_session_total_usage(self, session_id: str) -> TokenUsage:
        """Aggregate all runs' usage into a session-level total.

        RFC-0011: 遍历该 session 下所有 run 的 assistant messages，
        累加 metadata["usage"]。用于 session 级别的展示。
        """
        ...
```

###### 持久化流程总览

```
LLM 调用返回 ModelResponse (含 TokenUsage)
    │
    ├─→ Message.metadata["usage"] = usage.to_dict()    # 写入消息 metadata
    │       └─→ history.append(msg)
    │             └─→ HistoryList.flush()
    │                   └─→ AgentRunActionModel.append_messages  # 自动带 metadata
    │
    └─→ accumulator.record(usage)                       # 内存中聚合器累加
          └─→ run 结束时 accumulator.snapshot()          # 返回给调用方 + 发射事件
```

##### 4c. Executor 集成点

```python
# executor.py 伪代码 — 展示 UsageAccumulator 的接入位置

class Executor:
    def execute(self, config, messages, agent_state, global_storage) -> ExecutorOutput:
        # 1. 初始化 accumulator
        accumulator = UsageAccumulator(
            agent_name=config.name,
            run_id=agent_state.run_id,
            token_budget=config.token_budget,
        )
        agent_state.usage_accumulator = accumulator

        for iteration in range(max_iterations):
            # 2. Budget 检查
            status = accumulator.check_budget()
            if status != BudgetStatus.OK:
                # 发射 warning event，并根据策略决定是否中止
                ...

            # 3. LLM 调用
            response = llm_caller.call(messages, llm_config)

            # 4. 记录 usage
            usage: TokenUsage = response.usage  # 已经是 TokenUsage
            accumulator.record(usage)

            # 5. 写入 assistant message metadata
            assistant_msg = response.to_ump_message()
            assistant_msg.metadata["usage"] = usage.to_dict()
            history.append(assistant_msg)

            # 6. 发射 UsageUpdateEvent
            emit(UsageUpdateEvent(
                usage=usage,
                cumulative_usage=accumulator.total_usage,
                iteration=iteration,
            ))

            # ... tool execution, history update ...

        # 7. Run 结束，生成 summary 并发射事件
        summary = accumulator.snapshot()
        emit(UsageSummaryEvent(summary=summary))

        # 8. flush history（usage 已在每条 assistant message metadata 中，自动落盘）
        history.flush()

        return ExecutorOutput(content=..., usage_summary=summary)
```

##### 4d. Sub-Agent 集成

```python
# subagent_manager.py 伪代码

async def call_sub_agent(agent_name, message, agent_state, global_storage):
    # Sub-agent 执行后拿到 summary
    result = await sub_agent.run_async(message)
    child_summary = result.usage_summary

    # 向 parent accumulator 合并
    parent_accumulator = agent_state.usage_accumulator
    parent_accumulator.merge_child(child_summary)
```

#### Tracer 适配

`langfuse.py` 中的 `_sanitize_usage()` 改为直接消费 `TokenUsage.to_dict()`，不再需要过滤非 int 值。

```python
# 现有
def _sanitize_usage(usage: Mapping[str, object]) -> dict[str, int]:
    return {k: v for k, v in usage.items() if isinstance(v, int)}

# 改为
def _sanitize_usage(usage: TokenUsage) -> dict[str, int]:
    return usage.to_dict()
```

#### AgentConfig 扩展

```yaml
# agent.yaml
name: my_agent
llm_config:
  model: claude-sonnet-4-6
  ...

# RFC-0011: 新增 budget 配置
token_budget: 1000000        # 总 token 消耗上限（含 sub-agents）
budget_action: warn           # warn | stop (超出预算时的行为)
```

### 示例

#### 获取 run 的 usage summary

```python
agent = Agent(config=config, session_manager=session_mgr)
result = agent.run("Hello")

# 通过 result 直接获取
summary = result.usage_summary
print(f"Tokens: {summary.total_usage.total_tokens}")
print(f"Input: {summary.total_usage.input_tokens}")
print(f"Completion: {summary.total_usage.completion_tokens}")
print(f"LLM calls: {summary.llm_call_count}")

# 通过 session 查询历史
summaries = await session_mgr.get_usage_summaries(session_id)
for s in summaries:
    print(f"Run {s.run_id}: {s.total_usage.total_tokens} tokens")
```

#### 实时 usage 事件

```python
def handle_event(event: Event):
    match event:
        case UsageUpdateEvent():
            print(f"[iter {event.iteration}] +{event.usage.total_tokens} tokens, "
                  f"cumulative: {event.cumulative_usage.total_tokens}")
        case UsageSummaryEvent():
            print(f"Run total: {event.summary.total_usage.total_tokens} tokens")

agent.run("Complex task", event_handlers=[handle_event])
```

#### Token budget 限制

```yaml
name: budget_agent
llm_config:
  model: claude-opus-4-6
token_budget: 500000
budget_action: stop
```

当累计 token 消耗超出 500k 时，Executor 在下一轮迭代前中止并返回已有结果。

---

## 权衡取舍

### 考虑过的替代方案

#### A. 继续用 dict，仅补充 TypedDict

```python
class TokenUsageDict(TypedDict):
    input_tokens: int
    completion_tokens: int
    ...
```

**未采用原因**：TypedDict 是结构化的 dict，但不支持 `__add__`、不可冻结、不可自定义方法。累加和预算检查需要行为，不仅仅是形状约束。

#### B. 用 Pydantic BaseModel

**未采用原因**：`TokenUsage` 是高频创建对象（每次 LLM 调用），Pydantic 验证开销不必要。且 codebase 中 `ModelToolCall`、`ModelResponse` 等同级数据对象全部使用 stdlib dataclass，保持一致。

#### C. OpenHands 风格：全局 Metrics 单例

OpenHands 用全局 `Metrics` 对象。**未采用原因**：

- 全局单例不支持 NexAU 的 multi-agent / multi-session 并行；
- 每个 Agent/Executor 需要独立的 accumulator 实例，由 agent 树结构自然管理生命周期。

### 缺点

1. **`ModelResponse.usage` 类型变更**：从 `JsonDict | None` 到 `TokenUsage` 是 breaking change。但 `ModelResponse` 是内部 API（不在 `nexau.core` 公开），影响范围可控。
2. **内存开销**：每次 LLM 调用多创建一个 `TokenUsage` 小对象。`slots=True` 已优化，且 LLM 调用频率远低于对象创建的性能瓶颈。

---

## 实现计划

### 阶段划分

- [ ] **Phase 1: 数据模型** — `nexau/core/usage.py`（`TokenUsage`, `UsageSummary`, `BudgetStatus`）
- [ ] **Phase 2: 归一化迁移** — `_normalize_usage()` 返回 `TokenUsage`；`ModelResponse.usage` 类型改为 `TokenUsage`；迁移下游消费者
- [ ] **Phase 3: UsageAccumulator** — 聚合器实现、Executor 集成、SubAgentManager 集成
- [ ] **Phase 4: 事件** — `UsageUpdateEvent`、`UsageSummaryEvent`、AgentEventsMiddleware 接入
- [ ] **Phase 5: 持久化** — `Message.metadata["usage"]` 写入、SessionManager 查询接口（`get_run_usage` / `get_session_total_usage`，无 ORM schema 变更）
- [ ] **Phase 6: Token Budget** — `AgentConfig` 扩展、Executor budget 检查逻辑

### 相关文件

- `nexau/core/usage.py` — 新增：数据模型 + 聚合器
- `nexau/archs/main_sub/execution/model_response.py` — 修改：`_normalize_usage()` 返回 `TokenUsage`，`ModelResponse.usage` 类型变更
- `nexau/archs/main_sub/execution/executor.py` — 修改：集成 `UsageAccumulator`
- `nexau/archs/main_sub/execution/subagent_manager.py` — 修改：子 agent summary 合并
- `nexau/archs/llm/llm_aggregators/events.py` — 新增：`UsageUpdateEvent`、`UsageSummaryEvent`
- `nexau/archs/main_sub/execution/middleware/agent_events_middleware.py` — 修改：发射 usage 事件
- `nexau/archs/session/` — 修改：SessionManager 新增 `get_run_usage()` / `get_session_total_usage()` 查询接口（从 message metadata 累加，无 schema 变更）
- `nexau/archs/tracer/adapters/langfuse.py` — 修改：适配 `TokenUsage` 类型
- `nexau/archs/main_sub/execution/middleware/round_and_token_reminder.py` — 修改：可选读取 accumulator 数据

---

## 测试方案

### 单元测试

- `TokenUsage.__add__` 累加正确性（含 cache 字段）
- `TokenUsage.to_dict()` 序列化完整性
- `_normalize_usage()` 对 OpenAI / Anthropic / Gemini / None 输入的输出正确性
- `UsageAccumulator.record()` / `merge_child()` / `check_budget()` / `snapshot()` 线程安全验证
- `BudgetStatus` 在各阈值条件下的判定

### 集成测试

- 完整 Agent.run() 后 `result.usage_summary` 非零且合理
- Sub-agent 场景：parent summary 包含 child summaries，total 正确累加
- Token budget stop：设置极低 token_budget，验证 agent 提前中止
- Event stream：注册 event_handler，验证 `UsageUpdateEvent` / `UsageSummaryEvent` 按序发射
- 消息级持久化：assistant message `metadata["usage"]` 随 `HistoryList.flush()` 写入 DB，reload 后 metadata 完整可读
- Session 查询：`get_run_usage()` 从 message metadata 累加结果正确，`get_session_total_usage()` 跨 run 累加正确

### 手动验证

- 使用 `RoundAndTokenReminderMiddleware` 观察实时 token 消耗与 accumulator 数据一致性
- Langfuse dashboard 中 usage 数据与框架 `UsageSummary` 交叉验证

---

## 未解决的问题

1. **Streaming usage**：部分 provider（OpenAI Streaming）仅在 stream 最后一个 chunk 返回 usage，需确认 `llm_aggregators` 层是否已正确捕获。若未捕获，需在 Phase 2 补充。

2. **Image / audio token 计量**：Gemini 和 OpenAI 的多模态 token 计量格式不同（Gemini 报告 `cachedContentTokenCount`，OpenAI 在 `prompt_tokens_details` 中报告 `audio_tokens` / `image_tokens`）。Phase 1 暂不处理，后续按需扩展 `TokenUsage` 字段。

3. **Responses API `include` 选项**：OpenAI Responses API 需要 `include=["usage"]` 才返回 usage。当前 `llm_aggregators` 是否已正确设置？需确认。

4. **跨 session 的 budget**：当前 budget 作用域是 per-run。是否需要 per-session 或 per-user 级别的 budget？暂不实现，但 `SessionManager.get_session_total_usage()` 为此预留了查询接口。

---

## 参考资料

- **OpenHands** `openhands/llm/metrics.py` — `TokenUsage` dataclass, `Metrics` 累加器
- **OpenCode** `packages/core/src/session/usage.ts` — 4 层 usage 系统, `getUsage()` 结算, provider 兼容表
- **NexAU 现有实现** `nexau/archs/main_sub/execution/model_response.py` — `_normalize_usage()`, `_coerce_usage()`
- **NexAU tracer** `nexau/archs/tracer/adapters/langfuse.py` — `_sanitize_usage()`
- **NexAU events** `nexau/archs/llm/llm_aggregators/events.py` — `CompactionStartedEvent`, `CompactionFinishedEvent`
- **NexAU token counter** `nexau/archs/main_sub/utils/token_counter.py` — `TokenCounter` (context window counting, orthogonal to usage accounting)
