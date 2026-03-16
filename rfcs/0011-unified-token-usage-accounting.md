# RFC-0011: 统一 Token Usage 核算体系

- **状态**: draft
- **优先级**: P1
- **标签**: `architecture`, `observability`, `dx`
- **影响服务**: `nexau/archs/main_sub/execution/`, `nexau/archs/llm/llm_aggregators/`, `nexau/archs/session/`, `nexau/archs/tracer/`, `nexau/core/`
- **创建日期**: 2026-03-16
- **更新日期**: 2026-03-16

## 摘要

当前 NexAU 的 token usage 以 `JsonDict | None`（`dict[str, Any]`）松散传递，缺少类型化数据模型与 usage 事件流。本 RFC 提出统一核算体系：

1. **规范数据模型** — `TokenUsage` 不可变 dataclass，替代裸 dict；
2. **api_type 归一化** — 在 `_normalize_usage()` 出口处直接产出 `TokenUsage`，覆盖 OpenAI / Anthropic / Gemini / 兼容网关；
3. **事件与持久化** — 每次 LLM 调用后发射 `UsageUpdateEvent`，同时将 usage 写入 `Message.metadata` 随 history 自动落盘。

目标是让每一次 LLM 调用的 token 消耗可类型化追踪、可对外暴露、可持久化恢复，同时不破坏现有 `ModelResponse` / middleware / tracer 链路。

> **Scope 说明**：费用计算（cost estimation / pricing table / dollar budget）和 token budget 限额不在本 RFC 范围内，后续可作为独立 RFC 在本体系之上扩展。

---

## 动机

### 1) Usage 以裸 dict 传递，缺少类型安全

当前 `ModelResponse.usage` 类型为 `JsonDict | None`（即 `dict[str, Any] | None`），下游消费者只能靠 `.get("input_tokens", 0)` 猜键名。这直接违反项目 "Zero `Any`" 的类型安全方针，也导致以下问题：

- IDE 无法推断 usage 字段，无法自动补全；
- 没有编译期保护：拼错键名（如 `"imput_tokens"`）只有运行时才暴露；
- `_normalize_usage()` 返回 `dict[str, Any] | None`，标准化后仍然是松散 dict。

### 2) Usage 未持久化

当前 usage 只存在于单次 `ModelResponse` 的内存对象中：

- 单次 `ModelResponse` 中的 usage 没有随 message metadata 持久化，history replay 后无法恢复 usage；
- 一个 run 跨多轮迭代（iteration）的 token 消耗无法事后追溯。

### 3) 无 usage 事件流

`AgentEventsMiddleware` 已经桥接了 text / tool_call / thinking 等流式事件，但没有 usage 事件。用户（尤其是 transport 层和 UI）无法实时获知 token 消耗进度，只能在 run 结束后手动从 `ModelResponse` 中提取。

---

## 设计

### 概述

```
┌─────────────────────────────────────────────────────────┐
│              Layer 3: 事件 & 持久化                       │
│  UsageUpdateEvent → AgentEventsMiddleware → Transport/UI │
│  Message.metadata["usage"]  (随 history 自动落盘)        │
└────────────────────────┬────────────────────────────────┘
                         │ emits / persists
┌────────────────────────▼────────────────────────────────┐
│              Layer 2: api_type 归一化 (Converter)          │
│  UsageConverterRegistry.get(api_type) → converter        │
│  _normalize_usage(raw, api_type) → TokenUsage            │
│                                                          │
│  内置: OpenAI / Anthropic / Gemini                       │
│  扩展: UsageConverterRegistry.register("xxx", custom)    │
└────────────────────────┬────────────────────────────────┘
                         │ converts
┌────────────────────────▼────────────────────────────────┐
│              Layer 1: 规范数据模型                        │
│  TokenUsage    — 不可变 token 计数                       │
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
    api_type-specific extras (e.g. Anthropic cache tokens) are stored
    as explicit fields to allow precise accumulation via __add__.
    """

    input_tokens: int = 0
    """Prompt / input tokens (including cache contributions when applicable)."""

    completion_tokens: int = 0
    """Output / completion tokens."""

    reasoning_tokens: int = 0
    """Tokens consumed by chain-of-thought / extended thinking (0 if not applicable)."""

    total_tokens: int = 0
    """Grand total. If the api_type supplies it, use theirs; otherwise sum of above."""

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
- 如未来新增 api_type 有全新维度（如 audio tokens），再添加字段或引入 extras。

##### api_type 兼容表

| api_type                    | input_tokens | completion_tokens | reasoning_tokens | cache_creation | cache_read | total_tokens |
|-----------------------------|:------------:|:-----------------:|:----------------:|:--------------:|:----------:|:------------:|
| `openai_chat_completion`    | prompt_tokens | completion_tokens | completion_tokens_details.reasoning_tokens | — | — | total_tokens |
| `openai_responses`          | input_tokens  | output_tokens     | reasoning_tokens  | — | — | total_tokens |
| `anthropic_chat_completion` | input_tokens  | output_tokens     | — (thinking tokens billed as output) | cache_creation_input_tokens | cache_read_input_tokens | input + output |
| `gemini_rest`               | promptTokenCount | candidatesTokenCount | thoughtsTokenCount | — | — | totalTokenCount |

> **注意**：Anthropic 的 extended thinking tokens 计入 `output_tokens`，不单独报告 reasoning_tokens。`_normalize_usage()` 当前从 `completion_tokens_details` 或 `output_tokens_details` 提取 reasoning_tokens，对 Anthropic 该值为 0。

#### Layer 2: api_type 归一化 — UsageConverter

##### 设计思路

内置覆盖主流 api_type（OpenAI / Anthropic / Gemini），同时允许使用者注册自定义 converter 以支持私有网关或新兴 api_type。

##### `UsageConverter` Protocol

```python
from typing import Protocol

from nexau.core.usage import TokenUsage


class UsageConverter(Protocol):
    """Strategy for converting an api_type-specific usage dict into TokenUsage.

    RFC-0011: 可扩展的 api_type usage 转化协议。
    内置实现覆盖 OpenAI / Anthropic / Gemini，用户可注册自定义实现。
    """

    def convert(self, raw_usage: dict[str, object]) -> TokenUsage:
        """Convert raw api_type usage dict to canonical TokenUsage."""
        ...
```

##### 内置 Converter 实现

```python
class OpenAIChatUsageConverter:
    """OpenAI ChatCompletion format (prompt_tokens, completion_tokens, total_tokens)."""

    def convert(self, raw: dict[str, object]) -> TokenUsage:
        details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
        return TokenUsage(
            input_tokens=_int(raw.get("prompt_tokens", 0)),
            completion_tokens=_int(raw.get("completion_tokens", 0)),
            reasoning_tokens=_int(details.get("reasoning_tokens", 0)),
            total_tokens=_int(raw.get("total_tokens", 0)),
        )


class OpenAIResponsesUsageConverter:
    """OpenAI Responses API format (input_tokens, output_tokens)."""

    def convert(self, raw: dict[str, object]) -> TokenUsage:
        return TokenUsage(
            input_tokens=_int(raw.get("input_tokens", 0)),
            completion_tokens=_int(raw.get("output_tokens", 0)),
            reasoning_tokens=_int(raw.get("reasoning_tokens", 0)),
            total_tokens=_int(raw.get("total_tokens", 0)),
        )


class AnthropicUsageConverter:
    """Anthropic format with cache token accounting."""

    def convert(self, raw: dict[str, object]) -> TokenUsage:
        direct_input = _int(raw.get("input_tokens", 0))
        cache_creation = _int(raw.get("cache_creation_input_tokens", 0))
        cache_read = _int(raw.get("cache_read_input_tokens", 0))
        completion = _int(raw.get("output_tokens", 0))

        total_input = cache_creation + cache_read + direct_input
        return TokenUsage(
            input_tokens=total_input,
            completion_tokens=completion,
            reasoning_tokens=0,  # Anthropic thinking tokens 计入 output_tokens
            total_tokens=total_input + completion,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            input_tokens_uncached=direct_input,
        )


class GeminiUsageConverter:
    """Gemini REST format (promptTokenCount, candidatesTokenCount)."""

    def convert(self, raw: dict[str, object]) -> TokenUsage:
        input_t = _int(raw.get("promptTokenCount", 0))
        completion_t = _int(raw.get("candidatesTokenCount", 0))
        reasoning_t = _int(raw.get("thoughtsTokenCount", 0))
        return TokenUsage(
            input_tokens=input_t,
            completion_tokens=completion_t,
            reasoning_tokens=reasoning_t,
            total_tokens=_int(raw.get("totalTokenCount", input_t + completion_t + reasoning_t)),
        )
```

##### `UsageConverterRegistry` — 注册与查找

```python
class UsageConverterRegistry:
    """Registry mapping api_type to UsageConverter.

    RFC-0011: 内置 converter 默认注册；用户可通过 register() 覆盖或扩展。

    Usage::

        # 注册自定义 api_type
        UsageConverterRegistry.register("my_gateway", MyGatewayUsageConverter())

        # 覆盖内置 api_type 的转化逻辑
        UsageConverterRegistry.register("anthropic_chat_completion", MyCustomAnthropicConverter())
    """

    _converters: ClassVar[dict[str, UsageConverter]] = {
        "openai_chat_completion": OpenAIChatUsageConverter(),
        "openai_responses": OpenAIResponsesUsageConverter(),
        "anthropic_chat_completion": AnthropicUsageConverter(),
        "gemini_rest": GeminiUsageConverter(),
    }

    @classmethod
    def register(cls, api_type: str, converter: UsageConverter) -> None:
        """Register or override a converter for the given api_type."""
        cls._converters[api_type] = converter

    @classmethod
    def get(cls, api_type: str) -> UsageConverter | None:
        """Look up converter by api_type. Returns None if not registered."""
        return cls._converters.get(api_type)
```

##### `_normalize_usage()` 改为 Converter 驱动

```python
from nexau.core.usage import TokenUsage


def _normalize_usage(
    usage: dict[str, object] | None,
    api_type: str | None = None,
) -> TokenUsage:
    """Normalize api_type-specific usage dict into canonical TokenUsage.

    RFC-0011: 归一化入口。优先使用 api_type 查找已注册的 UsageConverter；
    未注册时回退到通用启发式逻辑（现有行为）。

    Returns TokenUsage (never None) — 无 usage 时返回零值 TokenUsage()。
    """
    if usage is None:
        return TokenUsage()

    # 1. 尝试通过 api_type 查找注册的 converter
    if api_type is not None:
        converter = UsageConverterRegistry.get(api_type)
        if converter is not None:
            return converter.convert(usage)

    # 2. 回退：通用启发式归一化（保持现有逻辑兼容未注册的 api_type）
    return _fallback_normalize(usage)
```

**设计决策 — 为何用 Registry 而非构造函数注入**：

- `_normalize_usage()` 在 `ModelResponse.from_openai_message` 等多个 `@classmethod` 中调用，这些是静态工厂方法，无法方便地传入实例依赖；
- Registry 是模块级单例，启动时注册一次，运行期零开销查找；
- 用户可在 agent 初始化阶段调用 `UsageConverterRegistry.register()` 完成自定义，不需要修改任何框架内部代码。

**向后兼容**：`ModelResponse.usage` 类型从 `JsonDict | None` 改为 `TokenUsage`。不提供 `__getitem__` / `get` 等 dict 兼容方法，直接一次性迁移所有下游消费者改用属性访问（如 `.input_tokens`）。`ModelResponse` 是内部 API，下游调用点有限，一步到位更干净。

#### Layer 3: 事件与持久化

##### 3a. Usage 事件

新增 `UsageUpdateEvent` 到 `nexau/archs/llm/llm_aggregators/events.py`：

```python
@dataclass
class UsageUpdateEvent(Event):
    """Emitted after each LLM call with that call's token usage.

    RFC-0011: 由 AgentEventsMiddleware 在 after_model hook 中发射。
    Transport 层可将此事件转发给 UI，实现实时 token 消耗展示。
    """

    type: str = field(default="USAGE_UPDATE", init=False)
    usage: TokenUsage = field(default_factory=TokenUsage)
```

##### 3b. History 持久化：`Message.metadata["usage"]`

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

> **注意**：本 RFC 不在 `SessionManager` 上新增 `get_run_usage()` / `get_session_total_usage()` 等便利方法。usage 数据已持久化在 message metadata 中，调用方可自行遍历累加。如后续有高频查询需求，再作为独立 PR 补充。

###### 持久化流程总览

```
LLM 调用返回 ModelResponse (含 TokenUsage)
    │
    ├─→ Message.metadata["usage"] = usage.to_dict()    # 写入消息 metadata
    │       └─→ history.append(msg)
    │             └─→ HistoryList.flush()
    │                   └─→ AgentRunActionModel.append_messages  # 自动带 metadata
    │
    └─→ emit(UsageUpdateEvent(usage=usage))             # 实时事件推送
```

##### 3c. 不变更 `Agent.run()` 返回值

`Agent.run()` 继续返回 `str`，`Executor.execute()` 继续返回 `tuple[str, list[Message]]`。Usage 通过事件和 message metadata 两条路径获取：

| 获取方式 | 时机 | 用途 |
|---|---|---|
| `UsageUpdateEvent` | 每次 LLM 调用后实时推送 | UI 实时展示、transport 转发 |
| `message.metadata["usage"]` | 持久化，跨进程可查 | 遍历 assistant messages 即可累加 |

##### 3d. Executor 集成点

```python
# executor.py 伪代码

class Executor:
    def execute(self, history, agent_state, ...) -> tuple[str, list[Message]]:

        for iteration in range(max_iterations):
            # 1. LLM 调用
            response = llm_caller.call(messages, llm_config)

            # 2. 写入 assistant message metadata（随 history 自动落盘）
            #    to_ump_message() 内部已将 response.usage 写入 metadata["usage"]
            assistant_msg = response.to_ump_message()
            history.append(assistant_msg)

            # 3. 发射 UsageUpdateEvent（由 AgentEventsMiddleware.after_model 完成）
            #    middleware 从 response.usage 读取，emit(UsageUpdateEvent(usage=usage))

            # ... tool execution, history update ...

        history.flush()
        return final_response, messages
```

> **Sub-Agent**：每个 sub-agent 有自己的 Executor 和 `AgentEventsMiddleware`，会独立发射 `UsageUpdateEvent` 并将 usage 写入各自的 message metadata。不做跨 agent 的 usage 合并——如需 run 树级汇总，由调用方按 `root_run_id` 遍历所有 agent 的 messages 累加。

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

### 示例

#### 获取 run 的 usage

```python
from nexau.archs.main_sub.agent import Agent
from nexau.core.usage import TokenUsage

agent = Agent(config=config, session_manager=session_mgr)

# Agent.run() 返回 str，不携带 usage
response: str = agent.run(message="Hello")

# 从 history 的 message metadata 中累加 usage
total = TokenUsage()
for msg in agent.history:
    usage_dict = msg.metadata.get("usage")
    if usage_dict:
        total = total + TokenUsage(**usage_dict)
print(f"Total: {total.total_tokens}, Input: {total.input_tokens}, Completion: {total.completion_tokens}")
```

#### 注册自定义 UsageConverter

```python
from nexau.core.usage import TokenUsage, UsageConverter, UsageConverterRegistry


class MyGatewayUsageConverter:
    """Custom converter for internal LLM gateway."""

    def convert(self, raw: dict[str, object]) -> TokenUsage:
        # 私有网关返回 {"in": 100, "out": 50, "cached": 20}
        input_t = int(raw.get("in", 0))
        output_t = int(raw.get("out", 0))
        cached = int(raw.get("cached", 0))
        return TokenUsage(
            input_tokens=input_t,
            completion_tokens=output_t,
            total_tokens=input_t + output_t,
            cache_read_tokens=cached,
        )


# 在 agent 启动前注册
UsageConverterRegistry.register("my_gateway", MyGatewayUsageConverter())

# 之后所有 api_type="my_gateway" 的 LLM 调用自动使用此 converter
```

#### 实时 usage 事件

事件通过 `AgentEventsMiddleware` 的 `on_event` 回调发射，与现有 text/tool_call 事件机制一致：

```python
from nexau.archs.llm.llm_aggregators import Event
from nexau.archs.llm.llm_aggregators.events import UsageUpdateEvent

def handle_event(event: Event):
    match event:
        case UsageUpdateEvent():
            print(f"+{event.usage.total_tokens} tokens "
                  f"(input: {event.usage.input_tokens}, completion: {event.usage.completion_tokens})")

# on_event 回调通过 AgentEventsMiddleware 注册
middleware = AgentEventsMiddleware(session_id="sess_123", on_event=handle_event)
```

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

**未采用原因**：TypedDict 是结构化的 dict，但不支持 `__add__`、不可冻结、不可自定义方法（如 `to_dict()`）。`TokenUsage` 需要行为（累加、序列化），不仅仅是形状约束。

#### B. 用 Pydantic BaseModel

**未采用原因**：`TokenUsage` 是高频创建对象（每次 LLM 调用），Pydantic 验证开销不必要。且 codebase 中 `ModelToolCall`、`ModelResponse` 等同级数据对象全部使用 stdlib dataclass，保持一致。

### 缺点

1. **`ModelResponse.usage` 类型变更**：从 `JsonDict | None` 到 `TokenUsage` 是 breaking change。但 `ModelResponse` 是内部 API（不在 `nexau.core` 公开），影响范围可控。
2. **内存开销**：每次 LLM 调用多创建一个 `TokenUsage` 小对象。`slots=True` 已优化，且 LLM 调用频率远低于对象创建的性能瓶颈。

---

## 实现计划

### 阶段划分

- [ ] **Phase 1: 数据模型 & Converter** — `nexau/core/usage.py`（`TokenUsage`、`UsageConverter` Protocol、内置 Converter 实现、`UsageConverterRegistry`）
- [ ] **Phase 2: 归一化迁移** — `_normalize_usage()` 改为 Converter 驱动，接收 `api_type` 参数；`ModelResponse.usage` 类型改为 `TokenUsage`；一次性迁移下游消费者
- [ ] **Phase 3: 事件 & 持久化** — `UsageUpdateEvent` 定义、`AgentEventsMiddleware.after_model` 发射事件、`ModelResponse.to_ump_message()` 写入 `metadata["usage"]`（随 `HistoryList.flush()` 自动落盘，无 ORM schema 变更）

### 相关文件

- `nexau/core/usage.py` — 新增：`TokenUsage` 数据模型、`UsageConverter` Protocol、内置 Converter（OpenAI / Anthropic / Gemini）、`UsageConverterRegistry`
- `nexau/archs/main_sub/execution/model_response.py` — 修改：`_normalize_usage()` 改为 Converter 驱动；`ModelResponse.usage` 类型变更；`to_ump_message()` 写入 `metadata["usage"]`
- `nexau/archs/llm/llm_aggregators/events.py` — 新增：`UsageUpdateEvent`
- `nexau/archs/main_sub/execution/middleware/agent_events_middleware.py` — 修改：`after_model` hook 中发射 `UsageUpdateEvent`
- `nexau/archs/tracer/adapters/langfuse.py` — 修改：`_sanitize_usage()` 适配 `TokenUsage` 类型

---

## 测试方案

### 单元测试

- `TokenUsage.__add__` 累加正确性（含 cache 字段）
- `TokenUsage.to_dict()` 序列化完整性
- 各内置 Converter（`OpenAIChatUsageConverter`、`AnthropicUsageConverter`、`GeminiUsageConverter`）转化正确性
- `UsageConverterRegistry.register()` 自定义 converter 注册与覆盖
- `_normalize_usage()` 对已注册 api_type 走 converter、未注册走 fallback
- `ModelResponse.to_ump_message()` 生成的 message 含 `metadata["usage"]`

### 集成测试

- 完整 Agent.run() 后 assistant message `metadata["usage"]` 非空且合理
- `UsageUpdateEvent` 在每次 LLM 调用后发射，payload 与 message metadata 一致
- 消息级持久化：assistant message `metadata["usage"]` 随 `HistoryList.flush()` 写入 DB，reload 后 metadata 完整可读

### 手动验证

- Langfuse dashboard 中 usage 数据与 message metadata 交叉验证

---

## 未解决的问题

1. ~~**Streaming usage**~~（已确认）：OpenAI ChatCompletion 已设置 `stream_options={"include_usage": True}`（`llm_caller.py:731`），Anthropic 从 `message_start` / `message_delta` / `message_stop` 三个事件捕获 usage。两者均已正确实现，无需额外处理。

2. ~~**Responses API `include` 选项**~~（已确认，需修复）：OpenAI Responses API 的 `include` 参数当前只添加了 `"reasoning.encrypted_content"`（`llm_caller.py:878`），**缺少 `"usage"`**。需在 Phase 2 补充 `include_list.append("usage")`，否则 Responses API 的 streaming 不返回 usage。

3. **Image / audio token 计量**：Gemini 和 OpenAI 的多模态 token 计量格式不同（Gemini 报告 `cachedContentTokenCount`，OpenAI 在 `prompt_tokens_details` 中报告 `audio_tokens` / `image_tokens`）。Phase 1 暂不处理，后续按需扩展 `TokenUsage` 字段。

4. **Token budget**：本 RFC 不实现 token budget 限额功能。`TokenUsage` 数据模型和 message metadata 持久化提供了基础数据，后续可作为独立 RFC 扩展。

---

## 参考资料

- **NexAU 现有实现** `nexau/archs/main_sub/execution/model_response.py` — `_normalize_usage()`, `_coerce_usage()`
- **NexAU tracer** `nexau/archs/tracer/adapters/langfuse.py` — `_sanitize_usage()`
- **NexAU events** `nexau/archs/llm/llm_aggregators/events.py` — 现有事件体系（`BaseEvent` / union type）
- **NexAU token counter** `nexau/archs/main_sub/utils/token_counter.py` — `TokenCounter` (context window counting, orthogonal to usage accounting)
