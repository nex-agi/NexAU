# RFC-0010: LLM 聚合器事件流设计

| 字段 | 内容 |
|---|---|
| **状态** | implemented |
| **优先级** | P1 |
| **创建日期** | 2026-04-16 |
| **最后更新** | 2026-04-16 |
| **作者** | NexAU core |
| **关联 RFC** | RFC-0002（Agent 间消息事件）、RFC-0003（LLM Failover，boundary）、RFC-0004（上下文溢出/压缩，boundary）、RFC-0006（RFC 目录补全总纲） |

## 摘要

NexAU 通过一个"**聚合器 + 统一事件流**"层（`nexau/archs/llm/llm_aggregators/`）把 Anthropic / OpenAI Chat Completions / OpenAI Responses / Gemini REST 这四种**互不兼容**的流式 API 收敛到**同一个 `Event` 联合类型**上。每个后端都实现一个 `Aggregator[InputT, OutputT]` 子类：一边通过 `aggregate()` 逐 chunk 吃原生流数据、最终 `build()` 出原生消息对象供重放/持久化；一边通过注入的 `on_event` 回调**同步**吐出以 `ag_ui.core` 为底的统一事件（Text / Thinking / ToolCall / Image 各自 START → CONTENT → END 三段式生命周期，外加 Run 生命周期、传输错误等）。消费侧由 `AgentEventsMiddleware` 完成——它在 `stream_chunk` hook 里按类型分发到正确的聚合器、在 `before_agent / after_agent / after_tool` 里补发运行时级别的 RunStarted / RunFinished / RunError / ToolCallResult 事件。

## 动机

NexAU 同时要接入多家 LLM 供应商，但它们的流式协议差异巨大：

| 后端 | 流式原子 | 消息/工具 token 方式 | thinking / reasoning |
|---|---|---|---|
| Anthropic | `RawMessageStreamEvent`（message_start / content_block_start / _delta / _stop / message_delta / message_stop） | 按 `content_block.index` 区分 text / tool_use / thinking 块 | `thinking` 块独立增量 |
| OpenAI Chat Completions | `ChatCompletionChunk`（delta.content / delta.tool_calls[i].function.arguments） | `tool_calls[i]` 按索引累加，arguments 是字符串拼接式增量 | `reasoning_content`（非标准扩展字段） |
| OpenAI Responses | `ResponseStreamEvent`（response.created / output_item.added / content_part.added / text.delta / reasoning_summary.* / function_call.arguments.delta / response.done） | 按 `output_item` 语义分组，工具调用独立 item | `reasoning_item` + `reasoning_summary` 分层结构 |
| Gemini REST | 纯 JSON dict（`candidates[0].content.parts[]`） | parts 里 text / functionCall / thought 按类型辨识 | `thought: true` 标记 |

如果上层 UI / 多会话广播 / 持久化每对接一家都要写一套专用逻辑，将导致 O(后端数 × 消费者数) 的爆炸复杂度，且任何一个供应商加新能力（如 GPT-5 Responses 的 reasoning summary）都会反复冲击上层代码。聚合器层的目标就是把这一维收敛：**上层只面对统一的 `Event`，后端细节藏在聚合器内部**。

同时，运行时的**会话级**事件（Run 开始 / 结束 / 错误 / 工具调用最终结果）并非任何一条 LLM chunk 能产出的——它们需要在 agent 执行前后由中间件显式发出。因此本 RFC 还要定义：事件**来源职责**（哪类事件谁负责 emit）和**路由规则**（多后端混合、并行 choice 如何不串线）。

## 设计

### 架构总览

```
                          ┌─────────────────────────────┐
                          │     AgentEventsMiddleware   │
                          │   (main_sub/.../middleware) │
                          ├──┬──────────────────────────┤
  before_agent ───────────┤  │→ RunStartedEvent          │
  after_agent  ───────────┤  │→ RunFinishedEvent /      │
                          │  │   RunErrorEvent           │
  after_tool   ───────────┤  │→ ToolCallResultEvent      │
                          │  │                           │
  stream_chunk(raw) ──────┤  │   dispatch by type        │──┐
                          └──┴──────────────────────────┘  │
                                                           │
                                            ┌──────────────┼──────────────┐
                                            ▼              ▼              ▼
                                     Anthropic     OpenAI Chat     OpenAI Resp.   Gemini REST
                                     EventAggregator Aggregator    Aggregator     EventAggregator
                                            │              │              │              │
                                            └──────── on_event ───────────┘──────────────┘
                                                           │
                                                           ▼
                                         统一 Event 流（TextMessage* / Thinking* /
                                         ToolCall* / Image* / Run* / Compaction* / …）
```

### 核心抽象：`Aggregator[InputT, OutputT]`

定义在 `nexau/archs/llm/llm_aggregators/events.py:56`：

```python
class Aggregator[AggregatorInputT, AggregatorOutputT](ABC):
    @abstractmethod
    def aggregate(self, item: AggregatorInputT) -> None: ...
    @abstractmethod
    def build(self) -> AggregatorOutputT: ...
    @abstractmethod
    def clear(self) -> None: ...
```

三个方法分工明确：

- **`aggregate(item)`**：吃一片原生 chunk，更新内部状态，并按需通过构造期注入的 `on_event` 回调**同步**发出统一事件（延迟调度会破坏 UI 顺序，因此契约是"同步调用"）。
- **`build()`**：在 `aggregate` 全部喂完后，产出**后端原生**消息对象（`ChatCompletion` / `Response` / `Message` 等），给持久化、重放、failover middleware 使用；它**不发**任何统一事件。
- **`clear()`**：重置内部状态以便同一实例复用（`run_id` 边界、retry、新一轮 agent step）。

公开的 4 个后端实现均位于同一包下（`llm_aggregators/__init__.py`）：

| 类 | 位置 | Generic 参数（InputT → OutputT） |
|---|---|---|
| `AnthropicEventAggregator` | `anthropic/anthropic_event_aggregator.py:46` | `RawMessageStreamEvent` → `None` |
| `OpenAIChatCompletionAggregator` | `openai_chat_completion/openai_chat_completion_aggregator.py:63` | `ChatCompletionChunk` → `ChatCompletion` |
| `OpenAIResponsesAggregator` | `openai_responses/openai_responses_aggregator.py:47` | `ResponseStreamEvent` → `Response` |
| `GeminiRestEventAggregator` | `gemini_rest/gemini_rest_event_aggregator.py` | `dict[str, object]` → `None` |

Anthropic 和 Gemini REST 的 `OutputT = None` 是有意选择——它们的上游调用方（`llm_caller.py` 里的 `AnthropicStreamAggregator` / `GeminiRestStreamAggregator`）自己另行维护重放态，不需要 `build()` 的返回值。

### 统一事件模型（以 `ag_ui.core` 为底）

所有事件都继承自 `ag_ui.core.events.BaseEvent`，NexAU 在 `events.py` 里**部分复用、部分扩展**（给 `TextMessageStartEvent` / `ThinkingTextMessageStartEvent` / `RunStartedEvent` 等加 `run_id` 字段以支持 RFC-0002 的多 agent 并发），然后用一个 `Event` 联合类型（`events.py:321-350`）收口所有子类型：

| 类别 | 事件（START → CONTENT → END） | 关联字段 |
|---|---|---|
| Text | `TextMessageStartEvent` / `TextMessageContentEvent` / `TextMessageEndEvent` | `message_id`、`run_id`（仅 Start） |
| Thinking | `ThinkingTextMessageStartEvent` / `ThinkingTextMessageContentEvent` / `ThinkingTextMessageEndEvent` | `thinking_message_id`、`parent_message_id`、`run_id`（仅 Start） |
| ToolCall | `ToolCallStartEvent` / `ToolCallArgsEvent` / `ToolCallEndEvent` | `tool_call_id`、`parent_message_id`、`run_id`（仅 Start） |
| Image | `ImageMessageStartEvent` / `ImageMessageContentEvent` / `ImageMessageEndEvent` | `message_id`、`mime_type`、`run_id`（仅 Start） |
| Run 生命周期 | `RunStartedEvent` / `RunFinishedEvent` / `RunErrorEvent` | `thread_id`、`run_id`、`root_run_id`、`agent_id`、`timestamp` |
| 工具结果 | `ToolCallResultEvent` | `tool_call_id`、`content`（JSON 字符串） |
| 压缩（**属 RFC-0004**） | `CompactionStartedEvent` / `CompactionFinishedEvent` | `phase ∈ {before_model, after_model, wrap_model_call}`、`mode ∈ {regular, emergency}` |
| 传输错误 | `TransportErrorEvent` | `message`、`timestamp` |
| 团队消息（**属 RFC-0002**） | `UserMessageEvent` / `TeamMessageEvent` | `content`、`to_agent_id`、`from_agent_id` |

**START → CONTENT → END 契约**：
1. 同一"消息/工具调用/图片"的所有事件共享同一个 ID（`message_id` / `tool_call_id` / `thinking_message_id`）。
2. Start 必须**先于**该 ID 的任何 Content；End 必须**晚于**该 ID 的所有 Content。
3. 只有 Start 事件携带 `run_id`（减少冗余），Content/End 通过 ID 关联回 Start。
4. 一个后端可以交错发出多条消息的事件（Anthropic 并发 content_block、Responses 并发 output_item），但**每条 ID 内部的顺序严格满足 (1)(2)(3)**。

### 聚合器实现套路

以 `AnthropicEventAggregator`（`anthropic_event_aggregator.py:46` 起）为例，四个后端共用同一套骨架：

1. **构造** `(on_event: Callable[[Event], None], run_id: str)`（关键字参数）。
2. **`aggregate(item)` 用 `match/case` 分派** 到不同原生事件类型。
3. **按"块索引/ID"维护状态字典**（Anthropic 用 `content_block.index`，Responses 用 `output_item.id`，Chat Completions 用 `tool_calls[i].index`），以区分同一流中交错的多个块。
4. **ID 生成策略因类型而异**：文本用后端提供的 `message.id`；工具调用用 block.id；thinking 由于 Anthropic 不给独立 ID，统一用 `uuid4` 生成 `thinking_message_id`。
5. **`on_event` 同步发事件**——禁止投入线程池或 async；UI 对事件顺序敏感。
6. **`clear()` 清空所有 per-index 状态**；`build()`（若实现）根据累计 state 重新组装完整消息对象。

子聚合器嵌套是允许的——`OpenAIResponsesAggregator` 内部为 `_ReasoningItemAggregator` / `_FunctionCallItemAggregator` / `_MessageItemAggregator` 各建一个子 `Aggregator[..., ...]`（`openai_responses_aggregator.py:256 / 393 / 553`），每个负责一种 output_item 的增量解析，这让单个巨大 Responses 事件流的状态机变得可局部推理。

### 消费侧：`AgentEventsMiddleware`

位于 `nexau/archs/main_sub/execution/middleware/agent_events_middleware.py:77`，是事件流的**唯一官方消费入口**。它实现中间件协议（`Middleware` ABC，详见 RFC-0011），关键职责：

1. **注入 `on_event`**：构造时接一个外部 `on_event: Callable[[Event], None]`（默认 `_noop_event_handler`，`agent_events_middleware.py:58`），该回调会被下述所有聚合器共享。

2. **懒启动聚合器**（`agent_events_middleware.py:121-151`）：
   ```python
   def openai_responses_aggregator(self, *, run_id: str) -> OpenAIResponsesAggregator:
       if self._openai_responses_aggregator is None:
           self._openai_responses_aggregator = OpenAIResponsesAggregator(
               on_event=self.on_event, run_id=run_id,
           )
       return self._openai_responses_aggregator
   ```
   Anthropic / Gemini / Responses 各自**单例懒启动**；Chat Completions 例外——用 `dict[str, OpenAIChatCompletionAggregator]` 按 `chunk.id` 一流一实例（`agent_events_middleware.py:113, 274`），以防并行 `n>1` choices 的 chunks 串线。

3. **`stream_chunk` 按类型分发**（`agent_events_middleware.py:240-303`）——用三条类型守卫函数快速判别来源：
   - `isinstance(chunk, ChatCompletionChunk)` → Chat Completions
   - `is_openai_responses_event(chunk)`（`agent_events_middleware.py:67-69`，靠 `chunk.__class__.__module__` 前缀） → Responses
   - `is_anthropic_event(chunk)`（`agent_events_middleware.py:63-64`） → Anthropic
   - `is_gemini_rest_chunk(chunk)`（`agent_events_middleware.py:72-74`，鸭子判断 `"candidates" in chunk`） → Gemini REST

4. **流边界清理**：
   - Anthropic：遇到 `message_start` 前 `clear()`（`agent_events_middleware.py:295-296`）
   - Responses：遇到 `response.created` 前 `clear()`（`agent_events_middleware.py:285-286`）
   - Gemini：按 `run_id` 变化 `clear()`（`agent_events_middleware.py:138-141`）
   - Chat Completions：一流一实例，无需 clear，流结束即可抛弃

5. **运行时事件补发**——这些事件**不来自任何 LLM chunk**，由中间件的三个 hook 直接 emit：
   - `before_agent` → `RunStartedEvent(thread_id, root_run_id, run_id, agent_id, timestamp)`（`agent_events_middleware.py:167-175`）
   - `after_agent` → 若 `stop_reason ∈ {ERROR_OCCURRED, CONTEXT_TOKEN_LIMIT}` 发 `RunErrorEvent`，否则发 `RunFinishedEvent(thread_id, run_id, result, timestamp)`（`agent_events_middleware.py:192-211`）
   - `after_tool` → `ToolCallResultEvent(tool_call_id, content=json.dumps(tool_output), timestamp)`（`agent_events_middleware.py:229-236`）

### `run_id` 与多 agent

`run_id` 是串联多级 agent 并发的关键（RFC-0002）。约定：

- **只有 Start 系列事件携带 `run_id`**（TextMessageStartEvent / ThinkingTextMessageStartEvent / ToolCallStartEvent / ImageMessageStartEvent / RunStartedEvent）；Content/End 通过 ID 关联回 Start，避免冗余。
- **一个 `AgentEventsMiddleware` 实例对应一个 session**（由 `session_id` 字段标识，`agent_events_middleware.py:111`），但**可为多个子 agent run 服务**——它在每次 hook / stream_chunk 里从 `agent_state.run_id` 动态取 run_id 注入聚合器。
- **子 agent 层级**：`RunStartedEvent` 额外带 `root_run_id` 和 `agent_id`，UI 端据此重建树形结构。

### 与其他 RFC 的边界

| RFC | 关系 | 具体边界 |
|---|---|---|
| RFC-0002（Agent Team） | **引用** | `UserMessageEvent` / `TeamMessageEvent` 类型定义在本 RFC 的 `events.py:286-315`，但**语义所有权属 RFC-0002**；本 RFC 只负责把它们纳入统一 `Event` 联合。 |
| RFC-0003（LLM Failover） | **不重叠** | Failover middleware 负责"请求失败 → 切换模型"；本 RFC 负责"chunk → event"。两者在 middleware 链上并列、互不调用。Failover 失败时发 `TransportErrorEvent`。 |
| RFC-0004（上下文溢出/压缩） | **引用** | `CompactionStartedEvent` / `CompactionFinishedEvent` 类型定义在本 RFC 的 `events.py:240-267`，但**emission 时机与 phase/mode 语义属 RFC-0004**；本 RFC 只承诺"能承载该事件"。 |
| RFC-0006 | **派生** | 本 RFC 是 RFC-0006 master plan 的 T4 子任务。 |

特别澄清：`llm_caller.py` 里另有 4 个同名不同姓的类——`OpenAIChatStreamAggregator`（`llm_caller.py:1288`）、`AnthropicStreamAggregator`（`llm_caller.py:1404`）、`OpenAIResponsesStreamAggregator`（`llm_caller.py:1676`）、`GeminiRestStreamAggregator`（`llm_caller.py:1844`）——它们是**内部 chunk → dict 重放助手**，不发事件、不在本 RFC 架构内。命名相似是历史包袱，**本 RFC 语境下的"聚合器"一律指 `llm_aggregators/` 包下的 `*EventAggregator` / `*Aggregator`**。

### 类型安全纪律

`llm_aggregators/CLAUDE.md` 明确禁止（enforced by ruff + pyright）：

- `# type: ignore` 注释（例外：`ag_ui.core.events` 的 Literal 字段赋值受 pydantic 变异性限制，只能 `# type: ignore[assignment]`，个别类已在源码标注）
- `Any` 类型
- `getattr` / `hasattr` 动态属性访问
- 字符串字面量类型比较（要用 `match/case` 配合 Literal 进行穷尽匹配）

新增后端必须同时通过 `pyright --strict` 与 ruff lint；违反上面四条任一的 PR 直接拒绝。

## 权衡取舍

1. **为什么用 ABC 而非 Protocol？**
   - 选择：显式继承 `Aggregator[InputT, OutputT]`。
   - 替代：`typing.Protocol`（结构子类型），允许非继承的后端就地实现三个方法。
   - 代价 vs 收益：Protocol 的好处是解耦（aggregators 不必依赖 ABC），但本项目三方聚合器可能性极低；而 ABC 提供`@abstractmethod` 的**运行时保证**（少实现一个方法直接抛 `TypeError: Can't instantiate abstract class`），配合 pyright 双重把关更稳。

2. **`on_event` 为什么同步？**
   - 选择：必须同步调用，禁止 `asyncio.create_task` / 线程池。
   - 替代：事件入队列异步消费。
   - 代价 vs 收益：同步保证 UI 看到的 chunk 顺序 = 底层 API 流顺序，无需额外排序逻辑；代价是 `on_event` 里做重活会阻塞 chunk 处理——由调用方自己负责把重活异步出去（典型做法：`on_event` 里只 put 到 asyncio.Queue，真正 IO 在别的协程消费）。

3. **为什么按 `chunk.id` 为 Chat Completions 建实例，其余后端单例？**
   - Chat Completions 在 `n>1` 或并行 tool call 时**同一请求内就会有多个 chunk.id**，如果共享一个 aggregator，不同 choice 的 `delta.content` 会串到一起。
   - 其余三个后端的原生协议本身就**每轮 LLM 请求只有一条流**（Anthropic 单消息、Responses 单 response、Gemini REST 单会话轮），`clear()` 时机明确、单例安全。

4. **Anthropic / Gemini 的 `OutputT = None` 会不会丢失重放能力？**
   - 丢不了——`llm_caller.py` 里另有专门的 `*StreamAggregator` 做重放持久化。拆成两套的代价是"两处要同时消费同一个 chunk"，收益是"事件职责"和"重放职责"解耦：未来要改 UI 事件协议不会波及重放格式，反之亦然。

5. **`run_id` 为什么只放 Start 事件？**
   - 替代：每个事件都带 `run_id`。
   - 收益：体积更小、减少 UI 层校验冗余。
   - 代价：UI 必须维护 "message_id → run_id" 映射表。现代 Web UI 本就用映射表管状态（redux / Zustand / Pinia），几乎零成本。

6. **为什么 `RunStartedEvent` / `RunFinishedEvent` / `ToolCallResultEvent` 由中间件发，而不是让聚合器发？**
   - 这些事件依赖 `agent_state.run_id / agent_id / root_run_id / session_id` 等**上下文**，聚合器只有 LLM chunk，没有这些信息。
   - 强制"谁有上下文谁发"的分工，避免把 middleware 层的状态下沉到 aggregator 层污染抽象。

## 实现计划

**已实现**（随 NexAU v0.3.x 落地，commit 历史见 `git log -- nexau/archs/llm/llm_aggregators/`）：

| # | 里程碑 | 位置 |
|---|---|---|
| M1 | Aggregator ABC + Event 联合类型 + `ag_ui.core` 复用 | `events.py` |
| M2 | Anthropic / OpenAI Chat Completions 聚合器 | 对应子包 |
| M3 | OpenAI Responses 聚合器（含 reasoning / thinking / 子聚合器嵌套） | `openai_responses/` |
| M4 | Gemini REST 聚合器 | `gemini_rest/` |
| M5 | `AgentEventsMiddleware`：stream_chunk 分发 + 三 hook 运行时事件 | `main_sub/execution/middleware/agent_events_middleware.py` |
| M6 | CompactionStarted/Finished 事件纳入（实际 emission 由 RFC-0004 驱动） | `events.py:240-267` |
| M7 | RFC-0002 的 UserMessageEvent / TeamMessageEvent 纳入统一 Event 联合 | `events.py:286-315, 321-350` |

**后续演进的触发点**（非本 RFC 交付）：

- 新增后端（如 Vertex AI、Bedrock）→ 新建 `<backend>/<backend>_event_aggregator.py` + 在 `AgentEventsMiddleware.stream_chunk` 增加守卫分支。
- 新增事件类型 → 在 `events.py` 追加类 + 加入 `Event` 联合 + 按需为 `AgentEventsMiddleware` 增加发射点。
- `ag_ui.core` 版本升级 → 关注 `BaseEvent` / Literal 字段的变异性变化，必要时更新 `# type: ignore[assignment]` 注释和 pydantic 版本约束。

## 测试方案

聚合器层的测试聚焦"chunk → event 序列"的契约，已实现形式：

1. **单聚合器单元测试**（`tests/archs/llm/llm_aggregators/<backend>/`）：
   - 用录制好的原生 chunk fixture（真实抓包或 mock），驱动 `aggregate()`。
   - 断言 `on_event` 捕获的事件序列**严格符合 START → CONTENT → END 契约**：同 ID 顺序 + 跨 ID 可交错。
   - 断言 `build()` 返回的原生消息对象与"同一流 non-stream 请求"的结果等价。
   - 覆盖 thinking / reasoning / 多工具调用 / image / 异常截断等分支。

2. **中间件集成测试**（`tests/archs/main_sub/execution/middleware/`）：
   - 模拟一个 agent run 全流程，验证 `before_agent / stream_chunk / after_tool / after_agent` 四处 hook 的事件发射次数、类型、run_id 正确。
   - 覆盖**同一 session 多 run**，确认聚合器按 run_id 正确 clear。

3. **跨后端等价性测试**：
   - 同一段 prompt 在 4 个后端分别跑一次（小模型 + 确定性参数），断言各自 `Event` 序列在"语义层"等价——具体做法是把事件序列 normalize 成 `[(type, message_id, delta?)]` 列表，去除后端特异 ID 后比较。该类测试标记为 `@pytest.mark.live`，CI 默认跳过，只在发版前手动跑。

4. **类型安全门禁**：
   - `pyright --strict` 必须对 `nexau/archs/llm/llm_aggregators/` 和 `AgentEventsMiddleware` 零错误。
   - ruff 规则覆盖 `CLAUDE.md` 里列出的 4 条禁令（`# type: ignore` / `Any` / 动态属性 / 字符串字面量比较）。

## 未解决的问题

1. **部分失败时的状态一致性**：聚合器 `aggregate()` 中途抛异常后，内部 state 可能处于半构建态。当前做法是**依赖外部 `clear()`**（`AgentEventsMiddleware` 在流边界主动 clear），但如果异常发生在单轮流中间且上层未触发 `clear()`，复用该实例会产出损坏事件。是否要在 ABC 层引入"进入损坏态后拒绝继续 aggregate"的机制（类似游标的 `failed` 标志）待定。

2. **Event 联合的开放封闭困境**：`Event = A | B | ...` 是**闭合**的 union，消费方可以用 `match/case` 穷尽；但每增加一个事件类型就要改 `events.py` 的 union 和所有下游 `match`。用 Protocol / 子类型树能开放扩展但丢失穷尽性保障。当前取穷尽性；长期若插件式事件增多，可能要重新评估。

3. **`CompactionStartedEvent` 的发射点归属**：类型定义在本包，但 emission 跨 `before_model / after_model / wrap_model_call` 三个时机，分散在 RFC-0004 的压缩实现里。若未来压缩策略变动（比如并行 summarize），事件 schema 可能被动变化——RFC-0004 与本 RFC 的边界要保持清晰的 schema owner：**字段结构在本 RFC，emission 语义在 RFC-0004**。

4. **UI 端的事件版本化**：`Event` 联合目前没有显式版本号。跨前后端部署时若事件类型 schema 变动（比如新增字段），老版 UI 可能因 pydantic 严格校验拒绝事件。是否需要引入 `schema_version: int` 字段、是否需要 "忽略未知字段" 模式待定（当前 `ag_ui.core.BaseEvent` 使用 pydantic，默认 forbid extra 与否由上游决定）。

## 参考资料

- 代码入口：`nexau/archs/llm/llm_aggregators/` （`__init__.py` 导出清单 + 各后端子包）
- 统一事件与 ABC：`nexau/archs/llm/llm_aggregators/events.py`
- 聚合器实现 CLAUDE 指南：`nexau/archs/llm/llm_aggregators/CLAUDE.md`
- 消费侧中间件：`nexau/archs/main_sub/execution/middleware/agent_events_middleware.py`
- 历史上游参考：[ag-ui protocol](https://github.com/ag-ui-protocol/ag-ui)（`ag_ui.core.events.BaseEvent` 提供基础字段与 START/CONTENT/END 约定）
- 相关 RFC：
  - RFC-0002: Agent Team — UserMessageEvent / TeamMessageEvent 语义
  - RFC-0003: LLM Failover Middleware — 中间件链上另一侧
  - RFC-0004: 上下文溢出与紧急压缩 — CompactionStarted/Finished 的 emission 语义
  - RFC-0006: RFC 目录补全总纲 — 本 RFC 的上级任务
