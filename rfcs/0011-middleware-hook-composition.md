# RFC-0011: 中间件钩子组合与语义

| 字段 | 内容 |
|---|---|
| **状态** | implemented |
| **优先级** | P1 |
| **创建日期** | 2026-04-16 |
| **最后更新** | 2026-04-16 |
| **作者** | NexAU core |
| **关联 RFC** | RFC-0002（Team — TeamMessageMiddleware）、RFC-0003（LLMFailoverMiddleware）、RFC-0004（ContextCompactionMiddleware）、RFC-0006（RFC 目录补全总纲）、RFC-0007（工具系统 — wrap_tool_call & before/after_tool 语义）、RFC-0010（LLM 聚合器事件流 — stream_chunk + AgentEventsMiddleware） |

## 摘要

NexAU 用一个**单一 `Middleware` 基类 + `MiddlewareManager` 编排器**统一了 agent 执行管线上的所有"切面"：从 agent 启动 / 终止、每轮 LLM 调用前后、每次工具调用前后、围绕模型调用的"环绕式"重试，到流式 chunk 的旁路观察。`Middleware`（`nexau/archs/main_sub/execution/hooks.py:254`）暴露 9 个可覆盖的钩子方法，缺省全部 no-op；`MiddlewareManager`（`hooks.py:493`）按**严格定义的顺序**驱动它们：`before_*` 类钩子按注册顺序**正向**迭代、`after_*` 类钩子**反向**迭代、`wrap_model_call` / `wrap_tool_call` 用递归 `call_next` 形成**洋葱式**包裹、`stream_chunk` 正向迭代且任一中间件返回 `None` 即丢弃 chunk。所有钩子都返回统一的 `HookResult`（`hooks.py:95`）数据类，其字段（`messages` / `parsed_response` / `tool_input` / `tool_output` / `agent_response` / `force_continue`）描述对管线状态的**增量修改**——上一个中间件的产出会被合入 `hook_input`，再传给下一个，从而形成**值传递的链式累积**。这套设计让 RFC-0002 的团队消息注入、RFC-0003 的 LLM failover、RFC-0004 的上下文压缩、RFC-0010 的统一事件发射等"看似无关"的能力都能以零改 executor 的方式自由组合。

## 动机

AI agent 框架不可避免地需要"在执行某关键点上插入逻辑"——日志、计费、注入提示、压缩历史、做工具结果脱敏、捕获事件给前端、失败重试 / 切模型……。这类需求的反模式是直接改 executor 主循环：每加一个能力就动核心代码、几个能力之间互相耦合、新写的特性常常忘了照顾既有特性。

NexAU 选择了**中间件抽象**统一应对：

- 所有切面都派生自同一个 `Middleware` 基类，**接口收敛**。
- 编排顺序由 `MiddlewareManager` 集中管理，**executor 主循环只调用 manager**，不感知具体中间件。
- 一类钩子 + 一种结果对象（`HookResult`）让"中间件输出 → 下一个中间件输入"的链路**类型可推断**，不需要每对邻居单独约定数据格式。
- 不同钩子位置对应不同的"管线时机"：lifecycle / per-iteration model call / per-tool-call / around model call / streaming chunk——同一中间件可以在多个时机挂钩（典型如 `AgentEventsMiddleware` 同时挂 `before_agent / after_agent / after_tool / stream_chunk`）。

## 设计

### 钩子矩阵

`Middleware` 暴露 9 个钩子（`hooks.py:254-288`），按**触发时机**分为 5 组：

| 组 | 钩子 | 触发时机 | 输入 dataclass | 顺序 | 可改的 `HookResult` 字段 |
|---|---|---|---|---|---|
| Agent 生命周期 | `before_agent(hook_input)` | run loop 开始之前 | `BeforeAgentHookInput`（agent_state, messages） | 正序 | `messages` |
| Agent 生命周期 | `after_agent(hook_input)` | run loop 结束之后（含错误终止） | `AfterAgentHookInput`（+ agent_response, stop_reason） | 反序 | `agent_response`, `messages` |
| 每轮 model call | `before_model(hook_input)` | 每次 LLM 调用前（agent 可能有多轮） | `BeforeModelHookInput`（agent_state, messages, max_iterations, current_iteration） | 正序 | `messages` |
| 每轮 model call | `after_model(hook_input)` | LLM 返回并解析后 | `AfterModelHookInput`（+ original_response, parsed_response, model_response） | 反序 | `parsed_response`, `messages`, `force_continue` |
| 每次 tool call | `before_tool(hook_input)` | 工具实际执行之前 | `BeforeToolHookInput`（agent_state, sandbox, tool_name, tool_call_id, tool_input） | 正序 | `tool_input` |
| 每次 tool call | `after_tool(hook_input)` | 工具返回结果之后 | `AfterToolHookInput`（+ tool_output） | 反序 | `tool_output` |
| 环绕式 model call | `wrap_model_call(params, call_next)` | 围绕单次 LLM 请求 | `ModelCallParams`（messages, max_tokens, ..., shutdown_event） | 洋葱（middleware[0] 最外层） | 整个 `ModelResponse` |
| 环绕式 tool call | `wrap_tool_call(params, call_next)` | 围绕单次工具执行 | `ToolCallParams`（agent_state, sandbox, tool_name, parameters, ...） | 洋葱 | 整个 tool 返回值 |
| 流式 chunk | `stream_chunk(chunk, params)` | 收到 LLM 流式 chunk 时 | 原生 chunk + `ModelCallParams` | 正序，返回 `None` 则丢弃 | 替换或丢弃 chunk |

所有钩子的返回值都是 `HookResult` 或其子类（`BeforeModelHookResult` / `AfterModelHookResult` / `AfterToolHookResult`，`hooks.py:152-202`，向后兼容旧调用方式），缺省返回 `HookResult.no_changes()`（`hooks.py:127-129`）。环绕式 `wrap_*` 是个例外——它们不返回 `HookResult`，而是返回真正的 `ModelResponse | None` / `Any`，因为它们的契约是"调用 `call_next` 拿到内层结果，然后返回（可能修改后的）该结果"。

### `HookResult`：值传递的增量修改

`HookResult`（`hooks.py:94-149`）是一个 `@dataclass`，所有字段缺省为 `None` / `False`：

```python
@dataclass
class HookResult:
    messages: list[Message] | None = None
    parsed_response: ParsedResponse | None = None
    force_continue: bool = False
    tool_output: Any | None = None
    tool_input: dict[str, Any] | None = None
    agent_response: str | None = None
```

**核心约定**：`None` 字段表示"我不改这一项"，非 `None` 字段表示"用我这个值替换 `hook_input` 里的同名字段"。`force_continue` 是布尔短路开关，触发 after_model 后跳过本轮的 tool/sub-agent 调用、直接进入下一轮 LLM。

`HookResult.no_changes()` 类方法（`hooks.py:127-129`）等价于"什么都不改"；`HookResult.with_modifications(...)`（`hooks.py:131-149`）是 named-arg 工厂，避免出现 7 个位置参数的调用。

### `MiddlewareManager`：统一编排器

`MiddlewareManager`（`hooks.py:493-704`）是核心 executor 路径与中间件之间的唯一接口。它被三个文件共 9 处显式调用——分布如下：`executor.py` 4 处（agent / model 生命周期）、`tool_executor.py` 3 处（tool 调用前 / 围绕 / 之后）、`llm_caller.py` 2 处（model wrap + stream chunk）：

| 调用位置 | manager 方法 | 关键行为 |
|---|---|---|
| `executor.py:419` | `run_before_agent` | 正序遍历 `middlewares`；每次把上一步产出的 `messages` 写回 `hook_input` 再传下家；异常仅记录 warning 不中断（`hooks.py:528-530`） |
| `executor.py:806` | `run_after_agent` | **反序**遍历；同时合并 `agent_response` 与 `messages` 修改（`hooks.py:541-552`） |
| `executor.py:503` | `run_before_model` | 正序；本地变量 `current_messages` 累积修改，避免污染 `hook_input` 的引用 |
| `executor.py:888` | `run_after_model` | 反序；除 messages/parsed_response 外，任何中间件设置 `force_continue=True` 即整体 `force_continue=True`（`hooks.py:598-599`） |
| `tool_executor.py:109` | `run_before_tool` | 正序；累积 `tool_input` 修改 |
| `tool_executor.py:231` | `wrap_tool_call(params, call_next)` | 与 wrap_model_call 同构的洋葱递归（`hooks.py:683-698`） |
| `tool_executor.py:280` | `run_after_tool(hook_input, initial_output)` | 反序；累积 `tool_output` 修改 |
| `llm_caller.py:319` | `wrap_model_call(params, call_next)` | 递归实现洋葱：`invoke(0, params)` → middleware[0].wrap_model_call(params, lambda p: invoke(1, p)) → … → `call_next(params)`；middleware[0] 是最外层（`hooks.py:640-655`） |
| `llm_caller.py:1239` | `stream_chunk(chunk, params)` | 正序；任一中间件返回 `None` 立刻 `return None` 丢弃（`hooks.py:667-672`）；返回新对象记 info 日志 |

#### 为什么 before 正序、after 反序？

这是经典 onion-skin（洋葱皮）模型的延伸：
- middleware[0] 在 `before_*` 链最先执行（最外层入），在 `after_*` 链最后执行（最外层出）——它**完整包裹**了所有更内层的中间件视野。
- 与 `wrap_*` 的 onion 顺序保持一致：写一个 logging middleware，把它放到链首，它的 before / after / wrap 三个钩子都自然地构成最外圈。

#### 为什么 wrap_* 用递归而非 for 循环？

环绕式语义需要"调用 next 之后还要做事"——典型场景：retry / failover / context compaction（先看 next 调用是否抛 `ContextOverflowError`，是则压缩历史再次 `call_next`）。for 循环做不到"中间件可以决定是否调用 next、调用几次"，递归 `call_next` 闭包是最直接的表达。

#### 异常隔离

`run_*` 钩子的循环里每个中间件都包了 `try/except`，**异常只记 warning、不中断后续中间件**（`hooks.py:528 / 553 / 572 / 600 / 616 / 636 / 679`）。这是有意的——一个观察类中间件（如 logging / event emitter）出问题不应让 agent 跑不下去。但 `wrap_model_call` / `wrap_tool_call` 没有这层保护：它们的异常会沿着 `call_next` 栈向上抛，因为环绕式中间件本来就承担**真实业务责任**（重试、压缩、超时），一旦失败就该让上层感知。

### `FunctionMiddleware`：旧接口兼容层

`FunctionMiddleware`（`hooks.py:291-339`）是个适配器，把"裸的钩子函数"包成 `Middleware` 实例。Executor 在 `_build_middleware_manager`（`executor.py:812-856`）里把外部传入的 `before_model_hooks / after_model_hooks / before_tool_hooks / after_tool_hooks` 列表逐一转成 `FunctionMiddleware`，追加到 `configured_middlewares` 之后构成最终列表，然后实例化 `MiddlewareManager`。

> **顺序结果**：显式 `Middleware` 实例先于 hook-style 注册的 `FunctionMiddleware`。before_* 时它们更早执行（外层），after_* 时它们更晚执行（外层退出）——保持 onion 一致性。

### 内置中间件清单

NexAU 在 `nexau/archs/main_sub/execution/middleware/` 与 `nexau/archs/main_sub/team/middleware/` 下提供 6 个生产级中间件（按"挂载的钩子"列出）：

| 类 | 位置 | 挂载钩子 | 主要职责 | 归属 RFC |
|---|---|---|---|---|
| `AgentEventsMiddleware` | `execution/middleware/agent_events_middleware.py:77` | before_agent / after_agent / after_tool / stream_chunk | LLM chunk → 统一 Event；agent 生命周期 / 工具结果事件 | RFC-0010 |
| `LLMFailoverMiddleware` | `execution/middleware/llm_failover.py:171` | wrap_model_call | 触发器命中 → 回退到备用 provider；带熔断器 | RFC-0003 |
| `LongToolOutputMiddleware` | `execution/middleware/long_tool_output.py:80` | after_tool | 截断超长 tool output，避免污染上下文 | （本 RFC 内嵌） |
| `RoundAndTokenReminderMiddleware` | `execution/middleware/round_and_token_reminder.py:29` | before_model | 注入"已用 N 轮 / M tokens"提醒系统消息 | （本 RFC 内嵌） |
| `ContextCompactionMiddleware` | `execution/middleware/context_compaction/middleware.py:58` | before_model + wrap_model_call + after_model | 阈值检测 → 主动压缩 / 异常驱动紧急压缩 | RFC-0004 |
| `TeamMessageMiddleware` | `team/middleware/team_message_middleware.py:41` | before_model | 把团队/用户消息总线里的新消息注入下一轮 LLM 输入 | RFC-0002 |

这一清单**没有约定优先级（注册顺序）**，由调用方 / yaml 配置决定。但有几条经验法则（详见"权衡取舍"）：

1. `LLMFailoverMiddleware` 应位于链尾（最内层 wrap_model_call），避免 retry 把 logging / events 重发多次；否则需要在 emit 端自己去重。
2. `AgentEventsMiddleware` 应位于链首（最外层），让它能看到所有内层中间件**最终修改后**的事件流（after_tool 反序 → 它最后看到，恰好对应"最终" tool_output）。
3. `ContextCompactionMiddleware` 的 `wrap_model_call` 必须在 `LLMFailoverMiddleware` 之外（更外层），否则压缩后的 messages 会被 failover 的内层重试流忽略。

### 与其他 RFC 的边界

| RFC | 关系 | 边界细节 |
|---|---|---|
| RFC-0002（Agent Team） | **使用** | `TeamMessageMiddleware` 是 RFC-0002 的载体；team bus / message routing 语义在 RFC-0002 内定义。本 RFC 只承诺"before_model 钩子能用来注入 messages"。 |
| RFC-0003（LLM Failover） | **使用** | `LLMFailoverMiddleware` 用 `wrap_model_call` 实现 retry/fallback；触发条件、熔断器策略在 RFC-0003 内定义。本 RFC 只承诺"wrap_model_call 是洋葱递归"。 |
| RFC-0004（Context Compaction） | **使用** | `ContextCompactionMiddleware` 三钩齐挂：`before_model`（轮间检测）、`wrap_model_call`（捕 context overflow 异常 → 压缩 → 再次 call_next）、`after_model`（压缩 metadata 写回）。算法在 RFC-0004 内定义。 |
| RFC-0007（Tool System） | **使用** | `before_tool / after_tool / wrap_tool_call` 钩子针对工具调用。工具定义、绑定、调度在 RFC-0007 内定义。本 RFC 只承诺"工具执行前后 / 围绕，三个钩子可用"。 |
| RFC-0010（LLM Aggregator Event Stream） | **使用** | `AgentEventsMiddleware.stream_chunk` 是事件聚合的唯一官方入口；事件 schema、聚合器协议在 RFC-0010 内定义。本 RFC 只承诺"stream_chunk 钩子能拿到原生 chunk + ModelCallParams"。 |
| RFC-0006 | **派生** | T5 子任务。 |

## 权衡取舍

1. **为什么用类继承而非 Protocol / 装饰器注册？**
   - 选择：`class MyMiddleware(Middleware): def before_model(...)` 显式覆盖。
   - 替代：`@register_middleware("before_model") def my_hook(...)` 装饰器风格。
   - 收益：一个中间件可以同时挂多个钩子时，**状态共享**很自然（实例字段）；装饰器模式要么手动用闭包/单例、要么强制每个钩子无状态。
   - 代价：必须知道 `Middleware` 基类的 9 个钩子签名；无 IDE 自动补全的话有学习曲线。
   - 折中：`FunctionMiddleware` 适配器允许 hook-style 用法，照顾遗留代码。

2. **为什么 `HookResult` 用 `None` 表示"不改"而非真正的 sentinel？**
   - 选择：`messages=None` 表示"沿用上一步的 messages"。
   - 替代：定义 `_UNCHANGED` 哨兵对象。
   - 收益：`None` 在 Python 类型注解里是一等公民，pyright/mypy 推断 `messages: list[Message] | None` 完全自然。
   - 代价：如果未来某字段的合法值包含 `None` 本身（比如"显式清空 messages"），需要换语义。当前所有字段都没有这个问题。

3. **为什么 `after_*` 反序？为什么 `wrap_*` 也按 middleware[0] 是最外层？**
   - 这是 koa.js / express middleware / asgi middleware 的标准 onion 模型。统一规则的好处：写中间件时"我在链首 / 链尾"的直觉与 before / after / wrap 完全一致——不需要为不同钩子记忆不同顺序。
   - 反例：如果 after_* 也正序，链首的中间件会"先 before、先 after"——它的视野就被切成两半，`__init__` 里维护的状态在 before 里设置、却被链尾的 after 提前消费，违反直觉。

4. **`stream_chunk` 为什么允许丢弃 chunk？**
   - 选择：返回 `None` 即停止 chain、整个 chunk 不再传给 aggregator。
   - 用例：`LongToolOutputMiddleware` 之外的"流式过滤"——比如未来加一个 PII 脱敏 middleware，可能会丢弃含敏感数据的 chunk。
   - 代价：丢 chunk 会破坏 LLM 流的完整性（aggregator 重建消息时少一段），调用方要小心；当前没有内置中间件真的这么做。

5. **为什么不集中提供"中间件注册顺序"的约束？**
   - 选择：完全交给调用方配置。
   - 替代：每个中间件声明 `priority: int` 或 `must_be_after: list[Type]`。
   - 代价：用户可能配错（典型：把 failover 放到 events 之外，导致事件被重发）。
   - 收益：核心保持简单、灵活；约束以"最佳实践文档 / yaml 校验器"形式落到外围（待办）。

6. **`run_*` 默默吞异常 vs `wrap_*` 抛异常 — 是不是不一致？**
   - 是有意为之的不一致。`run_*` 钩子里典型场景是观察 / 记录 / 提醒——一个 logger 挂掉不应让 agent 死掉。`wrap_*` 是真正的业务责任——retry 失败、压缩失败、网络挂了，**必须**让 executor 知道。

7. **没提供 async 钩子，会不会成为瓶颈？**
   - 当前所有钩子都是同步签名，因为 executor 主循环本身是同步的（与 LLM 客户端 / 工具执行的多线程协作）。需要 IO 的中间件可以用 `concurrent.futures.ThreadPoolExecutor` 自己处理。
   - 未来引入 asyncio 主循环时，可能需要并存 sync/async 两套钩子接口（类似 Django 的 sync/async 中间件）。

## 实现计划

**已实现**（位于 `nexau/archs/main_sub/execution/`）：

| # | 里程碑 | 位置 |
|---|---|---|
| M1 | `Middleware` 基类 + 6 个 dataclass 输入 + `HookResult` + `*HookResult` 子类 | `hooks.py:42-202` |
| M2 | `Middleware` 基类 9 个 no-op 钩子方法 | `hooks.py:254-288` |
| M3 | `FunctionMiddleware` 适配器 | `hooks.py:291-339` |
| M4 | `LoggingMiddleware`（标准库自带的观察实现） | `hooks.py:342-474` |
| M5 | `MiddlewareManager` 编排器（9 个 `run_* / wrap_* / stream_chunk` 方法） | `hooks.py:493-704` |
| M6 | Executor 集成：`_build_middleware_manager` + 9 个调用点 | `executor.py:143, 419, 503, 806, 888, 812-856, etc.` |
| M7 | 6 个内置生产中间件 | 见上节"内置中间件清单" |

**后续演进**（非本 RFC 交付）：

- yaml 配置层加"中间件链校验"——比如检测 `LLMFailover` 在 `AgentEvents` 之内时给 warning。
- `wrap_tool_call` 当前没有内置中间件使用它（外部能扩展，但官方没示范）；考虑加一个 `ToolTimeoutMiddleware` 演示。
- 引入 async 钩子 protocol，照顾未来 asyncio executor。

## 测试方案

中间件层的测试聚焦"组合行为"——单中间件容易写，关键是多个中间件互相影响时不出意外。

1. **单中间件单元测试**（`tests/archs/main_sub/execution/middleware/`，每个生产中间件一个文件）：
   - mock `MiddlewareManager`、直接调用钩子、断言 `HookResult` 字段值。
   - 内置 `LoggingMiddleware` 已有测试覆盖。

2. **MiddlewareManager 顺序契约测试**（`tests/archs/main_sub/execution/test_middleware_manager.py`）：
   - 注入 N=3 个 spy middleware，每个在 before/after/wrap 钩子里 append 一个标记；断言：
     - `run_before_*` 顺序 = `[m0, m1, m2]`
     - `run_after_*` 顺序 = `[m2, m1, m0]`
     - `wrap_*` 顺序 = `m0(m1(m2(call_next())))`
   - 验证 `HookResult.messages = [...]` 在中间件 i 设置后被中间件 i+1 看到。
   - 验证一个中间件抛异常后，run_* 仍把后续中间件跑完；wrap_* 立即向上抛。

3. **`force_continue` 短路测试**：构造 after_model 链，第二个 middleware 设 `force_continue=True`，断言 executor 跳过当轮 tool 调用。

4. **集成测试**（`tests/archs/main_sub/execution/test_executor_with_middlewares.py`）：
   - 把 6 个内置中间件按推荐顺序串联起来跑一个 toy agent，断言事件 / 日志 / 重试 / 压缩四类行为不互相干扰。
   - 重点用例：`LLMFailoverMiddleware` 在 `AgentEventsMiddleware` 之内时，确认 RunStartedEvent 不被 retry 多发。

## 未解决的问题

1. **链顺序约束没有 enforcement**。本 RFC 文档化了三条 "X 应在 Y 之外/之内" 的最佳实践，但没有运行时校验。错误顺序导致的事件重发 / 压缩失效是隐性 bug，需要补 yaml-loader 校验或 `MiddlewareManager.__init__` 的 sanity check。

2. **`HookResult.messages` 的"全量替换"语义在大消息列表上昂贵**。每次 before_model 修改 messages 都要复制一份；对长会话（几千条 message）可能成为 O(n²) 累积。是否要引入 `messages_diff: list[(int, Message)]` 增量结构待定。

3. **`stream_chunk` 中 `None` 与 "missing chunk" 的语义混淆**。当前 `None` 表示"丢弃"。如果未来某 chunk 类型本来就允许 `None`（比如 keep-alive），会冲突。考虑改用显式 `_DROP` 哨兵。

4. **Async 化的迁移路径**。当 executor 主循环转 asyncio 时，所有 9 个钩子的签名都要二选一（async vs sync）。Django 的双套接口方案是可行参考，但成本高。

5. **wrap_tool_call 缺乏内置示例**。外部可以扩展，但没有"官方推荐怎么写"的参照。后续应至少提供一个 `ToolTimeoutMiddleware` 或 `ToolRetryMiddleware`。

## 参考资料

- 核心实现：`nexau/archs/main_sub/execution/hooks.py`
- 编排器调用点：`nexau/archs/main_sub/execution/executor.py`（`MiddlewareManager` 构建在 `_build_middleware_manager:812-856`，9 处运行入口分散在 `:419 / :503 / :806 / :888` 等）
- 内置中间件目录：`nexau/archs/main_sub/execution/middleware/` 与 `nexau/archs/main_sub/team/middleware/`
- onion-skin 模型参考：[koa middleware](https://koajs.com/) / [express middleware](https://expressjs.com/en/guide/using-middleware.html)
- 相关 RFC：
  - RFC-0002: Agent Team — TeamMessageMiddleware 用例
  - RFC-0003: LLM Failover Middleware — LLMFailoverMiddleware 用例
  - RFC-0004: 上下文溢出与紧急压缩 — ContextCompactionMiddleware 用例
  - RFC-0007: 工具系统架构与绑定模型 — before/after/wrap_tool 钩子的工具侧契约
  - RFC-0010: LLM 聚合器事件流 — stream_chunk + AgentEventsMiddleware 用例
  - RFC-0006: RFC 目录补全总纲 — 上级任务
