# RFC-0026: HistoryList 写入收口 — typed REPLACE 走 FrameworkContext API + append-only 演进路线

- **状态**: implemented (Stage 1 + Stage 2a)
- **优先级**: P2
- **标签**: `architecture`, `dx`, `event-sourcing`, `cleanup`, `api`
- **影响服务**: NexAU (`HistoryList` / `Executor` / provider serializers / `RoundAndTokenReminderMiddleware` / `ContextCompactionMiddleware` / `HookResult` / `FrameworkContext` / `ModelCallParams`)
- **创建日期**: 2026-05-09
- **更新日期**: 2026-07-31

## 摘要

收口 RFC-0022 Phase 3 落地的 typed REPLACE 写入路径。把 "compaction middleware 直写 `agent_state.history.emit_typed_replace(...)` + `adopt_replaced_state` 重置 baseline 防双写" 这套两路写协同，重构成 **HookResult 透传 generic `HistoryEvent` discriminated union → executor 在 middleware 边界 dispatch by event type → `FrameworkContext.history.replace(...)` 等公开 API 写入**。

**FrameworkContext.history 是新加的 RPC-friendly 公开 API**——只暴露 `replace(messages, *, extra)` 一个方法，未来加 lambda tool 这种远程执行场景时它能直接当 RPC stub。AgentState 在 deprecation 路径上（替代品 `FrameworkContext`），本 RFC 删除其 `history` 字段并保证不增加新属性。

**对外影响**：旧的 `HookResult.with_modifications(messages=...)` 中间件调用方完全不受影响（additive）。`HistoryList.emit_typed_replace` / `adopt_replaced_state` 退化为薄 wrapper 保留。**`AgentState.history` 字段被删除**——内部无 production caller，只有 RFC-0022 Phase 3 期间加的 demo doc 已同步更新。

Stage 2a 不增加 `prompt_messages` 或第二套模型输入。真实用户输入继续使用 `USER`；框架内部
语言使用已有 `FRAMEWORK` role。Round/token reminder 始终追加独立 FRAMEWORK，持久化时因而是
APPEND，不再改写 USER。各 provider 只在序列化边界的临时视图中把相邻 USER/FRAMEWORK 合成
一个 user turn，canonical UMP 的 role、id、content、metadata 和顺序均不变。

## 动机

### Phase 3 落地的三处怪味

RFC-0022 Phase 3 引入 `CompactAutoVariant` / `UserClearVariant` 等 typed REPLACE extra，让消费者从 action 流就能区分 "这次 REPLACE 是 compaction / user clear / focused compact 还是 untyped 的兜底"，本意是好的。但实现路径选了"middleware 跨层直写持久化"，留下三处架构异味：

**① `agent_state.history` 反向引用** —— `AgentState` 本来是 per-execution 状态容器（`run_id` / `context` / `global_storage` / `sandbox_manager`），Phase 3 给它加了 `history: HistoryList | None`，让 middleware 能反向跨层调用持久化层。Middleware 现在能直接 `agent_state.history.emit_typed_replace(...)`，破坏 "middleware 通过 HookResult 影响状态 → executor 统一写持久化" 的分层。

**② `adopt_replaced_state` 名字温柔实际暴力** —— 公开方法，唯一调用者是同文件的另一个方法（`emit_typed_replace`），名字暗示"被动接受"实际是强制 reset 内存 list + baseline + pending_messages。它存在的唯一原因是：`emit_typed_replace` fire-and-forget 写完持久化后，必须**同步**重置内存 baseline，否则下一次 `flush()` 会发现"指纹和 baseline 不一致"→ 又写一次 untyped REPLACE → 同一次 compaction 落两条 REPLACE 行。

**③ Compaction 一次操作触发两路写** —— `ContextCompactionMiddleware` 的 `before_model` / `after_model` 路径既调用 `_emit_typed_replace_for_compaction(...)`（路径 A：直写 typed REPLACE + 重置 baseline）又 `return HookResult.with_modifications(messages=compacted_messages)`（路径 B：让 executor 看到新 messages）。两个不同代码路径协同才正确：
- 路径 A 失败（fire-and-forget 网络抖一下）→ 有 typed REPLACE 丢失但内存继续往前 → 持久化和内存分叉
- 注释里写"falls back to fingerprint-diff REPLACE on the next flush"其实是错的——baseline 已被路径 A 重置，flush 不会兜底

### 根因一句话

> **HistoryList 把"transient prompt mutations"和"persistent history events"塞进了同一个 channel（`HookResult.messages`），然后在 `flush()` 里用 fingerprint diff 反推 intent。**

从这个视角看，所有问题其实是**同一个设计 trade-off 的三种症状**——为了让 compaction 带上 typed extra，选择了"middleware 跨层直写"而不是"HookResult 透传 + executor 统一写"。

### 还有一个隐藏的真实 bug

顺带发现：`round_and_token_reminder` 曾把提示直接拼进最后一条 USER。Executor 随后把 working
messages 同步回 HistoryList，flush 把 USER fingerprint 变化误判成 untyped REPLACE。这既污染
真实用户输入，也让上层 UI 显示假的 compaction。

Stage 2a 直接复用 UMP 已有 role 修复：用户语言保持 USER，框架语言使用独立 FRAMEWORK。
FRAMEWORK 是 durable working context，正常持久化为 APPEND；是否展示由上层产品按 role 决定。
Provider 没有 framework role 时，只在序列化临界点将其投影为 user-shaped content。

## 设计

### 概述

所有 typed REPLACE 写入收敛到 `FrameworkContext.history.replace(messages, *, extra)` 这一个**公开 API 方法**，多条来源路径都走它：

```
                                    ┌────────────────────────────────────────┐
hook 路径（regular compaction /     │ HookResult.history_event=ReplaceEvent  │
  /clear / /compact / /undo）   ───▶│   ↓                                    │
                                    │ MiddlewareManager 写 outparam          │
                                    │   ↓                                    │
                                    │ Executor 读 hook_input.history_event   │
                                    │   → dispatch by .type:                 │
                                    │     ReplaceEvent → ctx.history.replace │
                                    │     UnknownEvent → skip (forward-compat)│
                                    └────────┬───────────────────────────────┘
                                             │
                                             ▼
                          ┌──────────────────────────────────────┐
                          │ FrameworkContext.history.replace(    │
                          │     messages, extra=variant)         │  ◀── 公开 API
                          │   ↓ (内部转 HistoryList)             │
                          │ HistoryList.replace_all(             │
                          │     messages, replace_extra=variant) │
                          │   - 重置内存 list                    │
                          │   - 更新 baseline                    │
                          │   - schedule_async typed REPLACE     │
                          └──────────────────────────────────────┘
                                             ▲
                                             │
                                    ┌────────┴─────────────────────────────┐
emergency 路径（wrap_model_call    │ ContextCompactionMiddleware 调用     │
  里的 LLM-retry 紧急压缩）  ─────▶│ params.framework_context.history     │
                                    │   .replace(...) — 同一个公开 API     │
                                    │ （不走 HookResult，因为不在 hook 里）│
                                    └──────────────────────────────────────┘
```

**为什么是 FrameworkContext.history 而不是 HistoryList**：FrameworkContext 是中间件 / tool / 未来 lambda tool 的统一公开 API 表面，narrow / 类型化 / RPC-friendly。HistoryList 是内部实现细节（一个 `list[Message]` 子类，方法签名重、跨进程序列化困难）。把"public API → 内部实现"分层做出来是 RFC-0026 的核心。

### 详细设计

#### 1. `FrameworkContext.history` —— 新增分组 API

```python
class HistoryAPI:
    """Write-side typed-event API for agent history.

    RPC-friendly by design — every public method takes only serializable
    arguments (Pydantic Message + ReplaceVariantBase subclass). When
    running in remote-tool mode (lambda / RPC future), this becomes a
    thin RPC stub.
    """
    def replace(
        self,
        messages: list[Message],
        *,
        extra: ReplaceVariantBase,  # required — typed channel only
    ) -> None: ...

class FrameworkContext:
    # ...existing tools / execution APIs...
    history: HistoryAPI  # NEW grouped API
```

**最小暴露面**：只有 `replace`。理由：
- 当前 typed-event 写入唯一的 production caller 就是 compaction
- APPEND 走 executor 内部流（`history.append/extend` 的 list 接口），没有 middleware 用例需要从外部 emit
- 读路径无 production caller（CLAUDE.md demo 已更新）
- 后续真有需求时再加（`append` / `read_messages` / `undo`），"narrow first" 优于"宽 API 后退"

**RPC 友好**：参数全部是 Pydantic 可序列化对象，无 in-process object handle 跨边界——lambda tool 把 ctx 远程化时只需把 `replace` 实现替换为 RPC stub。

#### 2. `HookResult.history_event` —— 中间件 → executor 的通用 typed-event 信号

```python
# nexau/archs/main_sub/execution/history_events.py (NEW)
class ReplaceEvent(BaseModel):
    type: Literal["replace"] = "replace"
    messages: list[Message]
    extra: ReplaceVariantBase

class AppendEvent(BaseModel):
    type: Literal["append"] = "append"
    messages: list[Message]
    extra: AppendExtra | None = None

class UndoEvent(BaseModel):
    type: Literal["undo"] = "undo"
    before_run_id: str
    extra: UndoExtra | None = None

class UnknownEvent(BaseModel):
    """Forward-compat fallback; old SDK reading new event types lands here."""
    type: str

# Discriminated union over .type, with callable Discriminator that
# routes unknown values to UnknownEvent (vs raising ValidationError).
HistoryEvent = Annotated[
    Annotated[ReplaceEvent, Tag("replace")] |
    Annotated[AppendEvent, Tag("append")] |
    Annotated[UndoEvent, Tag("undo")] |
    Annotated[UnknownEvent, Tag("unknown")],
    Discriminator(_discriminate_history_event),
]

# hooks.py
@dataclass
class HookResult:
    messages: list[Message] | None = None
    # ... 现有字段不变 ...
    history_event: HistoryEvent | None = None  # NEW — 通用 slot
```

**为什么是 `history_event` 不是 `replace_extra`**：

最初 RFC-0026 设计的是 `replace_extra: ReplaceVariantBase | None`——只能装 typed REPLACE。但未来 `/undo` 来时怎么办？要么塞 `undo_extra` 字段（slot 爆炸 N 个 typed slot），要么把 UndoExtra 硬塞 ReplaceVariantBase（语义错乱）。

通用 `history_event` slot + discriminated union 一次性解决：
- 加新 event type → union 里加一个 variant，0 caller 升级
- 老 SDK 看到新 type → fallback 到 `UnknownEvent`（callable Discriminator 实现），executor 跳过该事件，不 crash
- 每个 variant 内部带自己的 typed extra，编译期就能区分（`isinstance(event, ReplaceEvent)` 类型守卫）

**语义**：当 middleware 修改 `messages` 表达的是真实 history event（compaction / `/clear` / `/compact <focus>` / 未来 `/undo`）时，**同时**设置 `history_event` 携带显式事件。

**向后兼容**：默认 None；现有 middleware 只设 `messages` 不设 `history_event` → executor 走旧的 fingerprint-diff 兜底路径。

#### 3. `MiddlewareManager` —— outparam 模式

`BeforeModelHookInput` / `AfterModelHookInput` 加 `history_event: HistoryEvent | None = None` 字段（outparam）。`run_before_model` / `run_after_model` 在迭代 hook 链时收集 `hook_result.history_event`，最后写到 `hook_input.history_event`，executor 读取。

每次 `run_*` 开头清空 outparam，防止上一轮 stale 状态泄漏。

#### 4. Executor —— 边界处按 event type dispatch

```python
def _emit_pending_history_event(framework_context, event):
    if event is None:
        return
    if isinstance(event, ReplaceEvent):
        framework_context.history.replace(event.messages, extra=event.extra)
        return
    # AppendEvent / UndoEvent / UnknownEvent: 没有 ctx.history.* 公开方法
    # （RFC-0026 的"narrow first"原则）。silently skip——producer 上线时
    # 同时加 ctx.history.append/undo 方法 + 这里加 dispatch 分支。
    logger.debug("RFC-0026: skipping history_event %r — no producer wired", type(event).__name__)
```

在 `run_before_model` / `run_after_model` 返回后立刻调用：

```python
messages = self.middleware_manager.run_before_model(before_model_hook_input)
_emit_pending_typed_replace(
    framework_context,
    messages,
    before_model_hook_input.replace_extra,
)
```

**为什么是 eager-write 而不是延后到 end-of-iteration sync**：保留 Phase 3 原有语义——typed REPLACE 行的 `messages_after` 是 compaction 当时的状态（不含后续 assistant response）。延后会造成 REPLACE 行携带 "已经又 append 过的状态"。

#### 5. Emergency 路径 —— 中间件直接调 ctx.history.replace

`ContextCompactionMiddleware.wrap_model_call` 在 LLM 上下文溢出时做紧急压缩，因为不在 hook 链路里、不返回 `HookResult`：

```python
if params.framework_context is not None:
    params.framework_context.history.replace(
        compacted_messages,
        extra=self._build_compaction_variant(mode="emergency", ...),
    )
```

通过 `ModelCallParams.framework_context`（RFC-0026 新增字段）拿到 ctx，调用同一个公开 API。**不**读 `agent_state.history`——AgentState 上的 history 字段已删除。

与 hook-dispatched `ReplaceEvent` 不同，emergency direct path 没有
`HookResult` 返回通道。因此 `HistoryAPI.replace(...)` 同时记录一次
direct-replace outbox；`Executor` 在 `call_llm` / `call_llm_async` 返回后消费该
outbox，把局部 working `messages` 对齐到 `compacted_messages`，再追加 assistant
与 tool result。这个对齐是正确性要求：如果只写 `HistoryList` 而不更新
Executor 局部变量，run 末尾的 `_sync_history(...)` 会用旧 messages 反向覆盖
刚完成的 typed REPLACE，造成下一轮继续以未压缩 history 调用模型。

#### 6. AgentState —— 删除 history 字段

AgentState 在 `agent_context.py` / `framework_context.py` 演进路线上明确处于**逐步废弃**状态（替代方案是 `FrameworkContext`）。本 RFC：
- **删除** `history: HistoryList | None` 字段
- 不增加任何替代逃生口字段（`pending_replace_extra` 之类）
- Constructor 不再接受 `history=` 参数
- 所有 `agent_state.history` 内部引用迁移到 `ctx.history.replace(...)`

无 production caller 受影响（grep 验证：除 RFC-0022 Phase 3 自身实现 + 一个 demo doc 外没有其他用法）。

#### 7. `HistoryList.replace_all(replace_extra=)` —— 内部入口

```python
def replace_all(
    self,
    new_messages: list[Message],
    *,
    update_baseline: bool = False,
    replace_extra: ReplaceVariantBase | None = None,  # NEW
) -> None:
    self.clear()
    super().extend(new_messages)
    if self._persistence_enabled:
        self._pending_messages.clear()
        if update_baseline or replace_extra is not None:
            current_non_system = [m for m in self if m.role != Role.SYSTEM]
            self._baseline_fingerprints = self._compute_fingerprints(current_non_system)
        if replace_extra is not None:
            self._schedule_typed_replace(new_messages, replace_extra)
```

**关键不变量**：`replace_extra` 隐含 `update_baseline=True`——typed 写已经定下 post-REPLACE ground truth，下一次 flush 不能再用 fingerprint diff 把它当作变化重写一遍。

#### 8. `emit_typed_replace` / `adopt_replaced_state` —— 退化为 wrapper

外部 SDK 用户可能在 RFC-0022 Phase 3 时期就建立在这两个方法上，删除会破坏向后兼容。本 RFC 把它们改成薄 wrapper 转发到新的 canonical 路径：

```python
def emit_typed_replace(self, new_messages, *, extra):
    """DEPRECATED (RFC-0026): use HookResult.replace_extra from middleware,
    or replace_all(messages, replace_extra=...) from non-hook contexts."""
    self.replace_all(new_messages, update_baseline=True, replace_extra=extra)

def adopt_replaced_state(self, new_messages):
    """DEPRECATED (RFC-0026): use replace_all(messages, update_baseline=True)."""
    self.replace_all(new_messages, update_baseline=True)
```

行为完全等价，调用方零感知。

### 示例

**Before（Phase 3 原版）：**

```python
# context_compaction/middleware.py
def before_model(self, hook_input):
    compacted_messages = self._compact(hook_input.messages)
    self._emit_typed_replace_for_compaction(
        agent_state=hook_input.agent_state,
        mode="regular",
        messages_before=hook_input.messages,
        messages_after=compacted_messages,
        # ...
    )
    return HookResult.with_modifications(messages=compacted_messages)

# 上面 _emit_typed_replace_for_compaction 内部：
history = getattr(agent_state, "history", None)
if history is None:
    return
variant = CompactAutoVariant(...)
history.emit_typed_replace(compacted_messages, extra=variant)
# 路径 A 完成；执行器后续路径 B 通过 _origin_history.replace_all(messages)
# 同步内存（baseline 已被路径 A 重置 → 不会 fingerprint-diff 写第二次）
```

**After（RFC-0026）：**

```python
# context_compaction/middleware.py
def before_model(self, hook_input):
    compacted_messages = self._compact(hook_input.messages)
    variant = self._build_compaction_variant(
        mode="regular",
        messages_before=hook_input.messages,
        messages_after=compacted_messages,
        # ...
    )
    return HookResult.with_modifications(
        messages=compacted_messages,
        replace_extra=variant,
    )

# executor.py 自动读 hook_input.replace_extra → 立即 _write_typed_replace_if_pending
# → HistoryList.replace_all(replace_extra=variant)
```

## 权衡取舍

### 考虑过的替代方案

#### 方案 A：直接做 event sourcing（一步到位）

直接把 `HistoryList` 改成 `EventLog` 的 derived view，所有写入都是显式 `AppendEvent` / `ReplaceEvent` 事件，删除 fingerprint-diff 机制。

**为什么否决**：
- 工作量大（~5 文件 + 全部 middleware audit + reader 端 replay 重写）
- 一次性 breaking change 风险高
- 不符合 #528 当前作为 Phase 1+2+3 收尾 PR 的范畴

→ 拆为 RFC-0026 Stage 1（本 RFC，HookResult 透传） + 后续 Stage 2/3（事件流 + EventLog projection）渐进推进。

#### 方案 B：用 `ModelCallParams.replace_extra` 字段做 emergency 路径的逃生口

为了避免 emergency 路径直接读 `agent_state.history`，可以在 `ModelCallParams` 加 `replace_extra` 字段，让 middleware 先 set 再让 llm_caller 在 wrap_model_call 返回后读出来传回 executor。

**为什么否决**：
- llm_caller 不返回 params，需要改返回 tuple 或新 dataclass，传播链长
- emergency 路径需要 typed REPLACE **eager-write**（保留 Phase 3 语义），延后到 executor sync 会让 REPLACE 行携带"已经又 append 过的状态"，语义不符
- 直接调 `history.replace_all` 调用的是公开 canonical 方法，不是 Phase 3 那种私有 `emit_typed_replace`，破坏分层程度小得多

#### 方案 C：在 AgentState 加 `pending_replace_extra` 逃生口

最初的实现尝试。

**为什么否决**：AgentState 在演进路线上**明确处于废弃路径**（替代方案是 `FrameworkContext`），不能再加新属性——一加就增加迁移阻力。

#### 方案 D：把公开 API 命名成 `ctx.actions.*` 而不是 `ctx.history.*`

`AgentRunActionModel` / `RunActionType` / `nexau_agent_run_actions` 在持久化层都用 "action" 这个词；按命名一致性看，公开 API 也叫 `ctx.actions.replace(...)` 是顺理成章的。

**为什么否决**：

| | `ctx.history.replace(...)` | `ctx.actions.emit(ReplaceEvent(...))` |
|---|---|---|
| 抽象层级 | Domain operation（"我想改对话历史"） | Implementation detail（"往 action 流发一个 event"） |
| Caller 心智 | 业务语义 | 事件流语义 |
| 类型安全 | 每动作专属方法签名，IDE 补全清晰 | 单 `emit(event)`，类型靠 union |
| 暴露面 | 窄（每动作一个方法） | 宽（任何 event 都能发） |
| 类比 | Linux ``read(fd)`` | Linux ``syscall(SYS_read, ...)`` |

具体决策：**保持 `ctx.history.*` 是用户面 API，`actions` 留给 RFC-0022 的持久化层**。理由：

1. **没有 raw emit 的真实场景**：middleware 通过 `HookResult.history_event` 信号 → executor dispatch → `ctx.history.<verb>`，不存在"middleware 直接 emit 原始 event"的需求；`ctx.actions.emit(...)` 是 over-engineering
2. **"narrow first" 原则**：`ctx.actions` 现在加进去就是无 caller 的 API surface
3. **`history` 是更好的用户面词**：`ctx.history.read_messages()` 比 `ctx.actions.replay()` 直观；`ctx.history.replace(messages, extra)` 比 `ctx.actions.emit(ReplaceEvent(...))` 轻量
4. **`actions` 在 nexau 已经多义**：tool action / agent action / sub-agent action / RunAction 全叫 action，再加一个 `ctx.actions` 名词冲突；`history` 是单义的领域词

**两层关系（mental model，不暴露成两个 namespace）**：

```
┌──────────────────────────────────────────────────────────┐
│ Layer 1: 用户面 API（暴露给 middleware / tool 作者）     │
│   ctx.history.replace(messages, extra=variant)            │
│   (未来) ctx.history.append / undo / read_messages       │
└─────────────────┬────────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────────┐
│ Layer 2: 持久化实现（RFC-0022，不暴露给 middleware/tool）│
│   AgentRunActionModel    (DB 行)                          │
│   RunActionType          (action 类型 enum)               │
│   AgentRunActionService  (持久化 service)                 │
│   nexau_agent_run_actions (表)                            │
└──────────────────────────────────────────────────────────┘
```

**未来真要加 raw emit**：场景化方法（如 time-travel 调试器需要 `ctx.history.replay_events(events: list[HistoryEvent])`），针对**具体场景**加方法到 `ctx.history`，**不**在公开 API 引入 generic `ctx.actions.emit(event)` namespace。

### 缺点

- **仍保留 fingerprint-diff 兜底机制**：HistoryList 的 `flush()` 仍用旧 baseline diff 兼容会改写
  history 的 middleware。Round/token reminder 已改为 append-only FRAMEWORK；其他 producer（如
  `runtime_environment`）的持久化语义不在 Stage 2a 范围内。
- **Emergency 路径的 direct replace 需要局部 history 对齐**：`ctx.history.replace(...)`
  既要写 durable history，也要让 Executor 消费 direct-replace outbox 更新
  working messages。否则 end-of-run sync 会把旧 messages 写回，抵消 typed
  REPLACE。
- **`emit_typed_replace` / `adopt_replaced_state` 残留**：作为 deprecated wrapper 还在 public API surface，等下一个 minor 版本可以加 `@deprecated` decorator + warning，再下一个删。

## 实现计划

### 阶段划分

#### Stage 1（本 RFC，已实现）

`HookResult` 透传 typed extra + executor 通过 `FrameworkContext.history.replace` 单一公开 API + emergency 路径直调同 API。

- [x] `nexau/archs/main_sub/execution/history_events.py` 新增 — `HistoryEvent` discriminated union（ReplaceEvent / AppendEvent / UndoEvent / UnknownEvent + callable Discriminator forward-compat fallback）
- [x] `FrameworkContext.history` (HistoryAPI) 新增分组 API，`replace(messages, *, extra)` 唯一方法
- [x] `HookResult.history_event: HistoryEvent | None` 字段（additive，generic slot 而非 typed slot）
- [x] `BeforeModelHookInput` / `AfterModelHookInput` outparam `history_event` 字段
- [x] `MiddlewareManager.run_before_model` / `run_after_model` 收集 + publish outparam
- [x] `ModelCallParams.framework_context` 字段（emergency 路径用）
- [x] `LLMCaller.call_llm` / `call_llm_async` 接 `framework_context` 参数 → 写入 ModelCallParams
- [x] Executor 在 4 处 middleware 边界通过 `_emit_pending_history_event(framework_context, event)` dispatch by event type
- [x] Executor 在 `wrap_model_call` emergency direct replace 后消费
  `HistoryAPI` outbox，将 local working messages 对齐到 compacted history
- [x] FrameworkContext 构造 2 处加 `_history=` 参数
- [x] `HistoryList.replace_all` 增加 `replace_extra=` kwarg + `_schedule_typed_replace` 私有 helper（内部实现，没暴露公开）
- [x] `emit_typed_replace` / `adopt_replaced_state` 退化为 wrapper（保留 back-compat）
- [x] `ContextCompactionMiddleware` 三处迁移：regular before/after → `history_event=ReplaceEvent(...)`；emergency → `params.framework_context.history.replace(...)`
- [x] **删除** `AgentState.history` 字段 + import + constructor 参数
- [x] `Agent._run_async_inner` 不再传 `history=` 给 AgentState
- [x] `tool/CLAUDE.md` 示例更新（`agent_state.history` → 提示用 ctx.history）
- [x] 单元测试 `tests/unit/test_rfc0026_history_event_channel.py`（12 个 test：HookResult.history_event 默认 + ReplaceEvent round-trip + outparam funneling + UnknownEvent forward-compat fallback + 落库端到端 + back-compat shim 等）

#### Stage 2a（本 RFC，已实现）

使用已有 UMP messages 和 role 解决 reminder 污染，不引入平行 prompt channel：

- [x] `RoundAndTokenReminderMiddleware` 始终 append 独立 FRAMEWORK，不修改末尾 USER；
- [x] OpenAI Chat/Responses、Anthropic、Gemini 仅在 provider 临界点聚合相邻 USER/FRAMEWORK；
- [x] 没有 FRAMEWORK 时，相邻普通 USER 保持原行为，不被误合并；
- [x] 空 FRAMEWORK 保留在 canonical history，但不生成空 provider turn；
- [x] Gemini 在 functionResponse 后把 FRAMEWORK text 追加到同一个 provider USER content，
      保持 provider 的 user/model 交替；
- [x] public dict history 对 role 做大小写无关的已知值映射，`FRAMEWORK`/`framework` round-trip
      仍为 FRAMEWORK；未知 role 继续按旧兼容策略降级为 USER。

#### Stage 2（独立 PR，未启动）

逐个 producer 明确 APPEND/REPLACE 语义后，删除 fingerprint-diff 兜底机制。本阶段同样不引入
`prompt_messages`：模型输入和 canonical working history 继续使用一份 UMP messages；来源通过
现有 role 表达。

- [ ] audit 仍会改写既有 message 的 middleware，并改为显式 APPEND/REPLACE；
- [ ] `HookResult.history_event: AppendEvent | ReplaceEvent | None` 覆盖全部 durable mutation；
- [ ] HistoryList 删除 `_baseline_fingerprints` / `_pending_messages` / `_compute_fingerprints` / `_prepare_flush` 的 diff 逻辑；`flush` 退化为"等 background task 完成"
- [ ] 删 `emit_typed_replace` / `adopt_replaced_state` 两个 deprecated wrapper

#### Stage 3（独立 PR，未启动）

HistoryList 退化为 EventLog 的 derived view。

- [ ] 引入 `EventLog` 抽象（single-thread serial 写入器，append-only）
- [ ] `HistoryList(event_log)` 构造时 replay event_log 拼出初始 list
- [ ] `HistoryList.append/extend/replace_all` 内部转化为 emit `AppendEvent` / `ReplaceEvent` + 折叠到内存视图
- [ ] 持久化后端从 HistoryList 移到 EventLog（可插拔 in-memory / SQLite / Postgres）
- [ ] 跨线程 `_schedule_async` 复杂度移到 EventLog 单线程序列化队列（修今天 fire-and-forget 的乱序风险）
- [ ] AgentState.history backreference 完全移除

#### Stage 4（独立 PR，未启动）

Reader 端基于 EventLog.replay 重写。

- [ ] `load_messages_semantics` / `replay_oracle` 等 reader 路径全部建在 EventLog.replay 之上
- [ ] 删旧的 fingerprint-based reader 反推逻辑
- [ ] UNDO 作为 first-class event 简化处理

### 相关文件

Stage 1（本 RFC 实现的）：
- `nexau/archs/main_sub/execution/history_events.py` (NEW) — `HistoryEvent` discriminated union + `UnknownEvent` forward-compat fallback
- `nexau/archs/main_sub/framework_context.py` — 新增 `HistoryAPI` 分组 API + `ctx.history` 字段
- `nexau/archs/main_sub/execution/hooks.py` — `HookResult.history_event` + outparam + `ModelCallParams.framework_context` + MiddlewareManager 收集
- `nexau/archs/main_sub/execution/llm_caller.py` — `call_llm` / `call_llm_async` 接 `framework_context` → 写到 ModelCallParams
- `nexau/archs/main_sub/execution/executor.py` — `_emit_pending_history_event` dispatcher + 4 处 middleware 边界 + FrameworkContext 构造传 `_history=` + `_sync_history` 简化
- `nexau/archs/main_sub/history_list.py` — `replace_all(replace_extra=)` + `_schedule_typed_replace` + 两个 deprecated wrapper
- `nexau/archs/main_sub/agent_state.py` — **删除** `history` 字段 + `HistoryList` import
- `nexau/archs/main_sub/agent.py` — `AgentState(...)` 构造不再传 `history=`
- `nexau/archs/main_sub/execution/middleware/context_compaction/middleware.py` — `_build_compaction_variant` 纯 builder + 3 处调用点迁移（regular hook 路径 → ReplaceEvent；emergency → ctx.history.replace）
- `nexau/archs/tool/CLAUDE.md` — 示例更新
- `tests/unit/test_rfc0026_history_event_channel.py` — 12 个测试

Stage 2a：
- `nexau/archs/main_sub/execution/middleware/round_and_token_reminder.py`
- `nexau/core/serializers/user_projection.py`
- `nexau/core/serializers/{openai_chat,anthropic_messages,gemini_messages}.py`
- `tests/unit/test_{round_and_token_reminder,user_framework_projection,anthropic_gemini_serializers}.py`

Stage 2/3/4 涉及（未来 PR）：
- `nexau/archs/main_sub/execution/middleware/runtime_environment.py`
- `nexau/archs/main_sub/history_list.py`（彻底重写）
- `nexau/archs/session/...`（EventLog 引入）

## 测试方案

### 单元测试（Stage 1 已交付）

`tests/unit/test_rfc0026_history_event_channel.py`，8 个测试覆盖：

1. `HookResult.replace_extra` 默认 None / 显式赋值 round-trip
2. `MiddlewareManager` 把 hook 链返回的 `replace_extra` publish 到 `hook_input` outparam
3. `MiddlewareManager` 在每次 `run_*` 开头清空 outparam（防 stale）
4. `HistoryList.replace_all(replace_extra=variant)` 落地写一行 typed REPLACE row（端到端 SQLite）
5. `HistoryList.replace_all(messages)` 不带 extra → 不立即写（保留旧 fingerprint-diff 兜底）
6. `emit_typed_replace`（deprecated）写出与新路径完全等价的 typed REPLACE row
7. `adopt_replaced_state`（deprecated）正确重置 baseline，subsequent flush 不会重复写

### 回归测试

完整 unit suite：3641 passed / 0 failed（Stage 1 实现后跑的；先前是 3633，新增 8 个）。
- `test_run_action_lifecycle_service.py` / `test_run_action_db_roundtrip.py` / `test_run_action_typed_replace.py` 等 RFC-0022 Phase 1+2+3 测试全部通过
- `test_executor_coverage3.py` 一处 mock 断言更新（`replace_all` kwarg 签名加了 `replace_extra`）

### 集成测试

NAC（`china-qijizhifeng/nexau-cloud-runtime` PR #549）K8s Sandbox 测试覆盖端到端的 compaction → typed REPLACE 持久化路径，行为不变。

## 未解决的问题

1. **何时给 `emit_typed_replace` / `adopt_replaced_state` 加 `@deprecated` warning**：本 RFC 只把它们退化为 wrapper，没加 warning。下一个 minor 版本可以加，再下一个删。需要 SDK 用户群是否真有外部 caller 的 telemetry 数据。

2. **`agent_state.history` 反向引用何时彻底切断**：emergency 路径仍读它。等 AgentState 全面被 FrameworkContext 替代时，候选迁移路径有 `FrameworkContext.history_handle` 或 `ModelCallParams.history`，需要 RFC-0027 / Stage 3 阶段一并设计。

3. **剩余 producer 的迁移节奏**：`runtime_environment` 等会改写既有 message 的 middleware
   仍需逐个明确 APPEND/REPLACE 语义。Stage 2a 已证明不需要第二套 prompt channel；后续继续用
   UMP role + typed history event 收敛。

## 参考资料

- [RFC-0022](./0022-agent-run-action-lifecycle-and-typed-blocks.md) — 本 RFC 上游，Phase 1+2+3 落地的 typed REPLACE protocol
- [RFC-0021](./0021-history-archive-on-compaction.md) — compaction 归档语义
- 实现 commits（在本分支 `feat/rfc-0022-phase-3-typed-compact-replace`）：
  - `5ec839ba` — 第一版（含 `AgentState.pending_replace_extra` 逃生口）
  - `b96e8896` — follow-up 删除 AgentState 新字段，改用 emergency 路径直调 canonical 方法
