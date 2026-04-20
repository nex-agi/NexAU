# RFC-0008: 会话持久化与历史管理

- **状态**: implemented
- **优先级**: P1
- **标签**: `architecture`, `session`, `persistence`, `history`
- **影响服务**: `nexau/archs/session/`, `nexau/archs/main_sub/history_list.py`, `nexau/archs/main_sub/agent.py`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 的**会话层**（session layer）负责把跨请求的运行时状态——会话元数据、全局存储、消息历史、并发锁——透明地落到可插拔后端上。设计通过三层抽象实现：`SessionModel` 定义数据、`DatabaseEngine` 抽象后端、`SessionManager` 聚合服务。`HistoryList` 在 `list[Message]` 之上叠加批量持久化语义，以 run 为粒度写入三种事件（APPEND / REPLACE / UNDO），支持 event-sourcing 风格重建。本 RFC 记录已实现的完整架构，边界上排除 RFC-0001 已覆盖的停止路径持久化细节。

## 动机

### 现状

Agent 的每一次运行都会读写多种形态的运行时数据：
- **会话上下文**：`working_directory`、`username`、`date` 等跨 Agent 共享的 kv
- **全局存储**：Agent 间数据交换的容器（`GlobalStorage`，见 RFC-0012）
- **消息历史**：每个 Agent 在会话中累积的 `Message` 列表
- **Agent 元数据**：`user_id + session_id + agent_id + agent_name`
- **并发锁**：同一 (session, agent) 上只允许一个运行，需跨进程协同

这些数据需要：
1. 跨请求持久化——HTTP 无状态，REST 层每次都要"恢复"一个 Agent
2. 支持多种部署形态——内存（测试）、SQLite（单机）、JSONL（审计）、远端 HTTP（分布式）
3. 以 **run** 为最小事务单位——一次 `agent.run_async()` 要么全部落盘，要么完全失败
4. 支持**重建**而非**覆盖**——Compaction 会做 REPLACE、用户可以 UNDO，必须能回到任意时刻

### 不做会怎样

- ORM 与业务逻辑耦合，更换后端要改动 Agent 代码
- 每条消息一次写请求，热路径 I/O 放大
- 历史成为"最后一次写入的快照"，UNDO / REPLACE / 审计均不可实现
- 并发写入无仲裁，同一 session 双请求导致消息交错

## 设计

### 分层架构

```
┌────────────────────────────────────────────────────┐
│                   Agent / Executor                 │
│   （ business 层，通过 SessionManager 操作数据 ）   │
└─────────────────┬──────────────────────────────────┘
                  │
        ┌─────────▼─────────┐      ┌──────────────────┐
        │  SessionManager   │──┬──▶│ AgentLockService │
        │ (unified facade)  │  │   └──────────────────┘
        └─────────┬─────────┘  │   ┌──────────────────┐
                  │            ├──▶│AgentRunActionSvc │
                  │            │   └──────────────────┘
                  │            │   ┌──────────────────┐
                  │            ├──▶│   AgentService   │
                  │            │   └──────────────────┘
                  │            │   ┌──────────────────┐
                  │            └──▶│  TaskLockService │
                  │                └──────────────────┘
        ┌─────────▼─────────┐
        │  DatabaseEngine   │  （ 抽象 CRUD：find/create/update/delete/upsert ）
        └─────────┬─────────┘
    ┌─────┬──────┼───────┬───────┐
    ▼     ▼      ▼       ▼       ▼
 InMemory  SQL   JSONL  Remote  （ 4 种后端实现 ）
```

### 数据模型

`SessionModel` 位于 [session.py:30-80](nexau/archs/session/models/session.py#L30-L80)，以 `(user_id, session_id)` 为复合主键，存储：

| 字段 | 类型 | 用途 |
|------|------|------|
| `user_id` + `session_id` | str | 复合主键 |
| `created_at` / `updated_at` | datetime | 审计时间戳 |
| `context` | `dict[str, Any]` (JSON) | 跨 Agent 共享的会话上下文 |
| `storage` | `GlobalStorage` (JSON) | 全局存储（RFC-0012） |
| `sandbox_state` | `dict[str, Any]` (JSON) | 沙箱恢复快照（RFC-0009） |
| `root_agent_id` | str \| None | 根 Agent 的 ID，用于跨请求复用 |

其他表：
- `AgentModel` — Agent 元数据 `(user_id, session_id, agent_id, agent_name)`
- `AgentRunActionModel` — 事件流记录（APPEND / REPLACE / UNDO）
- `AgentLockModel` — Agent 级并发锁（带 TTL 和 heartbeat）
- `TaskLockModel` — Team 任务级锁

### 后端抽象

`DatabaseEngine` (抽象基类) 位于 [engine.py:39-134](nexau/archs/session/orm/engine.py#L39-L134)，公开六个纯 CRUD 抽象方法（`setup_models` / `find_first` / `find_many` / `create` / `create_many` / `update` / `delete` / `count`）和两个组合操作（`upsert` / `get_or_create`）。它不绑定任何特定 ORM，子类负责把 SQLModel 映射到自己的存储形式：

| 实现 | 文件 | 适用场景 |
|------|------|---------|
| `InMemoryDatabaseEngine` | [memory_engine.py](nexau/archs/session/orm/memory_engine.py) | 单进程测试 / CLI 临时会话 |
| `SQLDatabaseEngine` | [sql_engine.py](nexau/archs/session/orm/sql_engine.py) | SQLite / PostgreSQL 等任何 SQLAlchemy 兼容后端 |
| `JSONLDatabaseEngine` | [jsonl_engine.py](nexau/archs/session/orm/jsonl_engine.py) | 审计日志 / 离线回放 |
| `RemoteDatabaseEngine` | [remote_engine.py](nexau/archs/session/orm/remote_engine.py) | 分布式部署——把 CRUD 转成 HTTP 调用 |

### SessionManager：统一门面

`SessionManager` 位于 [session_manager.py:33](nexau/archs/session/session_manager.py#L33)，聚合四个服务并暴露会话级 API：
- `register_agent(user_id, session_id, agent_name, is_root)` ([session_manager.py:134](nexau/archs/session/session_manager.py#L134))——在 is_root 情况下复用已有 `root_agent_id`，保证跨请求的根 Agent ID 稳定
- `get_session` / `_get_or_create_session` / `_update_session`
- 属性 `agent_run_action` / `agent_lock` / `task_lock` 暴露子服务供 Agent 直接调用

Agent 在 `_init_session_state()`（[agent.py:248](nexau/archs/main_sub/agent.py#L248)）里完成：注册自身到 `AgentModel`、加载历史、初始化 `HistoryList`、恢复 `context` 与 `storage`。

### HistoryList：带批量持久化的消息列表

`HistoryList` 继承 `list[Message]`，位于 [history_list.py:34](nexau/archs/main_sub/history_list.py#L34)，核心字段：

| 字段 | 作用 |
|------|------|
| `_pending_messages` ([line 86](nexau/archs/main_sub/history_list.py#L86)) | 当前 run 累积、尚未 flush 的消息 |
| `_baseline_fingerprints` ([line 87](nexau/archs/main_sub/history_list.py#L87)) | run 开始时的消息指纹快照，用于 flush 时判定 append-only vs replace |
| `_persistence_enabled` | 是否挂接了 `SessionManager` |

关键行为：
1. **`append(item)`**（[line 94](nexau/archs/main_sub/history_list.py#L94)）——进入底层 list，并缓存到 `_pending_messages`（忽略 system 消息）
2. **`flush()`**（[line 233](nexau/archs/main_sub/history_list.py#L233)）——run 结束时调用：比对 fingerprint，决定生成 APPEND 还是 REPLACE 事件，再通过 `_persist_flush_async` 投递到 `AgentRunActionService`
3. **`update_context()`**（[line 288-318](nexau/archs/main_sub/history_list.py#L288-L318)）——切换 run 时**先自动 flush**，再写入新的 `run_id / root_run_id / parent_run_id` 并重建 baseline
4. **异常安全**——`_persist_flush_async` 内部捕获所有异常（[line 230-231](nexau/archs/main_sub/history_list.py#L230-L231)）并降级为 warning log，不破坏主流程

Agent 内共有 5 个 flush 调用点：正常返回前 ([agent.py:1101](nexau/archs/main_sub/agent.py#L1101))、两个错误路径 ([agent.py:1118, 1126](nexau/archs/main_sub/agent.py#L1118-L1126))、**finally 兜底** ([agent.py:1137](nexau/archs/main_sub/agent.py#L1137))、以及 stop 路径 ([agent.py:1252](nexau/archs/main_sub/agent.py#L1252))。finally 分支即 RFC-0001 强调的"CancelledError 也能写回"的最终保障。

### 事件模型：APPEND / REPLACE / UNDO

`AgentRunActionService` 位于 [agent_run_action_service.py:39](nexau/archs/session/agent_run_action_service.py#L39)。三类事件按 run 级 append-only 存储：

| 事件 | 方法 | 语义 |
|------|------|------|
| APPEND | [persist_append](nexau/archs/session/agent_run_action_service.py#L71) (line 71) | 常规 run 结束，追加当轮新增消息 |
| UNDO | [persist_undo](nexau/archs/session/agent_run_action_service.py#L128) (line 128) | `undo_before_run_id` 指向要回滚到的 run；回放时该 run 至 UNDO 之前的所有 APPEND 都被跳过 |
| REPLACE | [persist_replace](nexau/archs/session/agent_run_action_service.py#L166) (line 166) | 上下文压缩（RFC-0004）——用新的完整消息序列覆盖历史起点 |

**重建算法** `load_messages` ([line 232](nexau/archs/session/agent_run_action_service.py#L232))：
1. 以 `created_at_ns` 倒序分页扫描动作
2. 遇到 UNDO 时记录 `skip_until_run_id`，略过被 UNDO 的 run
3. 遇到 REPLACE 时固定基线，停止继续往前扫描
4. 正序合并：`base_replace.messages` → 反转后的 `appends_desc`
5. 以 `Message.id` 做去重 + 最新值覆盖，得到最终有序列表

### 并发锁：AgentLockService

位于 [agent_lock_service.py:47](nexau/archs/session/agent_lock_service.py#L47)。关键设计：
- **过期驱动**：锁在表中带 `expires_at_ns`，查询时过滤过期行，不需要后台清理任务
- **Run 级 heartbeat**：`lock_ttl=30s`，`heartbeat_interval=10s`（[line 71-72](nexau/archs/session/agent_lock_service.py#L71-L72)）——heartbeat 循环 ([line 113](nexau/archs/session/agent_lock_service.py#L113)) 每 10s 把 `expires_at_ns` 往后推 30s
- **holder_id = `{pid}:{uuid8}`**——区分不同进程、不同获取动作
- **约束** `heartbeat_interval < lock_ttl/2`（[line 85-86](nexau/archs/session/agent_lock_service.py#L85-L86)）——确保一次错过 heartbeat 不会立即过期
- **非阻塞**：busy 时立刻失败，不排队；调用方负责重试策略

### 与 RFC-0001 的边界

本 RFC 描述"正常运行路径上的持久化 + 历史模型"，**不**重复以下 RFC-0001 已覆盖的内容：
- `agent.stop()` 的语义、`AgentStopReason.USER_INTERRUPTED`
- `CancelledError` 触发 finally flush 的具体控制流
- "中断后继续"的用户交互流程

两个 RFC 共享 `HistoryList.flush()` 这一原语，但讨论的"调用者"不同。

## 权衡取舍

| 选择 | 代价 | 收益 |
|------|------|------|
| **4 种后端实现由抽象类分发**，而非配置驱动的 ORM | 新增后端要写完整的 `DatabaseEngine` 子类 | 彻底消除 Agent 代码对后端细节的感知；远端 HTTP 后端成为"一等公民" |
| **HistoryList 批量 flush**，而非每条消息都写 | flush 前崩溃会丢本 run 数据（由 finally 兜底缓解） | 热路径只做内存 append；I/O 次数从 O(msg) 降到 O(run) |
| **Event sourcing**（APPEND/REPLACE/UNDO）而非 snapshot | `load_messages` 要扫描 + 合并；REPLACE 前的 APPEND 成为"死记录" | 原生支持压缩、撤销、审计回放；任意历史点可重建 |
| **过期锁 + heartbeat**，而非 blocking acquire | Heartbeat 线程占用资源；holder 崩溃到锁过期间 `lock_ttl` 内不可被接手 | 不需要死锁检测；进程异常退出自动释放；跨进程一致 |
| **`SessionModel.context / storage` 以 JSON 列存** | 无法对字段做原生 SQL 查询 | 自由增删字段无需 schema 迁移；前向兼容 |
| **根 Agent ID 存在 `SessionModel.root_agent_id`** | 多了一次 `update_session` 写 | 跨请求复用 Agent ID；无需客户端记忆 |

## 实现计划

此 RFC 记录**已实现**的子系统，补全缺失的设计文档。当前完成度：

| 能力 | 状态 | 文件 |
|------|------|------|
| `SessionModel` + 其余 SQLModel | implemented | [models/](nexau/archs/session/models/) |
| `DatabaseEngine` 抽象与 4 实现 | implemented | [orm/](nexau/archs/session/orm/) |
| `SessionManager` 门面 | implemented | [session_manager.py](nexau/archs/session/session_manager.py) |
| `AgentRunActionService` (APPEND/REPLACE/UNDO/load) | implemented | [agent_run_action_service.py](nexau/archs/session/agent_run_action_service.py) |
| `AgentLockService` (TTL + heartbeat) | implemented | [agent_lock_service.py](nexau/archs/session/agent_lock_service.py) |
| `HistoryList` 批量 flush + fingerprint 合并 | implemented | [history_list.py](nexau/archs/main_sub/history_list.py) |
| RFC-0001 stop 路径集成 | implemented | RFC-0001 |

目前不计划结构性变更，未来演进已归入"未解决的问题"。

## 测试方案

现有保障：
- `tests/archs/session/` 覆盖 4 种 DatabaseEngine 的 CRUD 一致性、`upsert` / `get_or_create` 语义
- `AgentRunActionService` 测试：APPEND 顺序、UNDO 跳过、REPLACE 截断、系统消息过滤、load 去重
- `AgentLockService` 测试：基本 acquire/release、过期自动回收、heartbeat 延长、非法参数（`heartbeat_interval >= lock_ttl/2`）
- `HistoryList` 测试：append-only / replace 判定、系统消息不持久化、`update_context` 自动 flush

新增补充：
- 一次端到端回归：用 `InMemoryDatabaseEngine` 跑一个带 REPLACE + UNDO + APPEND 混合的 run 序列，断言 `load_messages` 结果等价于预期的最终序列
- `RemoteDatabaseEngine` 合约测试：在 mock HTTP server 上校验所有抽象方法都正确映射到 HTTP 端点

## 未解决的问题

1. **`load_messages` 全量扫描成本**：当前算法 O(N_actions) 线性扫描；对超长会话（10k+ runs）需要引入 checkpoint（例如每 1000 个 APPEND 自动生成一次 compaction REPLACE）
2. **多 session 数据清理策略**：表只增不删，`user_id` 级别的 GDPR 级联删除需要新接口
3. **`AgentLockService` 重试策略**：当前由调用方负责；可能需要在 `SessionManager` 或 Agent 层提供标准退避器
4. **`JSONLDatabaseEngine` 在多进程写入时的一致性**：需确认是否依赖外部文件锁，或限定为单写者后端
5. **sandbox_state 字段演进**：当前是 free-form JSON；随 RFC-0009 落地，需要明确 schema 与版本号

## 参考资料

- [`nexau/archs/session/models/session.py:30-80`](nexau/archs/session/models/session.py#L30-L80) — SessionModel
- [`nexau/archs/session/orm/engine.py:39-134`](nexau/archs/session/orm/engine.py#L39-L134) — DatabaseEngine 抽象
- [`nexau/archs/session/orm/memory_engine.py`](nexau/archs/session/orm/memory_engine.py) / [`sql_engine.py`](nexau/archs/session/orm/sql_engine.py) / [`jsonl_engine.py`](nexau/archs/session/orm/jsonl_engine.py) / [`remote_engine.py`](nexau/archs/session/orm/remote_engine.py) — 4 种后端实现
- [`nexau/archs/session/session_manager.py:33`](nexau/archs/session/session_manager.py#L33) — SessionManager
- [`nexau/archs/session/session_manager.py:134`](nexau/archs/session/session_manager.py#L134) — register_agent 根 Agent 复用逻辑
- [`nexau/archs/session/agent_run_action_service.py:71-207`](nexau/archs/session/agent_run_action_service.py#L71-L207) — APPEND / UNDO / REPLACE
- [`nexau/archs/session/agent_run_action_service.py:232-328`](nexau/archs/session/agent_run_action_service.py#L232-L328) — load_messages 重建
- [`nexau/archs/session/agent_lock_service.py:47-150`](nexau/archs/session/agent_lock_service.py#L47-L150) — AgentLockService + heartbeat
- [`nexau/archs/main_sub/history_list.py:34-318`](nexau/archs/main_sub/history_list.py#L34-L318) — HistoryList + flush
- [`nexau/archs/main_sub/agent.py:248`](nexau/archs/main_sub/agent.py#L248) — Agent._init_session_state
- [`nexau/archs/main_sub/agent.py:846`](nexau/archs/main_sub/agent.py#L846) — load_messages 调用点
- [`nexau/archs/main_sub/agent.py:1101, 1118, 1126, 1137`](nexau/archs/main_sub/agent.py#L1101) — 5 个 flush 调用点
- RFC-0001 Agent 中断时状态持久化（停止路径的详细语义）
- RFC-0006 NexAU RFC 目录补全总纲（本 RFC 所属主计划）
