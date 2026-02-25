# RFC-0002: AgentTeam — 多 Agent 协作框架

- **状态**: implemented
- **优先级**: P0
- **标签**: `architecture`, `agent`, `collaboration`
- **影响服务**: `nexau/archs/main_sub/`, `nexau/archs/session/`, `nexau/archs/transports/http/`, `nexau/archs/tool/`
- **创建日期**: 2026-02-17
- **更新日期**: 2026-02-25

## 摘要

在 NexAU 中新增 AgentTeam 能力：一个 leader agent 协调多个 teammate agents，通过共享任务列表 + 队内消息协作完成工作。同一 `(user_id, session_id)` 仅允许一个 AgentTeam，所有 agent 状态在该 session 内持久化。

## 动机

当前 NexAU 的 Agent 体系以单 agent 或 parent→sub-agent 的树形调用为主。Sub-agent 模式适合"委派-等待-汇总"的串行场景，但无法满足以下需求：

1. **并行协作**：多个 agent 同时工作，各自领取任务独立执行
2. **动态任务分配**：leader 可在运行时创建任务、分配给 teammate，teammate 也可自助领取
3. **队内通信**：agent 之间需要点对点消息和广播，而非仅通过 parent 中转
4. **统一流式输出**：客户端需要在同一 SSE 连接中看到所有 agent 的实时输出，并能区分来源

AgentTeam 填补了这一空白，提供 leader-teammate 协作模式，支持共享任务列表、队内消息、并发执行与统一流式输出。

## 设计

### 概述

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AgentTeam (session scope)                       │
│                                                                     │
│  ┌───────────────┐         ┌──────────────────────────────────┐    │
│  │  Leader Agent  │────────▶│       Shared Task Board          │    │
│  │  (coordinator) │         │  ┌──────┐ ┌──────┐ ┌──────┐    │    │
│  └───────┬───────┘         │  │ T-001│ │ T-002│ │ T-003│    │    │
│          │                  │  │pend. │ │in_pr.│ │comp. │    │    │
│          │ spawn/message    │  └──────┘ └──────┘ └──────┘    │    │
│          │                  └──────────────────────────────────┘    │
│  ┌───────▼───────────────────────────────────┐                     │
│  │            Teammate Agents                 │                     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐│                     │
│  │  │ coder-1  │  │ coder-2  │  │reviewer-1││                     │
│  │  │ (role:   │  │ (role:   │  │ (role:   ││                     │
│  │  │  coder)  │  │  coder)  │  │ reviewer)││                     │
│  │  └──────────┘  └──────────┘  └──────────┘│                     │
│  └───────────────────────────────────────────┘                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Team Message Bus (DB-backed, per-agent inbox)           │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼ SSE (TeamStreamEnvelope)
┌─────────────────────┐
│  Client / Frontend   │
│  multi-agent stream  │
└─────────────────────┘
```

核心组件：

- **AgentTeam**：Team 生命周期管理器，持有 leader + teammates 引用
- **TaskBoard**：共享任务列表，DB-backed，支持并发安全的 claim/release
- **TeamMessageBus**：队内消息投递，DB-backed，持久化到接收方 history
- **Team Tools**：注入到 leader 和 teammates 的协作工具集
- **TeamSSEMultiplexer**：多 agent 流式事件聚合，输出 `TeamStreamEnvelope`

### 详细设计

#### 1. 数据模型

##### 1.1 TeamModel（新增 SQLModel）

```python
class TeamModel(SQLModel, table=True):
    """Team metadata — one team per (user_id, session_id)."""

    __tablename__ = "teams"

    # 主键
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)

    # Leader
    leader_agent_id: str

    # 可用角色配置 (role_name -> agent_config_ref)
    # 所有 candidates 均可被实例化为一个或多个 team member
    candidates: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )

    # 配置
    max_teammates: int = Field(default=10)

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TeamMemberModel(SQLModel, table=True):
    """Teammate instance record."""

    __tablename__ = "team_members"

    # 主键
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    agent_id: str = Field(primary_key=True)  # e.g. "coder-1"

    # 角色
    role_name: str

    # Agent 独立 session_id（用于 history/state 隔离和 team 恢复）
    member_session_id: str = Field(default="")

    # 状态: idle | running | stopped
    status: str = Field(default="idle")

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
```

##### 1.2 TeamTaskModel（新增 SQLModel）

```python
class TeamTaskModel(SQLModel, table=True):
    """Shared task board for AgentTeam collaboration."""

    __tablename__ = "team_tasks"

    # 主键
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    task_id: str = Field(primary_key=True)  # 格式: "T-001"

    # 任务内容
    title: str
    description: str = ""
    priority: int = Field(default=0)  # 0=normal, 1=high, 2=critical

    # 状态机: pending -> in_progress -> completed
    status: str = Field(default="pending")  # pending | in_progress | completed

    # 依赖关系: 未全部 completed 的依赖视为 blocked
    dependencies: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )  # list of task_id

    # 分配
    assignee_agent_id: str | None = Field(default=None)

    # 元数据
    result_summary: str | None = Field(default=None)
    deliverable_path: str | None = Field(default=None)  # 相对路径: .nexau/tasks/{task_id}-{slug}.md
    created_by: str = ""  # agent_id of creator
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
```

##### 1.3 TeamTaskLockModel（新增 SQLModel）

```python
class TeamTaskLockModel(SQLModel, table=True):
    """Short-lived lock for task claim/update critical sections."""

    __tablename__ = "team_task_locks"

    # 主键 — 与 task 一一对应
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    task_id: str = Field(primary_key=True)

    # 锁持有者
    holder_id: str  # 格式: "{pid}:{uuid}"，沿用 AgentLockService 约定

    # TTL（由 TaskLockService 在创建时显式设置，无默认值）
    acquired_at_ns: int
    expires_at_ns: int
```

> 语义：仅用于 `claim_task` / `release_task` / `update_task_status` 的临界区保护（短持有 3–10s），不等价于"任务执行期锁"。

##### 1.4 TeamMessageModel（新增 SQLModel）

```python
class TeamMessageModel(SQLModel, table=True):
    """Persistent intra-team messages."""

    __tablename__ = "team_messages"

    # 主键
    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    message_id: str = Field(primary_key=True)  # UUID

    # 路由
    from_agent_id: str
    to_agent_id: str | None = Field(default=None)  # None = broadcast

    # 内容
    content: str
    message_type: str = Field(default="text")  # text | idle_notification

    # 投递状态
    delivered: bool = Field(default=False)
    delivered_at: datetime | None = Field(default=None)

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
```

#### 2. 锁机制：DB-backed TTL Lock

沿用 `AgentLockService` 的设计哲学，新增 `TaskLockService`：

```
acquire(task_id)
  │
  ├─ 查询 TeamTaskLockModel(task_id)
  │   ├─ 存在且 expires_at_ns > now → 返回冲突 (LockConflictError)
  │   └─ 不存在 / 已过期 → 创建锁记录
  │
  ├─ 执行临界区操作 (claim / update_status / release)
  │
  └─ 删除锁记录
```

设计要点：

- **TTL 短持有**：默认 5s，无需 heartbeat（临界区操作为单次 DB 写入）
- **无等待**：acquire 失败立即返回冲突，调用方重试选择其他任务
- **跨引擎兼容**：`InMemoryDatabaseEngine` / `SQLDatabaseEngine` / `RemoteDatabaseEngine` 均可工作
- **使用点**：`claim_task`、`release_task`、`update_task_status` 均在锁内执行

```python
class TaskLockService:
    """Short-lived DB-backed TTL lock for task operations."""

    def __init__(self, *, engine: DatabaseEngine, lock_ttl: float = 5.0) -> None:
        self._engine = engine
        self._lock_ttl = lock_ttl

    @asynccontextmanager
    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        team_id: str,
        task_id: str,
    ) -> AsyncGenerator[None, None]:
        # 1. 检查是否存在未过期锁
        existing = await self._find_valid_lock(user_id, session_id, team_id, task_id)
        if existing is not None:
            raise LockConflictError(f"Task {task_id} is locked by {existing.holder_id}")

        # 2. 创建锁记录
        holder_id = f"{os.getpid()}:{uuid4()}"
        now_ns = time.time_ns()
        lock = TeamTaskLockModel(
            user_id=user_id,
            session_id=session_id,
            team_id=team_id,
            task_id=task_id,
            holder_id=holder_id,
            acquired_at_ns=now_ns,
            expires_at_ns=now_ns + int(self._lock_ttl * 1_000_000_000),
        )
        await self._engine.create(lock)

        try:
            yield
        finally:
            # 3. 释放锁（仅删除自己持有的锁，防止竞争条件下误删他人锁）
            await self._engine.delete_where(
                TeamTaskLockModel,
                user_id=user_id,
                session_id=session_id,
                team_id=team_id,
                task_id=task_id,
                holder_id=holder_id,  # 必须匹配持有者
            )
```

#### 3. AgentTeam 生命周期

```
POST /team/stream (or /team/query)
  │
  ▼
AgentTeam.initialize()
  ├─ 从 TeamModel / TeamMemberModel 恢复 / 首次创建 team 记录
  ├─ 注册 DatabaseEngine models (TeamModel, TeamMemberModel, TeamTaskModel, TeamTaskLockModel, TeamMessageModel)
  ├─ 恢复已有 teammate 实例（若 session 中已存在）
  └─ 创建 Leader Agent 实例（注入 Team Tools，含 spawn_teammate）

AgentTeam.run(message)
  ├─ 保存主事件循环引用 (self._loop = asyncio.get_running_loop())
  ├─ Leader Agent 以 team_mode 执行（forever-run 循环）
  │   ├─ Leader 分析任务，调用 spawn_teammate(role_name) 按需实例化 teammates
  │   │   （spawn_teammate 通过 run_coroutine_threadsafe 在主循环上启动 teammate）
  │   ├─ Leader 调用 create_task() 创建任务
  │   ├─ Leader 调用 claim_task(task_id, assignee_agent_id) 分配任务给已 spawn 的 teammate
  │   │   （claim_task 通过 message 通知 teammate）
  │   ├─ Leader/Teammate 调用 message() / broadcast() 通信
  │   │   （消息通过 enqueue_message 即时注入目标 agent）
  │   └─ Leader 调用 finish_team(summary) 结束团队运行
  │       （finish_team 为 stop tool，触发 leader executor 退出循环）
  │
  ├─ Teammate Agent 并发执行（asyncio.run_coroutine_threadsafe）
  │   ├─ spawn_teammate 后立即以 forever-run 模式启动
  │   ├─ 每个 teammate 独立 Agent 实例 + 独立 executor
  │   ├─ 空闲时在 _message_available.wait() 上等待消息
  │   └─ 收到消息后唤醒，drain queued_messages，调用 LLM 处理
  │
  ├─ Leader 退出后，force_stop 所有 teammate
  └─ 持久化 team 元信息到 TeamModel / TeamMemberModel
```

##### 3.1 Teammate 管理

```python
class AgentTeam:
    """Team lifecycle manager."""

    def __init__(
        self,
        *,
        leader_config: AgentConfig,
        candidates: dict[str, AgentConfig],  # role_name -> config
        engine: DatabaseEngine,
        session_manager: SessionManager,
        user_id: str,
        session_id: str,
        max_teammates: int = 10,  # teammate 并发上限
    ) -> None: ...

    async def spawn_teammate(self, role_name: str) -> str:
        """Spawn a new teammate instance from candidates[role_name].

        Returns agent_id (e.g. "coder-1", "coder-2").
        Raises MaxTeammatesError if current active count >= max_teammates.
        """
        ...

    async def remove_teammate(self, agent_id: str) -> None:
        """Remove a teammate instance (marks idle, removes from active list)."""
        ...

    async def run(
        self,
        message: str,
        *,
        on_event: Callable[[TeamStreamEnvelope], None] | None = None,
    ) -> str:
        """Run the team: start leader, manage teammate lifecycle."""
        ...
```

- `candidates` 为初始化时预配置的 `dict[role_name, AgentConfig]`，仅定义可用角色，不预实例化
- Leader 通过 `spawn_teammate` tool 按需实例化 teammate（如分析任务后决定需要 2 个 coder + 1 个 reviewer）
- 同一 role 可多实例，通过 `agent_id` 寻址（如 `coder-1`, `coder-2`）
- 跨请求恢复时，已 spawn 的 teammate 从 `TeamMemberModel` 恢复

##### 3.1.1 Session 隔离

每个 agent（leader + teammates）拥有独立的 session，不共享 GlobalStorage 或 session context。
Agent 之间的所有通信仅通过 MessageBus 进行。

```
team_session_id  →  原始 session_id，仅用于 team 级数据（TeamModel, TaskBoard, MessageBus）
                    不直接传给任何 Agent

leader session   →  f"{team_session_id}:leader"
teammate session →  f"{team_session_id}:{agent_id}"   e.g. "sess-abc:code_agent-1"
```

设计要点：

- `AgentTeam._team_session_id` 仅用于 team 级数据 scope（TaskBoard、MessageBus、TeamModel 等）
- 每个 Agent 构造时传入独立的 `session_id`，拥有独立的 SessionModel（history、GlobalStorage、context）
- `TeamMemberModel.member_session_id` 记录每个 agent 的独立 session_id，用于 team 恢复
- 不传 `global_storage=` 参数，每个 Agent 自行创建独立的 GlobalStorage
- 此设计为未来分布式 teammate 提供基础：每个 agent 可独立部署，仅通过 MessageBus 通信

##### 3.2 Teammate 并发执行模型 — Forever-Run 模式

每个 teammate 以 "forever run" 模式运行：agent 的 executor 在 team_mode 下持续循环，
空闲时等待消息，所有通信通过 `enqueue_message` 实现。Agent 仅在 `force_stop()` 被调用时退出。

###### 3.2.1 Executor team_mode 扩展

`Executor` 新增以下支持：

- `team_mode: bool = False` — 启用永久运行循环
- `_message_available = threading.Event()` — 跨线程消息唤醒信号
- `enqueue_message()` 调用 `_message_available.set()` 唤醒等待中的循环
- `force_stop()` — 设置 `stop_signal` 并唤醒 `_message_available`

```
Executor loop (team_mode):
  while iteration < max_iterations:
    1. Check stop_signal → break
    2. Drain queued_messages → append to history
    3. LLM call → parse → execute tools
    4. If should_stop (no tool calls) and no stop_tool_result:
       → while not stop_signal and no queued_messages:
           _message_available.clear()
           _message_available.wait(timeout=30)
       → continue (re-enter loop to drain messages and call LLM)
    5. If stop_tool_result → break (finish_team triggered)
```

关键设计：team_mode 下非 stop_tool 触发的 `should_stop` 不退出循环，
而是在 while 循环中持续等待新消息或 `stop_signal`。
必须在内层 while 中等待，避免超时后以 assistant 消息结尾调用 LLM（LLM 会返回空响应）。

###### 3.2.2 跨线程 asyncio 调度

`spawn_teammate` 在 tool executor 的线程中执行。Tool 的 async 函数通过
`asyncio.run()` 在临时事件循环中运行（见 `Tool.execute()` line 296）。
**不能使用 `asyncio.create_task()`** — 该 task 会随临时循环销毁，teammate 永远不会启动。

解决方案：`AgentTeam.run()` 保存主事件循环引用，`spawn_teammate` 通过
`asyncio.run_coroutine_threadsafe()` 调度到主循环：

```python
# AgentTeam.run() 中保存主循环引用
self._loop = asyncio.get_running_loop()

# spawn_teammate 中跨线程调度
teammate_future = asyncio.run_coroutine_threadsafe(
    self._run_teammate_forever(agent_id),
    self._loop,
)
self._teammate_futures[agent_id] = teammate_future  # concurrent.futures.Future
```

使用 `concurrent.futures.Future`（而非 `asyncio.Task`）跟踪 teammate 运行状态。
`remove_teammate` 和 `_stop_all_teammates` 通过 `future.cancel()` + `future.result(timeout=...)` 管理生命周期。

###### 3.2.3 Teammate 永久运行

```python
async def _run_teammate_forever(self, agent_id: str) -> None:
    """Run teammate in forever-run mode. Exits only on force_stop."""
    agent = self._teammate_agents[agent_id]
    await self._update_member_status(agent_id, "running")
    self._watchdog.register(agent_id)
    try:
        await agent.run_async(
            message="You have been activated. Wait for task assignments and messages from the leader."
        )
    except Exception as e:
        logger.error(f"Teammate {agent_id} exited with error: {e}")
    finally:
        await self._update_member_status(agent_id, "idle")
        self._watchdog.unregister(agent_id)
```

##### 3.3 队内消息投递

```
Agent A 调用 message(to_agent_id="B", content="...")
  │
  ▼
TeamMessageBus.send(from="A", to="B", content="...")
  ├─ 写入 TeamMessageModel (持久化)
  │
  ▼
AgentTeam.message(to="B", content, from="A")
  ├─ 构造 system message: "[Team Message from A]: ..."
  ├─ 调用 agent.enqueue_message(msg)
  │   ├─ 追加到 executor.queued_messages
  │   └─ executor._message_available.set()  ← 唤醒等待中的 forever-run 循环
  │
  ▼
Agent B 的 executor 被唤醒
  ├─ 退出 _message_available.wait()
  ├─ 循环顶部 drain queued_messages → 追加到 history
  └─ 调用 LLM（history 中包含新消息）
```

关键约束：

- **即时唤醒**：消息通过 `_message_available.set()` 立即唤醒目标 agent，无需等待迭代边界
- **持久化**：所有消息写入 `TeamMessageModel`，跨请求可恢复
- **注入方式**：通过 `executor.enqueue_message()` 直接注入，替代原有的 middleware drain 方案

```python
def message(
    self,
    to_agent_id: str,
    content: str,
    from_agent_id: str,
) -> None:
    """Enqueue a message to a teammate or leader agent.

    RFC-0002: 向 teammate 或 leader 注入消息

    消息通过 executor.enqueue_message 注入，
    executor 的 _message_available 事件会唤醒等待中的永久运行循环。
    """
    enqueue_text = f"[Team Message from {from_agent_id}]: {content}"
    msg = {"role": "system", "content": enqueue_text}

    if to_agent_id == self._leader_agent_id:
        if self._leader_agent is not None:
            self._leader_agent.enqueue_message(msg)
    else:
        agent = self._teammate_agents.get(to_agent_id)
        if agent is not None:
            agent.enqueue_message(msg)
```

##### 3.4 全员空闲通知

Teammate 完成任务后进入 idle 等待状态。当所有 teammate 均处于 idle 时，watchdog 自动通知 leader：

```
所有 teammate 进入 idle 等待
  │
  ▼
TeammateWatchdog._check_all_idle() → True
  │
  ▼
TeammateWatchdog._notify_leader(
    "[All Idle] All agents are idle. Review task board status and decide next steps — assign new tasks, check results.",
    "watchdog"
)
  │
  ▼
Leader 收到通知，决策下一步（分配新任务 / 检查结果 / 调用 finish_team）
```

##### 3.5 Teammate Watchdog

`AgentTeam` 内置 watchdog 机制，检测全员空闲死锁（所有 agent 均在等待消息、无任何进展）：

```python
@dataclass(frozen=True)
class WatchdogConfig:
    """Idle detection configuration."""
    idle_check_interval_seconds: float = 10.0  # 检查间隔


class TeammateWatchdog:
    """Detects all-idle deadlock among teammates.

    RFC-0002: 全员空闲检测

    以独立 asyncio.Task 运行，定期检查所有 running 状态的 teammate。
    若所有 teammate 均处于 idle（等待消息），向 leader 发送全员空闲通知。
    Leader 可据此决定是否分配新任务、检查结果或调用 finish_team。
    """

    def __init__(
        self,
        *,
        config: WatchdogConfig,
        check_all_idle: Callable[[], bool] | None = None,
        notify_leader: Callable[[str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._start_times: dict[str, float] = {}  # agent_id -> start_timestamp
        self._stopped = False
        self._check_all_idle = check_all_idle    # 检查所有 agent 是否 idle 的回调
        self._notify_leader = notify_leader      # 通知 leader 的回调

    def stop(self) -> None:
        """Signal the watchdog loop to exit."""
        self._stopped = True

    def register(self, agent_id: str) -> None:
        """Register a teammate as running (called on spawn)."""
        self._start_times[agent_id] = time.monotonic()

    def unregister(self, agent_id: str) -> None:
        """Unregister a teammate (called on idle/stop)."""
        self._start_times.pop(agent_id, None)

    async def run(self) -> None:
        """Watchdog loop — runs as background asyncio.Task."""
        while not self._stopped:
            await asyncio.sleep(self._config.idle_check_interval_seconds)
            # 全员空闲检测 — 仅在有注册 teammate 时才唤醒 leader
            if self._check_all_idle is not None and self._notify_leader is not None and len(self._start_times) > 0:
                if self._check_all_idle():
                    self._notify_leader(
                        "[All Idle] All agents are idle. Review task board status and decide next steps — assign new tasks, check results.",
                        "watchdog",
                    )
```

Watchdog 在 `AgentTeam.run()` 中作为后台任务启动，team run 结束时取消：

```python
# AgentTeam.run() 中
watchdog_task = asyncio.create_task(self._watchdog.run())
try:
    result = await self._run_leader(message)
finally:
    watchdog_task.cancel()
```

与 RFC 初稿的差异：

- **初稿设计**：监控单个 teammate 超时（5 分钟），超时后发送 per-agent 超时通知
- **实际实现**：检测全员空闲死锁（所有 agent 均在等待），发送全员空闲通知
- **原因**：全员空闲死锁是更常见的卡死场景（leader 等 teammate，teammate 等 leader），单 agent 超时监控在 LLM 调用慢时会产生误报

Leader 收到全员空闲通知后，可选择：
- 调用 `list_tasks()` 检查任务状态，分配新任务
- 调用 `finish_team(summary)` 结束团队运行
- 发送 `message(to_agent_id, "...")` 催促特定 teammate

#### 4. AgentTeamState

扩展 `AgentState`，将所有 team 相关依赖作为强类型字段暴露，避免 `get_context_value` 的 `Any` 返回值：

```python
class AgentTeamState(AgentState):
    """Extended AgentState with typed team collaboration context.

    RFC-0002: Team 上下文强类型扩展

    所有 Team Tools 通过此类型获取 team 相关依赖，
    避免 get_context_value 的动态类型访问。
    """

    team: AgentTeam
    task_board: TaskBoard
    message_bus: TeamMessageBus
    is_leader: bool
```

##### 4.1 Team Tool 结果类型

所有 Tool 返回值使用强类型 dataclass，避免 `dict[str, str]` 的模糊语义：

```python
@dataclass(frozen=True)
class TeammateInfo:
    """Teammate instance info."""
    agent_id: str
    role_name: str
    status: str  # idle | running | stopped


@dataclass(frozen=True)
class TaskInfo:
    """Task board entry."""
    task_id: str
    title: str
    description: str
    status: str  # pending | in_progress | completed
    priority: int
    dependencies: list[str]
    assignee_agent_id: str | None
    result_summary: str | None
    created_by: str
    is_blocked: bool  # 依赖未全部 completed
    deliverable_path: str | None  # 可交付物路径


@dataclass(frozen=True)
class SpawnResult:
    """spawn_teammate 返回值。"""
    agent_id: str
    role_name: str


@dataclass(frozen=True)
class RemoveTeammateResult:
    """remove_teammate 返回值。"""
    agent_id: str
    role_name: str


@dataclass(frozen=True)
class CreateTaskResult:
    """create_task 返回值。"""
    task_id: str
    title: str
    description: str
    priority: int
    status: str
    deliverable_path: str  # 任务可交付物文件路径


@dataclass(frozen=True)
class ClaimTaskResult:
    """claim_task 返回值。"""
    task_id: str
    title: str
    status: str
    assignee_agent_id: str
    deliverable_path: str | None  # 可交付物路径（若已创建）


@dataclass(frozen=True)
class UpdateTaskStatusResult:
    """update_task_status 返回值。"""
    task_id: str
    title: str
    status: str
    result_summary: str | None = None


@dataclass(frozen=True)
class ReleaseTaskResult:
    """release_task 返回值。"""
    task_id: str
    title: str
    status: str


@dataclass(frozen=True)
class MessageResult:
    """message / broadcast 返回值。"""
    message_id: str
    delivered_to: list[str]  # 目标 agent_id 列表


@dataclass(frozen=True)
class FinishTeamResult:
    """finish_team 返回值。"""
    summary: str
    completed_tasks: int
    total_tasks: int


@dataclass(frozen=True)
class ToolError:
    """Tool 操作失败时的错误返回。"""
    error: str
    code: str  # permission_denied | conflict | blocked | not_found | invalid_state | busy
    status: str = "error"  # 固定为 "error"，用于 executor stop-tool 守卫
```

框架在创建 team agent 时构造 `AgentTeamState` 替代普通 `AgentState`：

```python
# AgentTeam.initialize() 中
team_state = AgentTeamState(
    # 继承 AgentState 的所有参数
    agent_name=agent_name,
    agent_id=agent_id,
    run_id=run_id,
    root_run_id=root_run_id,
    context=context,
    global_storage=global_storage,
    executor=executor,
    # Team 扩展字段
    team=self,
    task_board=self._task_board,
    message_bus=self._message_bus,
    is_leader=(agent_id == self._leader_agent_id),
)
```

> Tool 函数签名中使用 `agent_state: AgentTeamState`，框架自动注入时按类型匹配。

#### 5. Team Tools

所有 Team Tools 通过 `AgentTeamState` 的强类型字段获取 team 上下文。Leader 和 Teammate 注入不同的工具子集。

##### 5.1 工具清单

| 工具名 | Leader | Teammate | 说明 |
|--------|--------|----------|------|
| `spawn_teammate` | ✅ | ❌ | 从 candidates 实例化 teammate（leader-only） |
| `remove_teammate` | ✅ | ❌ | 移除 teammate 实例（leader-only） |
| `message` | ✅ | ✅ | 点对点消息 |
| `broadcast` | ✅ | ✅ | 广播消息 |
| `list_teammates` | ✅ | ✅ | 列出所有 teammate 及状态 |
| `list_tasks` | ✅ | ✅ | 列出任务列表（含状态、依赖、分配） |
| `create_task` | ✅ | ❌ | 创建任务（leader-only） |
| `claim_task` | ✅ | ✅ | 领取/分配任务（leader 可指定 assignee，teammate 只能 self-claim）|
| `update_task_status` | ✅ | ✅ | 更新任务状态 |
| `release_task` | ✅ | ✅ | 释放任务（取消分配） |
| `finish_team` | ✅ | ❌ | 结束团队运行（leader-only stop tool） |

##### 5.2 工具定义

```python
# --- spawn_teammate (leader-only) ---
async def spawn_teammate(
    role_name: str,
    agent_state: AgentTeamState,
) -> SpawnResult | ToolError:
    """Spawn a new teammate instance from candidates.

    RFC-0002: 从 candidates 动态实例化 teammate

    Leader 根据任务需求调用此工具，从预配置的 candidates 中
    实例化一个新的 teammate agent。同一 role 可多次 spawn
    产生多个实例（如 coder-1, coder-2）。

    Teammate 实例化后立即以独立 asyncio.Task 启动运行，
    等待通过 message 或 claim_task 接收工作指令。
    """
    if not agent_state.is_leader:
        return ToolError(error="Only leader can spawn teammates", code="permission_denied")

    try:
        agent_id = await agent_state.team.spawn_teammate(role_name)
    except MaxTeammatesError:
        return ToolError(
            error=f"Max teammates limit reached ({agent_state.team.max_teammates})",
            code="invalid_state",
        )
    return SpawnResult(agent_id=agent_id, role_name=role_name)


# --- remove_teammate (leader-only) ---
async def remove_teammate(
    agent_id: str,
    agent_state: AgentTeamState,
) -> RemoveTeammateResult | ToolError:
    """Remove a teammate instance.

    RFC-0002: 移除 teammate 实例

    仅可移除 idle 状态的 teammate。正在执行任务的 teammate
    需先完成或被 stop 后才能移除。
    """
    if not agent_state.is_leader:
        return ToolError(error="Only leader can remove teammates", code="permission_denied")

    await agent_state.team.remove_teammate(agent_id)
    return RemoveTeammateResult(agent_id=agent_id, role_name=...)


# --- message ---
async def message(
    to_agent_id: str,
    content: str,
    agent_state: AgentTeamState,
) -> MessageResult:
    """Send a message to a specific teammate.

    RFC-0002: 队内点对点消息

    消息将在目标 agent 的下一次迭代边界注入其上下文。
    """
    msg = await agent_state.message_bus.send(
        from_agent_id=agent_state.agent_id,
        to_agent_id=to_agent_id,
        content=content,
    )
    return MessageResult(message_id=msg.message_id, delivered_to=[to_agent_id])


# --- broadcast ---
async def broadcast(
    content: str,
    agent_state: AgentTeamState,
) -> MessageResult:
    """Broadcast a message to all teammates.

    RFC-0002: 队内广播消息
    """
    msg, recipients = await agent_state.message_bus.broadcast(
        from_agent_id=agent_state.agent_id,
        content=content,
    )
    return MessageResult(message_id=msg.message_id, delivered_to=recipients)


# --- list_teammates ---
async def list_teammates(
    agent_state: AgentTeamState,
) -> list[TeammateInfo]:
    """List all teammate agents and their current status.

    RFC-0002: 列出队友
    """
    return agent_state.team.get_teammate_info()


# --- list_tasks ---
async def list_tasks(
    status: str | None = None,
    agent_state: AgentTeamState = ...,
) -> list[TaskInfo]:
    """List tasks on the shared task board.

    RFC-0002: 列出共享任务列表

    Args:
        status: 可选过滤条件 (pending / in_progress / completed)
    """
    return await agent_state.task_board.list_tasks(status=status)


# --- create_task (leader-only) ---
async def create_task(
    title: str,
    description: str = "",
    priority: int = 0,
    dependencies: list[str] | None = None,
    agent_state: AgentTeamState = ...,
) -> CreateTaskResult | ToolError:
    """Create a new task on the shared task board.

    RFC-0002: 创建任务（仅 leader 可调用）
    """
    if not agent_state.is_leader:
        return ToolError(error="Only leader can create tasks", code="permission_denied")

    task = await agent_state.task_board.create_task(
        title=title,
        description=description,
        priority=priority,
        dependencies=dependencies or [],
        created_by=agent_state.agent_id,
    )
    return CreateTaskResult(
        task_id=task.task_id,
        title=task.title,
        description=task.description,
        priority=task.priority,
        status=task.status,
        deliverable_path=task.deliverable_path,
    )


# --- claim_task ---
async def claim_task(
    task_id: str,
    assignee_agent_id: str | None = None,
    agent_state: AgentTeamState = ...,
) -> ClaimTaskResult | ToolError:
    """Claim a task from the shared task board.

    RFC-0002: 领取/分配任务

    - task_id 必须显式指定（禁止 claim-next）
    - assignee_agent_id 为空时 self-claim（leader 和 teammate 均可）
    - assignee_agent_id 非空时为 leader assignment（校验 caller 为 leader）
    - 单任务约束：每个 agent 同时只能持有一个 in_progress 任务

    Teammate 自助领取流程: list_tasks() → 选择 task_id → claim_task(task_id)
    若 claim 冲突则重试选择其他任务。
    """
    caller_id = agent_state.agent_id
    actual_assignee = assignee_agent_id or caller_id

    # leader assignment 校验
    if assignee_agent_id is not None and not agent_state.is_leader:
        return ToolError(error="Only leader can assign tasks to others", code="permission_denied")

    # 单任务约束：检查 assignee 是否已有 in_progress 任务
    active_tasks = await agent_state.task_board.list_tasks(status="in_progress")
    existing = [t for t in active_tasks if t.assignee_agent_id == actual_assignee]
    if existing:
        current = existing[0]
        return ToolError(
            error=f"{actual_assignee} already has an active task: {current.task_id} ({current.title}). Finish or release it before claiming a new one.",
            code="busy",
        )

    try:
        await agent_state.task_board.claim_task(
            task_id=task_id, assignee_agent_id=actual_assignee,
        )

        # leader assignment 时通过 enqueue_message 通知 teammate
        if assignee_agent_id is not None:
            agent_state.team.send_message_to_agent(
                actual_assignee,
                f"Task assigned: {task_id}. Use list_tasks to see details and work on it.",
                agent_state.agent_id,
            )

        task_info = await agent_state.task_board.get_task_info(task_id)
        return ClaimTaskResult(
            task_id=task_id,
            title=task_info.title,
            status="claimed",
            assignee_agent_id=actual_assignee,
            deliverable_path=task_info.deliverable_path,
        )
    except LockConflictError:
        return ToolError(error=f"Task {task_id} claim conflict, retry with another task", code="conflict")
    except TaskBlockedError:
        return ToolError(error=f"Task {task_id} is blocked by unfinished dependencies", code="blocked")


# --- update_task_status ---
async def update_task_status(
    task_id: str,
    status: str,
    result_summary: str | None = None,
    agent_state: AgentTeamState = ...,
) -> UpdateTaskStatusResult | ToolError:
    """Update task status (pending -> in_progress -> completed).

    RFC-0002: 更新任务状态
    """
    task_info = await agent_state.task_board.update_status(
        task_id=task_id,
        status=status,
        result_summary=result_summary,
    )
    return UpdateTaskStatusResult(task_id=task_id, title=task_info.title, status=status, result_summary=result_summary)


# --- release_task ---
async def release_task(
    task_id: str,
    agent_state: AgentTeamState = ...,
) -> ReleaseTaskResult | ToolError:
    """Release a claimed task (unassign).

    RFC-0002: 释放任务
    """
    task_info = await agent_state.task_board.release_task(task_id=task_id)
    return ReleaseTaskResult(task_id=task_id, title=task_info.title, status="released")


# --- finish_team (leader-only stop tool) ---
async def finish_team(
    summary: str,
    agent_state: AgentTeamState = ...,
) -> FinishTeamResult | ToolError:
    """Finish the team run and return a summary.

    RFC-0002: 结束团队运行（leader-only stop tool）

    Leader 在所有工作完成后调用此工具结束团队运行。
    注册为 executor 的 stop tool，触发 leader 的 forever-run 循环退出。
    Leader 退出后 AgentTeam.run() 会 force_stop 所有 teammate。
    """
    if not agent_state.is_leader:
        return ToolError(error="Only the team leader can finish the team", code="permission_denied")

    all_tasks = await agent_state.task_board.list_tasks()
    completed = [t for t in all_tasks if t.status == "completed"]
    return FinishTeamResult(
        summary=summary,
        completed_tasks=len(completed),
        total_tasks=len(all_tasks),
    )
```

##### 5.3 工具注入

Team Tools 在 `AgentTeam.initialize()` 时动态注入到各 agent：

```python
# Leader: 注入全部工具（含 spawn/remove teammate + finish_team stop tool）
leader_tools = [
    spawn_teammate_tool, remove_teammate_tool,
    message_tool, broadcast_tool, list_teammates_tool,
    list_tasks_tool, create_task_tool, claim_task_tool,
    update_task_status_tool, release_task_tool,
    finish_team_tool,  # stop tool — 触发 leader executor 退出
]

# Teammate: 注入协作工具（无 spawn/remove/create_task/finish_team）
teammate_tools = [
    message_tool, broadcast_tool, list_teammates_tool,
    list_tasks_tool, claim_task_tool,
    update_task_status_tool, release_task_tool,
]
```

通过 `AgentConfig.tools` 扩展注入。框架在构造 agent 时创建 `AgentTeamState`（而非普通 `AgentState`），Team Tools 的 `agent_state: AgentTeamState` 参数由框架自动注入。

#### 5. Multi-Agent SSE

##### 5.1 TeamStreamEnvelope

Team SSE 不复用原 `/stream` 的裸 Event 输出，定义独立的封装格式：

```python
class TeamStreamEnvelope(BaseModel):
    """Envelope for multi-agent SSE events."""

    team_id: str
    agent_id: str
    role_name: str | None = None
    run_id: str | None = None
    event: Event  # 原始 Event payload (TextMessageContentEvent, etc.)
```

SSE 输出示例：

```
data: {"team_id":"team_abc","agent_id":"leader-001","role_name":"leader","run_id":"run_x1","event":{"type":"TEXT_MESSAGE_CONTENT","delta":"Let me "}}

data: {"team_id":"team_abc","agent_id":"coder-1","role_name":"coder","run_id":"run_x2","event":{"type":"TEXT_MESSAGE_CONTENT","delta":"def hello"}}

data: {"team_id":"team_abc","agent_id":"reviewer-1","role_name":"reviewer","run_id":"run_x3","event":{"type":"TEXT_MESSAGE_CONTENT","delta":"LGTM"}}
```

UI 可直接按 `agent_id` 分栏显示 token 流。

##### 5.2 TeamSSEMultiplexer

```python
class TeamSSEMultiplexer:
    """Multiplexes events from multiple agents into a single SSE stream."""

    def __init__(self, *, team_id: str) -> None:
        self._team_id = team_id
        self._queue: asyncio.Queue[TeamStreamEnvelope | None] = asyncio.Queue()

    def create_event_handler(
        self, agent_id: str, role_name: str
    ) -> Callable[[Event], None]:
        """Create an on_event callback for a specific agent.

        Each agent gets its own handler that wraps events in TeamStreamEnvelope.
        """
        def handler(event: Event) -> None:
            envelope = TeamStreamEnvelope(
                team_id=self._team_id,
                agent_id=agent_id,
                role_name=role_name,
                run_id=getattr(event, "run_id", None),
                event=event,
            )
            self._queue.put_nowait(envelope)
        return handler

    async def stream(self) -> AsyncGenerator[TeamStreamEnvelope, None]:
        """Yield envelopes as they arrive from any agent."""
        while True:
            envelope = await self._queue.get()
            if envelope is None:  # Sentinel: team run complete
                break
            yield envelope

    def close(self) -> None:
        """Signal end of stream."""
        self._queue.put_nowait(None)
```

##### 5.3 并发安全保证

现有 `AgentEventsMiddleware` 内部的 aggregator 是单实例，多 agent 并发会导致串流。解决方案：

- **每个 agent/run 使用独立的 `AgentEventsMiddleware` 实例**
- 每个 middleware 实例持有独立的 aggregator 状态
- 所有 middleware 共享同一个 `TeamSSEMultiplexer` 作为事件出口
- 多个 agent 并发时事件可交错，但 envelope 中的 `agent_id` 保证正确

```python
# 为每个 teammate 创建独立的 middleware
for agent_id, role_name in teammate_instances:
    handler = multiplexer.create_event_handler(agent_id, role_name)
    middleware = AgentEventsMiddleware(
        session_id=session_id,
        on_event=handler,  # 独立 handler → 独立 aggregator
    )
    # 应用到该 agent 的 config
    agent_config_with_middleware = apply_middleware(agent_config, middleware)
```

#### 6. HTTP 新 Endpoints

所有 Team endpoints 挂载在 `/team` 前缀下，与现有 `/stream`、`/query` 隔离。

##### 6.1 Endpoint 清单

| Method | Path | 说明 | 请求体 | 响应 |
|--------|------|------|--------|------|
| `POST` | `/team/stream` | 运行 leader + multiplex SSE | `TeamRunRequest` | SSE `TeamStreamEnvelope` |
| `POST` | `/team/query` | 运行 leader（同步返回） | `TeamRunRequest` | `dict[str, str]` |
| `GET` | `/team/teammates` | 列出 teammates | query: `user_id`, `session_id` | `list[dict]` |
| `GET` | `/team/tasks` | 列出任务 | query: `user_id`, `session_id`, `?status=` | `list[dict]` |
| `POST` | `/team/tasks` | 创建任务 | `CreateTaskRequest` | `dict` |
| `POST` | `/team/tasks/claim` | 领取任务 | `ClaimTaskRequest` | `dict` |
| `PATCH` | `/team/tasks/{task_id}` | 更新任务状态 | `UpdateTaskRequest` + query: `user_id`, `session_id` | `dict` |
| `POST` | `/team/message` | 发送队内消息 | `SendMessageRequest` | `dict` |
| `POST` | `/team/user-message` | 向 agent 注入用户消息（stream 期间） | `UserMessageRequest` | `dict` |
| `POST` | `/team/stop` | 强制停止整个 Team | `StopTeamRequest` | `dict` |
| `GET` | `/team/status` | 查询 Team 运行状态 | query: `user_id`, `session_id` | `dict` |
| `GET` | `/team/subscribe` | 重连 SSE（前端刷新后订阅新事件） | query: `user_id`, `session_id`, `after` | SSE `TeamStreamEnvelope` |

##### 6.2 请求/响应模型

```python
class TeamRunRequest(BaseModel):
    """Request to run the team."""
    user_id: str
    session_id: str
    message: str


class TeamStreamEnvelopeResponse(BaseModel):
    """SSE event wrapper."""
    type: str = "team_event"  # team_event | complete | error
    envelope: dict[str, object] | None = None  # 序列化后的 TeamStreamEnvelope
    session_id: str
    error: str | None = None  # 仅 type="error" 时填充


class CreateTaskRequest(BaseModel):
    user_id: str
    session_id: str
    title: str
    description: str = ""
    priority: int = 0
    dependencies: list[str] = []


class ClaimTaskRequest(BaseModel):
    user_id: str
    session_id: str
    task_id: str  # 必须显式指定
    assignee_agent_id: str | None = None  # None = self-claim by leader


class UpdateTaskRequest(BaseModel):
    status: str  # pending | in_progress | completed
    result_summary: str | None = None


class SendMessageRequest(BaseModel):
    user_id: str
    session_id: str
    from_agent_id: str
    to_agent_id: str | None = None  # None = broadcast
    content: str


class UserMessageRequest(BaseModel):
    """Request to enqueue a user message to an agent during streaming."""
    user_id: str
    session_id: str
    content: str
    to_agent_id: str = "leader"  # 默认发给 leader


class StopTeamRequest(BaseModel):
    """Request to force-stop all agents in a team."""
    user_id: str
    session_id: str
```

##### 6.3 SSE Streaming 实现

```python
# POST /team/stream
@router.post("/team/stream")
async def team_stream(request: TeamRunRequest) -> StreamingResponse:
    team = registry.get_or_create(request.user_id, request.session_id)

    # 运行结束后从注册表移除（通过 on_complete 回调）
    team.set_on_complete(lambda: registry.remove(request.user_id, request.session_id))

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for envelope in team.run_streaming(request.message):
                response = TeamStreamEnvelopeResponse(
                    type="team_event",
                    envelope=envelope.model_dump(),
                    session_id=request.session_id,
                )
                yield f"data: {response.model_dump_json()}\n\n"
        except Exception as exc:
            error_response = TeamStreamEnvelopeResponse(
                type="error",
                session_id=request.session_id,
                error=str(exc),
            )
            yield f"data: {error_response.model_dump_json()}\n\n"
            return

        # 完成信号
        yield f"data: {TeamStreamEnvelopeResponse(type='complete', session_id=request.session_id).model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
```

### 示例

#### 典型工作流：Leader 分配任务给 Teammates

```python
# 1. 用户发起 team 请求
# POST /team/stream
# { "user_id": "u1", "session_id": "s1", "message": "实现一个 TODO 应用" }

# 2. Leader 分析任务，按需 spawn teammates（此时 team 内无任何 teammate）
spawn_teammate(role_name="coder")
# → SpawnResult(agent_id="coder-1", role_name="coder")
spawn_teammate(role_name="coder")
# → SpawnResult(agent_id="coder-2", role_name="coder")
spawn_teammate(role_name="reviewer")
# → SpawnResult(agent_id="reviewer-1", role_name="reviewer")

# 3. Leader 创建任务
create_task(title="实现后端 API", description="FastAPI CRUD endpoints", priority=1)
# → CreateTaskResult(task_id="T-001", title="实现后端 API", status="created", ...)

create_task(title="实现前端页面", description="React TODO list", dependencies=["T-001"])
# → CreateTaskResult(task_id="T-002", title="实现前端页面", status="created", ...)

# 4. Leader 分配任务给已 spawn 的 teammate
claim_task(task_id="T-001", assignee_agent_id="coder-1")
# → ClaimTaskResult(task_id="T-001", title="实现后端 API", status="claimed", assignee_agent_id="coder-1", ...)

# 5. coder-1 开始执行 T-001
# coder-1 完成后:
update_task_status(task_id="T-001", status="completed", result_summary="API done")
# → UpdateTaskStatusResult(task_id="T-001", title="实现后端 API", status="completed")
# → Watchdog 检测到全员空闲，通知 leader

# 6. T-002 依赖解除，coder-2 自助领取:
list_tasks(status="pending")
# → [TaskInfo(task_id="T-002", status="pending", is_blocked=False, ...)]
claim_task(task_id="T-002")
# → ClaimTaskResult(task_id="T-002", title="实现前端页面", status="claimed", assignee_agent_id="coder-2", ...)
```

#### Teammate 自助领取流程

```
Teammate                          TaskBoard
   │                                  │
   ├─ list_tasks() ──────────────────▶│
   │◀─ [TaskInfo(T-003, pending)] ───│
   │                                  │
   ├─ claim_task("T-003") ──────────▶│
   │   ├─ acquire lock ──────────────▶│ (TTL 5s)
   │   ├─ check dependencies ────────│
   │   ├─ check not assigned ────────│
   │   ├─ set assignee ─────────────▶│
   │   └─ release lock ─────────────▶│
   │◀─ ClaimTaskResult(claimed) ────────│
   │                                  │
   │  (若冲突)                        │
   │◀─ ToolError(conflict) ──────────│
   │                                  │
   ├─ list_tasks() ──────────────────▶│  (重试选择其他任务)
   └─ claim_task("T-004") ──────────▶│
```

## 权衡取舍

### 考虑过的替代方案

#### 方案 A：基于 GlobalStorage 的任务管理

将任务列表存储在 `SessionModel.storage`（GlobalStorage）中。

**未采用原因**：
- GlobalStorage 是整块覆盖式持久化（last-write-wins），并发写会丢失数据
- 无法提供行级锁，claim 操作无法保证原子性
- 不兼容未来分布式扩展

#### 方案 B：基于文件锁的任务锁

使用文件系统锁（flock）实现任务 claim 互斥。

**未采用原因**：
- 仅适用于单机部署
- 不兼容 `InMemoryDatabaseEngine`（测试场景）和 `RemoteDatabaseEngine`（分布式场景）
- 与现有 `AgentLockService` 的 DB-backed 设计哲学不一致

#### 方案 C：复用现有 `/stream` endpoint 做 multi-agent SSE

在现有 SSE 事件中添加 `agent_id` 字段。

**未采用原因**：
- 破坏旧客户端兼容性
- 现有 Event 类型不含 team 上下文信息
- 独立 endpoint + envelope 封装更清晰，不影响单 agent 场景

### 缺点

1. **新增 5 张 DB 表**：`TeamModel`、`TeamMemberModel`、`TeamTaskModel`、`TeamTaskLockModel`、`TeamMessageModel`，增加了 schema 复杂度，但这是并发安全和数据独立性的必要代价
2. **`concurrent.futures.Future` vs `asyncio.Task`**：跨线程调度使用 `run_coroutine_threadsafe` 返回 `Future` 而非 `Task`，API 不同（`future.result(timeout=...)` vs `await task`），但这是 tool executor 使用 `asyncio.run()` 临时事件循环的必然结果
3. **单进程并发上限**：asyncio 单线程模型下，teammate 数量受限于 LLM API 并发能力
4. **全员空闲检测而非单 agent 超时**：Watchdog 仅检测全员空闲死锁，不监控单个 agent 的执行时长。若某个 teammate 长时间执行但未完成，leader 无法感知（除非 teammate 主动发消息）

## 实现计划

### 阶段划分

- [x] Phase 1: 数据模型与基础服务
  - 新增 `TeamModel`、`TeamMemberModel`、`TeamTaskModel`、`TeamTaskLockModel`、`TeamMessageModel`
  - 实现 `TaskLockService`（沿用 AgentLockService 模式）
  - 实现 `TaskBoard`（CRUD + claim/release + 依赖检查）
  - 实现 `TeamMessageBus`（send/broadcast/drain）
  - 单元测试覆盖

- [x] Phase 2: AgentTeam 核心 + Forever-Run 模式
  - 实现 `AgentTeam` 生命周期管理
  - Executor `team_mode` 扩展（`_message_available`, `force_stop()`）
  - Agent 传递 `team_mode=self._team_state is not None` 给 Executor
  - 跨线程调度：`asyncio.run_coroutine_threadsafe` 替代 `asyncio.create_task`
  - 消息投递：`enqueue_message` + `message` 替代 middleware drain
  - 实现 Teammate 永久运行模型（`_run_teammate_forever`）
  - 实现 `finish_team` stop tool（leader-only）
  - 集成测试

- [x] Phase 3: Team Tools
  - 实现全部 11 个 Team Tools（含 spawn/remove teammate + finish_team）
  - Tool YAML 定义 + binding
  - Leader/Teammate 差异化注入
  - 工具级单元测试

- [x] Phase 4: Multi-Agent SSE
  - 实现 `TeamStreamEnvelope` 模型
  - 实现 `TeamSSEMultiplexer`
  - 确保每个 agent/run 独立 aggregator 实例
  - SSE 流式输出集成测试

- [x] Phase 5: HTTP Endpoints
  - 实现 `/team/*` 路由
  - 请求/响应模型
  - 端到端测试

### 相关文件

- `nexau/archs/session/models/` - 新增 TeamModel, TeamMemberModel, TeamTaskModel, TeamTaskLockModel, TeamMessageModel
- `nexau/archs/session/task_lock_service.py` - 新增 TaskLockService
- `nexau/archs/main_sub/team/` - 新增 AgentTeam, TaskBoard, TeamMessageBus
- `nexau/archs/main_sub/team/tools/` - 新增 Team Tools（含 finish_team stop tool）
- `nexau/archs/main_sub/team/sse/` - 新增 TeamSSEMultiplexer, TeamStreamEnvelope
- `nexau/archs/main_sub/execution/executor.py` - 新增 team_mode, _message_available, force_stop()
- `nexau/archs/main_sub/agent.py` - 传递 team_mode 给 Executor
- `nexau/archs/transports/http/team_routes.py` - 新增 /team/* 路由
- `nexau/archs/transports/http/team_registry.py` - 新增 TeamRegistry 实例注册表
- `nexau/archs/transports/http/sse_server.py` - 集成 TeamRegistry
- `examples/agent_team/start_server.py` - SSE 服务器示例
- `examples/agent_team/frontend/` - React 多 Agent 前端

## 测试方案

### 单元测试

- **TaskBoard**：create / list / claim / release / update_status / 依赖阻塞 / 并发 claim 冲突
- **TaskLockService**：acquire / release / TTL 过期 / 并发竞争
- **TeamMessageBus**：send / broadcast / drain / 持久化验证 / delivered 状态
- **Team Tools**：每个工具的正常路径 + 错误路径（权限校验、冲突、blocked）

### 集成测试

- **端到端 Team 工作流**：Leader 创建任务 → 分配 → Teammate 执行 → 完成 → Idle 通知
- **并发 claim**：多个 teammate 同时 claim 同一 task，验证最多一个成功
- **依赖阻塞**：创建有依赖的任务，验证依赖未完成时 claim 失败
- **队内消息**：验证消息在迭代边界注入、持久化、跨请求恢复
- **Multi-agent SSE**：验证同一 SSE 连接能区分多个 agent 的流式输出

### 手动验证

1. 启动 Team SSE 连接，观察多个 agent 的交错输出
2. 通过 `/team/tasks` API 观察任务状态变化
3. 模拟 teammate 崩溃后重连，验证状态恢复
4. 验证 claim-next 被禁止（无 task_id 的 claim 请求应失败）

## 未解决的问题

> 以下问题已在设计评审中确认决策，记录于此供实现参考。

### 已决定

1. **Teammate 最大并发数**：需要配置上限。`AgentTeam.__init__` 接受 `max_teammates: int = 10`，`spawn_teammate` 超限时返回 `ToolError(code="invalid_state")`。
2. **任务优先级调度**：不需要强制排序。`priority` 字段仅作标记供 agent 参考，不影响 claim 逻辑。
3. **Team 跨 session 共享**：没有需求。Team 严格 session-scoped，不支持跨 session 共享。
4. **消息 TTL / 清理**：不需要。`TeamMessageModel` 随 session 生命周期管理，无独立过期机制。
5. **Teammate 超时**：不监控单 agent 超时，改为全员空闲死锁检测。`TeammateWatchdog` 以后台 asyncio.Task 运行，检测所有 agent 均处于 idle 时通知 leader 决策（见 §3.5）。
6. **Team 元信息存储**：使用独立的 `TeamModel` + `TeamMemberModel` 表，不依赖 `SessionModel.context`。Team 严格 session-scoped，但数据独立存储，便于并发安全和查询。

### 待定

（暂无）

## 参考资料

- `nexau/archs/session/agent_lock_service.py` — AgentLockService 实现（锁机制参考）
- `nexau/archs/session/models/agent_lock.py` — AgentLockModel（数据模型参考）
- `nexau/archs/main_sub/execution/middleware/agent_events_middleware.py` — 事件中间件（SSE 参考）
- `nexau/archs/transports/http/sse_server.py` — SSE Transport（endpoint 参考）
- `nexau/archs/main_sub/agent.py` — Agent 生命周期（并发执行参考）
- RFC-0001: Agent 中断时状态持久化（stop/interrupt 机制参考）
