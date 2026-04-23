# RFC-0006: FrameworkContext — 类型安全的框架上下文

- **状态**: draft
- **优先级**: P1
- **标签**: `architecture`, `dx`, `type-safety`
- **影响服务**: nexau-core (tool, agent, middleware)
- **创建日期**: 2026-03-07
- **更新日期**: 2026-03-07

## 摘要

将 `AgentState` 进化为 `FrameworkContext`，提供类型安全的分组 API，替代当前基于 `GlobalStorage` 字符串 KV、`inspect.signature()` 魔法注入、`getattr` 私有属性访问的模式。

## 动机

当前 `AgentState` 存在四类类型安全问题：

### 1. GlobalStorage 当 Service Locator

```python
# tool_executor.py — 返回 Any，零类型安全
tracer: BaseTracer | None = agent_state.get_global_value("tracer")

# skill.py — key 是字符串，拼错了运行时才发现
skills: dict[str, Skill] = agent_state.get_global_value("skill_registry", {})
```

`GlobalStorage` 的设计初衷是跨 agent 共享用户业务数据，但框架内部的 `tracer`、`skill_registry` 等服务也塞进去了，导致：
- 返回类型永远是 `Any`
- key 是魔法字符串，无 IDE 支持
- 框架内部服务和用户数据混放，边界模糊

### 2. 工具参数注入靠 inspect 魔法

```python
# tool.py L198-210 — 运行时检查函数签名
sig = inspect.signature(self.implementation)
if "agent_state" not in sig.parameters:
    filtered_params.pop("agent_state", None)
if "global_storage" in sig.parameters:
    filtered_params["global_storage"] = agent_state.global_storage
```

工具作者必须记住参数名叫 `agent_state`（不能叫 `state`、`ctx` 等），否则注入失败。这是隐式约定，不是类型约束。

### 3. getattr 访问私有属性

```python
# recall_sub_agent_tool.py — 完全绕过类型系统
executor = getattr(agent_state, "_executor", None)
subagent_manager = getattr(executor, "subagent_manager", None)
```

工具需要调用 sub-agent，但 `AgentState` 没有暴露公开 API，只能 `getattr` 挖私有属性。

### 4. extra_kwargs 边界模糊

`extra_kwargs` 本意是配置常量（`base_url`、`api_key`），但也被用来传递运行时服务（如 `tool_registry`）。两类用途混在一起，没有类型约束。

## 设计

### 概述

引入 `FrameworkContext` 类，作为工具/中间件作者与框架交互的唯一入口。核心原则：

1. **只暴露行为，不暴露实现** — 工具作者看到 `ctx.tools.search()`，看不到 `ToolRegistry`
2. **类型安全** — 每个方法有明确的入参和返回类型
3. **分组 API** — 按领域分组（`tools`、`skills`、`agents`、`sandbox`、`variables`），IDE 补全清晰
4. **单一注入点** — 工具函数声明 `ctx: FrameworkContext` 即可，替代 `agent_state` + `global_storage` + `extra_kwargs` 三路注入

### 详细设计

#### FrameworkContext 接口

```python
class FrameworkContext:
    """Typed framework context for tool and middleware authors.

    RFC-0006: 替代 AgentState，提供类型安全的框架服务访问。
    工具作者只需声明 ctx: FrameworkContext 参数即可获得所有框架能力。
    通过分组 API 按领域组织，IDE 补全时 ctx.tools. 只出现 tools 相关方法。
    """

    # ══════════════════════════════════════
    # 身份信息
    # ══════════════════════════════════════
    agent_name: str
    agent_id: str
    run_id: str
    root_run_id: str

    # ══════════════════════════════════════
    # 分组 API
    # ══════════════════════════════════════
    tools: ToolsAPI
    skills: SkillsAPI
    agents: AgentsAPI
    sandbox: SandboxAPI
    variables: VariablesAPI

    # ══════════════════════════════════════
    # Team（独立领域，保持 property）
    # ══════════════════════════════════════
    @property
    def team_state(self) -> AgentTeamState | None: ...

    # ══════════════════════════════════════
    # Tracing（框架内部）
    # ══════════════════════════════════════
    @property
    def tracer(self) -> BaseTracer | None: ...

    # ══════════════════════════════════════
    # 层级关系
    # ══════════════════════════════════════
    @property
    def parent(self) -> FrameworkContext | None: ...

    # ══════════════════════════════════════
    # 用户数据（跨 agent 共享业务数据）
    # ══════════════════════════════════════
    @property
    def global_storage(self) -> GlobalStorage: ...
```

#### 分组 API 定义

```python
class ToolsAPI:
    """Tools management API.

    RFC-0006: 封装 ToolRegistry，只暴露操作语义
    """

    def search(self, *, query: str, max_results: int = 5) -> list[Tool]:
        """Search deferred tools and inject matches.

        搜到即注入，下一轮 LLM 可直接 function call。
        支持 "+keyword" 强制匹配。
        """
        ...

    def add(self, *, tool: Tool) -> None:
        """Dynamically add an eager tool to the current execution.

        直接写入 ToolRegistry，不经过 Executor 间接层。
        Runtime-added deferred tools are not supported.
        """
        ...

    def get(self, *, name: str) -> Tool | None:
        """Look up a tool by name."""
        ...


class SkillsAPI:
    """Skills management API.

    RFC-0006: 替代 get_global_value("skill_registry")
    """

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name."""
        ...

    def list(self) -> list[str]:
        """List all available skill names."""
        ...


class AgentsAPI:
    """Sub-agent invocation API.

    RFC-0006: 替代 getattr(agent_state, "_executor").subagent_manager
    """

    def call(self, name: str, message: str) -> str:
        """Call a sub-agent by name and return its response."""
        ...


class SandboxAPI:
    """Sandbox access API.

    RFC-0006: 统一 sandbox 和 sandbox env 访问
    """

    def get(self) -> BaseSandbox | None:
        """Get the sandbox for file/shell operations."""
        ...

    def get_env(self, key: str, default: str | None = None) -> str | None:
        """Get a sandbox environment variable."""
        ...

    @property
    def all_env(self) -> dict[str, str]:
        """All sandbox environment variables."""
        ...


class VariablesAPI:
    """Runtime variables access API.

    RFC-0006: 统一运行时变量访问
    """

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get a runtime variable."""
        ...

    @property
    def all(self) -> dict[str, str]:
        """All runtime variables."""
        ...
```

#### 工具函数签名

```python
# ── Before ──

# tool_search.py — 旧方案通过显式参数传 ToolRegistry
def tool_search(query: str, tool_registry: ToolRegistry, max_results: int = 5):
    matched = tool_registry.search(query, max_results=max_results)

# skill.py — 通过 GlobalStorage 取 skill_registry
def load_skill(skill_name: str, agent_state: AgentState):
    skills = agent_state.get_global_value("skill_registry", {})
    skill = skills.get(skill_name)

# recall_sub_agent_tool.py — getattr 挖私有属性
def recall_sub_agent(name: str, message: str, agent_state: AgentState):
    executor = getattr(agent_state, "_executor", None)
    mgr = getattr(executor, "subagent_manager", None)
    result = mgr.call(name, message)

# read_file.py — 只为拿 sandbox
def read_file(file_path: str, agent_state: AgentState | None = None):
    sandbox = get_sandbox(agent_state)


# ── After ──

# tool_search.py
def tool_search(query: str, ctx: FrameworkContext, max_results: int = 5):
    matched = ctx.tools.search(query=query, max_results=max_results)

# skill.py
def load_skill(skill_name: str, ctx: FrameworkContext):
    skill = ctx.skills.get(skill_name)

# recall_sub_agent_tool.py
def recall_sub_agent(name: str, message: str, ctx: FrameworkContext):
    result = ctx.agents.call(name, message)

# read_file.py
def read_file(file_path: str, ctx: FrameworkContext | None = None):
    sandbox = ctx.sandbox.get() if ctx else None
```

#### 注入机制

`Tool.execute()` 中的注入逻辑简化：

```python
# Before: 多个 reserved key，逐个 inspect
reserved_keys = {"agent_state", "global_storage"}
sig = inspect.signature(self.implementation)
if "agent_state" not in sig.parameters:
    filtered_params.pop("agent_state", None)
if "global_storage" in sig.parameters:
    filtered_params["global_storage"] = agent_state.global_storage

# After: 单一 reserved key
reserved_keys = {"ctx"}
sig = inspect.signature(self.implementation)
if "ctx" in sig.parameters:
    filtered_params["ctx"] = framework_context
```

#### FrameworkContext 构建

由 `Executor` 在每次 run 开始时构建，内部持有所有引用但不暴露给工具作者：

```python
# executor.py
def _build_framework_context(self) -> FrameworkContext:
    return FrameworkContext(
        agent_name=self.agent_name,
        agent_id=self.agent_id,
        run_id=self.run_id,
        root_run_id=self.root_run_id,
        # 私有引用，API 方法内部使用
        _tool_registry=self._tool_registry,
        _skill_registry=self._skill_registry,
        _sandbox=self._sandbox,
        _sandbox_manager=self._sandbox_manager,
        _global_storage=self.global_storage,
        _variables=self._variables,
        _team_state=self._team_state,
        _tracer=self._tracer,
        _parent_context=self._parent_framework_context,
    )
```

分组 API 对象由 `FrameworkContext.__init__` 创建，持有对应的内部引用：

```python
class FrameworkContext:
    def __init__(self, *, _tool_registry, ...):
        self.tools = ToolsAPI(_tool_registry=_tool_registry)
        self.skills = SkillsAPI(_skill_registry=_skill_registry)
        self.agents = AgentsAPI(_executor=_executor)  # Phase 1: not yet implemented
        self.sandbox = SandboxAPI(_sandbox=_sandbox, _sandbox_manager=_sandbox_manager, _variables=_variables)
        self.variables = VariablesAPI(_variables=_variables)
```

#### GlobalStorage 清理

从 GlobalStorage 移除的框架内部 key：

| Key | 迁移到 |
|-----|-------|
| `"tracer"` | `ctx.tracer` |
| `"skill_registry"` | `ctx.skills.get()` / `ctx.skills.list()` |
| ~~`"tool_registry"`~~ | 不再通过 `GlobalStorage` 共享；由 `Executor` 构建 `FrameworkContext` 并通过 `ctx.tools.search()` 访问 |
| `"parallel_execution_id"` | 内化到 Executor 实现，通过 `BeforeToolHookInput` 传递 |

GlobalStorage 保留用于：用户自定义的跨 agent 共享业务数据。

#### extra_kwargs 边界

| 用途 | 机制 |
|------|------|
| 配置常量（`base_url`、`api_key`） | `extra_kwargs`（保留） |
| 框架服务（registry、sandbox、tracer） | `FrameworkContext` 分组 API（新增） |
| 用户跨 agent 数据 | `GlobalStorage`（保留，仅业务数据） |

### 示例

#### 自定义工具使用 FrameworkContext

```python
def my_custom_tool(query: str, ctx: FrameworkContext) -> str:
    """A custom tool that uses framework services."""

    # 类型安全地访问 sandbox
    sandbox = ctx.sandbox.get()
    if sandbox is not None:
        result = sandbox.execute(query)
        return result

    # 类型安全地获取变量
    api_key = ctx.variables.get("api_key")

    # 动态添加工具
    new_tool = Tool.from_dict(...)
    ctx.tools.add(tool=new_tool)

    # 搜索 deferred 工具
    matched = ctx.tools.search(query="web fetch")

    return "done"
```

#### Middleware 使用 FrameworkContext

```python
class MyMiddleware(Middleware):
    def before_tool(self, hook_input: BeforeToolHookInput) -> HookResult:
        ctx = hook_input.ctx
        tracer = ctx.tracer
        if tracer:
            tracer.start_span("tool_call")
        return HookResult()
```

#### Team 工具使用 FrameworkContext

```python
def broadcast(message: str, ctx: FrameworkContext) -> str:
    """Broadcast a message to all team members."""
    team = ctx.team_state
    if team is None:
        return "Not in a team context"
    team.message_bus.broadcast(sender=ctx.agent_id, content=message)
    return "Broadcast sent"
```

## 全量 API 覆盖验证

以下是对当前所有框架访问模式的逐一验证：

| 当前模式 | 迁移后 | 状态 |
|----------|--------|------|
| `agent_state.get_sandbox()` | `ctx.sandbox.get()` | ✓ |
| `agent_state.get_sandbox_env(key)` | `ctx.sandbox.get_env(key)` | ✓ |
| `agent_state.all_sandbox_env` | `ctx.sandbox.all_env` | ✓ |
| `agent_state.agent_id` | `ctx.agent_id` | ✓ |
| `agent_state.agent_name` | `ctx.agent_name` | ✓ |
| `agent_state.run_id` | `ctx.run_id` | ✓ |
| `agent_state.root_run_id` | `ctx.root_run_id` | ✓ |
| `agent_state.get_variable(key)` | `ctx.variables.get(key)` | ✓ |
| `agent_state.all_variables` | `ctx.variables.all` | ✓ |
| `agent_state.team_state` | `ctx.team_state` | ✓ |
| `agent_state.parent_agent_state` | `ctx.parent` | ✓ |
| `agent_state.add_tool(tool)` | `ctx.tools.add(tool)` | ✓ |
| `get_global_value("tracer")` | `ctx.tracer` | ✓ |
| `get_global_value("skill_registry")` | `ctx.skills.get(name)` / `.list()` | ✓ |
| 显式传递 `ToolRegistry` 参数 | `ctx.tools.search()` | ✓ |
| `getattr(agent_state, "_executor").add_tool()` | `ctx.tools.add(tool=tool)` | ✓ |
| `getattr(executor, "subagent_manager")` | `ctx.agents.call(name, msg)` | ✓ |
| `set_global_value("parallel_execution_id")` | 内化到 Executor，不暴露 | ✓ |
| `agent_state.global_storage` | `ctx.global_storage`（仅用户数据） | ✓ |
| `get_context_value("__nexau_full_trace_*")` | 中间件私有状态，不暴露 | ✓ |

## 权衡取舍

### 考虑过的替代方案

#### 方案 A：引入独立的 ToolContext

单独创建 `ToolContext` dataclass，仅用于工具注入，不替代 `AgentState`。

**不采用原因**：增加了一个新概念，工具作者需要理解 `AgentState` 和 `ToolContext` 两个对象的区别。不如直接进化 `AgentState`。

#### 方案 B：保持 AgentState，只加 typed properties

在现有 `AgentState` 上加 `@property` 暴露 typed 服务。

**不采用原因**：命名不准确 — `AgentState` 暗示"状态容器"，实际承担的是"框架上下文 + 服务定位"的角色。重命名更能反映真实职责。

#### 方案 C：平铺 API

所有方法直接放在 `FrameworkContext` 上（`ctx.search_tools()`、`ctx.get_skill()`）。

**不采用原因**：随着 API 增长，平铺会导致补全列表过长。分组后 `ctx.tools.` 只出现 tools 相关方法，更清晰。

### 缺点

1. **破坏性变更** — 所有使用 `agent_state` 参数的工具函数都需要迁移
2. **过渡期复杂度** — 需要同时支持旧 `agent_state` 和新 `ctx` 参数
3. **对象层级** — 分组 API 多了一层间接访问（`ctx.tools.get()` vs `ctx.get_tool()`），但换来了更好的组织性

## 实现计划

### 阶段划分

- [ ] Phase 1: 新增 `FrameworkContext` 类和分组 API 类（`ToolsAPI`、`SkillsAPI` 等）
- [ ] Phase 2: `Tool.execute()` 支持 `ctx` 参数注入（同时保留 `agent_state` 兼容）
- [ ] Phase 3: 迁移内置工具：`tool_search`、`load_skill`、`recall_sub_agent`、file/shell tools 等
- [ ] Phase 4: 迁移 middleware hook 的 `HookInput`，加入 `ctx` 字段
- [ ] Phase 5: 从 GlobalStorage 移除框架内部 key（`tracer`、`skill_registry`、`parallel_execution_id`）
- [ ] Phase 6: 废弃 `agent_state` 注入路径（deprecation warning）

### 相关文件

- `nexau/archs/main_sub/agent_state.py` — 现有 AgentState，将被 FrameworkContext 替代
- `nexau/archs/main_sub/agent_context.py` — GlobalStorage 定义
- `nexau/archs/tool/tool.py` — Tool.execute() 注入逻辑
- `nexau/archs/main_sub/execution/tool_executor.py` — ToolExecutor 参数准备
- `nexau/archs/main_sub/execution/executor.py` — Executor 构建上下文
- `nexau/archs/tool/builtin/` — 所有内置工具（迁移目标）
- `nexau/archs/main_sub/execution/middleware/` — 所有中间件（迁移目标）
- `nexau/archs/main_sub/team/tools/` — Team 工具（迁移目标）

## 测试方案

### 单元测试

- `FrameworkContext` 各分组 API 方法的正确性
- `ToolsAPI.search()` / `.add()` / `.get()` 委托到 ToolRegistry
- `SkillsAPI.get()` / `.list()` 委托到 skill registry
- `AgentsAPI.call()` 委托到 SubAgentManager
- `SandboxAPI.get()` 正确处理 sandbox_manager lazy init
- `Tool.execute()` 对 `ctx` 参数的注入
- `ctx` 和 `agent_state` 并存时的优先级（`ctx` 优先）
- `extra_kwargs` 与 `ctx` 不冲突

### 集成测试

- 端到端：工具通过 `ctx.tools.search()` 搜索并注入 deferred tool
- 端到端：工具通过 `ctx.agents.call()` 调用子 agent
- 端到端：middleware 通过 `ctx.tracer` 创建 span
- 端到端：team 工具通过 `ctx.team_state` 访问消息总线

### 手动验证

- 现有使用 `agent_state` 的第三方工具在 deprecation 期间正常工作
- IDE 自动补全验证：`ctx.tools.` 后只出现 `search`、`add`、`get`

## 未解决的问题

1. **命名**：`FrameworkContext` vs `AgentContext` vs `Context` — 需要确认最终命名
2. **Middleware 迁移**：`HookInput` 中的 `agent_state` 字段是否也改为 `ctx`，还是保持独立的迁移节奏
3. **`AgentState` 是否完全移除**：还是保留为 `FrameworkContext` 的内部实现
4. **`get_context_value` / `set_context_value` 替代方案**：中间件需要 per-execution 私有状态，当前依赖 AgentContext。迁移后可以考虑中间件 per-instance 状态或 `FrameworkContext._internal_state: dict`

## 参考资料

- RFC-0005: Tool Search — 工具按需动态注入（首次暴露 GlobalStorage 滥用问题）
- [CLAUDE.md Type Safety Guidelines](../CLAUDE.md) — 项目类型安全规范
