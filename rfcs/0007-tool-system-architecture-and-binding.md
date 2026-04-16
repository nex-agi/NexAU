# RFC-0007: 工具系统架构与绑定模型

- **状态**: implemented
- **优先级**: P0
- **标签**: `architecture`, `tools`, `tool-binding`
- **影响服务**: `nexau` (`archs/tool/`, `archs/main_sub/`)
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

本 RFC 为追溯式设计文档，记录 NexAU 工具系统的架构与 binding 模型。系统由 (a) 单一 `tool_registry` dict（按 tool 名索引）、(b) Agent-level binding 为主的注册模型、(c) 标准化的执行生命周期、(d) 显式 result envelope 四部分组成。RFC-0005（tool-search）在本系统之上叠加 deferred loading 与 search 能力，本 RFC 描述其底层。本 RFC 不引入新设计；仅固化已有设计的契约边界，便于未来重构与新人 onboarding。

## 动机

### 已落地的子系统缺乏 canonical 设计文档

`nexau/archs/tool/` 与 `nexau/archs/main_sub/` 中的 tool registry、tool binding 与 invocation lifecycle 已经在生产中长期运行，但其设计意图分散在多处 docstring 与测试中。具体痛点：

1. **Tool 来源多样但收敛点未文档化**：YAML、Python class、MCP server、运行时 `add_tool()`、sub-agent-as-tool 五种来源全部汇入同一 `tool_registry` dict（`nexau/archs/main_sub/agent.py:180`），但任何一处文档都没把这五条路径并列描述。
2. **Result envelope 与错误传播契约隐性**：tool 返回值被 coerce 成 dict（`nexau/archs/tool/tool.py:224-238`），异常被吞掉转为 `{"error": ..., "error_type": ..., "traceback": ...}` 字段（`nexau/archs/tool/tool.py:242-249`），这一契约没有被显式记录。
3. **名称冲突策略未声明**：注册表是 dict 直接赋值（`nexau/archs/main_sub/agent.py:1144`、`registry[tool.name] = tool`），后注册的会静默覆盖前者；这是一个隐性的"靠配置纪律"约定。
4. **RFC-0005 的依赖前提缺失**：RFC-0005 设计了 deferred loading 与 ToolSearch，但其前提（registry / binding / invocation lifecycle 的当前形态）没有 RFC 描述；导致 RFC-0005 的"how this plugs in"段落只能口语化引用代码。

### 不补全会怎样

- 跨子系统重构（如 RFC-0005 提议的把 dict registry 升级为 `ToolRegistry` 类）只能边读代码边猜契约，容易破坏 sub-agent / MCP / 运行时注入路径。
- 新成员理解 tool 调用全链路需要逐一读 `agent.py`、`tool.py`、`tool_executor.py`、`response_parser.py`、`config.py`、`mcp_client.py` 至少 6 个文件。

## 设计

### 概述

工具系统由四层组成：

```
Sources         Registry              Lifecycle             Envelope
-------         --------              ---------             ---------
YAML config  ┐                                              {result, error,
Python class │                                              error_type,
MCP server   ├──> tool_registry ──> ToolExecutor ──> Tool   traceback,
add_tool()   │    (dict by name)    (dispatch)     .execute _is_stop_tool, ...}
sub-agent    ┘                                              ─stripped→ LLM
```

所有 tool 来源汇入单一 `tool_registry: dict[str, Tool]`，由 `Agent.__init__` 一次性建立（`nexau/archs/main_sub/agent.py:180`）；`ToolExecutor.execute_tool()` 负责调度、参数验证、依赖注入、middleware 钩子与错误归一；最终结果是固定 schema 的 dict，经过 `returnDisplay` 剥离后再回填给 LLM。

### 关键设计决策

1. **单一扁平 registry，按 tool 名索引（无命名空间）**
   - 原因：LLM 的 tool call 是扁平字符串（`tool_name="read_file"`），不存在层级；为 LLM 端简化是设计目标。
   - 结果：所有 tool 来源最终都按 name 存入同一 dict；没有按 source 分组的 namespace。
   - 已知代价：tool name 必须在所有来源间全局唯一；冲突时静默后写入覆盖（见决策 4）。

2. **Agent-level binding 为默认，session-level binding 为例外**
   - 原因：LLM 的 tool 列表通常需要在请求开始前知道（structured tool calling 要求 schema 提前 declare）；agent-level binding 是行业标准。
   - 结果：`tool_registry` 在 `Agent.__init__` 期间一次构建；运行时只通过 `add_tool()`（`nexau/archs/main_sub/agent.py:1141`）扩展，且需要持锁更新 `structured_tool_payload`（`nexau/archs/main_sub/execution/executor.py:1336-1344`）。
   - 已知代价：动态 agent 组合受限；RFC-0005 的 deferred loading 是对该代价的补救。

3. **Tool 来源统一收敛，但 sub-agent 是例外（synthesize on demand）**
   - 原因：sub-agent 的 tool spec 由 agent name 反推（`nexau/archs/main_sub/agent.py:445-450`），不是预注册的 Tool 对象；这样可以避免在 sub-agent 列表变化时同步重建 registry。
   - 结果：sub-agent **不**进入 `tool_registry`，但其 schema 在 `_build_openai_tool_specs()` 中被合成插入 `structured_tool_payload`。这是 registry 唯一的 "shadow tool" 类型。
   - 已知代价：调试 sub-agent 调用时不能直接在 registry 中查；需要走 sub-agent dispatch 路径理解。

4. **名称冲突静默覆盖**
   - 原因：允许 skill wrapping 与 MCP extension 共存（同名 tool 由不同来源提供时，后注册者赢）；显式冲突解决会增加配置复杂度。
   - 结果：`tool_registry[tool.name] = tool` 直接赋值；无 warning、无 error。
   - 已知代价：配置错误可能被吞，无法早期发现；这是已知的 fragility，未来若有"工具仓库"管理需求，应考虑改为显式策略（`error` / `prefer-yaml` / `prefer-mcp` 等）。

5. **Result envelope 是 dict，而非 exception**
   - 原因：tool result 必须 serialize 给 LLM；exception 不能跨越该边界；因此异常必须在 tool 层就转为 dict 字段。
   - 结果：`Tool.execute()` 内置 try/except，把 exception 转为 `{"error": <str>, "error_type": <str>, "traceback": <str>}`（`nexau/archs/tool/tool.py:242-249`）；调用方（`ToolExecutor`）通过检查 dict key 而非 catch 异常来判定失败。
   - 已知代价：caller 必须主动检查 `error` / `error_type` 字段；不符合 Python 习惯的异常传播。

6. **`returnDisplay` 字段在送 LLM 前被剥离**
   - 原因：`returnDisplay` 用于 UI 端富文本展示（如 markdown 渲染），但 LLM 不需要、且复制到上下文会浪费 token。
   - 结果：`ToolExecutor` 在送回 LLM 前 `pop("returnDisplay", None)`（`nexau/archs/main_sub/execution/tool_executor.py:295`）。
   - 已知代价：UI 与 LLM 看到的结果不同；调试时需要明确"我看到的是哪一份"。

7. **`disable_parallel` 是当前唯一的 capability metadata**
   - 原因：tool 的 read-only / destructive / idempotent / result-size 等 capability 没有显式建模；只对 "并发安全" 这一项做了显式建模（`Tool.disable_parallel`，`nexau/archs/tool/tool.py:97`）。
   - 结果：批量调度（`Executor._execute_batch()`）按该字段决定是否串行化；其他 capability 由 LLM 通过 description 自行推断。
   - 已知代价：无法做自动安全检查（如禁止 destructive tool 并行调用）；将 capability 模型扩展到 read-only/destructive 是未来 RFC 的可能方向。

### 详细设计

#### 数据模型

```
class Tool:
    name: str                      # registry key, must be globally unique
    description: str               # for LLM
    parameters: JsonSchema         # for LLM, used by validate
    implementation: Callable | str # callable, or import path (lazy import)
    binding: dict | None           # agent_state / global_storage / sandbox 注入清单
    disable_parallel: bool = False # capability flag
    as_skill: bool = False         # 表示为 skill（不出现在 tool spec, 出现在 SKILL.md）
    lazy: bool = False             # 推迟 import 到首次调用
```

#### 注册路径（5 种来源）

| 来源 | 入口函数 | 实际操作 |
|------|----------|----------|
| YAML config | `AgentConfigBuilder.build_tools()` (`nexau/archs/main_sub/config/config.py:645`) | 解析 yaml `tools:` → 调 `Tool.from_yaml()` (`nexau/archs/tool/tool.py:111`) → 追加到 `agent_params["tools"]` |
| Python class direct | `AgentConfig.from_yaml()` 或手工实例化 `Tool(implementation=callable)` | 直接 append 到 `config.tools` |
| MCP server | `Agent._initialize_mcp_tools()` (`nexau/archs/main_sub/agent.py:398`) | 调 `sync_initialize_mcp_tools()` (`nexau/archs/tool/builtin/mcp_client.py`) → 把 MCP tools 包装成 NexAU `Tool` → `config.tools.extend(...)` |
| 运行时 API | `Agent.add_tool()` (`nexau/archs/main_sub/agent.py:1141`) | 持 `_tool_registry_lock` → 更新 `tool_registry` 与 `structured_tool_payload`（`nexau/archs/main_sub/execution/executor.py:1336-1344`） |
| Sub-agent-as-tool | `_build_openai_tool_specs()` (`nexau/archs/main_sub/agent.py:445-450`) | 合成 tool spec 插入 `structured_tool_payload`，**不**进 registry |

#### 调用生命周期（按时序）

1. **Parse**：LLM 响应回流 → `ResponseParser.parse_response()` (`nexau/archs/main_sub/execution/response_parser.py:39`) → 输出 `ToolCall[]`
2. **Batch dispatch**：`Executor._execute_batch()` 按 `disable_parallel` 切分串/并行批次，提交 thread pool
3. **Per-tool execution**：`ToolExecutor.execute_tool()` (`nexau/archs/main_sub/execution/tool_executor.py:68`)
   1. 持锁查 registry (`tool_executor.py:89-94`)
   2. `middleware_manager.run_before_tool()` 注入参数 / 拦截
   3. `Tool.execute()` (`nexau/archs/tool/tool.py:170`)：参数 schema 校验 → 注入 `agent_state` / `global_storage` / `sandbox` → 调 binding callable → 结果 coerce 为 dict (`tool.py:224-238`)
   4. 异常在 `Tool.execute()` 内被 catch → 转为 error dict (`tool.py:242-249`)
   5. `middleware_manager.run_after_tool()` 可改写 result dict
   6. 若 tool 名在 `stop_tools` set → 标记 `_is_stop_tool: true` (`tool_executor.py:264-268`)
   7. `pop("returnDisplay", None)` (`tool_executor.py:295`)
4. **Result envelope** 回填 message history → 进入下一轮 LLM 调用

#### 与其他子系统的接口契约

| 对端子系统 | 契约要点 | 详见 RFC |
|-----------|----------|----------|
| LLM 聚合器 | tool spec 的 OpenAI / Anthropic schema 由 `_build_tool_call_payload()` 在 `Agent.__init__` 一次性生成；运行时变更需通过 `add_tool()` 走持锁路径 | RFC-0010 |
| 中间件钩子 | `run_before_tool()` / `run_after_tool()` 是 tool 调用的两个固定 hook 点；middleware 可改写 args / result，但不能跳过 tool 调用本身 | RFC-0011 |
| Session / 历史 | tool result envelope（剥离 returnDisplay 后）作为 message 写入 history；error envelope 也作为 message 写入 | RFC-0008 |
| 子 Agent 委派 | sub-agent 的 tool spec 在 `_build_openai_tool_specs()` 合成；sub-agent 调用走专门 dispatch，**不**经过 registry lookup | RFC-0013 |
| MCP | MCP tools 通过 `_initialize_mcp_tools()` 在 agent 启动时一次性发现并 wrap 成 NexAU `Tool` 实例追加到 `config.tools`；运行时新发现需要重启 agent | RFC-0016 |
| 技能系统 | `as_skill=true` 的 tool 不出现在 LLM tool spec，而是被组合进 `SKILL.md` 内容；调用走 skill loading 路径 | RFC-0018 |
| Tool search (RFC-0005) | RFC-0005 设计了 `defer_loading` 属性与 `ToolRegistry` 类，把当前的 dict 升级为按需 inject 的对象；本 RFC 是其底层前提，本 RFC 描述的注册流不变 | RFC-0005 |

### 架构图

```mermaid
flowchart TB
    YAML[YAML config<br/>tools:]
    PyDirect[Python class<br/>Tool(impl=callable)]
    MCP[MCP server<br/>via mcp_client]
    AddTool[add_tool() runtime API]
    SubAgent[Sub-agent<br/>synthesized spec]

    YAML --> Reg
    PyDirect --> Reg
    MCP --> Reg
    AddTool -->|locked update| Reg
    SubAgent -.shadow spec, not in registry.-> Payload

    Reg[tool_registry<br/>dict by name]
    Payload[structured_tool_payload<br/>OpenAI / Anthropic schema]

    Reg -->|build at init| Payload

    LLM[LLM response] --> Parser[ResponseParser]
    Parser --> Exec[Executor._execute_batch]
    Exec -->|disable_parallel?| ToolExec[ToolExecutor.execute_tool]

    ToolExec --> BeforeMW[middleware<br/>run_before_tool]
    BeforeMW --> ToolCall[Tool.execute<br/>schema validate +<br/>dependency inject +<br/>binding call]
    ToolCall --> Coerce[result coerce to dict<br/>or error envelope]
    Coerce --> AfterMW[middleware<br/>run_after_tool]
    AfterMW --> Strip[strip returnDisplay]
    Strip --> History[history.append]

    Reg -.lookup by name.-> ToolExec

    style Reg fill:#FED7AA,stroke:#EA580C
    style ToolExec fill:#FEE2E2,stroke:#DC2626
    style ToolCall fill:#D1FAE5,stroke:#10B981
    style Coerce fill:#FEF3C7,stroke:#CA8A04
```

## 权衡取舍

### 考虑过的替代方案

| 方案 | 优点 | 缺点 | 决定 |
|------|------|------|------|
| Namespace registry（如 `mcp:read_file` vs `local:read_file`） | 多来源同名 tool 共存 | LLM tool call 必须解析 namespace；schema 复杂度高 | 否（已采用扁平 registry） |
| Tool 调用结果用 exception 传播 | 符合 Python 习惯 | exception 不能 serialize 给 LLM；需要在 boundary 转换 | 否（已采用 envelope dict） |
| 完整 capability 元数据（read-only / destructive / idempotent / size） | 自动安全检查 + 调度优化空间 | 当前用户没有该需求；增加 schema 复杂度 | 否（仅保留 `disable_parallel` 一项） |
| 动态注册 = first-class（每个 turn 重建 registry） | sub-agent / 临时 tool 注入更自然 | 与 LLM structured tool calling 的"前置 declare"契约冲突；性能开销大 | 否（保留 init-time binding，运行时仅 `add_tool` 例外） |
| Sub-agent 也走 registry | registry 单一信源 | sub-agent 列表变化频繁；同步成本高 | 否（保留 synthesized spec 例外） |

### 缺点

1. **冲突静默**：tool name 冲突时无 error 提示，配置错误可能被吞。
2. **Capability 模型贫乏**：仅 `disable_parallel`；自动化安全 / 调度增益空间未释放。
3. **Sub-agent 是 registry 的 "shadow"**：调试 sub-agent 调用时不能直接 introspect；需要走 dispatch 路径。
4. **运行时注册必须持锁**：`add_tool()` 与 LLM 调用并发时锁竞争；目前规模未触发性能问题，未来高频动态注册需要重新评估。
5. **`returnDisplay` 双视图**：UI 与 LLM 看到的结果不同；调试时需明确视角。

## 实现计划

### 阶段划分

本 RFC 是追溯式设计文档，对应子系统已经实现并稳定。无新增 phase。

- [x] Phase 1: tool_registry 与 binding 模型（既已实施）
- [x] Phase 2: ToolExecutor 与 middleware 钩子（既已实施）
- [x] Phase 3: result envelope 与 returnDisplay 剥离（既已实施）
- [x] Phase 4: MCP / sub-agent-as-tool / lazy import 三类特殊路径（既已实施）

### 子任务分解

> 本 RFC 为追溯式设计文档，无新增子任务。

### 相关文件

- `nexau/archs/tool/tool.py` — `Tool` 类、execute、coerce、error envelope
- `nexau/archs/tool/tool_registry.py` — RFC-0005 引入的新文件（仅在该 RFC 落地后存在）
- `nexau/archs/tool/builtin/mcp_client.py` — MCP tool 发现与 wrap
- `nexau/archs/main_sub/agent.py` — `tool_registry` 构建、`_build_tool_call_payload`、`add_tool`、sub-agent spec 合成
- `nexau/archs/main_sub/execution/tool_executor.py` — 调度、middleware hook、result envelope 处理
- `nexau/archs/main_sub/execution/executor.py` — batch dispatch 与持锁 registry 更新
- `nexau/archs/main_sub/execution/response_parser.py` — LLM response 解析为 ToolCall
- `nexau/archs/main_sub/config/config.py` — YAML tool 加载入口

## 测试方案

### 单元测试

已存在的 unit tests 覆盖：

- `tests/unit/tool/` — Tool 实例化、from_yaml、execute、error envelope coerce
- `tests/unit/main_sub/test_agent_tools.py`（如存在；按实际路径调整）— registry 构建、add_tool 锁路径、sub-agent spec 合成

本 RFC 不引入新的代码改动，因此不引入新的 unit tests。

### 集成测试

已存在的 integration tests 覆盖：

- `tests/integration/` — YAML 加载 → agent 启动 → tool 调用 → result envelope 完整链路
- `tests/integration/mcp/`（如存在）— MCP server 起动 → tool 发现 → 注入 registry → LLM 调用
- `tests/integration/sub_agent/`（如存在）— sub-agent dispatch 路径

### 手动验证

不适用（追溯式 RFC）。

## 未解决的问题

1. **Capability 模型扩展**：是否需要把 read-only / destructive / idempotent / max_result_size 等 capability 显式建模？目前仅 `disable_parallel` 一项；未来若引入"安全护栏"或"自动调度优化"，需要新 RFC。
2. **冲突策略**：当前静默后写入覆盖，是否升级为可配置策略（`error` / `prefer-source-X` / `merge`）？需要先收集冲突实际发生频率与 root cause。
3. **Sub-agent 是否纳入 registry**：synthesize spec 路径能简化，但同步成本上升；待 RFC-0013（子 Agent 委派）评估。
4. **运行时注册的并发模型**：`_tool_registry_lock` 是粗粒度锁；高频 `add_tool` 场景下需评估是否升级为 RW lock 或 copy-on-write。
5. **`as_skill=true` 与 tool 的边界**：技能系统（RFC-0018）应明确 skill 与 tool 的差异、互操作；本 RFC 仅声明该字段存在。

## 参考资料

- `rfcs/0005-tool-search.md` — 在本系统之上叠加的 deferred loading + tool search
- `rfcs/0006-rfc-catalog-completion-master-plan.md` — 本 RFC 的母 RFC（master plan）
- `nexau/archs/tool/CLAUDE.md` — tool 子系统现有 docstring 与设计指引
- `nexau/archs/main_sub/CLAUDE.md` — main_sub（agent / executor）现有 docstring 与设计指引
