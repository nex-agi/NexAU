# RFC-0017: CLI 与 YAML 配置加载

- **状态**: implemented
- **优先级**: P2
- **标签**: `cli`, `config`, `yaml`, `pydantic`, `dx`, `architecture`
- **影响服务**: `nexau/cli/`, `nexau/cli_wrapper.py`, `nexau/archs/main_sub/config/`, `nexau/archs/main_sub/utils/common.py`, `nexau/archs/config/config_loader.py`
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 把"启动一个 agent"抽象成两条路径：(1) `nexau` Python CLI（argparse 子命令 `chat` / `serve http` / `serve stdio`）；(2) `nexau-cli` Node.js 包装器（legacy 兼容）。所有路径最终都聚焦到一件事——**从 YAML 配置文件构造 `AgentConfig` 实例**。本 RFC 描述：(1) 两个 entry point (`nexau` / `nexau-cli`) 在 `pyproject.toml:36-39` 的注册分工；(2) 三 subcommand 与 transport 的对应（chat → SQLite REPL、serve http → SSETransportServer、serve stdio → StdioTransport）；(3) **两阶段 YAML 加载**：先 `load_yaml_with_vars`（文本层 `${env.X}` / `${variables.X}` / `${this_file_dir}` 替换 + `yaml.safe_load`）→ 后 `AgentConfigSchema.from_yaml`（Pydantic 校验 + discriminated union）→ 最后 `AgentConfigBuilder` 11 段链式构造；(4) builder 链每一步只填充 `agent_params` 一个字典字段，错误聚合为 `ConfigError`；(5) sub-agent / tool yaml_path 支持 `package:resource` 与 filesystem 两种解析；(6) `_finalize` 模型后置校验自动注入 `recall_sub_agent` 工具 + `load_skill` 工具 + 解析 `CompositeTracer`；(7) deprecated 入口 `load_agent_config` + `overrides` 参数的 v0.4.0 退役计划。

## 动机

NexAU 必须解决的设计问题：

1. **多 transport 统一入口**：HTTP/SSE 服务、Stdio JSON-RPC 服务、本地交互 REPL 三个完全不同的运行模式，但都"加载同一个 agent.yaml 然后跑起来"。需要统一的命令行 dispatcher。
2. **YAML 不能是纯静态**：用户经常需要把 token / API key / 路径塞进 YAML，但又不想硬编码。必须支持环境变量插值（`${env.GITHUB_TOKEN}`）、文件相对路径（`${this_file_dir}`）、跨字段引用（`${variables.shared_url}`）。
3. **YAML schema 必须 strict**：`mcp_servers` 配错 type、tool 缺 yaml_path、middleware 写错 import string——错误必须**早期失败**并定位到字段，而不是运行时偶发崩溃。Pydantic discriminated union + `extra="forbid"` 是工业级做法。
4. **YAML 字段不直接等于 Python 对象**：YAML 里 `tools: [{name, yaml_path, binding, ...}]` 只是声明；实际要 import binding 函数 + 加载 tool YAML + 实例化 `Tool` 对象。需要分两阶段：schema 校验 → builder 构造。
5. **配置错误必须可定位**：单一 `ConfigError` 异常类型 + "Error loading X 'name'" 模板让用户立刻知道是哪个 tool / sub-agent / hook 配错。
6. **sub-agent 配置可嵌套**：sub-agent 自己也是 `AgentConfig.from_yaml`，递归 builder。`config_path` 必须支持相对父 YAML 的路径与 `package:resource` 安装包路径。
7. **历史欠债不能阻断新工作**：`load_agent_config`（已 deprecated）和 `overrides` 参数（v0.4.0 移除）需要在保留兼容的同时引导用户迁移。
8. **Node.js CLI 共存**：早期版本有 Node.js CLI，迁移到 Python 时为不破坏用户脚本，保留 `nexau-cli` 入口包装 Node.js binary。
9. **`recall_sub_agent` / `load_skill` 是隐式工具**：用户 YAML 里不写但运行时必需（sub-agent 委派 / skill 系统）。必须在 `_finalize` 后置校验里自动注入。

## 设计

### 概述

```
                            ┌─────────────────────────────────────────────┐
                            │ pyproject.toml [project.scripts] (:36-39)   │
                            │  - nexau     → nexau.cli.main:main          │
                            │  - nexau-cli → nexau.cli_wrapper:main       │
                            │   (Node.js wrapper, legacy)                 │
                            └────────┬────────────────────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────────────────────────────────┐
                            │ nexau/cli/main.py                           │
                            │  - create_parser (:12) argparse             │
                            │  - subcommands: chat / serve {http,stdio}   │
                            │  - main (:68) dispatch via args.func        │
                            └────────┬────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
   ┌────────────────────┐  ┌──────────────────┐  ┌────────────────────┐
   │ commands/chat.py   │  │ commands/http.py │  │ commands/stdio.py  │
   │  - REPL with /clear│  │  - SSETransport  │  │  - StdioTransport  │
   │  - SQLite session  │  │  - Pydantic args │  │  - Pydantic args   │
   └────────┬───────────┘  └────────┬─────────┘  └────────┬───────────┘
            │                       │                     │
            └─────────────┬─────────┴─────────────────────┘
                          ▼
            ┌─────────────────────────────────────────────────────┐
            │ load_yaml_with_vars (utils/common.py:57)            │
            │  - read text                                        │
            │  - replace ${this_file_dir} (:62)                   │
            │  - replace ${env.X} (:65, regex)                    │
            │  - yaml.safe_load (:76)                             │
            │  - extract `variables:` block                       │
            │  - replace ${variables.x.y} (:90)                   │
            │  - 2nd yaml.safe_load (:108)                        │
            └────────┬────────────────────────────────────────────┘
                     ▼
            ┌─────────────────────────────────────────────────────┐
            │ AgentConfigSchema.from_yaml (schema.py:109)         │
            │  - Pydantic validation                              │
            │  - MCPServerConfig discriminated union (:75)        │
            │  - extra="forbid" (:28, :41, :49)                   │
            │  - _require_llm_config validator (:103)             │
            └────────┬────────────────────────────────────────────┘
                     ▼
            ┌─────────────────────────────────────────────────────┐
            │ AgentConfigBuilder (config.py:308)                  │
            │  set_overrides → build_core_properties              │
            │  → build_llm_config → build_mcp_servers             │
            │  → build_hooks → build_tracers → build_tools        │
            │  → build_sub_agents → build_skills                  │
            │  → build_system_prompt_path → build_sandbox         │
            │  → get_agent_config                                 │
            └────────┬────────────────────────────────────────────┘
                     ▼
            ┌─────────────────────────────────────────────────────┐
            │ AgentConfig (config.py:94)                          │
            │  + @model_validator _finalize (:231)                │
            │    - 注入 recall_sub_agent tool (:236-244)          │
            │    - 注入 load_skill tool (:261-263)                │
            │    - 解析 CompositeTracer (:266-271)                │
            └─────────────────────────────────────────────────────┘
```

### 详细设计

#### 1. 两个 entry point 与三 subcommand

`pyproject.toml:36-39` 注册：

| 命令 | 入口函数 | 用途 |
|------|---------|------|
| `nexau` | `nexau.cli.main:main` | 新 Python 统一 CLI |
| `nexau-cli` | `nexau.cli_wrapper:main` | Node.js wrapper（legacy 向后兼容）|

`nexau/cli/main.py:12-65` argparse 注册三 subcommand：

```
nexau chat <agent.yaml> [--query] [--user-id] [--session-id] [--verbose]
nexau serve http  <agent.yaml> [--host] [--port] [--log-level] [--cors-origins]
nexau serve stdio <agent.yaml> [--verbose]
```

各 subcommand 在 `nexau/cli/commands/` 下独立实现：

- **chat (`chat.py:212` main)** ：交互 REPL + `~/.nexau/nexau.db` SQLite 持久化（`commands/chat.py:236-241`）；`/clear` 命令重建 session（`commands/chat.py:134-140`）；`--query` 切非交互单次查询模式。
- **http (`http.py:75` main)** ：`SSETransportServer` 启动 FastAPI（RFC-0015 描述的 transport），`HTTPConfig`（`commands/http.py:128-133`）封装 host/port/cors。
- **stdio (`stdio.py:46` server_main)** ：`StdioTransport` 启动 JSON-RPC stdio loop（RFC-0015），所有日志和 banner 重定向到 stderr 保护 stdout 协议帧（`commands/stdio.py:75-78`）。

`main.py:68-93` 顶层 dispatcher 用 `args.func(args)` 调子命令，捕获 `KeyboardInterrupt → 130` / 其他异常 → 1。

#### 2. `load_yaml_with_vars`：文本层变量替换 + 两次 yaml.safe_load

`nexau/archs/main_sub/utils/common.py:57-111` 是所有 YAML 加载的最底层。流程：

1. **读文件文本**（utils/common.py:58-59）：保留原始文本，便于后续正则替换。
2. **`${this_file_dir}` 替换**（utils/common.py:62）：用绝对路径替换，使 YAML 内可写 `system_prompt: ${this_file_dir}/system.md` 实现"配置文件目录相对引用"。
3. **`${env.VAR}` 正则替换**（utils/common.py:65-73）：未设置环境变量时抛 `ConfigError`，避免静默 fallback 引入 bug。
4. **第一次 `yaml.safe_load`**（utils/common.py:76）：得到 dict 后取出 `variables:` 顶层块。
5. **`${variables.x.y}` 跨字段引用**（utils/common.py:90-105）：路径解析 `path.split(".")`，必须解析到 scalar；解析到 dict/list 抛错（防止 YAML 字段被嵌入字符串中产生形态错乱）。
6. **第二次 `yaml.safe_load`**（utils/common.py:107-108）：用替换后的 text 再次解析；最后 `pop("variables", None)` 删除已用变量块。

**为什么两次 safe_load**：第一次是为了读 `variables:` 块（YAML structured），第二次是替换完文本后才能 parse 出最终 dict。这避免了"YAML 解析中插值"的复杂性（如 `!env` 自定义 tag 需要 SafeLoader 子类化）。

#### 3. `AgentConfigSchema`：Pydantic 校验层

`schema.py:86` 是顶层 schema。关键：

- **`extra="forbid"`** 在 `ToolConfigEntry` (schema.py:28) / `SubAgentConfigEntry` (schema.py:41) / `MCPServerBaseModel` (schema.py:49) 都开启——拼写错误的字段（如 `commnad` 而非 `command`）立刻报错。
- **`MCPServerConfig` discriminated union**（schema.py:75）：`MCPStdIOServer | MCPHttpServer | MCPSseServer`，按 `type: Literal["stdio"|"http"|"sse"]` 字段自动 dispatch。这让 stdio server 必须有 `command` 字段、http/sse 必须有 `url` 字段——schema 层强制。
- **`_require_llm_config`** model_validator（schema.py:103-107）：`llm_config` 缺失即抛 `ValueError`（被 `from_yaml` 的 try/except 转 `ConfigError`）。
- **`from_yaml`**（schema.py:109-152）：调 `load_yaml_with_vars` → `normalize_agent_config_dict`（先 round-trip 一次清理）→ `cls.model_validate(config)`；ValidationError 用 `_format_validation_error`（schema.py:216-223）压成"path->msg; path->msg" 单行。
- **`apply_agent_name_overrides`**（schema.py:159-196）：deprecated 路径，按 agent name 覆盖配置；v0.4.0 删除，取代为"先加载 config 再 mutate `.key = value`"。

#### 4. `AgentConfigBuilder`：11 段链式构造

`config.py:308` 起的 builder 类。`from_yaml`（config.py:134-185）调用模式：

```python
agent_builder = AgentConfigBuilder(schema_dict, config_path.parent)
agent_config = (
    agent_builder.set_overrides(overrides)         # config.py:861
    .build_core_properties()                       # config.py:403
    .build_llm_config()                            # config.py:749
    .build_mcp_servers()                           # config.py:438
    .build_hooks()                                 # config.py:493
    .build_tracers()                               # config.py:610
    .build_tools()                                 # config.py:645
    .build_sub_agents()                            # config.py:698
    .build_skills()                                # config.py:665
    .build_system_prompt_path()                    # config.py:828
    .build_sandbox()                               # config.py:851
    .get_agent_config()                            # config.py:873
)
```

每步只更新 `self.agent_params` 字典中的某个 key；最终 `get_agent_config()` 用 `AgentConfig(**self.agent_params)` 实例化。

**关键转换**：

- **build_core_properties**（config.py:403-436）：纯字段 mapping，提供默认值（`max_iterations=100` / `tool_call_mode="openai"` 等）。
- **build_mcp_servers**（config.py:438-491）：再次 type-narrow 校验（schema 已校验 type 字段，这里防御性二次确认 + 列出所有 MCP 配置）。
- **build_hooks**（config.py:493-608）：5 类 hooks（`middlewares` / `after_model_hooks` / `after_tool_hooks` / `before_model_hooks` / `before_tool_hooks`），每类逻辑相同：`_import_and_instantiate` 把字符串 / dict 转成 callable。
- **build_tracers**（config.py:610-643）：tracer 必须 `isinstance(tracer, BaseTracer)`，否则报错。
- **build_tools**（config.py:645-663 + `_load_tool_from_config` config.py:882-937）：解析 `yaml_path`（支持 `pkg:resource` 与文件路径），`extra_kwargs` 不允许包含 `agent_state` / `global_storage` 这两个 reserved key（防止 user override 框架字段）。
- **build_sub_agents**（config.py:698-747）：递归 `AgentConfig.from_yaml`，`config_path` 同样支持 `pkg:resource` 与文件路径。

#### 5. Hook 配置三态

`_import_and_instantiate`（config.py:323-369）支持三种形式：

```yaml
# 形式 1：字符串 import string
middlewares:
  - "nexau.archs.middleware.context_compaction:ContextCompactionMiddleware"

# 形式 2：dict with import + params
middlewares:
  - import: "nexau.archs.middleware.llm_failover:LLMFailoverMiddleware"
    params:
      retry_count: 3
      backoff_path: "./backoff.json"   # 自动相对 base_path 解析

# 形式 3：直接 callable（仅 overrides 路径用）
middlewares: [my_callable_function]
```

**`_path` / `_file` 后缀字段相对解析**（config.py:357-363）：dict 形式 params 中所有以 `_path` / `_file` 结尾的字符串值如果不是绝对路径，自动用 `base_path / value` 解析。

**`_instantiate_hook_object`**（config.py:371-401）区分 class（用 `**params_dict` 实例化）与 factory function（带 params 时调用、不带 params 时直接当 hook 对象）。

#### 6. `_finalize` 后置自动注入

`AgentConfig._finalize`（config.py:231-274）是 Pydantic `model_validator(mode="after")`：

- **注入 `recall_sub_agent` 工具**（config.py:236-244）：仅当 `sub_agents` 非空时；从 `nexau/archs/tool/builtin/description/recall_sub_agent_tool.yaml` 加载。
- **注入 `load_skill` 工具**（config.py:261-263）：从 skills 列表构造一个聚合 tool，让 LLM 可以"按需加载技能"。
- **解析 `resolved_tracer`**（config.py:266-271）：单 tracer 直接赋值；多 tracer 包装 `CompositeTracer`；零 tracer 设 `None`。
- **`_is_finalized` 防重入**（config.py:234-235）：避免后续 `model_dump` → revalidate 时重复注入。

#### 7. Sub-agent / Tool YAML 路径解析

`build_sub_agents`（config.py:716-734）与 `_load_tool_from_config`（config.py:919-927）共用同一套路径解析逻辑：

```python
if ":" in path_string:
    pkg, resource_path = path_string.split(":", 1)
    from importlib.resources import as_file, files
    resource = files(pkg).joinpath(resource_path)
    with as_file(resource) as resolved_path:
        ...  # 用 resolved_path
else:
    p = Path(path_string)
    if not p.is_absolute():
        p = base_path / p
    ...  # 用 p
```

这让框架内置 tool / sub-agent（`nexau.archs.tool.builtin:description/foo.yaml`）与用户自定义文件（`./my_tool.yaml`）共用一套配置语法。

#### 8. Deprecated 兼容层

- **`load_agent_config`** (`archs/config/config_loader.py:26`)：旧版加载器，被 `@deprecated` 标记，建议替换为 `Agent.from_yaml(...)`；保留以避免 0.x → 0.4.0 强制断裂。
- **`overrides` 参数**（schema.py:129-137 + config.py:151-158）：deprecated，v0.4.0 移除；新模式：`config = AgentConfig.from_yaml(...)` → `config.key = value` 手动覆盖。
- **`ConfigError` 多源定义**（config.py:65, schema.py:19, common.py:9, config_loader.py:13）：四处独立定义同名 exception，跨模块 try/except 时不一定能捕到——属于历史欠债，未解决。

### 示例

YAML 配置 `examples/code_agent/code_agent.yaml`（简化）：

```yaml
variables:
  shared_model: claude-sonnet-4-5
  api_url: ${env.ANTHROPIC_API_BASE}

llm_config:
  model: ${variables.shared_model}
  api_base: ${variables.api_url}

system_prompt: ${this_file_dir}/systemprompt.md
system_prompt_type: file

tools:
  - name: read_file
    yaml_path: nexau.archs.tool.builtin:description/read_file.yaml
    binding: nexau.archs.tool.builtin.file:read_file_impl
  - name: my_tool
    yaml_path: ./tools/my_tool.yaml
    binding: my_pkg.tools:my_tool_impl
    extra_kwargs:
      max_size: 1000

sub_agents:
  - name: searcher
    config_path: ./sub_agents/searcher.yaml

mcp_servers:
  - name: github
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: ${env.GITHUB_TOKEN}
```

启动：

```bash
nexau chat ./code_agent.yaml --query "list this repo's TODOs"
nexau serve http ./code_agent.yaml --port 8080
echo '{"jsonrpc":"2.0-stream","method":"agent.query",...}' | nexau serve stdio ./code_agent.yaml
```

加载流程：

1. `chat.main` → `build_agent` → `load_yaml_with_vars`
2. 文本层：`${this_file_dir}` / `${env.GITHUB_TOKEN}` / `${env.ANTHROPIC_API_BASE}` 替换
3. 第一次 `yaml.safe_load` → 取 `variables:` → `${variables.shared_model}` 替换
4. 第二次 `yaml.safe_load` → dict
5. `AgentConfigSchema.model_validate` → Pydantic 校验（mcp_server.type=stdio → 必须有 command）
6. `AgentConfigBuilder` 11 段链：注册 LLM → MCP → hooks → tracers → tools → sub-agents → skills → system_prompt → sandbox
7. `_finalize` 注入 `recall_sub_agent`（因 sub_agents 非空）+ `load_skill` + 设置 tracer
8. `Agent(config=config, session_manager, user_id, session_id)` 实例化
9. 进入 REPL 循环

## 权衡取舍

### 考虑过的替代方案

1. **用 `click` / `typer` 替代 argparse**：拒绝原因：argparse 标准库零依赖、subcommand 写法成熟；本 CLI 复杂度低（3 子命令），不需要装饰器糖。
2. **YAML 用单次 safe_load + custom `!env` tag**：拒绝原因：custom tag 需要 SafeLoader 子类化、`${variables.x}` 跨字段引用难表达、调试复杂；当前文本层正则替换简单透明。
3. **builder 链改为单函数 `from_yaml(dict) -> AgentConfig`**：拒绝原因：单函数会膨胀到 800+ 行；按职责分 11 段后每段 30-100 行可单测，错误定位精确。
4. **`mcp_servers` schema 不用 discriminated union，用单个超集 dict**：拒绝原因：超集 dict 无法在 schema 层区分 stdio 必须 `command` vs http 必须 `url`，校验需要散落在 builder；discriminated union 让 Pydantic 一次校验完。
5. **不支持 `pkg:resource` 路径，强制文件路径**：拒绝原因：框架内置 tool / sub-agent 散落在 site-packages 中，文件路径方案要求用户硬编码安装位置；`importlib.resources` 是标准做法。
6. **不引入 deprecated 包装层，直接 break**：拒绝原因：用户脚本已经在用 `load_agent_config` 与 `overrides`，强制 break 引发反弹；`@deprecated` + warning 给一个版本的迁移窗口。
7. **`ConfigError` 全局单一定义**：理论上对，但 `ConfigError` 在 4 个模块各定义一份是历史欠债；统一时需要协调跨包 import 顺序，作为后续 cleanup（未解决问题 #4）。
8. **保留 Node.js CLI（`nexau-cli`）作为主要入口**：拒绝原因：Node.js 入口仅是包装器，核心逻辑全在 Python；新 `nexau` 命令是首选，`nexau-cli` 仅向后兼容。
9. **`recall_sub_agent` / `load_skill` 工具让用户在 YAML 显式列出**：拒绝原因：这两个是框架隐式契约（有 sub_agents 必有 recall、有 skills 必有 load_skill），让用户写是冗余且易漏；自动注入是对的。

### 缺点

1. **`ConfigError` 跨 4 处独立定义**（config_loader.py:13, schema.py:19, common.py:9, config.py:65）：用户 `try/except ConfigError` 时取的是哪个 import 不确定，潜在不可靠的错误处理。
2. **`load_yaml_with_vars` 两次 `yaml.safe_load`**：大 YAML 文件（含 base64 system prompt 等）时双倍解析开销；当前未优化，因 YAML 通常 < 1MB。
3. **builder 链 11 步硬编码顺序**：调换某两步（如 `build_skills` 在 `build_tools` 之前）会失败，但顺序未在代码注释中说明；新人易踩坑。
4. **`overrides` deprecation 周期长**：`@deprecated` warning 在 v0.4.0 才硬删，在此之前 `overrides` 路径与 `apply_agent_name_overrides` 都需要维护。
5. **`system_prompt_path` 作为可重入步骤**：`build_system_prompt_path` 在 `from_yaml` 链与 `chat.py:74` `build_agent_from_config` 中各调用一次，重复了；功能正确（idempotent），但语义不清晰。
6. **Node.js wrapper（`cli_wrapper.py`）四个 fallback 路径**（cli_wrapper.py:33-77）：维护成本高，每次包结构变化都需更新；待 Node.js CLI 完全弃用后整体删除。
7. **`_import_and_instantiate` 不区分 class 与 factory function 类型注解**：用户 hook 写错 import string 时报"TypeError: ... takes 0 positional arguments but 2 given"，错误信息不友好。
8. **YAML schema 无 `$schema` 引用**：用户在 IDE 写 YAML 没有 autocomplete / 校验；理论上可生成 JSON Schema 暴露给 vscode-yaml extension，但目前未做。
9. **`load_dotenv()` 在 config.py:50 模块级隐式调用**：副作用埋在 import 里；测试中需要清理 env 时易混乱。

## 实现计划

### 阶段划分

- [x] Phase 1: argparse subcommand 框架 (`cli/main.py:12-65`)
- [x] Phase 2: chat / serve http / serve stdio 三入口 (`cli/commands/*.py`)
- [x] Phase 3: `load_yaml_with_vars` 三种插值 + 两次 safe_load (`utils/common.py:57-111`)
- [x] Phase 4: `AgentConfigSchema` Pydantic 校验 + discriminated union (`schema.py:86, 75`)
- [x] Phase 5: `AgentConfigBuilder` 11 段链 (`config.py:308-880`)
- [x] Phase 6: hook 三态 + `_path`/`_file` 自动相对解析 (`config.py:323-401`)
- [x] Phase 7: sub-agent / tool 双路径解析 (`pkg:resource` + filesystem) (`config.py:716-734, 919-927`)
- [x] Phase 8: `_finalize` 自动注入 `recall_sub_agent` / `load_skill` + tracer (`config.py:231-274`)
- [x] Phase 9: deprecated 兼容层 (`config_loader.py:26`, `schema.py:155-196`, `config.py:151-158`)
- [x] Phase 10: pyproject 双 entry point (`nexau` + `nexau-cli`)
- [ ] Phase 11（未来）：`ConfigError` 收敛到单一定义
- [ ] Phase 12（未来）：从 schema 自动生成 JSON Schema 暴露给 IDE
- [ ] Phase 13（未来）：v0.4.0 移除 `overrides` 与 `load_agent_config` + Node.js wrapper

### 相关文件

- `nexau/cli/main.py` - argparse dispatcher
- `nexau/cli/__main__.py` - `python -m nexau.cli` 入口
- `nexau/cli/commands/chat.py` - 交互 REPL 子命令
- `nexau/cli/commands/http.py` - HTTP server 子命令
- `nexau/cli/commands/stdio.py` - Stdio server 子命令
- `nexau/cli_wrapper.py` - Node.js CLI 包装器（legacy）
- `nexau/archs/main_sub/utils/common.py` - `load_yaml_with_vars` 与 `import_from_string`
- `nexau/archs/main_sub/config/schema.py` - Pydantic schema
- `nexau/archs/main_sub/config/config.py` - `AgentConfig` + `AgentConfigBuilder`
- `nexau/archs/config/config_loader.py` - deprecated `load_agent_config`
- `pyproject.toml` - `[project.scripts]` entry points

## 测试方案

### 单元测试

- **`load_yaml_with_vars` ${env.X} 替换**：set `os.environ["FOO"]="bar"`；YAML 内 `key: ${env.FOO}`；断言 `result["key"] == "bar"`。未设置时抛 `ConfigError`。
- **`load_yaml_with_vars` ${this_file_dir} 替换**：YAML 内 `path: ${this_file_dir}/sub.yaml`；断言 `result["path"]` 为 YAML 文件目录的绝对路径 + `/sub.yaml`。
- **`load_yaml_with_vars` ${variables.x.y} 嵌套**：YAML 含 `variables: {a: {b: "v"}}` 与 `key: ${variables.a.b}`；断言 `result["key"] == "v"`。解析到 dict 抛错。
- **`AgentConfigSchema` discriminated union**：构造 `mcp_servers: [{type: "stdio", name: "x", command: "npx"}]` 通过；改成 `{type: "stdio", name: "x", url: "..."}` 抛 ValidationError（缺 `command` + 多余 `url`）。
- **`AgentConfigSchema._require_llm_config`**：YAML 缺 `llm_config:`，断言 `from_yaml` 抛 `ConfigError` 含 "llm_config is required"。
- **`AgentConfigBuilder._import_and_instantiate` 三态**：分别传 string / dict / callable；断言返回 callable，dict 形式 params 中 `_path` 后缀字段被相对 `base_path` 解析。
- **`AgentConfigBuilder.build_tools` reserved key 拒绝**：tool config 含 `extra_kwargs: {agent_state: 1}`；断言抛 `ConfigError` 含 "reserved keys"。
- **`AgentConfigBuilder.build_sub_agents` package 路径**：`config_path: "nexau.archs.tool.builtin:description/X.yaml"`；mock `importlib.resources.files`；断言走 package 分支。
- **`_finalize` 注入 recall_sub_agent**：构造 `AgentConfig(sub_agents={"x": ...}, ...)`；断言 `tools` 包含 name="recall_sub_agent" 工具。无 sub_agents 时不注入。
- **`_finalize` CompositeTracer**：传 `tracers=[t1, t2]`；断言 `resolved_tracer` 是 `CompositeTracer` 实例。

### 集成测试

- **`nexau chat <yaml> --query`**：用 fixture YAML（最小 llm_config + 一个 echo tool）跑 `nexau chat fixture.yaml --query "hi"`；断言 stdout 含 echo 响应、退出 0。
- **`nexau serve http`**：`subprocess.Popen(["nexau", "serve", "http", "fixture.yaml", "--port", "0"])`；poll `/health` 200；POST `/query` 收到响应；SIGINT 优雅退出。
- **`nexau serve stdio`**：`subprocess` 启 stdio server，写 JSON-RPC `agent.query` 到 stdin；读 stdout 一行解析；断言 `result` 字段非空。
- **YAML loading 错误定位**：故意拼错 YAML 字段（`commnad`），断言 `ConfigError` 文本含字段路径与原因。
- **deprecated `load_agent_config` 仍工作**：导入并调用，断言收到 `DeprecationWarning` 但成功返回 Agent。

### 手动验证

1. 写一个含 `${env.X}` `${variables.y}` `${this_file_dir}` 三种插值的复杂 YAML，跑 `nexau chat`，确认变量正确注入。
2. 在 `tools[].extra_kwargs` 写 `agent_state: 1`，确认报错信息清晰指向 reserved key。
3. 用 `nexau-cli`（Node.js wrapper）跑同 YAML，确认 legacy 兼容。

## 未解决的问题

1. **`ConfigError` 多源定义统一**：4 处独立定义同名 exception，何时统一到 `nexau.errors`？需要梳理 import 顺序避免循环依赖。
2. **builder 链顺序约束文档化**：11 段调用顺序硬编码但无注释说明依赖关系（如 `build_skills` 必须在 `build_tools` 后）。是否补一段 invariant 注释或 dependency graph？
3. **`overrides` v0.4.0 退役 timeline**：何时硬删 `apply_agent_name_overrides` + `from_yaml(overrides=...)`？需要发版本公告 + migration guide。
4. **YAML schema → JSON Schema 自动暴露**：是否生成 `nexau.schema.json` 让 vscode-yaml extension 提供自动补全 / 实时校验？
5. **`load_dotenv()` 模块级副作用**：是否改为显式调用以提升测试可控性？
6. **Node.js wrapper 退役时机**：`nexau-cli` 何时硬删？需要统计仍在使用的用户脚本数量。
7. **YAML hot-reload**：长跑 server（`nexau serve http`）是否支持 `agent.yaml` 修改后热重载 agent 配置？目前必须重启进程。
8. **`recall_sub_agent` / `load_skill` 自动注入是否可选**：某些用户可能希望"自己实现 sub-agent 调度"——需要 opt-out flag？

## 参考资料

- `pyproject.toml:36` — `[project.scripts]` 节
- `pyproject.toml:38` — `nexau-cli = "nexau.cli_wrapper:main"`
- `pyproject.toml:39` — `nexau = "nexau.cli.main:main"`
- `nexau/cli/__main__.py:3` — `from nexau.cli.main import main`
- `nexau/cli/main.py:12` — `def create_parser`
- `nexau/cli/main.py:31-36` — chat subcommand
- `nexau/cli/main.py:39-47` — serve 子命令组
- `nexau/cli/main.py:50-55` — serve http
- `nexau/cli/main.py:58-63` — serve stdio
- `nexau/cli/main.py:68` — `def main(argv)`
- `nexau/cli/main.py:88-90` — KeyboardInterrupt → 130
- `nexau/cli/commands/chat.py:54` — `def build_agent`
- `nexau/cli/commands/chat.py:71` — `load_yaml_with_vars(config_path)`
- `nexau/cli/commands/chat.py:74` — `AgentConfigBuilder(...).build_system_prompt_path().get_agent_config()`
- `nexau/cli/commands/chat.py:134` — `/clear` 命令处理
- `nexau/cli/commands/chat.py:212` — `def main(args)`
- `nexau/cli/commands/chat.py:236-238` — SQLite 路径与目录创建
- `nexau/cli/commands/http.py:21` — `class ServerArgs(BaseModel)`
- `nexau/cli/commands/http.py:75` — `def main(args)`
- `nexau/cli/commands/http.py:118` — `AgentConfig.from_yaml(agent_path)`
- `nexau/cli/commands/http.py:128` — `HTTPConfig(...)` 构造
- `nexau/cli/commands/http.py:136` — `SSETransportServer(...)`
- `nexau/cli/commands/stdio.py:21` — `class ServerArgs(BaseModel)`
- `nexau/cli/commands/stdio.py:46` — `def server_main(args)`
- `nexau/cli/commands/stdio.py:76` — `stream=sys.stderr` 日志重定向
- `nexau/cli/commands/stdio.py:99` — `StdioTransport(...)`
- `nexau/cli_wrapper.py:27` — `def find_node_cli`
- `nexau/cli_wrapper.py:82` — `def main()` Node.js wrapper
- `nexau/archs/main_sub/utils/common.py:18` — `def import_from_string`
- `nexau/archs/main_sub/utils/common.py:57` — `def load_yaml_with_vars`
- `nexau/archs/main_sub/utils/common.py:62` — `${this_file_dir}` 替换
- `nexau/archs/main_sub/utils/common.py:65` — `${env.X}` 正则
- `nexau/archs/main_sub/utils/common.py:69-71` — env 未设置抛 ConfigError
- `nexau/archs/main_sub/utils/common.py:76` — 第一次 `yaml.safe_load`
- `nexau/archs/main_sub/utils/common.py:90-92` — `${variables.x.y}` 正则
- `nexau/archs/main_sub/utils/common.py:101-104` — variable 解析到 non-scalar 抛错
- `nexau/archs/main_sub/utils/common.py:107` — 第二次 `yaml.safe_load`
- `nexau/archs/main_sub/utils/common.py:110` — `pop("variables", None)`
- `nexau/archs/main_sub/config/schema.py:25-35` — `class ToolConfigEntry` extra=forbid
- `nexau/archs/main_sub/config/schema.py:46-54` — `class MCPServerBaseModel` extra=forbid
- `nexau/archs/main_sub/config/schema.py:57-60` — `MCPStdIOServer(type=Literal["stdio"], command)`
- `nexau/archs/main_sub/config/schema.py:63-66` — `MCPHttpServer(type=Literal["http"], url)`
- `nexau/archs/main_sub/config/schema.py:69-72` — `MCPSseServer(type=Literal["sse"], url)`
- `nexau/archs/main_sub/config/schema.py:75` — `MCPServerConfig = MCPStdIOServer | MCPHttpServer | MCPSseServer`
- `nexau/archs/main_sub/config/schema.py:86` — `class AgentConfigSchema`
- `nexau/archs/main_sub/config/schema.py:103-104` — `_require_llm_config` validator (装饰器 + def)
- `nexau/archs/main_sub/config/schema.py:109-110` — `def from_yaml` classmethod (装饰器 + def)
- `nexau/archs/main_sub/config/schema.py:155-196` — `apply_agent_name_overrides` (deprecated)
- `nexau/archs/main_sub/config/schema.py:199` — `def normalize_agent_config_dict`
- `nexau/archs/main_sub/config/schema.py:216` — `def _format_validation_error`
- `nexau/archs/main_sub/config/config.py:50` — `dotenv.load_dotenv()` 模块级
- `nexau/archs/main_sub/config/config.py:65` — `class ConfigError(Exception)`
- `nexau/archs/main_sub/config/config.py:94` — `class AgentConfig(...)`
- `nexau/archs/main_sub/config/config.py:134-135` — `from_yaml` classmethod (装饰器 + def)
- `nexau/archs/main_sub/config/config.py:151-158` — `overrides` deprecation warning
- `nexau/archs/main_sub/config/config.py:170-183` — builder 链 11 段
- `nexau/archs/main_sub/config/config.py:231-232` — `_finalize` model_validator (装饰器 + def)
- `nexau/archs/main_sub/config/config.py:236-244` — 注入 `recall_sub_agent` 工具
- `nexau/archs/main_sub/config/config.py:261-263` — 注入 `load_skill` 工具
- `nexau/archs/main_sub/config/config.py:266-271` — `resolved_tracer` 解析
- `nexau/archs/main_sub/config/config.py:308` — `class AgentConfigBuilder`
- `nexau/archs/main_sub/config/config.py:323` — `_import_and_instantiate`
- `nexau/archs/main_sub/config/config.py:357-363` — `_path`/`_file` 后缀自动相对解析
- `nexau/archs/main_sub/config/config.py:371` — `_instantiate_hook_object`
- `nexau/archs/main_sub/config/config.py:403` — `build_core_properties`
- `nexau/archs/main_sub/config/config.py:438` — `build_mcp_servers`
- `nexau/archs/main_sub/config/config.py:493` — `build_hooks`
- `nexau/archs/main_sub/config/config.py:610` — `build_tracers`
- `nexau/archs/main_sub/config/config.py:645` — `build_tools`
- `nexau/archs/main_sub/config/config.py:665` — `build_skills`
- `nexau/archs/main_sub/config/config.py:698` — `build_sub_agents`
- `nexau/archs/main_sub/config/config.py:716-734` — sub-agent `pkg:resource` / filesystem 双路径解析
- `nexau/archs/main_sub/config/config.py:749` — `build_llm_config`
- `nexau/archs/main_sub/config/config.py:828` — `build_system_prompt_path`
- `nexau/archs/main_sub/config/config.py:841-842` — 相对 `base_path` 解析
- `nexau/archs/main_sub/config/config.py:851` — `build_sandbox`
- `nexau/archs/main_sub/config/config.py:861` — `set_overrides`
- `nexau/archs/main_sub/config/config.py:873` — `get_agent_config`
- `nexau/archs/main_sub/config/config.py:882` — `_load_tool_from_config`
- `nexau/archs/main_sub/config/config.py:912-917` — `extra_kwargs` reserved key 校验
- `nexau/archs/main_sub/config/config.py:919-927` — tool `pkg:resource` / filesystem 双路径解析
- `nexau/archs/config/config_loader.py:13` — `class ConfigError` (重复定义)
- `nexau/archs/config/config_loader.py:19` — `@deprecated` decorator
- `nexau/archs/config/config_loader.py:26` — `def load_agent_config` (deprecated)

### 相关 RFC

- `rfcs/0006-rfc-catalog-completion-master-plan.md` — 本 RFC 的任务 T11
- `rfcs/0007-tool-system-architecture-and-binding.md` — `Tool.from_yaml` 是 `_load_tool_from_config` 的下游
- `rfcs/0011-middleware-hook-composition.md` — `build_hooks` 五类 hook 配置的下游消费
- `rfcs/0013-subagent-delegation-and-context-propagation.md` — `build_sub_agents` 与 `recall_sub_agent` 自动注入的下游
- `rfcs/0014-prompt-engineering-and-templating.md` — `build_system_prompt_path` 与 jinja prompt 渲染衔接
- `rfcs/0015-transport-routing-and-dispatch.md` — `nexau serve http` / `nexau serve stdio` 的下游 transport 实现
- `rfcs/0016-mcp-resource-discovery-and-recovery.md` — `mcp_servers` 配置的下游消费
