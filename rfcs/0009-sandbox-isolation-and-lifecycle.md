# RFC-0009: 沙箱隔离与生命周期管理

- **状态**: implemented
- **优先级**: P1
- **标签**: `architecture`, `sandbox`, `lifecycle`, `isolation`
- **影响服务**: `nexau/archs/sandbox/`, `nexau/archs/main_sub/agent.py` (sandbox 集成路径)
- **创建日期**: 2026-04-16
- **更新日期**: 2026-04-16

## 摘要

NexAU 用一个**统一的 Sandbox 抽象**封装所有"工具与文件系统的交互边界"——bash 命令、代码执行、文件读写、文件传输、glob、目录操作等。`BaseSandbox` 定义对外接口，`BaseSandboxManager` 处理生命周期（创建 / 暂停 / 恢复 / 停止 / 状态持久化），两套并行实现 `LocalSandbox`（开发与本机调试）与 `E2BSandbox`（远端隔离）共存。沙箱默认以**懒启动 + 双检锁**的方式在第一次工具调用时创建，并通过 `SessionModel.sandbox_state` 跨请求恢复。本 RFC 记录已实现的设计意图，明确 `status_after_run` 三态生命周期、Team 共享 sandbox 模式以及与 RFC-0008（session 持久化）/ RFC-0013（子 Agent 委派）的接口。

## 动机

### 现状

工具系统（RFC-0007）的所有副作用都需要落到某个执行环境里。直接把 `subprocess.run` 暴露给工具有以下问题：
1. **隔离不可选**——开发期能跑本机进程很方便，生产期必须能切换到远端容器
2. **生命周期混乱**——每次 LLM 调用都重启沙箱浪费时间；不停又会泄漏资源
3. **跨请求状态丢失**——HTTP 无状态恢复要求沙箱能从持久化状态重连
4. **多 Agent 共享场景**——`AgentTeam` 里多个 Agent 应共享同一个沙箱，避免文件复制
5. **输出爆炸**——长 stdout 会撑爆 LLM context；需要本地文件做完整记录 + 智能截断

### 不做会怎样

- 工具实现要为每种部署形态写两份逻辑
- E2B token 没有限流时容易因为 leaks 翻车
- "中断后继续"丢失沙箱里建好的临时文件，用户重做工作
- Team 模式无法做"shared workspace"

## 设计

### 分层

```
┌────────────────────────────────────────────────────┐
│       Tool / Skill / Agent.execute                 │
│   （ 调用 sandbox.execute_bash / read_file / ... ）  │
└──────────────────────┬─────────────────────────────┘
                       │
              ┌────────▼─────────┐
              │   BaseSandbox    │  抽象操作接口
              │   (ABC)          │  (bash/code/file/upload/glob)
              └─────┬────────┬───┘
                    │        │
              ┌─────▼──┐ ┌───▼──────┐
              │Local-  │ │E2B-      │
              │Sandbox │ │Sandbox   │
              └────────┘ └──────────┘

┌─────────────────────────────────────────────────────┐
│          BaseSandboxManager[TSandbox]               │
│  • prepare_session_context() — 仅注册上下文          │
│  • start_sync() — 双检锁 + 在调用方事件循环内启动     │
│  • on_run_complete() / pause / stop                 │
│  • persist_sandbox_state / load_sandbox_state       │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────▼────────┐  ┌────────────────────┐
       │ LocalSandbox-  │  │ E2BSandbox-        │
       │ Manager        │  │ Manager            │
       └────────────────┘  └────────────────────┘
```

### 配置模型

`SandboxConfig` 是 Pydantic 判别联合（discriminated union），位于 [base_sandbox.py:108](nexau/archs/sandbox/base_sandbox.py#L108)：

| 字段 | LocalSandboxConfig ([line 76](nexau/archs/sandbox/base_sandbox.py#L76)) | E2BSandboxConfig ([line 83](nexau/archs/sandbox/base_sandbox.py#L83)) |
|------|---|---|
| `type` | `"local"` | `"e2b"` |
| `work_dir` | `$SANDBOX_WORK_DIR` 或 `cwd` | `/home/user`（E2B 默认账户） |
| `envs` | dict[str, str] | 同左 |
| `status_after_run` | `"stop"`（默认） | `"pause"`（默认，保留容器） |
| `output_char_threshold` / `truncate_head_chars` / `truncate_tail_chars` | 共享 | 共享 |
| `template` / `timeout` / `api_url` / `api_key` / `metadata` / `sandbox_id` | — | E2B 专属 |
| `keepalive_interval` / `force_http` | — | E2B 专属 |

`parse_sandbox_config()`（[line 113](nexau/archs/sandbox/base_sandbox.py#L113)）支持从 dict（YAML 来源）或 typed model 解析；`type` 缺失时默认 `"local"`，向后兼容。

### BaseSandbox：操作接口

`BaseSandbox` 在 [base_sandbox.py:312](nexau/archs/sandbox/base_sandbox.py#L312) 定义，覆盖五类能力：

| 能力 | 抽象方法 |
|------|---------|
| Bash | `execute_bash` |
| 后台任务 | `get_background_task_status`, `kill_background_task`, `list_background_tasks` |
| 代码执行 | `execute_code` (CodeLanguage.PYTHON) |
| 文件 | `read_file` / `write_file` / `delete_file` / `edit_file` / `list_files` / `file_exists` / `get_file_info` / `create_directory` / `glob` |
| 上传下载 | `upload_file` / `download_file` / `upload_directory` / `download_directory` |

非抽象的便利方法：`upload_skill`（[line 735](nexau/archs/sandbox/base_sandbox.py#L735)）把 Skill 资源目录上传到 `<work_dir>/.skills/<name>`；`dict()`（[line 752](nexau/archs/sandbox/base_sandbox.py#L752)）只序列化 `init=True` 的字段，用于 `sandbox_state` 持久化。

输出契约统一返回 dataclass：`CommandResult`（[line 147](nexau/archs/sandbox/base_sandbox.py#L147)）/ `CodeExecutionResult`（[line 227](nexau/archs/sandbox/base_sandbox.py#L227)）/ `FileOperationResult`（[line 290](nexau/archs/sandbox/base_sandbox.py#L290)），都包含 `SandboxStatus`（[line 130](nexau/archs/sandbox/base_sandbox.py#L130)：RUNNING/STOPPED/ERROR/TIMEOUT/SUCCESS）。

### Smart truncation：保完整 + 显示精华

执行结果中 stdout/stderr 默认写入 `/tmp/nexau_bash_tool_results/<uuid8>/{stdout,stderr,command}.txt`（基于 `BASH_TOOL_RESULTS_BASE_PATH`，[line 42](nexau/archs/sandbox/base_sandbox.py#L42)）。`smart_truncate_output()`（[line 182](nexau/archs/sandbox/base_sandbox.py#L182)）行为：
- 总字符数 < `threshold`（默认 10_000） → 原样返回
- 否则每个流保留 `head_chars` + `tail_chars`（默认 5_000 + 5_000），中间放 `[N characters omitted]` + 文件路径提示
- 在返回值上额外置 `truncated=True` 与 `original_*_length`，供调用方决定是否拉完整文件

这套机制由 `LocalSandbox.execute_bash` 与 `E2BSandbox.execute_bash` 共用——后台任务 / 前台超时 / 正常退出三条路径都会经过截断。

### 进程级生命周期：local 的 graceful kill

`LocalSandbox._graceful_kill()`（[local_sandbox.py:82](nexau/archs/sandbox/local_sandbox.py#L82)）三段式终止：
1. `os.killpg(pgid, SIGTERM)` 让整个进程组有清理机会
2. `process.wait(grace_period)` 等待优雅退出（默认 5s）
3. 仍存活则 `SIGKILL` + 终态 `wait(10)`

由于命令通过 `start_new_session=True` 启动（[local_sandbox.py:239, 310](nexau/archs/sandbox/local_sandbox.py#L239)），这个 SIGKILL 能覆盖派生的子进程，避免 timeout 后僵尸。

### Manager：懒启动 + 双检锁 + 状态持久化

`BaseSandboxManager` 位于 [base_sandbox.py:792](nexau/archs/sandbox/base_sandbox.py#L792)。关键方法：

| 方法 | 行 | 作用 |
|------|----|------|
| `prepare_session_context` | [934](nexau/archs/sandbox/base_sandbox.py#L934) | Agent 初始化时**只**保存上下文，不启沙箱（避免错误事件循环） |
| `start_sync` | [986](nexau/archs/sandbox/base_sandbox.py#L986) | 第一次工具调用时同步启动，使用 `_start_lock` 双检锁防重复创建 |
| `start_no_wait` | [956](nexau/archs/sandbox/base_sandbox.py#L956) | 后台线程预启动（提供 `start_future`），但不阻塞 |
| `instance` (property) | [820](nexau/archs/sandbox/base_sandbox.py#L820) | 自动等 `start_future`、必要时根据 `_session_context` 自愈式恢复 |
| `add_upload_assets` | [1022](nexau/archs/sandbox/base_sandbox.py#L1022) | 已运行则立即上传；未启动则记入 deferred 列表 |
| `persist_sandbox_state` / `load_sandbox_state` | [867](nexau/archs/sandbox/base_sandbox.py#L867) / [901](nexau/archs/sandbox/base_sandbox.py#L901) | 将 `sandbox.dict() + sandbox_type` 写入 `SessionModel.sandbox_state` 字段（RFC-0008） |
| `on_run_complete` | [859](nexau/archs/sandbox/base_sandbox.py#L859) | 一次 run 结束的钩子，无论 status_after_run 都调用（E2B 用来停 keepalive 线程） |
| `pause` / `stop` / `pause_no_wait` | abstract / [1048](nexau/archs/sandbox/base_sandbox.py#L1048) | 子类实现具体语义；`pause_no_wait` 投到独立 Thread |

懒启动的设计动机：E2B SDK 的 httpx client 必须在它将要被使用的事件循环里创建。Agent 初始化常常发生在 FastAPI 启动期或测试线程里，与真正使用沙箱的事件循环不同。`prepare_session_context` + `start_sync` 把启动推迟到工具实际调用所在的循环内，从根源上消除 cross-event-loop 错误。

### 状态恢复路径

每次 `start()` 实现都会先调 `load_sandbox_state()` 取出旧 `sandbox_state`：
- `LocalSandboxManager.start()`（[local_sandbox.py:1293](nexau/archs/sandbox/local_sandbox.py#L1293)）——若有 `sandbox_id`，用 `extract_dataclass_init_kwargs` 反序列化重建 `LocalSandbox` 实例（仅恢复元数据）
- `E2BSandboxManager.start()`（[e2b_sandbox.py:1523](nexau/archs/sandbox/e2b_sandbox.py#L1523)）——若有 `sandbox_id`，调用 E2B SDK `Sandbox.connect(id)` 复用既有容器；失败回退到新建

恢复成功后立刻再 `persist_sandbox_state()` 一次，把当前实例的字段（含 work_dir、envs、output 配置）写回 session。

### Agent 集成

Agent 在 `_initialize_sandbox()`（[agent.py:544](nexau/archs/main_sub/agent.py#L544)）里二选一：
- **共享模式**（[agent.py:558-571](nexau/archs/main_sub/agent.py#L558-L571)）——`_shared_sandbox_manager` 由 `AgentTeam` 注入，多个 sub-agent 共用一个沙箱；只 `add_upload_assets` 不再 `prepare_session_context`，cleanup 由 Team 统一负责
- **独立模式**（[agent.py:573-607](nexau/archs/main_sub/agent.py#L573-L607)）——按 config 类型选 `LocalSandboxManager` 或 `E2BSandboxManager`，调用 `prepare_session_context()` 注册上下文，并 `cleanup_manager.register_sandbox_manager()` 挂到全局清理钩子

**工具访问 sandbox 的真正路径**走 `AgentState`：Agent 在每次 run 起点构造 `AgentState`（[agent.py:902-914](nexau/archs/main_sub/agent.py#L902-L914)），把 `sandbox_manager` 而不是 sandbox 实例注入；工具调用时通过 `AgentState.get_sandbox()`（[agent_state.py:159-169](nexau/archs/main_sub/agent_state.py#L159-L169)）触发 `start_sync()`，懒启动正好发生在工具实际所在的事件循环里。RFC-0007 描述的 `tool_executor` 通过 `AgentState` 把这条路径暴露给工具上下文。

另一处对 `self.sandbox_manager.instance` 的直接读取在 [agent.py:765-772](nexau/archs/main_sub/agent.py#L765-L772)，但用途不同——它是 RFC-0032 `sandbox_env` 的运行时注入路径：若 sandbox 已启动，原地更新 `envs`；否则把新 envs 合并回 `session_context.sandbox_config`，等下次懒启动一起带上。

Run 结束后的生命周期（[agent.py:941-952](nexau/archs/main_sub/agent.py#L941-L952)）：
1. 共享模式：完全跳过本块（Team 决定何时停）
2. 独立模式：先调 `on_run_complete()` 清理 run 级资源，再根据 `status_after_run` 决定 `pause_no_wait()` / `stop()` / 完全交给调用方（`"none"` 用于 RL 训练）

### 与其他 RFC 的边界

- **RFC-0008（session 持久化）**：本 RFC 只描述 `sandbox_state` 的**写入与回读**契约；该字段如何编码到 `SessionModel.sandbox_state` 列、以何种 backend 存，是 RFC-0008 的职责
- **RFC-0007（工具系统）**：本 RFC 不涉及"工具如何描述、如何选定"；它只定义工具能用哪些副作用 API
- **RFC-0013（子 Agent 委派）**：Team 共享 sandbox 的设计动机出现在本 RFC（工程对称性），但 Team 自身的生命周期在 RFC-0013 详述
- **RFC-0001（stop 持久化）**：用户中断时 `_persist_session_state` 会顺带把 sandbox state 同步落盘；本 RFC 不重述中断路径细节

## 权衡取舍

| 选择 | 代价 | 收益 |
|------|------|------|
| **判别联合 SandboxConfig**（local / e2b 两个 model） | 新增类型要更新 union | 类型安全、能直接 from YAML 反序列化 |
| **懒启动 + 双检锁**（vs eager start） | 第一次工具调用多 100~500ms | 消除 cross-event-loop 错误；Agent 初始化路径不依赖 E2B 可用性 |
| **start_sync 同步执行 sandbox 启动**（vs 全异步） | 同步阻塞调用方协程 | E2B SDK 强约束 httpx client 必须在使用的 loop 内创建；同步路径最稳 |
| **SmartTruncate 默认开**（10k 阈值，5k+5k） | 超长输出失去中段细节 | LLM context 不会爆；完整输出仍在文件可拉取 |
| **status_after_run 默认 local=stop / e2b=pause** | E2B 不停会持续计费 | local 廉价；e2b 暂停状态恢复秒级，对话连贯性优先 |
| **sandbox_state 用 free-form JSON** | 字段演化无强约束 | 沙箱实现可独立扩展元数据；与 RFC-0008 的 GenericJSON 列对齐 |
| **共享 sandbox by manager injection** vs by id 查找 | 额外的对象传递路径 | 生命周期所有权清晰：sub-agent 不会误停 Team 的 sandbox |
| **graceful kill 三段式**（5s SIGTERM → SIGKILL → 10s wait） | 最坏路径 15s 才返回 | 真实工具（编辑器、构建脚本）能保存状态；不会留僵尸子进程 |

## 实现计划

本 RFC 是已有子系统的追溯式文档化，三大块均已就绪：

| 能力 | 状态 | 主要文件 |
|------|------|---------|
| 操作接口（BaseSandbox + Local + E2B） | implemented | [base_sandbox.py](nexau/archs/sandbox/base_sandbox.py), [local_sandbox.py](nexau/archs/sandbox/local_sandbox.py), [e2b_sandbox.py](nexau/archs/sandbox/e2b_sandbox.py) |
| 生命周期（Manager + 懒启动 + state 恢复） | implemented | `BaseSandboxManager` ([base_sandbox.py:792](nexau/archs/sandbox/base_sandbox.py#L792)) |
| Agent 集成（共享/独立 / status_after_run） | implemented | [agent.py:544-607, 941-952](nexau/archs/main_sub/agent.py#L544-L607) |

未来工作纳入"未解决的问题"。

## 测试方案

现有保障：
- `tests/archs/sandbox/test_local_sandbox.py` — 覆盖 bash 前后台、超时、graceful kill、文件 CRUD、glob、edit 三种语义、smart truncation 阈值
- `tests/archs/sandbox/test_e2b_sandbox.py` — 在 E2B mock 下覆盖 connect / create / pause / resume / keepalive
- `tests/archs/sandbox/test_sandbox_manager.py` — 双检锁并发启动、`prepare_session_context` 与 `start_sync` 的事件循环正确性、状态持久化与回读

新增补充：
- 端到端：local + e2b 两套配置跑同一个 fixture（一段 bash + 一个 python 文件 + 上传 skill），断言可观察行为一致
- `status_after_run` 三态行为：`stop` → `is_running()=False`；`pause` → `pause_no_wait` 完成后 `is_running()=False` 且重新 `start_sync` 能 connect；`none` → 调用方负责
- 测试 `_shared_sandbox_manager` 注入路径下 sub-agent 不会触发 `pause`/`stop`

## 未解决的问题

1. **`_graceful_kill` 在 Windows 上不可移植**——`os.killpg` / `signal.SIGTERM` 流程要等 LocalSandbox 跨平台时再设计
2. **E2B 容器的 cold-start 与 keepalive 策略**：当前 `keepalive_interval=60` 是经验值；缺少在多并发会话下的成本/延迟测量
3. **共享 sandbox 的并发隔离**——同一目录下多个 sub-agent 同时写文件没有显式仲裁；目前依赖工具语义（`apply_patch`、`edit_file` 单文件原子）
4. **`sandbox_state` schema 演化**——现在是 free-form JSON；如果未来 sandbox 子类增加新字段，旧 session 的恢复路径需要 schema 版本号
5. **Smart truncation 阈值是否要按模型 context 自适应**：10k 字符在 GPT-4o 32k context 下是合理的，但在 Claude 200k 下显然过紧
6. **Sandbox 内部进程的资源配额**：local 没有 cgroup 限制；e2b 受平台模板约束。需要 cap 表达层

## 参考资料

- [`nexau/archs/sandbox/base_sandbox.py:42`](nexau/archs/sandbox/base_sandbox.py#L42) — `BASH_TOOL_RESULTS_BASE_PATH`
- [`nexau/archs/sandbox/base_sandbox.py:64-110`](nexau/archs/sandbox/base_sandbox.py#L64-L110) — `BaseSandboxConfig` / `LocalSandboxConfig` / `E2BSandboxConfig` / 联合类型
- [`nexau/archs/sandbox/base_sandbox.py:113`](nexau/archs/sandbox/base_sandbox.py#L113) — `parse_sandbox_config`
- [`nexau/archs/sandbox/base_sandbox.py:130-308`](nexau/archs/sandbox/base_sandbox.py#L130-L308) — 状态枚举与三类结果 dataclass
- [`nexau/archs/sandbox/base_sandbox.py:182`](nexau/archs/sandbox/base_sandbox.py#L182) — `smart_truncate_output`
- [`nexau/archs/sandbox/base_sandbox.py:312`](nexau/archs/sandbox/base_sandbox.py#L312) — `BaseSandbox` 抽象类
- [`nexau/archs/sandbox/base_sandbox.py:735`](nexau/archs/sandbox/base_sandbox.py#L735) — `upload_skill`
- [`nexau/archs/sandbox/base_sandbox.py:792`](nexau/archs/sandbox/base_sandbox.py#L792) — `BaseSandboxManager`
- [`nexau/archs/sandbox/base_sandbox.py:820, 856-1056`](nexau/archs/sandbox/base_sandbox.py#L820) — 懒启动 / 双检锁 / 状态持久化 / 生命周期
- [`nexau/archs/sandbox/base_sandbox.py:1059-1080`](nexau/archs/sandbox/base_sandbox.py#L1059-L1080) — 异常类型层级
- [`nexau/archs/sandbox/local_sandbox.py:59`](nexau/archs/sandbox/local_sandbox.py#L59) — `LocalSandbox`
- [`nexau/archs/sandbox/local_sandbox.py:82`](nexau/archs/sandbox/local_sandbox.py#L82) — `_graceful_kill` 三段式
- [`nexau/archs/sandbox/local_sandbox.py:1281`](nexau/archs/sandbox/local_sandbox.py#L1281) — `LocalSandboxManager` 生命周期与状态恢复
- [`nexau/archs/sandbox/e2b_sandbox.py:97`](nexau/archs/sandbox/e2b_sandbox.py#L97) — `E2BSandbox`
- [`nexau/archs/sandbox/e2b_sandbox.py:1523`](nexau/archs/sandbox/e2b_sandbox.py#L1523) — `E2BSandboxManager`
- [`nexau/archs/main_sub/agent.py:544-607`](nexau/archs/main_sub/agent.py#L544-L607) — Agent `_initialize_sandbox`（共享 / 独立模式）
- [`nexau/archs/main_sub/agent.py:902-914`](nexau/archs/main_sub/agent.py#L902-L914) — AgentState 构造：注入 `sandbox_manager` 而非 sandbox 实例
- [`nexau/archs/main_sub/agent_state.py:159-169`](nexau/archs/main_sub/agent_state.py#L159-L169) — `AgentState.get_sandbox()` → `start_sync()`：工具访问 sandbox 的真正懒启动入口
- [`nexau/archs/main_sub/agent.py:765-772`](nexau/archs/main_sub/agent.py#L765-L772) — RFC-0032 `sandbox_env` 运行时注入路径（与工具执行路径不同）
- [`nexau/archs/main_sub/agent.py:941-952`](nexau/archs/main_sub/agent.py#L941-L952) — Run 结束后的生命周期处理
- RFC-0006 NexAU RFC 目录补全总纲（本 RFC 所属主计划）
- RFC-0008 会话持久化与历史管理（`sandbox_state` 字段所在）
