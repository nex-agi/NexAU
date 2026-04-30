# Windows 支持基线与验证矩阵（RFC-0020 / T1）

> 内部基线文档。
>
> RFC-0020 T1 的目标不是直接对外发布完整 Windows 指南，而是在仓库内落地一份可被
> CI、测试和文档共同引用的统一基线，避免后续在 workflow、测试说明和公开文案中出现多套表述。

## 1. 支持范围基线

RFC-0020 对外支持声明必须统一为以下语义：

- **Supported OS**: Windows 10、Windows 11
- **Default shell backend**: PowerShell（`pwsh.exe` → `powershell.exe` → `cmd.exe`）
- **Optional shell backend**: Git Bash（Git for Windows，bash-compatible mode）
- **Primary entrypoint**: `nexau`
- **Compatible entrypoints to keep validated**:
  - `npm run run-agent` / `npm run agent`
  - legacy `nexau-cli`
  - Windows 兼容脚本入口（如 `run-agent.cmd`）
- **Windows official support boundary**: `Windows 10 / 11 + default PowerShell backend`
- **Git Bash compatibility boundary**: explicit Git Bash backend or bash-only command scenarios only

### 明确不包含

以下内容不属于 RFC-0020 的 Windows 正式支持承诺：

- PowerShell 自动兼容所有 bash-only 命令、bash heredoc 或 POSIX 专有语法
- 未安装 Git Bash 时的自动交互式安装流程
- WSL 作为默认前置依赖
- E2B SaaS / Self-host 远端 Linux 语义被视为“Windows 本地支持”的一部分
- `rg`、`ffmpeg`、Node.js 等外部依赖能力矩阵的细粒度公开承诺
- Makefile / 开发工具链的完整 Windows 可移植化

---

## 2. Windows Required Checks 基线

Windows CI 的目标是成为 **阻塞合并** 的 required checks，但不机械复制所有 Linux-only job。
本基线先固定 required checks 的**命名、边界与职责**，具体 workflow 实现由 RFC-0020 T2 落地。

### 2.1 计划中的 required checks

| Check 名称 | Runner | 阻塞合并 | 覆盖范围 | 明确不覆盖 |
| --- | --- | --- | --- | --- |
| `windows-quality` | `windows-latest` | 是 | Python / `uv` / Node.js 准备、默认 PowerShell backend 校验、lint、format-check、typecheck | E2B SaaS / Self-host、人工交互安装 |
| `windows-target-tests` | `windows-latest` | 是 | Windows 目标 pytest 集合、平台 helper、路径/进程差异、shell backend 差异、关键降级路径回归 | 全量 Linux-only 测试复制 |
| `windows-entrypoint-smoke` | `windows-latest` | 是 | 三类启动入口 smoke / integration、默认 PowerShell 链路、显式 Git Bash 缺失依赖链路（fail-fast、上层接管提示、重探测） | 发布前人工验收、真实交互安装 |

### 2.2 Required checks 边界规则

1. **必须阻塞合并**：上述 3 个 Windows checks 必须进入 PR 阻塞面，不能作为允许失败的观察性 job。
2. **不复制 Linux-only job**：`test-saas`、`test-selfhost` 继续留在 Linux runner，不作为 Windows required checks 的组成部分。
3. **不依赖交互式安装**：CI 中默认验证 PowerShell backend；Git Bash 作为可选 backend，以 runner 现有环境校验或测试层可控模拟为主，不依赖交互式安装。
4. **日志必须可诊断**：CI 日志至少能区分环境准备失败、默认 shell backend 前提失败、显式 Git Bash backend 前提失败、测试回归、入口 smoke 失败五类问题。
5. **不能靠大面积 skip 过关**：新增平台 skip / marker 必须有显式原因，并能在 CI 日志中追踪。

---

## 3. 启动入口与 shell backend 链路责任矩阵

RFC-0020 明确持续验证对象分为 **三类启动入口** 与 **两类 shell backend 链路**。
后续任务必须按下表收敛，避免验证职责分散。

此外，后续自动化验证与人工验收的场景基线，至少必须覆盖以下三类：

- `powershell-default-healthy`
- `git-bash-explicit-healthy`
- `git-bash-explicit-missing`
- `git-bash-explicit-unusable`

### 3.1 三类启动入口

| 验证对象 | 代表命令 / 入口 | 主验证层 | 归属任务 | 说明 |
| --- | --- | --- | --- | --- |
| Python CLI 主入口 | `nexau ...` | Windows smoke / integration | T4 | 主支持路径 |
| npm / 脚本入口 | `npm run agent` / `npm run run-agent` / `run-agent.cmd` | Windows smoke / integration | T4 | 对用户可见，未弃用前必须持续验证 |
| legacy wrapper | `nexau-cli` | Windows smoke / integration | T4 | 继续验证直到另有弃用 RFC |

### 3.2 两类 shell backend 链路

| 验证对象 | 关键分支 | 主验证层 | 归属任务 | 说明 |
| --- | --- | --- | --- | --- |
| PowerShell 默认链路 | `pwsh.exe` 探测成功 | CI + 单元测试 + smoke | T2 / T3 / T4 | 默认优先路径 |
| PowerShell 默认链路 | 降级到 `powershell.exe` | 单元测试 + integration | T3 / T4 | 验证默认 backend fallback |
| PowerShell 默认链路 | 最后兜底 `cmd.exe` | 单元测试 + integration | T3 / T4 | 仅承诺基础诊断能力，不扩张为推荐 shell |
| Git Bash 可选链路 | 显式 backend 探测成功 | CI + 单元测试 + smoke | T2 / T3 / T4 | 仅在显式 Git Bash / bash-only 场景需要 |
| Git Bash 可选链路 | 显式 backend 缺失依赖 fail-fast | 单元测试 + integration | T3 / T4 | 以模拟验证为主，不依赖真实交互安装 |
| Git Bash 可选链路 | 文件存在但不可用 / 健康检查失败 | 单元测试 + integration | T3 / T4 | 需与 missing 场景区分，验证错误分类、提示文案与探测稳定性 |
| Git Bash 可选链路 | 上层接管提示 | 单元测试 + integration | T3 / T4 | 固化错误信息与提示文案 |
| Git Bash 可选链路 | 重探测 / 重新进入显式 Git Bash 路径 | integration | T4 | 验证缺失分支后的恢复链路 |

---

## 4. 平台差异验证归属矩阵

以下高风险点必须有明确归属，避免“某处顺手测了一下”但没有持续保护：

| 风险点 | 主要验证层 | 归属任务 | 说明 |
| --- | --- | --- | --- |
| 路径转换（Windows 原生路径 ↔ PowerShell / Git Bash 路径） | 单元测试 | T3 | 验证不回退到 `/tmp` 等硬编码，且转换按 backend 分流 |
| 进程生命周期（启动、轮询、终止） | 单元测试 + 目标 pytest 集合 | T3 / T2 | Windows / POSIX 差异集中治理 |
| `rg` 缺失 fallback | 单元测试 + 目标 pytest 集合 | T3 / T2 | 属于关键降级路径 |
| `ffmpeg` 缺失降级 | 单元测试 + 目标 pytest 集合 | T3 / T2 | 属于关键降级路径 |
| 解释器调用差异（如 `python3` / `sys.executable` / shell path） | 单元测试 + 目标 pytest 集合 | T3 / T2 | 防止 Windows 反斜杠与命令选择回归 |
| 实际命令执行路径必须走 active shell backend | smoke / integration | T4 | 防止残留 `shell=True` → 宿主默认 shell 回退 |
| 公开文档表述与支持边界 | 文档评审 + 引用该基线 | T5 | README / docs / Windows 页统一口径 |

---

## 5. T2 / T3 / T4 / T5 的引用约束

### T2（Windows CI）必须引用本基线

- Workflow / job 命名应与本文件中的 required checks 名称保持一致，除非在 RFC-0020 中追加修订。
- Windows job 的日志与失败分类应沿用本文件第 2 节的边界定义。
- CI 中不应自行扩张支持承诺，例如把“Windows + PowerShell 默认，Git Bash 可选”写成“所有 Windows shell 全支持”。

### T3（测试基建）必须引用本基线

- marker / fixture / helper 的设计，应围绕本文件第 3、4 节定义的验证对象与风险点组织。
- 对缺失依赖、降级分支、平台差异的注入能力，应优先服务于 `windows-target-tests` 与 `windows-entrypoint-smoke`。

### T4（入口与 shell backend 链路）必须引用本基线

- 自动化 smoke / integration 的覆盖面必须至少包含本文件第 3 节列出的三类入口与两类 shell backend 链路。
- 报错与提示文案校验应与“显式 Git Bash 缺失依赖 fail-fast + 上层接管提示”语义一致；默认 PowerShell 路径不得因缺失 Git Bash 失败。

### T5（Windows 文档）必须引用本基线

- `docs/windows.md`、`README.md`、`README_CN.md`、`docs/getting-started.md`、`docs/index.md` 的公开支持表述，必须复用本文件第 1 节的统一语义。
- 文档只承诺 `Windows 10 / 11 + default PowerShell backend`，并将 Git Bash 描述为可选 bash-compatible backend；不擅自扩张到未被验证的 shell、runner 或依赖能力矩阵。

---

## 6. 当前仓库现状（T1 落地时记录）

截至 RFC-0020 T1 落地时：

- `.github/workflows/ci.yml` 仅包含 Ubuntu runner 的 `lint`、`typecheck`、`test-saas`、`test-selfhost`。
- Windows required checks **尚未接入**，这是 T2 的直接输入。
- `pytest.ini` 目前仅定义通用 marker，尚未形成面向 Windows 平台差异的集中约定，这是 T3 的直接输入。
- README / Getting Started / docs index 尚未形成独立 Windows 页面入口，这是 T5 的直接输入。

因此，本文件的角色是：

1. 为 T2 提供固定的 required checks 命名和边界；
2. 为 T3 提供统一的验证对象与平台风险点清单；
3. 为 T4 提供入口 / shell backend 链路的最小覆盖面；
4. 为 T5 提供统一的公开支持表述。
