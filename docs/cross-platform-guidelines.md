# NexAU 跨平台编码规范

本规范适用于所有涉及平台差异的代码变更，尤其是 RFC-0019（Windows 运行时支持）与 RFC-0020（Windows CI / 测试 / 文档）相关实现。

规范提炼自对成熟跨平台项目（OpenAI Codex CLI — 原生支持 Linux / macOS / Windows 的 Rust 代码库）的系统性分析，并适配为 NexAU 的 Python 语境。

---

## 原则 1：统一接口，平台后端分模块

**核心思想**：业务代码调用统一 API，平台差异封装在集中的兼容边界模块内。

**要求**：

- 平台适配逻辑集中在 `nexau/archs/platform/`（或等效集中位置），不散落在业务模块中。
- 该兼容边界至少承载四类职责：
  - **shell backend**：bash 发现与调用（Unix 系统 bash / Windows Git Bash）
  - **process compat**：进程创建、终止、信号处理
  - **path helpers**：临时目录、输出路径、Python 原生路径 ↔ Git Bash 路径转换
  - **Git Bash setup**：探测、安装协助、重探测
- `local_sandbox.py`、`run_shell_command.py`、内置工具等业务模块通过统一接口调用平台能力，不直接做平台判断。

```python
# ❌ 散落在业务代码中
if sys.platform == "win32":
    process = subprocess.Popen([git_bash, "-c", cmd], ...)
else:
    process = subprocess.Popen(cmd, shell=True, ...)

# ✅ 调用统一接口
from nexau.archs.platform import shell_backend
process = shell_backend.execute(cmd, cwd=cwd, envs=envs, background=background)
```

**参考**：Codex `sleep-inhibitor` 模式 — 外部暴露统一 `inhibit()` API，内部按平台拆分 `linux_inhibitor` / `macos` / `windows_inhibitor` / `dummy` 四个模块。

---

## 原则 2：平台选择在初始化时完成，不在每次调用时判断

**核心思想**：Python 没有编译期 `#[cfg]`，但可以在模块加载时一次性选定后端实例，后续调用不再做运行时分支。

**要求**：

- 平台后端的选择发生在模块初始化或工厂函数中，而非每次 `execute_bash()` / `_graceful_kill()` 都检查 `sys.platform`。
- 后端实例创建后，业务代码持有的引用类型不变，行为由后端实现决定。

```python
# ❌ 每次调用都判断平台
def _graceful_kill(self, process):
    if sys.platform == "win32":
        process.terminate()
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# ✅ 初始化时选定策略
_process_killer = WindowsProcessKiller() if sys.platform == "win32" else PosixProcessKiller()

def _graceful_kill(self, process):
    _process_killer.graceful_kill(process)
```

**参考**：Codex `exit_status.rs` — Unix 版本处理 signal → 128+signal，Windows 版本只按 exit code 处理，编译期隔离，运行时不分支。

---

## 原则 3：抽象能力模型，不强行统一实现

**核心思想**：跨平台抽象的是"能力"（可超时终止、可创建临时目录），不是具体 API（`killpg` vs `TerminateProcess`）。

**要求**：

- 跨平台接口定义**做什么**（capability），不规定**怎么做**（implementation）。
- 允许不同平台的实现完全不同，只要满足相同的语义契约。

| 能力 | Unix 实现 | Windows 实现 |
|------|----------|-------------|
| 优雅终止进程 | `SIGTERM` → wait → `SIGKILL`（process group） | `terminate()` → wait → `kill()` |
| 后台进程隔离 | `start_new_session=True` | `CREATE_NEW_PROCESS_GROUP` 或等效 |
| 临时目录创建 | `tempfile.mkdtemp()` | `tempfile.mkdtemp()` |
| Shell 命令执行 | `bash -c "command"` | `git-bash.exe -c "command"` |

**参考**：Codex `sandboxing/manager.rs` — 统一 `MacosSeatbelt | LinuxSeccomp | WindowsRestrictedToken` 枚举，抽象"沙箱能力"而非实现。

---

## 原则 4：显式降级或拒绝，不偷偷弱化语义

**核心思想**：当平台能力不满足时，系统必须明确告知用户，而不是静默跳过或伪装成功。

**要求（四层降级策略）**：

1. **能力可用** → 正常执行
2. **能力缺失但有 fallback** → 降级执行 + 明确告知用户
3. **能力缺失且无 fallback** → 明确报错 + 安装指引
4. **语义无法等价保证** → 明确拒绝，不静默弱化承诺

**NexAU 具体应用**：

| 场景 | 降级行为 |
|------|---------|
| `rg` (ripgrep) 缺失 | 降级到 Python fallback grep |
| `ffmpeg` 缺失 | 降级到仅图片支持，视频处理返回清晰提示 |
| Git Bash 缺失 | 提示用户确认 → 尝试安装协助 → 安装失败则返回手动指引 |
| WSL1 / 不支持的平台 | 明确拒绝，不尝试半支持 |

```python
# ❌ 静默忽略能力缺失
try:
    result = run_with_rg(pattern)
except Exception:
    return []  # 静默返回空结果

# ✅ 明确降级
if rg_available(sandbox):
    result = run_with_rg(pattern)
else:
    logger.info("ripgrep not found, falling back to Python grep")
    result = python_grep(pattern)
```

**参考**：Codex WSL1 fail-closed、`sleep-inhibitor` dummy backend、Windows sandbox 限制写进 README。

---

## 原则 5：路径是一等公民

**核心思想**：凡是涉及路径生成、传递、解析的地方，都必须把 Windows 路径差异当作正式设计约束。

**要求**：

### 5.1 路径格式区分

NexAU 在 Windows 上存在两种路径格式：

| 格式 | 示例 | 使用场景 |
|------|------|---------|
| Python 原生路径 | `C:\Users\wn\AppData\Local\Temp\nexau_xxx` | `pathlib`、`subprocess.Popen(cwd=...)`、Python 文件 API |
| Git Bash POSIX 路径 | `/c/Users/wn/AppData/Local/Temp/nexau_xxx` | Git Bash 命令字符串内部 |

### 5.2 路径规则

- Python 层使用 Python 原生路径（`pathlib` / `os.path`）。
- Git Bash 命令字符串中的路径通过统一 helper 转为 POSIX 风格。
- `subprocess.Popen` 的 `cwd` 参数使用 Python 原生路径。
- 跨越 Python 层 ↔ Git Bash 层的路径必须通过统一 helper 显式转换。
- `shlex.quote()` / command building 在路径转换之后执行，避免反斜杠被误处理。
- 不硬编码 `/tmp` — 使用 `tempfile.gettempdir()` 或 `tempfile.mkdtemp()`。
- E2B 远端 Linux 路径（`/tmp`、`/home/user`）保持不变，不做 Windows 化。

### 5.3 路径解析

- 解析 `filepath:lineNumber:content` 格式时，必须处理 Windows 驱动器号（`C:\path:42:content` 中的第一个 `:` 不是字段分隔符）。
- 不依赖 shell 命令输出路径（如 `ls` 的输出）驱动 Python 文件 API — 改用 Python 原生遍历（`pathlib.glob()`、`os.listdir()`）。

```python
# ❌ 硬编码 /tmp
BASH_TOOL_RESULTS_BASE_PATH = "/tmp/nexau_bash_tool_results"

# ✅ 平台感知
import tempfile
BASH_TOOL_RESULTS_BASE_PATH = str(Path(tempfile.gettempdir()) / "nexau_bash_tool_results")

# ❌ 用 ls 输出驱动 Python 读文件
ls_result = sandbox.execute_bash(f"ls -1 {tmp_dir}/frame_*.jpg | sort")
frame_paths = ls_result.stdout.splitlines()
for path in frame_paths:
    sandbox.read_file(path)  # ls 输出的路径格式可能与 Python 不兼容

# ✅ 用 Python 原生遍历（仅适用于 Local / 同机文件系统）
frame_paths = sorted(Path(tmp_dir).glob("frame_*.jpg"))
for path in frame_paths:
    sandbox.read_file(str(path))
```

> 注意：上述 `Path(tmp_dir).glob(...)` 示例仅适用于 **LocalSandbox** 或“遍历与读取发生在同一侧文件系统”的场景。
> 对于 E2B / Remote Sandbox，应使用远端同侧的文件遍历抽象（如 sandbox 侧 list/read 能力），避免在本地宿主机枚举路径、再去远端环境读取文件。
 
**参考**：Codex `tui/markdown_render.rs:810` — 专门处理 `C:/...` 驱动器路径避免误判；`windows-sandbox-rs/path` — 独立路径规范化模块。
---

## 原则 6：平台适配集中到边界模块，不散落到业务代码

**核心思想**：与原则 1 互补 — 不仅正面要求"统一接口"，还反向禁止"散落分支"。

**要求**：

- 业务代码（sandbox、工具、中间件）**不直接 import** `os.killpg`、`os.getpgid`、`signal.SIGKILL` 等平台特定 API。
- 需要平台行为的场景通过 `nexau/archs/platform/` 的统一接口调用。
- 如果某个模块必须做平台判断（如 `cleanup_manager.py` 的 signal 注册），该判断应封装在该模块内部的一个明确位置，不散落到多个方法中。

```python
# ❌ 业务代码直接使用 POSIX API
import os, signal
pgid = os.getpgid(process.pid)      # Windows: AttributeError
os.killpg(pgid, signal.SIGTERM)     # Windows: not available

# ✅ 通过平台兼容层
from nexau.archs.platform.process_compat import graceful_kill
graceful_kill(process, grace_period=5.0)
```

**参考**：Codex `windows-sandbox-rs` — 独立 crate，包含 ACL、token、process、ConPTY、path normalization 等 10+ 子模块，core 代码不直接调用 Windows API。

---

## 原则 7：构建、测试、文档也要跨平台

**核心思想**：跨平台不仅是业务代码的事，CI / 测试 / 文档命令 / 安装说明也必须考虑平台差异。

**要求**：

### 7.1 CI

- Windows CI 必须作为 required check 阻塞合并。
- Windows CI 使用 Windows 原生命令 + 官方 setup action，不依赖 `make` 或 `curl | sh`。
- Windows CI 中 Git Bash 以 runner 预装为主，不依赖交互式安装协助。

### 7.2 测试

- 平台 fixture / marker / helper 集中管理（如 `tests/conftest.py` 或 `tests/utils/platform.py`），不在测试文件中散落 `skipif`。
- 核心平台差异（路径、换行、shell、进程）有明确 fixture 承载。
- 降级路径（`rg` 缺失、`ffmpeg` 缺失、Git Bash 缺失）有可控模拟验证。
- Windows CI 的接入不得减少 Linux CI 上现有测试的执行数量。

### 7.3 文档

- 文档中的命令和路径必须标注平台差异或使用平台无关写法。
- Windows 相关说明集中在独立文档页，README / Getting Started 做最小必要引用。

**参考**：Codex `cargo-bin` — 构建资源定位主动避免 Windows path length 问题；CI 矩阵同时覆盖三平台。

---

## Checklist（代码审查用）

提交涉及平台差异的代码时，逐项检查：

- [ ] 平台判断是否集中在兼容边界模块，而非散落在业务代码
- [ ] 是否在模块初始化时选定后端，而非每次调用时判断
- [ ] 能力抽象是否只定义"做什么"，不强制具体平台 API
- [ ] 能力缺失时是否有明确的降级或拒绝，而非静默忽略
- [ ] 路径生成是否使用 `tempfile` / `pathlib`，而非硬编码 `/tmp`
- [ ] 跨 Python ↔ Git Bash 边界的路径是否通过统一 helper 转换
- [ ] 路径解析是否处理了 Windows 驱动器号（`C:\...` 中的 `:`）
- [ ] 是否避免了依赖 shell 命令输出路径驱动 Python 文件 API
- [ ] 新增测试是否有平台 marker / fixture，而非散落 `skipif`
- [ ] 文档命令是否标注了平台差异
