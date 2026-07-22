# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
E2B sandbox implementation for secure cloud-based code execution.

This implementation uses E2B (https://e2b.dev) to provide isolated cloud sandboxes
for secure code execution and file operations. E2B provides proper security isolation
and is suitable for production use.
"""

# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar, override

if TYPE_CHECKING:
    from e2b import FileType as E2BFileType
    from e2b import Sandbox as E2BRawSandbox
    from e2b.sandbox.filesystem.filesystem import WriteEntry

from packaging.version import Version

from .base_sandbox import (
    E2B_DEFAULT_WORK_DIR,
    HEREDOC_PATTERN,
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    E2BSandboxConfig,
    FileInfo,
    FileOperationResult,
    SandboxConfig,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
    _env_float,
    contains_heredoc,
    extract_dataclass_init_kwargs,
    smart_truncate_output,
)

logger = logging.getLogger(__name__)

BASH_TOOL_RESULTS_BASE_PATH = "/tmp/nexau_bash_tool_results"
"""Remote Linux directory used for E2B shell stdout/stderr artifacts."""

_T = TypeVar("_T")


# =============================================================================
# E2B SDK dynamic import
# =============================================================================
# e2b is an optional dependency, loaded dynamically via import_module.
# Type annotations use real e2b types imported under TYPE_CHECKING.
# Fallback values only prevent NameError; actual code paths are guarded by E2B_AVAILABLE.

Sandbox: type[E2BRawSandbox] | None = None
FileType: type[E2BFileType] | None = None
NotFoundException: type[Exception] | None = None

try:
    _e2b = import_module("e2b")

    Sandbox = _e2b.Sandbox
    FileType = _e2b.FileType
    NotFoundException = import_module("e2b.exceptions").NotFoundException
    _e2b_available = True
except (ImportError, ModuleNotFoundError, AttributeError):
    logger.warning("E2B SDK not installed. Install it with: pip install e2b")
    _e2b_available = False

E2B_AVAILABLE = _e2b_available


# =============================================================================
# Self-host (force_http) sandbox construction primitives — NAC#1304
# =============================================================================

_TRANSPORT_REBUILD_LOCK = threading.Lock()
"""Serializes "reset transport singleton -> construct Sandbox" sequences.

The SDK's ``get_transport()`` is an unlocked check-then-set on the
process-wide ``TransportWithLogger.singleton``; this lock only serializes the
construction paths that we control (manager start paths and wrapper
reconnects). Remaining races are handled by the pre-seed/detach protocol in
``_locked_build_http_sandbox``.
"""


def _locked_build_http_sandbox(
    sandbox_id: str,
    domain: str,
    envd_access_token: str,
    envd_version: Version,
) -> E2BRawSandbox:
    """Build an HTTP raw Sandbox bound to a private transport (self-host only).

    NAC#1304: 自建部署的 raw sandbox 统一构造原语（manager start 路径与
    wrapper reconnect 路径共用）。

    Ensures all internal SDK components (``_envd_api``, ``_filesystem``,
    ``_commands``, ...) use the HTTP URL plus the ``X-Access-Token`` header,
    and that the new object's connection pool is not shared with api-server
    calls (cached api-server HTTPS connections in a shared pool interfere with
    HTTP streaming to envd and make ``commands.run`` hang indefinitely):

    - pre-seed: a fresh transport is installed into the SDK singleton slot
      *before* construction, while holding the module lock. ``get_transport()``
      never overwrites a non-None singleton, so the constructed Sandbox is
      guaranteed to bind this fresh transport rather than adopting one that a
      concurrent api call already polluted.
    - detach: after construction the singleton is reset to ``None`` so the new
      object keeps its transport as a de-facto private reference. Known
      residual: between pre-seed and detach a concurrent ``get_transport()``
      caller may still grab the fresh transport (microsecond window; accepted,
      净效果远优于共享单例的现状).
    """
    assert Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"

    from e2b.api import limits
    from e2b.api.client_sync import TransportWithLogger
    from e2b.connection_config import ConnectionConfig

    connection_config = ConnectionConfig(
        sandbox_url=f"http://49983-{sandbox_id}.{domain}",
        extra_sandbox_headers={"X-Access-Token": envd_access_token},
    )
    with _TRANSPORT_REBUILD_LOCK:
        # 1. pre-seed：锁内先占位 singleton，杜绝构造期收养他人污染的 transport
        fresh = TransportWithLogger(limits=limits, proxy=connection_config.proxy)
        TransportWithLogger.singleton = fresh
        try:
            sandbox = Sandbox(  # type: ignore[call-arg]
                sandbox_id=sandbox_id,
                sandbox_domain=domain,
                envd_access_token=envd_access_token,
                traffic_access_token=None,
                connection_config=connection_config,
                envd_version=envd_version,
            )
        finally:
            # 2. detach：解绑全局单例，新对象的 pool 私有化
            TransportWithLogger.singleton = None
        if sandbox._transport is not fresh:
            # 理论不可达（singleton 非 None 时 get_transport 不覆写，置 None 仅在本锁内），
            # 保留探测以防 SDK 行为变化。
            logger.warning(
                "Transport singleton was contested during locked build; sandbox %s may share its connection pool",
                sandbox_id[:16],
            )
        return sandbox


def _rebuild_raw_sandbox_for_http(
    raw: E2BRawSandbox,
    fallback: E2BRawSandbox | None = None,
) -> E2BRawSandbox:
    """Extract domain/token from a connect() response and rebuild for HTTP.

    NAC#1304: reconnect 专用（fail-closed）。token 优先取新 connect 响应
    （resume 后可能轮换），响应缺失时回退旧对象的 token 并告警；两者皆缺则抛
    ``SandboxError`` —— 在 force_http 部署上绝不安装 SDK 默认的 HTTPS 对象。
    """
    domain = raw.sandbox_domain
    if fallback is not None and fallback.sandbox_domain and domain != fallback.sandbox_domain:
        # 防御：connect 响应缺 domain 时会被 SDK 默认域(property 回退)静默兜底，
        # 此处显式暴露新旧不一致而不是放任构错 URL。
        logger.warning(
            "connect response domain %r differs from previous %r; using response value",
            domain,
            fallback.sandbox_domain,
        )
    token = getattr(raw, "_envd_access_token", None)
    if not token and fallback is not None:
        token = getattr(fallback, "_envd_access_token", None)
        if token:
            logger.warning("connect response missing envd access token; falling back to previous token")
    if not (domain and token):
        raise SandboxError(
            f"force_http rebuild requires sandbox domain and envd access token; got domain={domain!r}, token={'<set>' if token else None}"
        )
    return _locked_build_http_sandbox(raw.sandbox_id, domain, token, raw._envd_version)


@dataclass(kw_only=True)
class E2BSandbox(BaseSandbox):
    """
    E2B cloud sandbox implementation for secure code execution.

    This implementation uses E2B's cloud infrastructure to provide isolated,
    secure sandboxes for code execution and file operations. It's suitable
    for production use and provides proper security isolation.

    Note: Configuration (template, timeout, api_key, etc.) is managed by E2BSandboxManager.
    This class only contains runtime state.
    """

    default_user: str = field(default="user")
    work_dir: str | Path | None = field(default=E2B_DEFAULT_WORK_DIR)
    envd_version: str | None = field(default=None)
    max_retries: int = field(default=5)
    transient_retry_window: float = field(default=60.0)
    """Retry time budget (seconds) for transient network errors, measured from
    the first failure of an operation (NAC#1312). Sized to ride out a
    data-plane outage (e.g. sandbox-proxy restart) by stalling and retrying
    instead of failing fast. ``<= 0`` falls back to legacy count-based retries
    governed by ``max_retries``."""
    _api_key: str | None = field(default=None, repr=False)
    _api_url: str | None = field(default=None, repr=False)
    _force_http: bool = field(default=False, repr=False)
    """Self-host marker (NAC#1304): when True, ``_reconnect`` rebuilds the raw
    sandbox with the HTTP ConnectionConfig instead of keeping the SDK-default
    HTTPS object. Injected at construction (manager start paths pass
    ``sandbox_config.force_http``); init=True so it round-trips through
    ``dict()`` persistence and restore."""
    _static_reconnect: bool = field(default=False, repr=False)
    """Explicit-target marker (NAC#1312 CR): when True, ``_reconnect`` rebuilds
    in place from the current object's own domain/token via
    ``_locked_build_http_sandbox`` and never touches the control-plane
    ``Sandbox.connect()`` — the binding parameters are static facts supplied
    at construction (A4A DevBox direct binding), and a connect round-trip
    would re-introduce SDK-default-domain fallback and out-of-band NotFound
    failure modes. Implies HTTP rebuild semantics regardless of
    ``_force_http``. init=True so it round-trips through ``dict()``."""

    # Unserialized fields
    _sandbox: E2BRawSandbox | None = field(default=None, repr=False, init=False)
    _reconnect_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, init=False)
    """Single-flight guard for ``_reconnect`` (NAC#1304)."""
    _sandbox_generation: int = field(default=0, repr=False, init=False)
    """Bumped on every successful ``_reconnect``; used to deduplicate a
    thundering herd of concurrent reconnect attempts."""

    def __post_init__(self) -> None:
        if self.work_dir is not None:
            self.work_dir = str(PurePosixPath(Path(self.work_dir).as_posix()))

    def get_temp_dir(self) -> str:
        """Return the E2B remote Linux temp directory."""
        return "/tmp"

    @property
    def sandbox(self) -> E2BRawSandbox | None:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: E2BRawSandbox):
        self._sandbox = sandbox
        self.sandbox_id = sandbox.sandbox_id

    def set_api_credentials(self, api_key: str | None, api_url: str | None) -> None:
        self._api_key = api_key
        self._api_url = api_url

    # Commands larger than this threshold (bytes) are written to a script file
    # inside the sandbox and executed via `bash <script>` instead of being
    # passed inline to `commands.run()`.  This avoids ConnectRPC message-size
    # limits (~4 MB) and Linux MAX_ARG_STRLEN (~128 KB) that cause large
    # heredocs / echo payloads to fail with exit-code 2.
    _LARGE_CMD_THRESHOLD: int = 65_536  # 64 KB

    # Re-export for backward compatibility; canonical definition lives in base_sandbox.
    _HEREDOC_PATTERN: re.Pattern[str] = HEREDOC_PATTERN

    def _prepare_output_dir(self, command: str, user: str = "user") -> str:
        """Create an output directory and write command.txt in the sandbox.

        stdout.txt and stderr.txt are created by shell-level redirection
        (``{ cmd; } > stdout.txt 2> stderr.txt``), so this method only sets
        up the directory and the command metadata file.

        Returns:
            The absolute output directory path.
        """
        if self._sandbox is None:
            raise SandboxError("Sandbox not started. Call start() first.")

        output_dir = f"{BASH_TOOL_RESULTS_BASE_PATH}/{uuid.uuid4().hex[:8]}"
        # NAC#1312: 主命令前的准备步骤同样要能扛数据面瞬断（幂等，可安全重试），
        # 否则 execute_shell 在真正带重试的 commands.run 之前就快速失败了
        self._retry_on_transient(
            lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                f"mkdir -p {output_dir} && : > {output_dir}/stdout.txt && : > {output_dir}/stderr.txt",
                user=user,
            )
        )
        self._retry_on_transient(lambda: self._sandbox._filesystem.write(f"{output_dir}/command.txt", command))  # type: ignore[union-attr]
        return output_dir

    def _maybe_scriptify(self, command: str, output_dir: str, user: str = "user") -> str:
        """If *command* exceeds the large-command threshold **or contains
        heredoc syntax**, write it to a temporary bash script inside the
        sandbox and return a short command that sources it.  Otherwise
        return *command* unchanged.

        Heredoc commands must be scriptified because the compound-command
        wrapper ``{ cmd; } > ... 2> ...`` appends ``; }`` after the heredoc
        delimiter, violating bash's requirement that the delimiter occupy
        its own line — causing a syntax error (exit_code=2).

        This also lets arbitrarily large commands work without hitting RPC
        or OS limits.
        """
        has_heredoc = contains_heredoc(command)
        is_large = len(command.encode("utf-8", errors="replace")) > self._LARGE_CMD_THRESHOLD

        if not has_heredoc and not is_large:
            return command

        script_path = f"{output_dir}/run.sh"
        # NAC#1312: 幂等写，纳入瞬断重试
        self._retry_on_transient(lambda: self._sandbox._filesystem.write(script_path, command))  # type: ignore[union-attr]
        logger.info(
            "[e2b] Command scriptified (%d bytes, heredoc=%s) – wrote to %s",
            len(command.encode("utf-8", errors="replace")),
            has_heredoc,
            script_path,
        )
        return f"bash {script_path}"

    def _read_output_files(
        self,
        output_dir: str,
        *,
        budget: float | None = None,
    ) -> tuple[str, str, str | None]:
        """Read stdout.txt and stderr.txt from the output directory in the sandbox.

        NAC#1312（CR 加固 x2）：
        - 两个文件共享**一份**重试预算，而不是各吃一个完整窗口；
        - 输出可用性作为独立维度返回（第三个元素），绝不抛出——"输出读
          不到" ≠ "命令失败"。上一版的 strict 异常会把已成功（且可能有
          副作用）的命令报成 ERROR，诱导 agent 重跑造成双重执行；同时
          stderr 读失败会连带丢弃已读到的 stdout。现在两个文件各自尽力
          读取，成功的保留，失败的记录进 note。

        Args:
            budget: 共享重试预算（秒）。``None`` = ``self.transient_retry_window``。
                轮询类调用方应传小值——状态轮询本身就是重复调用，单次失败
                下一轮自然补上，不该单次吃满窗口。

        Returns:
            (stdout, stderr, unavailable_note)——note 为 ``None`` 表示两个
            文件都读到了（NotFound 视为空输出，不算不可用）；否则描述哪些
            输出因预算耗尽不可用，调用方应把它并入结果的 error 字段显性
            呈现，而不是让输出静默变空。
        """
        assert self._sandbox is not None

        window = self.transient_retry_window if budget is None else budget
        start = time.monotonic()
        notes: list[str] = []

        def _read_one(path: str) -> str:
            try:
                if window > 0:
                    remaining = window - (time.monotonic() - start)
                    if remaining > 0.01:
                        raw = self._retry_on_transient(
                            lambda: self._sandbox._filesystem.read(path, format="bytes"),  # type: ignore[union-attr]
                            retry_window=remaining,
                        )
                    else:
                        # 预算已被前一个文件耗尽：只试一次，不再退避
                        raw = self._retry_on_transient(
                            lambda: self._sandbox._filesystem.read(path, format="bytes"),  # type: ignore[union-attr]
                            max_retries=0,
                            retry_window=0.01,
                        )
                else:
                    raw = self._retry_on_transient(
                        lambda: self._sandbox._filesystem.read(path, format="bytes")  # type: ignore[union-attr]
                    )
                return raw.decode("utf-8", errors="replace")
            except Exception as e:
                if NotFoundException is not None and isinstance(e, NotFoundException):
                    return ""
                notes.append(f"{path}: {e}")
                return ""

        stdout = _read_one(f"{output_dir}/stdout.txt")
        stderr = _read_one(f"{output_dir}/stderr.txt")
        note = "; ".join(notes) if notes else None
        return stdout, stderr, note

    def _resolve_path(self, path: str, cwd: str | None = None) -> str:
        """Resolve a relative path to an absolute path.

        E2B SaaS envd resolves relative paths server-side, but self-host envd
        (e.g. 0.4.2) does not. This method ensures consistent behavior across
        both environments by always resolving on the client side.

        Args:
            path: File or directory path (absolute or relative).
            cwd: Working directory for resolution. Defaults to self.work_dir.
        """
        if path.startswith("/"):
            return path
        base = cwd or str(self.work_dir)
        return f"{base}/{path}"

    # Transient error patterns that warrant a reconnect + retry.
    # Aligned with NexQ's _RETRYABLE_E2B_TEXT_PAT for comprehensive coverage.
    # Matched case-insensitively via _is_transient_error().
    _TRANSIENT_PATTERNS = (
        "event loop is closed",
        "server disconnected",
        "connection reset",
        "connection refused",
        "connection closed",
        "connection aborted",
        "remoteprotocolerror",
        "incomplete chunked read",
        "peer closed connection",
        "context deadline exceeded",
        "timed out",
        "temporarily unavailable",
        "temporary failure",
        "nodename nor servname",
        "name or service not known",
        "ssl handshake",
        "tlsv1 alert",
        "gateway timeout",
        "bad gateway",
        "service unavailable",
        "502 bad gateway",
        "502 server error",
        "503 service unavailable",
        "internal server error",
    )

    def _is_transient_error(self, exc: Exception) -> bool:
        """Return True if the exception looks like a transient network error."""
        msg = str(exc).lower()
        return any(p in msg for p in self._TRANSIENT_PATTERNS)

    def _reconnect(self, gen_seen: int | None = None) -> None:
        """Attempt to reconnect to the sandbox by sandbox_id (single-flight).

        NAC#1304: 重连必须保真——force_http 部署下用与初始构造相同的 HTTP
        ConnectionConfig 重建，而不是安装 ``Sandbox.connect()`` 返回的 SDK
        默认 HTTPS 对象（自建部署 :443 无监听，安装后本 run 内所有沙箱操作
        将永久 Connection refused）。

        NAC#1312 CR: ``_static_reconnect=True``（explicit-target 直绑路径）时
        完全不走控制面 ``Sandbox.connect()``，直接用当前对象自带的
        domain/token 原位重建——该路径的连接参数是显式配置的静态事实，
        走 connect 反而引入两类回归：响应缺 domain 被 SDK 默认域
        （``e2b.app``）静默兜底绕过 fail-closed；带外沙箱在 SM 无记录时
        connect 抛 NotFound（非 transient）中止整个重试窗口。

        Args:
            gen_seen: ``_sandbox_generation`` observed by the caller right
                before the failed operation. When another thread has already
                reconnected since (generation advanced), this call becomes a
                no-op so a thundering herd performs exactly one reconnect.
        """
        assert Sandbox is not None, "E2B SDK not installed."
        if not self.sandbox_id:
            raise SandboxError("Sandbox ID not set; cannot reconnect.")
        with self._reconnect_lock:
            # 1. double-check：他人已重连则直接复用其结果
            if gen_seen is not None and self._sandbox_generation != gen_seen:
                return
            # 2. 网络调用持 wrapper 锁：single-flight 语义即要求串行。慢 resume
            #    会让同 wrapper 的其他等待者阻塞于此，净效果仍优于 N 路并发
            #    connect 风暴。
            if self._static_reconnect:
                old = self._sandbox
                if old is None:
                    raise SandboxError("Static reconnect requires an existing sandbox object.")
                domain = old.sandbox_domain
                token = old._envd_access_token
                if not (domain and token):
                    raise SandboxError(
                        f"Static reconnect requires sandbox domain and envd access token; "
                        f"got domain={domain!r}, token={'<set>' if token else None}"
                    )
                raw = _locked_build_http_sandbox(old.sandbox_id, domain, token, old._envd_version)
            else:
                raw = Sandbox.connect(
                    sandbox_id=self.sandbox_id,
                    api_key=self._api_key,
                    api_url=self._api_url,
                )
                # 3. 自建部署：保真重建（fail-closed，失败时旧对象保留、gen 不增）
                if self._force_http:
                    raw = _rebuild_raw_sandbox_for_http(raw, fallback=self._sandbox)
            self._sandbox = raw
            self._sandbox_generation += 1

    def _retry_on_transient(
        self,
        fn: Callable[[], _T],
        max_retries: int | None = None,
        retry_window: float | None = None,
    ) -> _T:
        """Execute *fn* with automatic reconnect + retry on transient errors.

        NAC#1312: 重试预算以时间窗口为主。断连类故障（如 sandbox-proxy 重启）
        的恢复时间与重试次数无关——瞬时失败（connection refused）会在几秒内
        耗光任何次数制预算。窗口内不限次数，指数退避封顶 5s，数据面恢复后
        当前操作直接成功返回，而不是把错误抛给工具层。

        Args:
            fn: Zero-arg callable that performs the SDK operation.
            max_retries: Optional attempt cap. When passed, retrying stops at
                whichever budget exhausts first (attempts or window) — for
                auxiliary reads that must stay low-latency (e.g. best-effort
                pid probes). ``None`` = unlimited attempts within the window.
            retry_window: Retry time budget in seconds, measured from the
                first transient failure (fn() 自身耗时不计入，预算只约束
                "还要不要再试"). Defaults to ``self.transient_retry_window``;
                ``<= 0`` falls back to legacy count-based retries
                (``max_retries`` or ``self.max_retries``).

        Returns:
            Whatever *fn* returns on success.

        Raises:
            The original exception if it is not transient or the budget is
            exhausted.
        """
        window = retry_window if retry_window is not None else self.transient_retry_window
        # 窗口关闭(<=0)时退回纯次数制，保底 self.max_retries
        count_cap = max_retries if max_retries is not None else (self.max_retries if window <= 0 else None)
        deadline: float | None = None
        attempt = 0
        while True:
            # 每次 fn() 之前采集 generation：表达"失败的这次调用跑在哪个对象上"。
            # 循环前只采一次会让第 2+ 次重试被 double-check 误拦（漏重连）；
            # 在 except 里采集则会让惊群去重失效（冗余重连）。
            # 已知有界代价：采集与 fn() 解引用之间他线程完成 swap 时，本次失败
            # 的 reconnect 会被去重跳过、白耗一轮 backoff，下一轮自愈。
            gen_seen = self._sandbox_generation
            try:
                return fn()
            except Exception as e:
                if not self._is_transient_error(e):
                    raise
                now = time.monotonic()
                if window > 0:
                    if deadline is None:
                        deadline = now + window
                    elif now >= deadline:
                        raise
                if count_cap is not None and attempt >= count_cap:
                    raise
                # 指数退避封顶 5s + 抖动；不睡过 deadline。
                # attempt 参与 2**n 前先 cap——大窗口配置下 attempt 可达数百，
                # 2**1024 转 float 会 OverflowError（NAC#1312 CR finding）。
                base = min(0.5 * (2 ** min(attempt, 10)), 5.0)
                delay = base + random.uniform(0, base * 0.25)
                if deadline is not None:
                    # clamp 到剩余预算（此处恒 > 0，超窗已在上方 raise）：
                    # 总耗时严格 ≤ window，临期的余量本身就是末班车尝试
                    delay = min(delay, deadline - now)
                logger.warning(
                    "Transient error (attempt %d%s), reconnecting and retrying in %.1fs: %s",
                    attempt + 1,
                    f", window {window:.0f}s" if window > 0 else f"/{count_cap + 1}" if count_cap is not None else "",
                    delay,
                    e,
                )
                time.sleep(delay)
                try:
                    self._reconnect(gen_seen)
                except Exception as reconnect_err:
                    if self._is_transient_error(reconnect_err):
                        # 断连窗口期控制面也可能抖（SM/sidecar 短暂不可达）：
                        # 不中止预算，下一轮重试时再尝试 reconnect
                        logger.warning("Reconnect failed with transient error (will retry): %s", reconnect_err)
                    else:
                        # 确定性失败（沙箱已销毁、鉴权失效、force_http fail-closed）：
                        # 立即上抛，不对着已死沙箱空耗窗口
                        logger.error(f"Reconnect failed: {reconnect_err}")
                        raise e from reconnect_err
                attempt += 1

    @override
    def execute_shell(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
    ) -> CommandResult:
        """
        Execute a shell command in the E2B sandbox.

        命令启动时即通过 shell 级重定向将 stdout/stderr 写入临时文件，
        执行完毕后从文件读取输出并按需智能截断。

        Args:
            command: The shell command to execute
            timeout: Optional timeout in milliseconds (overrides default)
            cwd: Optional working directory
            user: Optional user to run the command as
            envs: Optional environment variables
            background: Optional flag to run the command in the background

        Returns:
            CommandResult containing execution results
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        from e2b.exceptions import TimeoutException

        if self._sandbox is None:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        start_time = time.time()

        if timeout is None:
            timeout = 120000  # Default: 2 minutes

        timeout_seconds = timeout / 1000.0

        try:
            # Strip leading/trailing whitespace to avoid syntax errors when
            # wrapping multiline commands (a bare `;` after a newline is invalid).
            command_stripped = command.strip()

            if background:
                # 1. 创建输出目录，命令通过 shell 重定向写入文件
                output_dir = self._prepare_output_dir(command, user=user)
                command_stripped = self._maybe_scriptify(command_stripped, output_dir, user=user)
                wrapped_cmd = f"{{ {command_stripped}; }} > {output_dir}/stdout.txt 2> {output_dir}/stderr.txt"

                # Background mode: retry on transient network errors
                handle = self._retry_on_transient(
                    lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                        cmd=wrapped_cmd,
                        background=True,
                        timeout=int(timeout_seconds),
                        cwd=cwd or str(self.work_dir),
                        user=user,
                        envs=self._merge_envs(envs),
                    )
                )

                bg_pid: int = handle.pid  # background=True → CommandHandle

                # E2B CommandHandle requires iterating events to populate _result.
                # 输出已重定向到文件，consumer 线程只需等待进程结束获取 exit code。
                task_info: dict[str, Any] = {
                    "handle": handle,
                    "command": command,
                    "start_time": start_time,
                    "finished": False,
                    "exit_code": -1,
                    "error": None,
                    "std_output_dir": output_dir,
                }

                def _consume_events(h: object, info: dict[str, Any]) -> None:
                    try:
                        for _stdout_chunk, _stderr_chunk, _pty in h:  # type: ignore[attr-defined]
                            pass  # 输出已重定向到文件，仅消费事件以跟踪完成状态
                    except StopIteration:
                        pass
                    except Exception as exc:
                        # Streaming connection dropped (e.g. proxy timeout after ~15 min).
                        # The sandbox process may still be running — do NOT mark as
                        # finished.  Set a stream_error flag so callers can fall back
                        # to file-based status checking.
                        info["stream_error"] = str(exc)
                        logger.warning(
                            "[e2b] Background consumer stream error (process may still be running): %s",
                            exc,
                        )
                        return
                    # Only mark finished when the handle has a real result
                    # (process exited and envd reported it).
                    if h._result is not None:  # type: ignore[attr-defined]
                        info["exit_code"] = h._result.exit_code  # type: ignore[attr-defined]
                        if h._result.exit_code != 0:  # type: ignore[attr-defined]
                            info["error"] = info.get("error") or f"Command failed with exit code {h._result.exit_code}"  # type: ignore[attr-defined]
                    info["finished"] = True

                consumer_thread = threading.Thread(
                    target=_consume_events,
                    args=(handle, task_info),
                    daemon=True,
                )
                consumer_thread.start()
                task_info["thread"] = consumer_thread

                self._background_tasks[bg_pid] = task_info
                duration_ms = int((time.time() - start_time) * 1000)

                stdout_msg = f"Background task started (pid: {bg_pid})\nOutput will be saved to {output_dir}/"

                return CommandResult(
                    status=SandboxStatus.SUCCESS,
                    stdout=stdout_msg,
                    stderr="",
                    exit_code=0,
                    duration_ms=duration_ms,
                    background_pid=bg_pid,
                    output_dir=output_dir,
                    stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                    stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
                )

            # Foreground mode: 统一走 "pid 守卫后台启动 + 自适应轮询"
            # （NAC#1312 CR 彻底修，取代原先 600s 阈值分叉的同步/后台两套路径）：
            # - 命令在沙箱内后台执行，stdout/stderr/exitcode 全部落沙箱本地
            #   文件，与连接解耦——数据面断连不影响命令本体，轮询容错等恢复
            # - pid 文件守卫使启动幂等：瞬断重试绝不双跑非幂等命令
            #   （旧同步路径的重试 = 整条命令原样重跑，会重复执行副作用）
            # - 轮询间隔 0.2s 起 ×1.5 递增封顶 10s：短命令保持低延迟，长命令
            #   避免高频轮询，同时天然规避网关 ~15min 长连接掐断
            import shlex as _shlex

            output_dir = self._prepare_output_dir(command, user=user)
            command_stripped = self._maybe_scriptify(command_stripped, output_dir, user=user)
            wrapped_cmd = f"{{ {command_stripped}; }} > {output_dir}/stdout.txt 2> {output_dir}/stderr.txt"

            exitcode_path = f"{output_dir}/exitcode.txt"
            pid_path = f"{output_dir}/pid.txt"
            # 用户命令必须套进孙 shell（bash -c）：`exit N` 这类命令若直接
            # 内联在子 shell 里会把整个子 shell 退掉，`echo $? > exitcode`
            # 永不执行 → 轮询永远等不到 DONE（真实 envd 上实测复现）。
            # CR 加固：外层再包 GNU timeout——服务端强制超时（等价旧同步路径
            # envd 的 timeout 语义），到点 TERM 命令、10s 后 KILL 兜底，防止
            # runtime 侧只能 best-effort kill launcher 子 shell 而留下孤儿
            # 进程树；GNU timeout 超时退出码 124 由下方映射回 TIMEOUT。
            inner_cmd = f"timeout -k 10 {max(int(timeout_seconds), 1)} bash -c " + _shlex.quote(wrapped_cmd)
            bg_script = (
                f"cd {_shlex.quote(cwd or str(self.work_dir))} && "
                f"if [ ! -s {pid_path} ]; then "
                f"({inner_cmd}; echo $? > {exitcode_path}) </dev/null >/dev/null 2>/dev/null & "
                f"echo $! > {pid_path}; "
                f"fi\n"
            )
            bg_start_cmd = "bash -lc " + _shlex.quote(bg_script)
            self._retry_on_transient(
                lambda: self._sandbox.commands.run(bg_start_cmd, timeout=0, user=user, envs=self._merge_envs(envs))  # type: ignore[union-attr]
            )

            # 状态判定完全在沙箱侧完成（pid 由沙箱自己 cat，runtime 不需要
            # 单独读 pid 文件——省一次 RTT 也省掉旧路径的 sleep(1) 预热）：
            # exitcode 非空 → DONE；进程活着 → RUNNING；否则 DEAD（异常死亡，
            # `-s` 而非 `-f`：排除 "文件已创建、退出码尚未落盘" 的空文件窗口）
            status_cmd = (
                f"if [ -s {exitcode_path} ]; then echo DONE; "
                f"elif [ -s {pid_path} ] && kill -0 $(cat {pid_path}) 2>/dev/null; then echo RUNNING; "
                f"else echo DEAD; fi"
            )

            # 轮询循环（CR 加固 x3）：
            # - 先查后睡：短命令 launch 返回时多半已完成，首查零延迟
            # - deadline 用 monotonic（系统时钟被 NTP 回拨时 wall-clock 会把
            #   超时拉长/缩短）；服务端 GNU timeout 是权威超时，本地 deadline
            #   只是断连期间的兜底判定，故加 5s 余量避免与服务端竞态
            # - 轮询连续失败达阈值时触发一次保真 _reconnect：普通 refused
            #   由连接池自愈，但 token 轮换/对象陈旧类故障必须重建对象——
            #   裸 except-continue 会把这类可恢复故障拖到超时
            exit_code = 0
            poll_interval = 0.2
            _bg_timed_out = False
            poll_fail_streak = 0
            poll_deadline = time.monotonic() + timeout_seconds + 5.0
            while True:
                try:
                    check = self._sandbox.commands.run(status_cmd, timeout=15, user="root")  # type: ignore[union-attr]
                    poll_fail_streak = 0
                    st = (check.stdout or "").strip()
                    if st in ("DONE", "DEAD"):
                        break
                except Exception:
                    # 断连期间命令在沙箱内不受影响；容错等待数据面恢复
                    poll_fail_streak += 1
                    if poll_fail_streak >= 3:
                        poll_fail_streak = 0
                        try:
                            self._reconnect()
                        except Exception as reconnect_err:
                            logger.warning("Poll-loop reconnect failed (will keep polling): %s", reconnect_err)
                if time.monotonic() >= poll_deadline:
                    _bg_timed_out = True
                    break
                time.sleep(min(poll_interval, max(poll_deadline - time.monotonic(), 0.05)))
                poll_interval = min(poll_interval * 1.5, 10.0)

            # Always try to read exit code (process may have finished right as
            # we timed out or after DEAD detection). 完整瞬断窗口：命令大概率
            # 已被服务端 GNU timeout 终结并写下 124/真实退出码，值得等断连
            # 恢复拿到权威结论，而不是急着误报
            try:
                raw_ec = self._retry_on_transient(
                    lambda: self._sandbox._filesystem.read(exitcode_path),  # type: ignore[union-attr]
                )
                ec_str = raw_ec.decode().strip() if isinstance(raw_ec, (bytes, bytearray)) else str(raw_ec).strip()
                exit_code = int(ec_str)
            except Exception:
                exit_code = -1

            if exit_code == 124 or (_bg_timed_out and exit_code == -1):
                # 124 = GNU timeout 已在服务端终结命令（权威超时）；
                # -1 + 本地超时 = 断连期间无法确认，best-effort 清理后按超时报
                if exit_code == -1:
                    try:
                        self._sandbox.commands.run(  # type: ignore[union-attr]
                            f"kill -TERM $(cat {pid_path}) 2>/dev/null || true",
                            timeout=0,
                            user="root",
                        )
                    except Exception:
                        pass
                # Raise TimeoutException so outer handler returns SandboxStatus.TIMEOUT.
                raise TimeoutException(f"Command timed out after {timeout}ms")

            duration_ms = int((time.time() - start_time) * 1000)

            # 从文件读取完整输出。输出不可用是独立维度（output_note），并入
            # error 显性呈现——但绝不改写命令本身的成败结论（exit code 是
            # 权威事实；把已成功的副作用命令报成 ERROR 会诱导 agent 重跑）
            stdout, stderr, output_note = self._read_output_files(output_dir)

            # 智能截断
            t_stdout, t_stderr, was_truncated, orig_stdout_len, orig_stderr_len = smart_truncate_output(
                stdout,
                stderr,
                output_dir,
                threshold=self.output_char_threshold,
                head_chars=self.truncate_head_chars,
                tail_chars=self.truncate_tail_chars,
            )

            error_parts: list[str] = []
            if exit_code != 0:
                error_parts.append(f"Command failed with exit code {exit_code}")
            if output_note:
                error_parts.append(f"command completed (exit {exit_code}) but output unavailable: {output_note}")
            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=t_stdout,
                stderr=t_stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                error="; ".join(error_parts) or None,
                truncated=was_truncated,
                original_stdout_length=orig_stdout_len,
                original_stderr_length=orig_stderr_len,
                output_dir=output_dir,
                stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
            )

        except TimeoutException:
            duration_ms = int((time.time() - start_time) * 1000)
            return CommandResult(
                status=SandboxStatus.TIMEOUT,
                stdout="",
                stderr="",
                exit_code=-1,
                duration_ms=duration_ms,
                error=f"Command timed out after {timeout}ms",
                truncated=False,
            )

        except Exception as e:
            # Other unexpected exceptions
            duration_ms = int((time.time() - start_time) * 1000)
            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_ms=duration_ms,
                error=f"Command execution error: {str(e)[:200]}",
                truncated=False,
            )

    @override
    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
    ) -> CommandResult:
        """Deprecated legacy alias for execute_shell."""
        return self.execute_shell(
            command,
            timeout=timeout,
            cwd=cwd,
            user=user,
            envs=envs,
            background=background,
        )

    @override
    def get_background_task_status(self, pid: int) -> CommandResult:
        """
        Get the status and output of a background task.

        从 output_dir 下的 stdout.txt / stderr.txt 读取输出（文件由 shell 重定向写入）。

        Args:
            pid: The process ID of the background task

        Returns:
            CommandResult with current status and accumulated output
        """
        if pid not in self._background_tasks:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=f"Background task not found: pid={pid}",
                exit_code=-1,
                error=f"Background task not found: pid={pid}",
            )

        task_info = self._background_tasks[pid]
        duration_ms = int((time.time() - task_info["start_time"]) * 1000)
        output_dir: str | None = task_info.get("std_output_dir")
        finished = bool(task_info["finished"])

        # 从文件读取输出。NAC#1312 CR：预算按语义分——
        # - 未结束的轮询：短预算（本方法被高频重复调用，单次失败下一轮自然
        #   补上，不该单次吃满 60s 窗口把轮询方拖成分钟级卡顿）
        # - 已结束：这次读取就是最终输出，用完整预算；输出不可用并入 error
        #   显性呈现，但不改写命令本身的成败结论（exit code 是权威事实，
        #   把已成功的副作用命令报成 ERROR 会诱导 agent 重跑双重执行）
        stdout = ""
        stderr = ""
        output_read_error: str | None = None
        if output_dir:
            stdout, stderr, note = self._read_output_files(
                output_dir,
                budget=None if finished else 5.0,
            )
            if note and finished:
                output_read_error = f"command completed but output unavailable: {note}"

        # 智能截断
        if output_dir:
            t_stdout, t_stderr, was_truncated, o_out, o_err = smart_truncate_output(
                stdout,
                stderr,
                output_dir,
                threshold=self.output_char_threshold,
                head_chars=self.truncate_head_chars,
                tail_chars=self.truncate_tail_chars,
            )
        else:
            t_stdout, t_stderr, was_truncated, o_out, o_err = stdout, stderr, False, None, None

        if finished:
            exit_code = task_info["exit_code"]
            base_error = task_info.get("error")
            combined_error = "; ".join(x for x in (base_error, output_read_error) if x) or None
            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=t_stdout,
                stderr=t_stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=combined_error,
                truncated=was_truncated,
                original_stdout_length=o_out,
                original_stderr_length=o_err,
                background_pid=pid,
                output_dir=output_dir,
                stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
            )

        # Stream error but process might still be running — use kill -0 to check.
        # NOTE: when the stream drops we lose the authoritative exit code from
        # envd.  We report exit_code=-1 with a descriptive error as a safe
        # default; callers should treat this as "indeterminate" rather than a
        # definitive failure.
        if task_info.get("stream_error") and self._sandbox is not None:
            try:
                # NAC#1312: 纳入瞬断重试——stream 掉线常与数据面故障同源，
                # 状态探测若也快速失败会把仍在运行的任务误报为 indeterminate。
                # CR 加固：短预算（max_retries=2）——本方法是轮询 API，单次
                # 探测失败下一轮补上，不叠加成 3x 窗口的卡顿
                check = self._retry_on_transient(
                    lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                        f"kill -0 {pid} 2>/dev/null && echo ALIVE || echo DEAD",
                        timeout=10,
                        user="root",
                    ),
                    max_retries=2,
                    retry_window=10.0,
                )
                if "DEAD" in (check.stdout or ""):
                    task_info["finished"] = True
                    task_info["exit_code"] = -1
                    task_info["error"] = f"Process exited but exit code is unknown (event stream lost: {task_info['stream_error']})"
                    return CommandResult(
                        status=SandboxStatus.ERROR,
                        stdout=t_stdout,
                        stderr=t_stderr,
                        exit_code=-1,
                        duration_ms=duration_ms,
                        error=task_info["error"],
                        truncated=was_truncated,
                        original_stdout_length=o_out,
                        original_stderr_length=o_err,
                        background_pid=pid,
                        output_dir=output_dir,
                        stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                        stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
                    )
            except Exception:
                pass  # Can't check, assume still running

        # Task is still running, return accumulated output so far
        return CommandResult(
            status=SandboxStatus.RUNNING,
            stdout=t_stdout,
            stderr=t_stderr,
            exit_code=-1,
            duration_ms=duration_ms,
            truncated=was_truncated,
            original_stdout_length=o_out,
            original_stderr_length=o_err,
            background_pid=pid,
            output_dir=output_dir,
            stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
            stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
        )

    @override
    def kill_background_task(self, pid: int) -> CommandResult:
        """
        Kill a background task.

        Args:
            pid: The process ID of the background task

        Returns:
            CommandResult with the kill operation result
        """
        if pid not in self._background_tasks:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=f"Background task not found: pid={pid}",
                exit_code=-1,
                error=f"Background task not found: pid={pid}",
            )

        task_info = self._background_tasks[pid]
        handle = task_info["handle"]

        try:
            killed = handle.kill()
            duration_ms = int((time.time() - task_info["start_time"]) * 1000)
            del self._background_tasks[pid]
            return CommandResult(
                status=SandboxStatus.SUCCESS if killed else SandboxStatus.ERROR,
                stdout=f"Background task (pid={pid}) killed successfully" if killed else f"Failed to kill task (pid={pid})",
                exit_code=0 if killed else -1,
                duration_ms=duration_ms,
                background_pid=pid,
            )
        except Exception as e:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=str(e),
                exit_code=-1,
                error=f"Failed to kill background task: {str(e)[:200]}",
                background_pid=pid,
            )

    @override
    def execute_code(
        self,
        code: str,
        language: CodeLanguage | str,
        timeout: int | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionResult:
        """
        Execute Python code in the E2B sandbox.

        Args:
            code: The Python code to execute
            language: Programming language (must be "python" or CodeLanguage.PYTHON)
            timeout: Optional timeout in milliseconds (overrides default)
            user: Optional user to run the code as (default: root)
            envs: Optional environment variables to set

        Returns:
            CodeExecutionResult containing execution results and outputs
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        start_time = time.time()

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                return CodeExecutionResult(
                    status=SandboxStatus.ERROR,
                    language=CodeLanguage.PYTHON,
                    error_type="ValueError",
                    error_value=f"Unsupported language: {language}. Only Python is supported.",
                    duration_ms=0,
                )

        if language != CodeLanguage.PYTHON:
            return CodeExecutionResult(
                status=SandboxStatus.ERROR,
                language=language,
                error_type="ValueError",
                error_value=f"Unsupported language: {language}. Only Python is supported.",
                duration_ms=0,
            )

        temp_file_path = None
        try:
            # Create temporary Python file in the sandbox temp directory.
            temp_dir = self.get_temp_dir()
            temp_filename = f"tmp_{uuid.uuid4().hex[:8]}.py"
            temp_file_path = self.join_path(temp_dir, temp_filename)

            # Write code to temp file
            self.write_file(temp_file_path, code)

            # Execute the temp file
            result = self.execute_shell(
                f"{self.get_python_command()} {temp_filename}",
                timeout,
                cwd=temp_dir,
                user=user,
                envs=envs,
            )

            outputs: list[dict[str, Any]] = []
            if result.stdout:
                outputs.append({"type": "stdout", "text": result.stdout})
            if result.stderr:
                outputs.append({"type": "stderr", "text": result.stderr})

            return CodeExecutionResult(
                status=result.status,
                language=language,
                outputs=outputs,
                error_type=None if result.status == SandboxStatus.SUCCESS else "ExecutionError",
                error_value=result.error,
                traceback=[result.stderr] if result.stderr else None,
                duration_ms=result.duration_ms,
                truncated=result.truncated,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to execute code: {e}")
            return CodeExecutionResult(
                status=SandboxStatus.ERROR,
                language=language,
                error_type=type(e).__name__,
                error_value=str(e),
                duration_ms=duration_ms,
            )
        finally:
            # Clean up temp file
            if temp_file_path:
                try:
                    self.execute_shell(f"rm -f {temp_file_path}", timeout=5000)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

    @override
    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file from the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox
            encoding: File encoding (default: utf-8)
            binary: Whether to read file in binary mode

        Returns:
            FileOperationResult containing file content
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            resolved_path = self._resolve_path(file_path)

            raw_content = self._retry_on_transient(
                lambda: self._sandbox._filesystem.read(resolved_path, format="bytes")  # type: ignore[union-attr]
            )
            content: str | bytearray
            if binary:
                content = raw_content
            else:
                content = raw_content.decode(encoding)

            # Get file size
            file_info = self.get_file_info(resolved_path)
            file_size = file_info.size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                content=content,
                size=file_size,
            )

        except Exception as e:
            # Use debug for UnicodeDecodeError - expected when reading binary as text, avoid log spam
            if isinstance(e, UnicodeDecodeError):
                logger.debug("Skipped binary file (cannot decode as text): %s", file_path)
            else:
                logger.error(f"Failed to read file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR, file_path=file_path, error=f"Failed to read file: {str(e)}", content=None
            )

    @override
    def write_file(
        self,
        file_path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        binary: bool = False,
        create_directories: bool = True,
        user: str | None = None,
    ) -> FileOperationResult:
        """
        Write content to a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox
            content: Content to write (string or bytes)
            encoding: File encoding (default: utf-8)
            binary: Whether to write file in binary mode
            create_directories: Whether to create parent directories if they don't exist
            user: Optional user to run the create_directories command as

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            resolved_path = self._resolve_path(file_path)

            # Create parent directories if needed
            if create_directories:
                parent_dir = str(Path(resolved_path).parent)
                if parent_dir and parent_dir != ".":
                    self._retry_on_transient(
                        lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                            cmd=f"mkdir -p {parent_dir}",
                            user=user,
                        )
                    )

            # Write file using E2B filesystem API (with retry)
            self._retry_on_transient(
                lambda: self._sandbox._filesystem.write(resolved_path, content, request_timeout=300.0)  # type: ignore[union-attr]
            )

            # Calculate size from content directly (avoid extra round trip)
            if isinstance(content, (bytes, bytearray)):
                size = len(content)
            else:
                size = len(content.encode(encoding))

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=resolved_path,
                size=size,
            )

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to write file: {str(e)}",
            )

    @override
    def delete_file(self, file_path: str) -> FileOperationResult:
        """
        Delete a file from the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            resolved_path = self._resolve_path(file_path)

            # Check if file exists
            if not self.file_exists(resolved_path):
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=resolved_path,
                    error=f"File does not exist: {resolved_path}",
                )

            # Use E2B filesystem remove（NAC#1312: 纳入瞬断重试）。
            # CR 加固：重试中撞 NotFound = 首发已删成功、响应丢失后重发撞空
            # （上方 file_exists 已确认过文件存在）——目标状态已达成，视为
            # 成功而不是把删除成功报成 ERROR 误导调用方重试。
            try:
                self._retry_on_transient(lambda: self._sandbox._filesystem.remove(resolved_path))  # type: ignore[union-attr]
            except Exception as e:
                if NotFoundException is not None and isinstance(e, NotFoundException):
                    logger.info("delete_file: %s already gone after transient retry; treating as success", resolved_path)
                else:
                    raise

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
            )

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to delete file: {str(e)}",
            )

    @override
    def list_files(
        self,
        directory_path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileInfo]:
        """
        List files in a directory in the E2B sandbox.

        Args:
            directory_path: Path to the directory in the sandbox
            recursive: Whether to list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of FileInfo objects for matching files
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        assert FileType is not None
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            resolved_path = self._resolve_path(directory_path)

            # Check if directory exists
            if not self.file_exists(resolved_path):
                raise SandboxFileError(f"Directory does not exist: {directory_path}")

            # Use E2B filesystem list（NAC#1312: 纳入瞬断重试）
            entries = self._retry_on_transient(lambda: self._sandbox._filesystem.list(resolved_path))  # type: ignore[union-attr]

            files: list[FileInfo] = []
            for entry in entries:
                # Detect readable/writable from permissions
                readable = True
                writable = True
                if entry.mode:
                    # Check owner read permission (bit 8)
                    readable = bool(entry.mode & 0o400)
                    # Check owner write permission (bit 7)
                    writable = bool(entry.mode & 0o200)

                # Convert E2B EntryInfo to our FileInfo
                file_info = FileInfo(
                    path=entry.path,
                    exists=True,
                    is_file=entry.type == FileType.FILE,
                    is_directory=entry.type == FileType.DIR,
                    size=entry.size,
                    mode=entry.mode,
                    permissions=entry.permissions,
                    modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                    symlink_target=entry.symlink_target,
                    readable=readable,
                    writable=writable,
                    encoding=None,  # Skip encoding detection in list for performance
                )
                files.append(file_info)

                # Recursively list subdirectories if requested
                if recursive and file_info.is_directory:
                    try:
                        subdir_files = self.list_files(file_info.path, recursive=True, pattern=pattern)
                        files.extend(subdir_files)
                    except Exception as e:
                        logger.warning(f"Failed to list subdirectory {file_info.path}: {e}")

            # Apply pattern filtering if specified
            if pattern:
                import fnmatch

                files = [f for f in files if fnmatch.fnmatch(Path(f.path).name, pattern)]

            return files

        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {e}")
            raise SandboxFileError(f"Failed to list files: {str(e)}")

    @override
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            True if file exists, False otherwise
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        # NAC#1304 掩蔽治理：只有"确实不存在"才返回 False；其余 stat 失败
        # （连接错误/权限/超时等）一律上抛。旧行为把 Connection refused 吞成
        # False，工具层随即误报 "Directory not found: /"，掩盖真实故障并误导
        # 模型分支决策。注意 SDK 的 exists() 已在内部把 not_found 转为 False，
        # NotFoundException 分支是防御性兜底；禁止改写成"已知错误类型白名单
        # 吞掉"——那会让文案不在名单里的连接错误（如 SSL EOF）重新被掩蔽。
        # NAC#1312: 纳入瞬断重试——它是工具 dir check 的入口，曾是断连期间
        # 最先毫秒级快速失败的裸调。NotFound 非 transient，会立刻穿出重试循环。
        try:
            resolved_path = self._resolve_path(file_path)
            return self._retry_on_transient(lambda: self._sandbox._filesystem.exists(resolved_path))  # type: ignore[union-attr]
        except Exception as e:
            if NotFoundException is not None and isinstance(e, NotFoundException):
                return False
            raise

    @override
    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileInfo object containing file metadata
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        assert FileType is not None
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            resolved_path = self._resolve_path(file_path)

            # Check if file exists
            exists = self.file_exists(resolved_path)

            if not exists:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Use E2B filesystem get_info API
            # NAC#1312 掩蔽治理（同 file_exists 口径）：只有 NotFound 才视为
            # "不存在"；连接类错误走瞬断重试，窗口耗尽后如实上抛，不再吞成
            # exists=False 误导调用方
            try:
                entry = self._retry_on_transient(lambda: self._sandbox._filesystem.get_info(resolved_path))  # type: ignore[union-attr]
            except Exception as e:
                if NotFoundException is not None and isinstance(e, NotFoundException):
                    return FileInfo(
                        path=file_path,
                        exists=False,
                    )
                raise

            # Detect readable/writable from permissions
            readable = True
            writable = True
            if entry.mode:
                # Check owner read permission (bit 8)
                readable = bool(entry.mode & 0o400)
                # Check owner write permission (bit 7)
                writable = bool(entry.mode & 0o200)

            if entry.type == FileType.FILE:
                raw_data = self._retry_on_transient(
                    lambda: self._sandbox._filesystem.read(resolved_path, format="bytes")  # type: ignore[union-attr]
                )
                encoding = self._detect_file_encoding(bytes(raw_data))
            else:
                encoding = None

            return FileInfo(
                path=file_path,
                exists=True,
                is_file=entry.type == FileType.FILE,
                is_directory=entry.type == FileType.DIR,
                size=entry.size,
                mode=entry.mode,
                permissions=entry.permissions,
                modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                symlink_target=entry.symlink_target,
                readable=readable,
                writable=writable,
                encoding=encoding,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise SandboxFileError(f"Failed to get file info: {str(e)}")

    @override
    def create_directory(self, directory_path: str, parents: bool = True, user: str | None = None) -> bool:
        """
        Create a directory in the E2B sandbox.

        Args:
            directory_path: Path to the directory to create
            parents: Whether to create parent directories if they don't exist

        Returns:
            True if directory created successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            cmd = f"mkdir -p {directory_path}" if parents else f"mkdir {directory_path}"
            result = self._retry_on_transient(
                lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                    cmd=cmd,
                    cwd=str(self.work_dir),
                    user=user,
                )
            )

            if result.exit_code != 0:
                raise SandboxFileError(f"Failed to create directory: {result.stderr}")

            return True

        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise SandboxFileError(f"Failed to create directory: {str(e)}")

    @override
    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
    ) -> FileOperationResult:
        """
        Edit a file by replacing old_string with new_string.

        Supports three operations:
        1. CREATE: Set old_string to empty string to create a new file
        2. UPDATE: Provide both old_string and new_string to update existing content
        3. REMOVE_CONTENT: Set new_string to empty string to remove the old_string content

        Args:
            file_path: Path to the file to edit
            old_string: String to replace (empty for file creation)
            new_string: Replacement string (empty for content removal)

        Returns:
            FileOperationResult containing operation status and details
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            file_exists = self.file_exists(file_path)

            # Determine operation type
            if old_string == "" and new_string != "":
                operation = "CREATE"
            elif old_string != "" and new_string == "":
                operation = "REMOVE_CONTENT"
            else:
                operation = "UPDATE"

            # Validate operation
            if operation == "CREATE" and file_exists:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File already exists: {file_path}. Use UPDATE operation instead.",
                )

            if operation != "CREATE" and not file_exists:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}. Use CREATE operation instead.",
                )

            # Read original content
            if file_exists:
                read_result = self.read_file(file_path, binary=False)
                if read_result.status != SandboxStatus.SUCCESS:
                    return read_result
                original_content = read_result.content
            else:
                original_content = ""

            assert isinstance(original_content, str), f"Unexpected content type: {type(original_content)}"

            # Validate string matching for UPDATE/REMOVE operations
            if operation != "CREATE":
                if old_string not in original_content:
                    # Try to normalize common escape sequence issues from LLM
                    def _normalize_escape_sequences(value: str) -> str:
                        return (
                            value.replace("\\\\n", "\n")
                            .replace("\\n", "\n")
                            .replace("\\\\t", "\t")
                            .replace("\\t", "\t")
                            .replace("\\\\r", "\r")
                            .replace("\\r", "\r")
                        )

                    normalized_old_string = _normalize_escape_sequences(old_string)
                    normalized_new_string = _normalize_escape_sequences(new_string)

                    if normalized_old_string != old_string and normalized_old_string in original_content:
                        # Use normalized version
                        old_string = normalized_old_string
                        new_string = normalized_new_string
                    else:
                        return FileOperationResult(
                            status=SandboxStatus.ERROR,
                            file_path=file_path,
                            error=f"String to replace not found in file: {file_path}",
                        )

                matches = original_content.count(old_string)
                if matches > 1:
                    return FileOperationResult(
                        status=SandboxStatus.ERROR,
                        file_path=file_path,
                        error=(
                            f"Found {matches} matches of the string to replace. "
                            "For safety, this tool only supports replacing exactly one occurrence at a time. "
                            "Add more lines of context to your edit and try again."
                        ),
                    )

            # Apply the edit
            if operation == "CREATE":
                updated_content = new_string
            elif operation == "REMOVE_CONTENT":
                updated_content = original_content.replace(old_string, "", 1)
            else:
                updated_content = original_content.replace(old_string, new_string, 1)

            # Write the updated content
            write_result = self.write_file(file_path, updated_content)

            return write_result

        except Exception as e:
            logger.error(f"Failed to edit file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to edit file: {str(e)}",
            )

    @override
    def glob(self, pattern: str, recursive: bool = True, user: str | None = None) -> list[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '*.py', '**/*.txt')
            recursive: Whether to search recursively (default: True)

        Returns:
            List of file paths matching the pattern
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        from e2b import CommandExitException

        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            # Normalize repeated slashes (e.g. //foo -> /foo) to avoid
            # accidentally computing root '/' as the search base.
            pattern = re.sub(r"/{2,}", "/", pattern)

            # Use find command with pattern matching
            if recursive:
                if "**" in pattern:
                    # Split on the first occurrence of ** to get search_dir and remainder.
                    # e.g. "/home/user/project/**/*.py" → search_dir="/home/user/project", file_pattern="*.py"
                    # e.g. "src/**/*.ts"               → search_dir="src",                file_pattern="*.ts"
                    # e.g. "**/*.py"                   → search_dir=".",                  file_pattern="*.py"
                    # e.g. "/home/user/project/**"     → search_dir="/home/user/project", file_pattern="*"
                    # e.g. "**"                        → search_dir=".",                  file_pattern="*"
                    idx = pattern.index("**")
                    search_dir = pattern[:idx].rstrip("/") or "."
                    remainder = pattern[idx + 2 :].lstrip("/")  # skip "**" and any trailing /
                    # If remainder contains more **, take only the final filename pattern
                    if "**" in remainder:
                        remainder = remainder.rsplit("**/", 1)[-1].lstrip("/")
                    file_pattern = remainder if remainder else "*"
                    cmd = f'find "{search_dir}" -type f -name "{file_pattern}" 2>/dev/null'
                elif "/" in pattern:
                    search_dir, file_pattern = pattern.rsplit("/", 1)
                    search_dir = search_dir or "."
                    # If the directory portion contains wildcards, let find handle
                    # them via -path instead of quoting the dir (which would cause
                    # bash to fail when the glob path doesn't literally exist).
                    if "*" in search_dir or "?" in search_dir:
                        # Find the deepest non-glob ancestor as the search root
                        parts = search_dir.lstrip("/").split("/")
                        root_parts = []
                        for part in parts:
                            if "*" in part or "?" in part:
                                break
                            root_parts.append(part)
                        root_dir = ("/" if search_dir.startswith("/") else "") + "/".join(root_parts) or "."
                        # Reconstruct the full path pattern (strip trailing slash)
                        full_pattern = pattern.rstrip("/")
                        # Limit search depth to avoid full filesystem traversal
                        # when root_dir is "/" (e.g. pattern "//foo*/bar.py"
                        # after normalization becomes "/foo*/bar.py" with root "/").
                        depth = len(full_pattern.strip("/").split("/"))
                        maxdepth = f" -maxdepth {depth}" if root_dir == "/" else ""
                        if file_pattern:
                            cmd = f'find "{root_dir}"{maxdepth} -path "{full_pattern}" 2>/dev/null'
                        else:
                            cmd = f'find "{root_dir}"{maxdepth} -type d -path "{full_pattern}" -print 2>/dev/null'
                    else:
                        cmd = f'find "{search_dir}" -name "{file_pattern}" 2>/dev/null || true'
                else:
                    search_dir = "."
                    file_pattern = pattern
                    cmd = f'find "{search_dir}" -name "{file_pattern}" 2>/dev/null || true'
            else:
                cmd = f'ls -1 "{pattern}" 2>/dev/null || true'

            result = self._retry_on_transient(
                lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                    cmd=cmd,
                    cwd=str(self.work_dir),
                    user=user,
                )
            )

            if result.exit_code != 0 and result.exit_code != 1:
                raise SandboxFileError(f"Glob command failed: {result.stderr}")

            # Parse output
            stdout_text = result.stdout or ""
            matches = [line.strip() for line in stdout_text.split("\n") if line.strip()]

            return sorted(matches)

        except CommandExitException as e:
            # find returns exit code 1 when it encounters permission errors
            # (e.g. /proc, /sys) but may still have found valid matches in stdout.
            stdout = getattr(e, "stdout", "") or ""
            matches = [line.strip() for line in stdout.split("\n") if line.strip()]
            return sorted(matches)

        except Exception as e:
            logger.error(f"Failed to glob pattern {pattern}: {e}")
            raise SandboxFileError(f"Failed to glob: {str(e)}")

    @override
    def upload_file(
        self,
        local_path: str,
        sandbox_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Upload a file from the local filesystem to the E2B sandbox.

        Args:
            local_path: Path to the file on the local filesystem
            sandbox_path: Destination path in the sandbox
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            local_file = Path(local_path)

            if not local_file.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=sandbox_path,
                    error=f"Source file does not exist: {local_path}",
                )

            # Read local file
            with open(local_file, "rb") as f:
                content = f.read()

            # Write to sandbox
            return self.write_file(sandbox_path, content, binary=True, create_directories=create_directories)

        except Exception as e:
            logger.error(f"Failed to upload file from {local_path} to {sandbox_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=sandbox_path,
                error=f"Failed to upload file: {str(e)}",
            )

    @override
    def download_file(
        self,
        sandbox_path: str,
        local_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Download a file from the E2B sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the file in the sandbox
            local_path: Destination path on the local filesystem
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Read from sandbox
            read_result = self.read_file(sandbox_path, binary=True)

            if read_result.status != SandboxStatus.SUCCESS:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=local_path,
                    error=f"Failed to read file from sandbox: {read_result.error}",
                )

            # Write to local filesystem
            local_file = Path(local_path)

            if create_directories:
                local_file.parent.mkdir(parents=True, exist_ok=True)

            with open(local_file, "wb") as f:
                content = read_result.content
                if content is None:
                    return FileOperationResult(
                        status=SandboxStatus.ERROR,
                        file_path=local_path,
                        error="Empty content returned from sandbox read",
                    )
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
                    f.write(bytes(content))

            file_size = local_file.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=local_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to download file from {sandbox_path} to {local_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=local_path,
                error=f"Failed to download file: {str(e)}",
            )

    @override
    def upload_directory(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> bool:
        """
        Upload a directory from the local filesystem to the E2B sandbox.

        Uses write_files() for batch upload to minimize round trips.

        Args:
            local_path: Path to the directory on the local filesystem
            sandbox_path: Destination path in the sandbox

        Returns:
            True if directory uploaded successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            local_dir = Path(local_path)

            if not local_dir.exists():
                raise SandboxFileError(f"Source directory does not exist: {local_path}")

            if not local_dir.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {local_path}")

            resolved_sandbox_path = self._resolve_path(sandbox_path)

            # 1. Collect all files and directories to create
            files_to_write: list[WriteEntry] = []
            parent_dirs: set[str] = {resolved_sandbox_path}

            for item in local_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(local_dir)
                    dest_path = f"{resolved_sandbox_path}/{rel_path}"
                    parent_dirs.add(str(Path(dest_path).parent))

                    with open(item, "rb") as f:
                        files_to_write.append({"path": dest_path, "data": f.read()})

            if not files_to_write:
                return True

            # 2. Create all parent directories in one shot（NAC#1312: 幂等，纳入瞬断重试）
            dirs_cmd = " ".join(f'"{d}"' for d in sorted(parent_dirs))
            self._retry_on_transient(lambda: self._sandbox.commands.run(cmd=f"mkdir -p {dirs_cmd}", user="user"))  # type: ignore[union-attr]

            # 3. Batch-write all files（幂等，纳入瞬断重试）。
            # CR 权衡（两轮意见相反后的裁决）：request_timeout=300s 的单次
            # 尝试可长阻塞，纯窗口制下慢失败会叠加 600s+；但 cap=1 又让
            # refused 类快失败 1 秒内耗尽预算失去断连自愈。取 cap=3——
            # 快失败场景 4 次尝试跨 ~7s 退避可自愈短断连，慢失败场景有界
            self._retry_on_transient(
                lambda: self._sandbox._filesystem.write_files(files_to_write, request_timeout=300.0),  # type: ignore[union-attr]
                max_retries=3,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to upload directory from {local_path} to {sandbox_path}: {e}")
            raise SandboxFileError(f"Failed to upload directory: {str(e)}")

    @override
    def download_directory(
        self,
        sandbox_path: str,
        local_path: str,
    ) -> bool:
        """
        Download a directory from the E2B sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the directory in the sandbox
            local_path: Destination path on the local filesystem

        Returns:
            True if directory downloaded successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if source directory exists
            if not self.file_exists(sandbox_path):
                raise SandboxFileError(f"Source directory does not exist: {sandbox_path}")

            # Create local directory
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)

            # List all files recursively
            files = self.list_files(sandbox_path, recursive=True)

            # Resolve sandbox_path to absolute path for comparison
            resolved_sandbox_path = self._resolve_path(sandbox_path)

            # Download all files
            for file_info in files:
                if file_info.is_file:
                    # Calculate relative path using absolute paths
                    file_abs_path = Path(file_info.path)
                    sandbox_abs_path = Path(resolved_sandbox_path)

                    try:
                        rel_path = file_abs_path.relative_to(sandbox_abs_path)
                    except ValueError:
                        # If relative_to fails, try to extract the relative part manually
                        file_path_str = str(file_abs_path)
                        sandbox_path_str = str(sandbox_abs_path)
                        if file_path_str.startswith(sandbox_path_str):
                            rel_path = Path(file_path_str[len(sandbox_path_str) :].lstrip("/"))
                        else:
                            logger.warning(f"Cannot determine relative path for {file_info.path}")
                            continue

                    dest_path = local_dir / rel_path

                    # Download file
                    result = self.download_file(file_info.path, str(dest_path), create_directories=True)

                    if result.status != SandboxStatus.SUCCESS:
                        logger.warning(f"Failed to download {file_info.path}: {result.error}")

            return True

        except Exception as e:
            logger.error(f"Failed to download directory from {sandbox_path} to {local_path}: {e}")
            raise SandboxFileError(f"Failed to download directory: {str(e)}")


@dataclass(kw_only=True)
class E2BSandboxManager(BaseSandboxManager[E2BSandbox]):
    """
    Manager for E2B sandbox lifecycle and configuration.

    This class handles:
    - Creating and configuring E2B sandboxes
    - Connecting to existing sandboxes
    - Persisting and loading sandbox state
    - Managing sandbox lifecycle (start/stop)
    """

    # E2B configuration fields
    work_dir: str | Path = field(default=E2B_DEFAULT_WORK_DIR)
    template: str = field(default_factory=lambda: os.getenv("E2B_TEMPLATE", "base"))
    # crash-safe: 与 E2B_TRANSIENT_RETRY_WINDOW 同款治理（空串/非法值回退默认）
    timeout: int = field(default_factory=lambda: int(_env_float("E2B_TIMEOUT", 300.0)))
    api_key: str | None = field(default_factory=lambda: os.getenv("E2B_API_KEY"))
    api_url: str | None = field(default_factory=lambda: os.getenv("E2B_API_URL"))
    metadata: dict[str, str] = field(default_factory=lambda: {})
    envs: dict[str, str] = field(default_factory=lambda: {})

    # Keepalive state (not init params)
    _keepalive_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _keepalive_stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = str(PurePosixPath(Path(self.work_dir).as_posix()))

    def _maybe_rebuild_for_http(
        self,
        e2b_sandbox_raw: E2BRawSandbox,
        sandbox_config: E2BSandboxConfig,
    ) -> E2BRawSandbox:
        """Conditionally rebuild a Sandbox instance for HTTP (self-host).

        If force_http is enabled and the raw sandbox has the required domain
        and access token, rebuilds the instance with an HTTP ConnectionConfig.
        Otherwise returns the original SDK instance unchanged (SaaS path).

        Args:
            e2b_sandbox_raw: Sandbox instance from connect() or beta_create()
            sandbox_config: Sandbox configuration with force_http flag

        Returns:
            The original or rebuilt Sandbox instance
        """
        if not sandbox_config.force_http:
            return e2b_sandbox_raw

        domain = e2b_sandbox_raw.sandbox_domain
        envd_token = getattr(e2b_sandbox_raw, "_envd_access_token", None)

        if domain and envd_token:
            return self._build_sandbox_with_connection_config(
                sandbox_id=e2b_sandbox_raw.sandbox_id,
                domain=domain,
                envd_access_token=envd_token,
                envd_version=e2b_sandbox_raw._envd_version,
            )

        logger.warning(
            f"force_http=True but cannot rebuild sandbox {e2b_sandbox_raw.sandbox_id[:16]}...: "
            f"domain={domain!r}, envd_token={'<set>' if envd_token else None}. "
            "Keeping original instance."
        )
        return e2b_sandbox_raw

    def _build_sandbox_with_connection_config(
        self,
        sandbox_id: str,
        domain: str,
        envd_access_token: str,
        envd_version: Version,
    ) -> E2BRawSandbox:
        """Build an HTTP Sandbox instance using ConnectionConfig (self-host only).

        Ensures all internal SDK components (_envd_api, _filesystem._envd_api, etc.)
        use the HTTP URL and X-Access-Token header from initialization.

        Important: Must reset TransportWithLogger.singleton before creating a new
        Sandbox instance. Otherwise the new instance reuses the httpcore.ConnectionPool
        from beta_create/connect, whose cached HTTPS connections to the API server
        interfere with HTTP streaming connections to envd, causing commands.run to hang.

        Args:
            sandbox_id: E2B sandbox ID
            domain: Sandbox domain for URL construction
            envd_access_token: Access token for envd API authentication
            envd_version: envd version from Sandbox Manager response

        Returns:
            Configured Sandbox instance with HTTP ConnectionConfig
        """
        # NAC#1304: 与 wrapper reconnect 路径共用同一构造原语（pre-seed + detach），
        # 消除两份实现漂移；本方法保持原签名与调用方 _maybe_rebuild_for_http 的
        # fail-open 语义不变。
        return _locked_build_http_sandbox(
            sandbox_id=sandbox_id,
            domain=domain,
            envd_access_token=envd_access_token,
            envd_version=envd_version,
        )

    @override
    def start(self, session_manager: Any, user_id: str, session_id: str, sandbox_config: SandboxConfig) -> E2BSandbox:
        """
        Start an E2B sandbox for a session.

        Args:
            session_manager: Session manager instance
            user_id: User ID
            session_id: Session ID
            sandbox_config: Typed sandbox configuration (E2BSandboxConfig expected)

        Returns:
            Configured and started E2BSandbox instance
        """
        assert E2B_AVAILABLE and Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"

        if not isinstance(sandbox_config, E2BSandboxConfig):
            raise ValueError(f"E2BSandboxManager requires E2BSandboxConfig, got {type(sandbox_config)}")

        # Override api_key/api_url from config if provided
        if sandbox_config.api_key:
            self.api_key = sandbox_config.api_key
        if sandbox_config.api_url:
            self.api_url = sandbox_config.api_url
        if sandbox_config.template and sandbox_config.template != "base":
            self.template = sandbox_config.template

        # Priority 1: Check if config contains an existing sandbox to connect to
        config_sandbox_id = sandbox_config.sandbox_id

        if config_sandbox_id:
            try:
                logger.info(f"Connecting to existing sandbox from config: {config_sandbox_id[:16]}...")

                # Use Sandbox.connect() — domain/token come from API response
                e2b_sandbox_raw = Sandbox.connect(
                    sandbox_id=config_sandbox_id,
                    timeout=sandbox_config.timeout,
                    api_key=self.api_key,
                    api_url=self.api_url,
                )

                # Extract connection info from response and rebuild with ConnectionConfig
                e2b_sandbox = self._maybe_rebuild_for_http(e2b_sandbox_raw, sandbox_config)
                envd_ver = e2b_sandbox_raw._envd_version

                sandbox = E2BSandbox(
                    work_dir=sandbox_config.work_dir,
                    output_char_threshold=sandbox_config.output_char_threshold,
                    truncate_head_chars=sandbox_config.truncate_head_chars,
                    truncate_tail_chars=sandbox_config.truncate_tail_chars,
                    max_retries=sandbox_config.max_retries,
                    transient_retry_window=sandbox_config.transient_retry_window,
                    # NAC#1304: reconnect 保真重建的判据，随 dict() 持久化
                    _force_http=sandbox_config.force_http,
                )
                sandbox.set_api_credentials(self.api_key, self.api_url)
                sandbox.envd_version = str(envd_ver)

                sandbox.sandbox = e2b_sandbox
                sandbox.sandbox_id = config_sandbox_id

                logger.info(f"Connected to existing sandbox: {config_sandbox_id[:16]}...")

                self._instance = sandbox
                self._start_keepalive(config_sandbox_id, sandbox_config.keepalive_interval)
                return sandbox

            except Exception as e:
                logger.warning(f"Failed to connect to sandbox from config: {e}. Will try session state or create new.")

        # Priority 2: Load existing sandbox state from session_manager if available
        sandbox_state = self.load_sandbox_state(session_manager, user_id, session_id)

        # Try to restore from saved state
        if sandbox_state and sandbox_state.get("sandbox_id"):
            try:
                logger.info(f"Attempting to restore E2B sandbox from state: {sandbox_state.get('sandbox_id')}")

                # Create sandbox instance from saved state
                sandbox_kwargs = extract_dataclass_init_kwargs(E2BSandbox, sandbox_state)
                sandbox = E2BSandbox(**sandbox_kwargs)
                sandbox.set_api_credentials(self.api_key, self.api_url)

                if not sandbox.sandbox_id:
                    raise SandboxError("Sandbox ID not found in state, failed to restore.")

                # Try to reconnect to existing E2B sandbox
                # Same as Priority 1: only self-host needs rebuild
                e2b_sandbox_raw = Sandbox.connect(
                    sandbox_id=sandbox.sandbox_id,
                    api_key=self.api_key,
                    api_url=self.api_url,
                )

                e2b_sandbox = self._maybe_rebuild_for_http(e2b_sandbox_raw, sandbox_config)
                envd_ver = e2b_sandbox_raw._envd_version

                sandbox.envd_version = str(envd_ver)
                sandbox.sandbox = e2b_sandbox

                logger.info(f"Successfully reconnected to E2B sandbox: {sandbox.sandbox_id}")

                self._instance = sandbox
                self._start_keepalive(sandbox.sandbox_id, sandbox_config.keepalive_interval)
                return sandbox

            except Exception as e:
                logger.warning(f"Failed to restore sandbox from state: {e}. Creating new sandbox.")

        # Create new sandbox
        logger.info(f"Creating new E2B sandbox with template: {self.template}")

        # Create E2B sandbox via SDK (Step 1: create to get sandbox_id and connection info)
        # Disable auto_pause if status_after_run is not "pause" (for RL training scenarios)
        e2b_sandbox = Sandbox.beta_create(
            template=self.template,
            timeout=self.timeout,
            api_key=self.api_key,
            api_url=self.api_url,
            metadata=self.metadata or None,
            envs=self.envs or None,
            auto_pause=sandbox_config.status_after_run == "pause",
        )
        created_sandbox_id = e2b_sandbox.sandbox_id
        envd_ver = e2b_sandbox._envd_version

        # Step 2: Self-host needs rebuild with HTTP + X-Access-Token
        e2b_sandbox = self._maybe_rebuild_for_http(e2b_sandbox, sandbox_config)

        # Create our wrapper instance
        sandbox = E2BSandbox(
            work_dir=sandbox_config.work_dir,
            output_char_threshold=sandbox_config.output_char_threshold,
            truncate_head_chars=sandbox_config.truncate_head_chars,
            truncate_tail_chars=sandbox_config.truncate_tail_chars,
            max_retries=sandbox_config.max_retries,
            transient_retry_window=sandbox_config.transient_retry_window,
            # NAC#1304: reconnect 保真重建的判据，随 dict() 持久化
            _force_http=sandbox_config.force_http,
        )
        sandbox.set_api_credentials(self.api_key, self.api_url)
        sandbox.envd_version = str(envd_ver)

        sandbox.sandbox = e2b_sandbox
        sandbox.sandbox_id = created_sandbox_id

        logger.info(f"E2B sandbox created with ID: {sandbox.sandbox_id}")

        # NAC self-host provisions and bind-mounts the canonical work directory
        # before publishing Sandbox Ready. Reissuing mkdir through envd adds a
        # redundant synchronous data-plane round trip to every create (and
        # amplifies concurrent-create latency). SaaS and custom paths retain the
        # defensive ensure because they do not share that platform contract.
        platform_provisioned_work_dir = sandbox_config.force_http and PurePosixPath(str(sandbox.work_dir)) == PurePosixPath(
            E2B_DEFAULT_WORK_DIR
        )
        if not platform_provisioned_work_dir:
            try:
                sandbox.create_directory(str(sandbox.work_dir))
                logger.debug(f"Work directory ensured: {sandbox.work_dir}")
            except Exception as e:
                logger.warning(f"Failed to create work directory {sandbox.work_dir}: {e}")

        # Persist sandbox state
        self.persist_sandbox_state(session_manager, user_id, session_id, sandbox)

        self._instance = sandbox
        self._start_keepalive(sandbox.sandbox_id, sandbox_config.keepalive_interval)
        return sandbox

    # -------------------------------------------------------------------------
    # Keepalive
    # -------------------------------------------------------------------------

    def _start_keepalive(self, sandbox_id: str | None, interval: int) -> None:
        """Start a daemon thread that periodically calls set_timeout.

        Prevents the idle checker from pausing the sandbox during long agent runs.
        """
        if interval <= 0 or not sandbox_id or Sandbox is None:
            return
        self._keepalive_stop_event.clear()
        # Capture in local vars so the closure doesn't depend on module-level None check
        sandbox_cls = Sandbox
        mgr_timeout = self.timeout
        mgr_api_url = self.api_url
        mgr_api_key = self.api_key

        def _loop() -> None:
            while not self._keepalive_stop_event.wait(interval):
                try:
                    sandbox_cls.set_timeout(
                        sandbox_id,
                        mgr_timeout,
                        api_url=mgr_api_url,
                        api_key=mgr_api_key,
                    )
                    logger.debug(f"Keepalive sent: {sandbox_id[:8]}...")
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if "409" in err_msg or "not running" in err_msg:
                        logger.warning(f"Sandbox paused, auto-resuming: {sandbox_id[:8]}...")
                        try:
                            sandbox_cls.connect(
                                sandbox_id=sandbox_id,
                                timeout=mgr_timeout,
                                api_key=mgr_api_key,
                                api_url=mgr_api_url,
                            )
                            logger.info(f"Sandbox auto-resumed: {sandbox_id[:8]}...")
                        except Exception as resume_err:
                            logger.error(f"Auto-resume failed: {resume_err}")
                    else:
                        logger.warning(f"Keepalive failed: {sandbox_id[:8]}..., error={exc}")

        self._keepalive_thread = threading.Thread(target=_loop, daemon=True)
        self._keepalive_thread.start()
        logger.debug(f"Keepalive thread started: {sandbox_id[:8]}..., interval={interval}s")

    def _stop_keepalive(self) -> None:
        """Stop the keepalive thread if running."""
        if self._keepalive_thread is not None:
            self._keepalive_stop_event.set()
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None
            logger.debug("Keepalive thread stopped")

    @override
    def on_run_complete(self) -> None:
        """Stop keepalive thread when agent execution completes."""
        self._stop_keepalive()

    @override
    def stop(self) -> bool:
        """
        Stop the E2B sandbox for a session.

        Kills the sandbox and clears ``_instance`` so that subsequent
        calls to ``start_sync()`` / ``instance`` will create a new one.
        """
        self._stop_keepalive()
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"

        instance: E2BSandbox | None = self._instance

        if not instance:
            logger.info("No sandbox instance found, nothing to stop")
            return False

        try:
            if instance.sandbox:
                instance.sandbox.kill(api_key=self.api_key, api_url=self.api_url)
                logger.info(f"E2B sandbox {instance.sandbox_id} destroyed")
            # 清除 _instance，沙箱已销毁，无法重连
            self._instance = None
            return True
        except Exception as e:
            logger.error(f"Failed to destroy E2B sandbox: {e}")
            return False

    @override
    def pause(self) -> bool:
        """
        Pause the E2B sandbox for a session.

        After pausing, clears ``_instance`` so that subsequent calls to
        ``start_sync()`` / ``instance`` detect the sandbox is gone and
        re-create via ``start()``.  The current ``sandbox_id`` is written
        into ``_session_context`` so ``start()`` can reconnect to the
        paused sandbox instead of creating a brand-new one.
        """
        self._stop_keepalive()
        try:
            instance = self._instance
            if not instance or not instance.sandbox:
                logger.warning("No E2B sandbox instance to pause; skipping pause")
                return False
            sandbox_id = instance.sandbox_id

            # Pass api_key/api_url explicitly — E2B SDK's beta_pause() does not
            # inherit the key used at creation time, so without this the call
            # fails when E2B_API_KEY env var is not set (e.g. self-hosted).
            instance.sandbox.beta_pause(api_key=self.api_key, api_url=self.api_url)

            # 1. 将 sandbox_id 写入 session_context，使下次 start() 能通过
            #    Priority 1 (config.sandbox_id) 重新连接到暂停的沙箱
            if sandbox_id and self._session_context:
                cfg = self._session_context.get("sandbox_config")
                if isinstance(cfg, E2BSandboxConfig):
                    self._session_context["sandbox_config"] = cfg.model_copy(update={"sandbox_id": sandbox_id})

            # 2. 清除 _instance，防止后续 start_sync() 返回已暂停的陈旧引用
            self._instance = None
            logger.info(f"E2B sandbox {sandbox_id} paused and instance cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to pause E2B sandbox: {e}")
            return False

    @override
    def is_running(self) -> bool:
        if self._instance is None or self._instance.sandbox is None:
            return False
        try:
            return self._instance.sandbox.is_running()
        except Exception as e:
            logger.error(f"Failed to check E2B sandbox status: {e}")
            return False
