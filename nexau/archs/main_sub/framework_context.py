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

"""Typed framework context for tool and middleware authors.

RFC-0006: FrameworkContext — 类型安全的框架上下文

替代 AgentState，提供分组 API 访问框架服务。
工具函数声明 ctx: FrameworkContext 即可获得所有框架能力。
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexau.archs.session.models.agent_run_action_model import ReplaceVariantBase
    from nexau.archs.tool.tool import Tool
    from nexau.archs.tool.tool_registry import ToolRegistry
    from nexau.core.messages import Message

    from .history_list import HistoryList


logger = logging.getLogger(__name__)


class HistoryAPI:
    """Write-side typed-event API for agent history.

    RFC-0026: replaces the RFC-0022 Phase 3 ``agent_state.history`` direct
    backreference. ContextCompactionMiddleware (and any future writer of
    typed REPLACE events — ``/clear`` / ``/compact <focus>``) emits through
    this API instead of reaching into HistoryList directly.

    RPC-friendly by design: every public method takes only serializable
    arguments (Pydantic ``Message`` + ``ReplaceVariantBase`` subclass).
    No HistoryList object handle ever crosses the API boundary, so when
    running in remote-tool mode (lambda / RPC future), this becomes a
    thin RPC stub with no in-process state to marshal.

    Intentional minimal surface — only operations with concrete callers
    today are exposed:
      - :meth:`replace` — typed REPLACE (compaction, /clear, /compact)
    Read access (``inspect prior messages``) and other event types
    (``append`` / ``undo``) are deliberately NOT pre-exposed; they get
    added when a real production caller appears.

    Internal: holds an optional ``HistoryList`` handle. None when the
    agent has no SessionManager (in-process tests) — all methods become
    safe no-ops, matching the existing HistoryList persistence-disabled
    semantics.
    """

    def __init__(
        self,
        *,
        _history: HistoryList | None,
    ) -> None:
        self._history = _history

    def replace(
        self,
        messages: list[Message],
        *,
        extra: ReplaceVariantBase,
    ) -> None:
        """Emit a typed REPLACE event into agent history.

        Used for compaction (auto / manual / focused), ``/clear``, and any
        future state-reset operation that carries semantic intent. The
        ``extra`` discriminates the reason
        (``CompactAutoVariant`` / ``UserClearVariant`` /
        ``CompactFocusedVariant`` / ...) so reader-side replay can render
        or aggregate by reason without inspecting message content.

        Args:
            messages: New full message list to install as the post-replace
                state. The persisted action row carries ``messages`` as
                ``replace_messages`` (RFC-0022 Phase 1 column) and the
                in-memory list is realigned synchronously so subsequent
                flushes don't double-write.
            extra: Required typed variant. There is no untyped path through
                this API — untyped REPLACE inferred from the
                fingerprint-diff fallback inside ``HistoryList.flush()``
                stays as the back-compat path for middleware that hasn't
                migrated yet (RFC-0026 Stage 2 will close that gap).
        """
        if self._history is None:
            return
        self._history.replace_all(messages, replace_extra=extra)


class ExecutionAPI:
    """Execution lifecycle API for stop-aware tools.

    RFC-0006: 封装 shutdown 信号，提供语义化的执行控制接口

    隐藏底层 threading.Event 实现细节，tool 作者只需调用
    ``ctx.execution.is_shutting_down()`` 即可判断是否应中止。
    """

    def __init__(
        self,
        *,
        _shutdown_event: threading.Event,
    ) -> None:
        self._shutdown_event = _shutdown_event

    def is_shutting_down(self) -> bool:
        """Check if the agent is being stopped.

        RFC-0006: 语义化的停止检查

        Returns:
            True if a shutdown has been signaled.
        """
        return self._shutdown_event.is_set()


class ToolsAPI:
    """Tools management API.

    RFC-0006: 封装 ToolRegistry，只暴露操作语义
    """

    def __init__(
        self,
        *,
        _tool_registry: ToolRegistry,
    ) -> None:
        self._tool_registry = _tool_registry

    def search(self, *, query: str, max_results: int = 5) -> list[Tool]:
        """Search deferred tools and inject matches.

        RFC-0006: 封装 ToolRegistry.search()

        搜到即注入，下一轮 LLM 可直接 function call。
        支持 "+keyword" 强制匹配。

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of matched and injected tools
        """
        return self._tool_registry.search(query, max_results=max_results)

    def add(self, *, tool: Tool) -> None:
        """Dynamically add an eager tool to the current execution.

        RFC-0006: 直接写入 ToolRegistry

        Deferred runtime additions are intentionally unsupported for now.

        Args:
            tool: Tool instance to add
        """
        if tool.defer_loading:
            raise ValueError(
                "Runtime-added deferred tools are not supported. Register deferred tools during agent initialization instead.",
            )

        self._tool_registry.add_source("runtime", [tool])

    def get(self, *, name: str) -> Tool | None:
        """Look up a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tool_registry.get_tool(name)


class FrameworkContext:
    """Typed framework context for tool and middleware authors.

    RFC-0006: 替代 AgentState，提供类型安全的分组 API。
    工具函数声明 ctx: FrameworkContext 即可获得所有框架能力。

    当前实现阶段：Tools API + Execution API（Phase 1）
    后续阶段将加入 skills / agents / sandbox / variables 等分组 API。
    """

    def __init__(
        self,
        *,
        agent_name: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        _tool_registry: ToolRegistry,
        _shutdown_event: threading.Event,
        _history: HistoryList | None = None,
        session_id: str = "",
        tool_name: str = "",
        allow_rules: list[str] | None = None,
        deny_rules: list[str] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.run_id = run_id
        self.root_run_id = root_run_id

        # RFC-0024: caller-supplied W3C trace id (32-hex). Opaque to nexau —
        # populated by ``Agent.run_async(trace_id=...)``, threaded into the
        # Executor → FrameworkContext, and inherited explicitly by sub-agents
        # via ``call_sub_agent(trace_id=...)``. Surfaced into RUN_START rows
        # and the live ``RunStartedEvent`` so a single trace links the whole
        # call tree. None when caller didn't supply one.
        self.trace_id: str | None = trace_id

        # RFC-0019: 权限数据字段
        self.session_id = session_id
        self.tool_name = tool_name
        self.allow_rules: list[str] = allow_rules if allow_rules is not None else ["**"]
        self.deny_rules: list[str] = deny_rules if deny_rules is not None else []

        # 分组 API
        self.tools = ToolsAPI(
            _tool_registry=_tool_registry,
        )
        self.execution = ExecutionAPI(
            _shutdown_event=_shutdown_event,
        )
        # RFC-0026: typed-event write API for compaction / /clear / etc.
        # Replaces the deprecated ``agent_state.history`` direct backref.
        self.history = HistoryAPI(_history=_history)

        # RFC-0019: 保留内部引用以便 for_tool_call 复用
        self._tool_registry = _tool_registry
        self._shutdown_event = _shutdown_event
        self._history = _history

    def for_tool_call(
        self,
        *,
        tool_name: str,
        allow_rules: list[str],
        deny_rules: list[str],
    ) -> FrameworkContext:
        """Create a per-tool-call context with tool-specific permission data.

        RFC-0019: 每次 tool call 前构造独立 FrameworkContext

        共享 ToolsAPI / ExecutionAPI / HistoryAPI 实例，但 permission 字段独立，
        保证并行 tool call 之间不互相干扰。
        """
        return FrameworkContext(
            agent_name=self.agent_name,
            agent_id=self.agent_id,
            run_id=self.run_id,
            root_run_id=self.root_run_id,
            _tool_registry=self._tool_registry,
            _shutdown_event=self._shutdown_event,
            _history=self._history,
            session_id=self.session_id,
            tool_name=tool_name,
            allow_rules=allow_rules,
            deny_rules=deny_rules,
            trace_id=self.trace_id,
        )

    @classmethod
    def for_testing(
        cls,
        *,
        agent_name: str = "test_agent",
        agent_id: str = "test-agent-id",
        run_id: str = "test-run-id",
        root_run_id: str = "test-root-run-id",
        shutdown_event: threading.Event | None = None,
        session_id: str = "",
        tool_name: str = "",
        allow_rules: list[str] | None = None,
        deny_rules: list[str] | None = None,
    ) -> FrameworkContext:
        """Create a FrameworkContext for unit testing tool functions.

        RFC-0006: 测试工厂方法

        提供合理默认值，用户只需覆盖关心的字段。
        tools API 使用空的 ToolRegistry（支持 search/add/get 但无预注册工具）。

        Example::

            def test_my_tool():
                ctx = FrameworkContext.for_testing()
                result = my_tool("hello", ctx=ctx)
                assert result == "expected"


            def test_my_tool_with_shutdown():
                event = threading.Event()
                ctx = FrameworkContext.for_testing(shutdown_event=event)
                event.set()
                assert ctx.execution.is_shutting_down()
        """
        from nexau.archs.tool.tool_registry import ToolRegistry

        return cls(
            agent_name=agent_name,
            agent_id=agent_id,
            run_id=run_id,
            root_run_id=root_run_id,
            _tool_registry=ToolRegistry(),
            _shutdown_event=shutdown_event or threading.Event(),
            session_id=session_id,
            tool_name=tool_name,
            allow_rules=allow_rules,
            deny_rules=deny_rules,
        )

    def __repr__(self) -> str:
        return f"FrameworkContext(agent_name='{self.agent_name}', agent_id='{self.agent_id}')"
