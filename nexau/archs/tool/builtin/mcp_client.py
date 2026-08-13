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

"""Run-scoped MCP integration backed exclusively by the official SDK.

RFC-0029: NexAU owns configuration, permissions and result adaptation.  The
official MCP Python SDK owns protocol negotiation, JSON-RPC, transports,
subprocesses and their cleanup.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import warnings
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp.client import Client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult
from mcp.types import Tool as MCPToolType
from pydantic import BaseModel

from nexau.archs.permissions.helpers import check_mcp_permission

from ..tool import Tool
from .mcp_auth import MCPAuthContext, MCPAuthHost, build_http_auth, build_http_client, redact_sensitive_data
from .mcp_result import adapt_call_tool_result, format_mcp_tool_output_for_llm, list_all_tools

if TYPE_CHECKING:
    from nexau.archs.main_sub.framework_context import FrameworkContext

logger = logging.getLogger(__name__)


def _sanitized_error(error: Exception) -> RuntimeError:
    """Detach transport exceptions from potentially credential-bearing text."""
    return RuntimeError(str(redact_sensitive_data(str(error))))


@dataclass
class MCPServerConfig:
    """Compatibility value object for programmatic MCP configuration.

    Agent YAML is validated by the discriminated Pydantic models in
    ``main_sub.config.schema``.  This class remains callable for existing
    Python integrations and is normalized to the same runtime contract.
    """

    name: str
    type: str = "stdio"
    command: str | None = None
    args: list[str] | None = field(default=None, repr=False)
    env: dict[str, str] | None = field(default=None, repr=False)
    url: str | None = field(default=None, repr=False)
    headers: dict[str, str] | None = field(default=None, repr=False)
    timeout: float | None = 30
    disable_parallel: bool = False
    source_id: str | None = None
    permissions: dict[str, list[str]] | None = None
    tool_permissions: dict[str, dict[str, list[str]] | None] | None = None
    auth: object | None = field(default=None, repr=False)


def _config_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return cast(Mapping[str, Any], value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if isinstance(value, MCPServerConfig):
        return vars(value)
    raise TypeError(f"MCP server configuration must be a mapping or model, got {type(value).__name__}")


def _normalize_server_config(value: object) -> MCPServerConfig:
    if isinstance(value, MCPServerConfig):
        return value
    data = _config_mapping(value)
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("MCP server configuration requires a non-empty name")
    return MCPServerConfig(
        name=name,
        type=str(data.get("type") or "stdio"),
        command=str(data["command"]) if data.get("command") is not None else None,
        args=[str(item) for item in cast(Sequence[object], data.get("args") or [])] or None,
        env={str(key): str(item) for key, item in cast(Mapping[object, object], data.get("env") or {}).items()} or None,
        url=str(data["url"]) if data.get("url") is not None else None,
        headers={str(key): str(item) for key, item in cast(Mapping[object, object], data.get("headers") or {}).items()} or None,
        timeout=float(data["timeout"]) if data.get("timeout") is not None else None,
        disable_parallel=bool(data.get("disable_parallel", False)),
        source_id=str(data["source_id"]) if data.get("source_id") is not None else None,
        permissions=cast(dict[str, list[str]] | None, data.get("permissions")),
        tool_permissions=cast(dict[str, dict[str, list[str]] | None] | None, data.get("tool_permissions")),
        auth=data.get("auth"),
    )


class MCPRuntimeFactory:
    """Loop-agnostic factory for per-run MCP connection scopes."""

    def __init__(
        self,
        server_configs: Sequence[object],
        *,
        auth_host: MCPAuthHost | None = None,
        auth_context: MCPAuthContext | None = None,
    ) -> None:
        configs = [_normalize_server_config(config) for config in server_configs]
        names = [config.name for config in configs]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique within a runtime factory")
        self.server_configs = tuple(configs)
        # RFC-0029: retain one host so its TokenStorage survives across run scopes.
        self.auth_host = auth_host or MCPAuthHost()
        self.auth_context = auth_context

    def open_scope(self) -> MCPRunScope:
        """Create a fresh scope; no SDK session is retained on this factory."""
        return MCPRunScope(self)

    def auth_context_for(self, config: MCPServerConfig) -> MCPAuthContext:
        base = self.auth_context
        return MCPAuthContext(
            identity=base.identity if base is not None else self.auth_host.identity,
            server_name=config.name,
            source_id=config.source_id,
        )


_ACTIVE_SCOPE: contextvars.ContextVar[MCPRunScope | None] = contextvars.ContextVar(
    "nexau_active_mcp_scope",
    default=None,
)


class MCPRunScope:
    """Own official SDK clients and transports for exactly one Agent run.

    Every SDK context is entered, used, and exited by the task that owns this
    scope.  Server failures are isolated; successful servers remain usable.
    """

    def __init__(self, factory: MCPRuntimeFactory) -> None:
        self.factory = factory
        self._stack: AsyncExitStack | None = None
        self._clients: dict[str, Client] = {}
        self._tools: list[MCPTool] = []
        self._failures: dict[str, Exception] = {}
        self._active_token: contextvars.Token[MCPRunScope | None] | None = None
        self._owner_task: asyncio.Task[Any] | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    @property
    def failures(self) -> Mapping[str, Exception]:
        return dict(self._failures)

    async def __aenter__(self) -> MCPRunScope:
        if self._stack is not None:
            raise RuntimeError("MCPRunScope cannot be entered more than once")
        self._owner_task = asyncio.current_task()
        self._owner_loop = asyncio.get_running_loop()
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self._active_token = _ACTIVE_SCOPE.set(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._assert_owner()
        token, self._active_token = self._active_token, None
        if token is not None:
            _ACTIVE_SCOPE.reset(token)
        stack, self._stack = self._stack, None
        self._clients.clear()
        self._tools.clear()
        if stack is None:
            return False
        await stack.__aexit__(exc_type, exc, traceback)
        return False

    def _assert_owner(self) -> None:
        self._assert_active()
        if asyncio.current_task() is not self._owner_task:
            raise RuntimeError("MCPRunScope lifecycle must be managed by the task that entered it")

    def _assert_active(self) -> None:
        if self._stack is None:
            raise RuntimeError("MCPRunScope is not active")
        if asyncio.get_running_loop() is not self._owner_loop:
            raise RuntimeError("MCPRunScope cannot be used across event loops")

    async def _connect_server(self, config: MCPServerConfig) -> Client:
        self._assert_owner()
        existing = self._clients.get(config.name)
        if existing is not None:
            return existing
        if self._stack is None:  # narrowed by _assert_owner; keeps static analyzers honest
            raise RuntimeError("MCPRunScope is not active")

        timeout = config.timeout
        if config.type == "stdio":
            if not config.command:
                raise ValueError(f"MCP stdio server '{config.name}' requires command")
            params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                # The SDK combines these values with its safe environment list.
                env=config.env,
            )
            transport = stdio_client(params)
            client = Client(
                transport,
                mode="auto",
                read_timeout_seconds=timeout,
            )
        elif config.type == "http":
            if not config.url:
                raise ValueError(f"MCP HTTP server '{config.name}' requires url")
            http_client = await self._stack.enter_async_context(
                build_http_client(
                    vars(config),
                    auth_host=self.factory.auth_host,
                    auth_context=self.factory.auth_context_for(config),
                )
            )
            client = Client(
                streamable_http_client(config.url, http_client=http_client),
                mode="auto",
                read_timeout_seconds=timeout,
            )
        elif config.type == "sse":
            if not config.url:
                raise ValueError(f"MCP SSE server '{config.name}' requires url")
            auth = await self._stack.enter_async_context(
                build_http_auth(
                    vars(config),
                    auth_host=self.factory.auth_host,
                    auth_context=self.factory.auth_context_for(config),
                )
            )
            request_timeout = timeout if timeout is not None else 5.0
            read_timeout = timeout if timeout is not None else 300.0
            client = Client(
                sse_client(
                    config.url,
                    headers=config.headers,
                    timeout=request_timeout,
                    sse_read_timeout=read_timeout,
                    auth=auth,
                ),
                mode="legacy",
                read_timeout_seconds=timeout,
            )
        else:
            raise ValueError(f"Unsupported MCP server type for '{config.name}': {config.type!r}")

        entered = await self._stack.enter_async_context(client)
        self._clients[config.name] = entered
        return entered

    async def discover_tools(self) -> list[MCPTool]:
        """Connect and paginate tools/list for every server, fail-soft per server."""
        self._assert_owner()
        discovered: list[MCPTool] = []
        self._failures.clear()
        for config in self.factory.server_configs:
            try:
                client = await self._connect_server(config)
                sdk_tools = await list_all_tools(client)
                discovered.extend(MCPTool(tool, self.factory, config) for tool in sdk_tools)
            except Exception as error:
                safe_error = _sanitized_error(error)
                self._failures[config.name] = safe_error
                logger.warning(
                    "MCP server '%s' (%s) unavailable for this run: %s",
                    config.name,
                    config.type,
                    safe_error,
                )
        self._tools = discovered
        return list(discovered)

    def get_tools(self) -> list[MCPTool]:
        self._assert_owner()
        return list(self._tools)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> CallToolResult:
        """Call a tool through the official high-level Client API."""
        self._assert_active()
        config = next((item for item in self.factory.server_configs if item.name == server_name), None)
        if config is None:
            raise KeyError(f"Unknown MCP server: {server_name}")
        client = self._clients.get(server_name)
        if client is None:
            if asyncio.current_task() is not self._owner_task:
                failure = self._failures.get(config.name)
                raise RuntimeError(f"MCP server '{server_name}' is unavailable") from failure
            try:
                client = await self._connect_server(config)
            except Exception as error:
                safe_error = _sanitized_error(error)
                self._failures[config.name] = safe_error
                raise RuntimeError(f"MCP server '{server_name}' is unavailable: {safe_error}") from safe_error
        try:
            return await client.call_tool(
                tool_name,
                dict(arguments or {}),
                read_timeout_seconds=config.timeout,
            )
        except Exception as error:
            safe_error = _sanitized_error(error)
            raise RuntimeError(f"MCP tool '{server_name}/{tool_name}' failed: {safe_error}") from safe_error


class MCPTool(Tool):
    """Stable NexAU descriptor that routes execution to the active run scope."""

    def __init__(
        self,
        mcp_tool: MCPToolType,
        runtime_factory: MCPRuntimeFactory,
        server_config: MCPServerConfig,
    ) -> None:
        self.mcp_tool = mcp_tool
        self.runtime_factory = runtime_factory
        self.server_config = server_config
        self._server_name = server_config.name
        self._raw_tool_name = mcp_tool.name

        resolved_permissions: dict[str, list[str]] | None = None
        if server_config.tool_permissions is not None and self._raw_tool_name in server_config.tool_permissions:
            resolved_permissions = server_config.tool_permissions[self._raw_tool_name]
        elif server_config.permissions is not None:
            resolved_permissions = server_config.permissions

        super().__init__(
            name=f"mcp__{server_config.name}__{mcp_tool.name}",
            description=mcp_tool.description or "",
            input_schema=dict(mcp_tool.input_schema),
            implementation=self._execute_sync,
            disable_parallel=server_config.disable_parallel,
            formatter=format_mcp_tool_output_for_llm,
            permissions=resolved_permissions,
            source_id=server_config.source_id,
        )
        self._has_native_async_execute = True

    @staticmethod
    def _filtered_arguments(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            sorted(
                ((key, value) for key, value in kwargs.items() if key not in {"agent_state", "global_storage", "sandbox", "ctx"}),
                key=lambda item: item[0],
            )
        )

    def _check_permission(self, kwargs: Mapping[str, Any]) -> None:
        context = cast("FrameworkContext | None", kwargs.get("ctx"))
        if context is not None:
            check_mcp_permission(context, self._server_name, self._raw_tool_name)

    def _execute_sync(self, **kwargs: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute_async(**kwargs))
        raise RuntimeError("MCPTool.execute() cannot run inside an event loop; use await execute_async()")

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute from a synchronous host using a fresh official SDK scope."""
        return self._execute_sync(**kwargs)

    async def execute_async(self, **kwargs: Any) -> dict[str, Any]:
        """Execute in the active Agent run, or a fully closed one-shot scope."""
        self._check_permission(kwargs)
        arguments = self._filtered_arguments(kwargs)
        scope = _ACTIVE_SCOPE.get()
        if scope is not None and scope.factory is self.runtime_factory:
            result = await scope.call_tool(self._server_name, self._raw_tool_name, arguments)
        else:
            # Deprecated standalone/bootstrap tools remain callable without
            # retaining an SDK client across event loops.
            async with self.runtime_factory.open_scope() as one_shot_scope:
                result = await one_shot_scope.call_tool(self._server_name, self._raw_tool_name, arguments)
        return cast(dict[str, Any], adapt_call_tool_result(result).tool_output)


class MCPClient:
    """Deprecated standalone facade; every operation opens and closes a scope."""

    def __init__(self, *, auth_host: MCPAuthHost | None = None) -> None:
        warnings.warn(
            "MCPClient is deprecated; use MCPRuntimeFactory.open_scope()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.servers: dict[str, MCPServerConfig] = {}
        self.sessions: dict[str, object] = {}
        self.tools: dict[str, MCPTool] = {}
        self._auth_host = auth_host or MCPAuthHost()

    def add_server(self, config: MCPServerConfig | object) -> None:
        normalized = _normalize_server_config(config)
        self.servers[normalized.name] = normalized

    async def connect_to_server(self, server_name: str) -> bool:
        if server_name not in self.servers:
            return False
        factory = MCPRuntimeFactory([self.servers[server_name]], auth_host=self._auth_host)
        tools: list[MCPTool] = []
        async with factory.open_scope() as scope:
            tools = await scope.discover_tools()
        self.tools = {tool.name: tool for tool in tools}
        return server_name not in scope.failures

    async def discover_tools(self, server_name: str) -> list[MCPTool]:
        if server_name not in self.servers:
            return []
        prefix = f"mcp__{server_name}__"
        if not any(name.startswith(prefix) for name in self.tools):
            await self.connect_to_server(server_name)
        return [tool for name, tool in self.tools.items() if name.startswith(prefix)]

    def get_all_tools(self) -> Sequence[Tool]:
        return list(self.tools.values())

    def get_tool(self, tool_name: str) -> MCPTool | None:
        """Return one cached compatibility descriptor by its NexAU name."""
        return self.tools.get(tool_name)

    async def disconnect_server(self, server_name: str) -> None:
        prefix = f"mcp__{server_name}__"
        self.tools = {name: tool for name, tool in self.tools.items() if not name.startswith(prefix)}

    async def disconnect_all(self) -> None:
        self.sessions.clear()
        self.tools.clear()


class MCPManager:
    """Deprecated compatibility manager with no global or persistent sessions."""

    def __init__(self, *, auth_host: MCPAuthHost | None = None) -> None:
        warnings.warn(
            "MCPManager is deprecated; use MCPRuntimeFactory.open_scope()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.client = MCPClient(auth_host=auth_host)
        self.auto_connect = True

    def add_server(
        self,
        name: str,
        server_type: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        disable_parallel: bool = False,
        source_id: str | None = None,
        permissions: dict[str, list[str]] | None = None,
        tool_permissions: dict[str, dict[str, list[str]] | None] | None = None,
        auth: object | None = None,
    ) -> None:
        self.client.add_server(
            MCPServerConfig(
                name=name,
                type=server_type,
                command=command,
                args=args,
                env=env,
                url=url,
                headers=headers,
                timeout=timeout,
                disable_parallel=disable_parallel,
                source_id=source_id,
                permissions=permissions,
                tool_permissions=tool_permissions,
                auth=auth,
            )
        )

    async def initialize_servers(self) -> dict[str, list[MCPTool]]:
        discovered: dict[str, list[MCPTool]] = {}
        for name in self.client.servers:
            if await self.client.connect_to_server(name):
                discovered[name] = await self.client.discover_tools(name)
        return discovered

    def get_available_tools(self) -> Sequence[Tool]:
        return self.client.get_all_tools()

    async def shutdown(self) -> None:
        await self.client.disconnect_all()


def get_mcp_manager() -> MCPManager:
    """Return a fresh deprecated manager; no process-global state is retained."""
    warnings.warn(
        "get_mcp_manager() is deprecated and no longer returns a global singleton",
        DeprecationWarning,
        stacklevel=2,
    )
    return MCPManager()


async def initialize_mcp_tools(
    server_configs: Sequence[object],
    *,
    auth_host: MCPAuthHost | None = None,
    auth_context: MCPAuthContext | None = None,
) -> Sequence[Tool]:
    """Bootstrap tool metadata in a fully entered and closed SDK scope."""
    factory = MCPRuntimeFactory(server_configs, auth_host=auth_host, auth_context=auth_context)
    tools: list[MCPTool] = []
    async with factory.open_scope() as scope:
        tools = await scope.discover_tools()
    return tools


def sync_initialize_mcp_tools(
    server_configs: Sequence[object],
    *,
    auth_host: MCPAuthHost | None = None,
    auth_context: MCPAuthContext | None = None,
) -> Sequence[Tool]:
    """Synchronous bootstrap wrapper for non-async constructors and scripts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            initialize_mcp_tools(
                server_configs,
                auth_host=auth_host,
                auth_context=auth_context,
            )
        )
    raise RuntimeError("sync_initialize_mcp_tools() cannot be called from an async context; use await initialize_mcp_tools()")


__all__ = [
    "MCPClient",
    "MCPManager",
    "MCPRunScope",
    "MCPRuntimeFactory",
    "MCPServerConfig",
    "MCPTool",
    "get_mcp_manager",
    "initialize_mcp_tools",
    "sync_initialize_mcp_tools",
]
