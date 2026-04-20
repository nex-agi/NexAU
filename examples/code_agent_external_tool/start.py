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

"""Code agent example with **all tools as external tools** (RFC-0018).

Contrast with ``examples/code_agent/start.py``:
- Origin example: tools have local bindings; Executor invokes them inside the
  agent loop.
- This example: every tool is declared ``kind: external``. The agent loop
  pauses whenever the LLM calls one, returning pending tool calls to *this*
  script. We then dispatch them to the real NexAU builtin implementations —
  the exact same functions the origin example uses — and feed results back via
  ``ToolResultBlock`` messages on a second ``run_async()`` call.

The simulated "remote" execution happens in :func:`_execute_external_call`
below. Structurally it is just a dispatch table from tool name → real builtin
callable, invoked through :meth:`Tool.execute_async` so the framework's param
filtering / validation / formatter pipeline still runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nexau import Agent, AgentConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware, Event
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool import Tool
from nexau.archs.tool.builtin.file_tools import list_directory, read_file, write_file
from nexau.archs.tool.builtin.session_tools import complete_task
from nexau.archs.tool.builtin.shell_tools import run_shell_command
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer
from nexau.core.messages import Message, Role, ToolResultBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parent
ORIGIN_TOOLS_DIR = EXAMPLE_DIR.parent / "code_agent" / "tools"


def _get_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _handle_event(event: Event) -> None:
    # Brief type-only log; full event payloads can stall the tty when tool
    # args / model chunks are large.
    print(f"[event] {type(event).__name__}")


# --------------------------------------------------------------------------- #
# External tool runner
# --------------------------------------------------------------------------- #
#
# Build a parallel set of **bound** Tools using the same YAMLs but with real
# Python implementations. This is the "host application" side: it owns the
# real tool execution. When the agent pauses on a pending external call, we
# dispatch it here.


def _build_local_runner_tools() -> dict[str, Tool]:
    """Build bound Tool instances reusing the origin example's YAML schemas.

    This mirrors how ``examples/code_agent/code_agent.yaml`` wires its tools —
    same YAML, same binding — but lives in caller-land instead of inside the
    agent's tool registry.
    """
    return {
        "read_file": Tool.from_yaml(str(ORIGIN_TOOLS_DIR / "read_file.tool.yaml"), read_file),
        "write_file": Tool.from_yaml(str(ORIGIN_TOOLS_DIR / "write_file.tool.yaml"), write_file),
        "list_directory": Tool.from_yaml(str(ORIGIN_TOOLS_DIR / "list_directory.tool.yaml"), list_directory),
        "run_shell_command": Tool.from_yaml(str(ORIGIN_TOOLS_DIR / "run_shell_command.tool.yaml"), run_shell_command),
        "complete_task": Tool.from_yaml(str(ORIGIN_TOOLS_DIR / "complete_task.tool.yaml"), complete_task),
    }


def _build_runner_state(agent: Agent) -> tuple[AgentState, FrameworkContext, AgentContext]:
    """Construct a minimal AgentState + FrameworkContext for external execution.

    We reuse the agent's sandbox_manager and global_storage so file/shell tools
    resolve paths against the same workspace the agent's system prompt advertised.
    The ``AgentContext`` is returned so the caller can close it on shutdown.
    """
    runner_registry = ToolRegistry()
    shutdown_event = threading.Event()
    run_id = f"external_runner_{uuid.uuid4().hex[:8]}"

    # AgentContext must be entered to be valid; caller is responsible for exit.
    ctx_mgr = AgentContext(context={})
    ctx_mgr.__enter__()

    state = AgentState(
        agent_name="external_runner",
        agent_id="external_runner",
        run_id=run_id,
        root_run_id=run_id,
        context=ctx_mgr,
        global_storage=agent.global_storage,
        tool_registry=runner_registry,
        sandbox_manager=agent.sandbox_manager,
    )
    framework_ctx = FrameworkContext(
        agent_name=state.agent_name,
        agent_id=state.agent_id,
        run_id=state.run_id,
        root_run_id=state.root_run_id,
        _tool_registry=runner_registry,
        _shutdown_event=shutdown_event,
    )
    return state, framework_ctx, ctx_mgr


async def _execute_external_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    runner_tools: dict[str, Tool],
    runner_state: AgentState,
    framework_ctx: FrameworkContext,
) -> ToolResultBlock:
    """Run a pending tool call locally and wrap the result for the agent."""
    tool = runner_tools.get(name)
    if tool is None:
        return ToolResultBlock(
            tool_use_id=call_id,
            content=json.dumps({"error": f"Unknown tool: {name}"}),
            is_error=True,
        )

    print(f"  [external] executing {name}({json.dumps(arguments, ensure_ascii=False)})")
    try:
        result = await tool.execute_async(
            **arguments,
            agent_state=runner_state,
            sandbox=runner_state.get_sandbox(),
            ctx=framework_ctx,
        )
    except Exception as exc:  # Defensive: runner mirrors a remote host boundary
        logger.exception("external tool %s failed", name)
        return ToolResultBlock(
            tool_use_id=call_id,
            content=json.dumps({"error": str(exc), "type": type(exc).__name__}),
            is_error=True,
        )

    # Tool output is always a dict; drop UI-only fields before handing to the LLM.
    is_error = "error" in result
    result.pop("returnDisplay", None)
    return ToolResultBlock(
        tool_use_id=call_id,
        content=json.dumps(result, ensure_ascii=False, default=str),
        is_error=is_error,
    )


async def _run_with_external_tools(
    agent: Agent,
    user_message: str,
    context: dict[str, Any],
    stop_tools: set[str],
) -> tuple[str, AgentContext]:
    """Drive the pause/resume loop until the agent returns a final string.

    RFC-0018 contract:
    - ``run_async`` returns ``str`` when the agent finishes normally.
    - It returns ``(response, {"stop_reason": ..., "pending_tool_calls": [...]})``
      when paused on external tools. The caller resumes by sending back a
      ``Role.TOOL`` ``Message`` containing ``ToolResultBlock`` entries.

    Stop-tool shortcut: if the LLM called a ``stop_tool`` (e.g. ``complete_task``)
    while paused, we execute it locally and return its result as the final answer
    without resuming the loop. Otherwise the next ``run_async`` iteration would
    just re-call the LLM with a "task already done" context and yield an empty
    response.

    Returns the final response and the ``AgentContext`` the caller must exit.
    """
    runner_tools = _build_local_runner_tools()
    runner_state, framework_ctx, ctx_mgr = _build_runner_state(agent)

    next_message: str | list[Message] = user_message
    next_context: dict[str, Any] | None = context

    # Pin Langfuse trace_id across pause/resume turns so all observations
    # (initial LLM call + external tool result roundtrips + final stop) land on
    # a single trace. Without this, each ``run_async`` creates a brand-new trace.
    # 32-char hex satisfies Langfuse's W3C trace-id format.
    trace_id = uuid.uuid4().hex
    langfuse_tracers = [t for t in (agent.config.tracers or []) if isinstance(t, LangfuseTracer)]
    for tracer in langfuse_tracers:
        tracer.set_trace_id(trace_id)

    while True:
        result = await agent.run_async(message=next_message, context=next_context)
        # Subsequent resume turns don't need context again — history carries it.
        next_context = None

        if isinstance(result, str):
            return result, ctx_mgr

        response_text, meta = result
        if meta.get("stop_reason") != AgentStopReason.EXTERNAL_TOOL_CALL.name:
            return response_text, ctx_mgr  # Other stop reasons surface verbatim.

        pending = meta.get("pending_tool_calls") or []
        print(f"\n[agent paused] {len(pending)} external tool call(s) pending")

        tool_results = [
            await _execute_external_call(
                call_id=call["id"],
                name=call["name"],
                arguments=call.get("input") or {},
                runner_tools=runner_tools,
                runner_state=runner_state,
                framework_ctx=framework_ctx,
            )
            for call in pending
        ]

        # RFC-0018 + stop_tools: if the LLM called a declared stop tool in this
        # pause, extract its ``result`` (or ``returnDisplay``) and return. We don't
        # resume the agent because the framework's stop-tool short-circuit only
        # runs on the local-execution path — external calls skip step 10 in the
        # executor loop (they break for pause instead).
        for call, block in zip(pending, tool_results):
            if call["name"] in stop_tools:
                payload = call.get("input") or {}
                return str(payload.get("result") or block.content), ctx_mgr

        next_message = [Message(role=Role.TOOL, content=list(tool_results))]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def _amain() -> bool:
    print("Testing Code Agent (external-tool mode)")
    print("=" * 60)

    print("Loading agent from YAML configuration...")
    config = AgentConfig.from_yaml(config_path=EXAMPLE_DIR / "code_agent.yaml")
    event_middleware = AgentEventsMiddleware(session_id="test", on_event=_handle_event)
    config.middlewares = [*(config.middlewares or []), event_middleware]
    stop_tools = set(config.stop_tools or [])

    agent = await Agent.create(config=config, global_storage=GlobalStorage())
    print("Agent loaded (all tools are external)")

    # Sandbox startup is blocking I/O (session_manager load + sandbox spawn) that
    # internally calls ``asyncio.run`` via ``run_async_function_sync``. Running it
    # inline would deadlock the current event loop, so hop to a worker thread.
    # Once ``_instance`` is cached, later ``start_sync()`` calls fast-path out.
    sandbox = await asyncio.to_thread(lambda: agent.sandbox_manager.instance)
    if sandbox is None:
        raise RuntimeError("Sandbox failed to start")
    print(f"Sandbox ready: {sandbox.sandbox_id}, work_dir={sandbox.work_dir}")

    user_message = input("\nEnter your task: ")
    print(f"\nUser: {user_message}\nAgent Response:\n" + "-" * 30)

    working_dir = str(sandbox.work_dir)
    context = {
        "date": _get_date(),
        "username": os.getenv("USER"),
        "working_directory": working_dir,
        "env_content": {
            "date": _get_date(),
            "username": os.getenv("USER"),
            "working_directory": working_dir,
        },
    }

    ctx_mgr: AgentContext | None = None
    try:
        response, ctx_mgr = await _run_with_external_tools(agent, user_message, context, stop_tools)
        print("\n" + "=" * 60)
        print(response)
        return True
    finally:
        if ctx_mgr is not None:
            ctx_mgr.__exit__(None, None, None)


def _install_sigint_handler() -> None:
    """Make Ctrl+C forcibly terminate even when background threads are alive.

    NexAU spawns a sandbox ThreadPoolExecutor and async-engine bridges that hold
    non-daemon resources. The default SIGINT handler raises KeyboardInterrupt in
    the main thread, which ``asyncio.run`` may absorb while awaiting a blocking
    future. Re-raising as ``os._exit`` on the second press ensures exit.
    """
    presses = {"n": 0}

    def _handler(signum: int, frame: Any) -> None:
        presses["n"] += 1
        if presses["n"] == 1:
            print("\n[Ctrl+C] shutting down — press again to force-exit")
            raise KeyboardInterrupt
        os._exit(130)

    signal.signal(signal.SIGINT, _handler)


def main() -> bool:
    _install_sigint_handler()
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        print("Interrupted")
        return False
    except Exception as exc:
        import traceback

        print(f"Error: {exc}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
