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

"""Hook interfaces, middleware abstractions, and utilities for agent execution."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from anthropic.types import ToolParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.sandbox.base_sandbox import BaseSandbox
from nexau.core.messages import Message

from .model_response import ModelResponse
from .parse_structures import ParsedResponse

if TYPE_CHECKING:
    from ..agent_state import AgentState
    from .executor import AgentStopReason


logger = logging.getLogger(__name__)


@dataclass
class BeforeAgentHookInput:
    """Input passed to before_agent hooks prior to the run loop."""

    agent_state: AgentState
    messages: list[Message]


@dataclass
class AfterAgentHookInput:
    """Input passed to after_agent hooks once execution finishes."""

    agent_state: AgentState
    messages: list[Message]
    agent_response: str
    stop_reason: AgentStopReason | None = None


@dataclass
class BeforeModelHookInput:
    """Input data passed to before_model_hooks.

    This class encapsulates all the information that hooks receive:
    - agent_state: The AgentState containing agent context and global storage
    - messages: The current conversation history
    - max_iterations: The maximum number of iterations
    - current_iteration: The current iteration
    """

    agent_state: AgentState
    max_iterations: int
    current_iteration: int
    messages: list[Message]


@dataclass
class AfterModelHookInput(BeforeModelHookInput):
    """Input data passed to after_model_hooks.

    This class encapsulates all the information that hooks receive:
    - original_response: The raw response from the LLM
    - parsed_response: The parsed structure containing tool/agent calls
    """

    original_response: str
    parsed_response: ParsedResponse | None = None
    model_response: ModelResponse | None = None


HookResultT = TypeVar("HookResultT", bound="HookResult")


@dataclass
class HookResult:
    """Unified result object for all middleware hook phases."""

    messages: list[Message] | None = None
    parsed_response: ParsedResponse | None = None
    force_continue: bool = False
    tool_output: Any | None = None
    tool_input: dict[str, Any] | None = None
    agent_response: str | None = None

    def has_messages(self) -> bool:
        return self.messages is not None

    def has_parsed_response(self) -> bool:
        return self.parsed_response is not None

    def has_tool_output(self) -> bool:
        return self.tool_output is not None

    def has_agent_response(self) -> bool:
        return self.agent_response is not None

    def has_modifications(self) -> bool:
        return (
            self.has_messages()
            or self.has_parsed_response()
            or self.force_continue
            or self.has_tool_output()
            or self.tool_input is not None
            or self.has_agent_response()
        )

    @classmethod
    def no_changes(cls: type[HookResultT]) -> HookResultT:
        return cls()

    @classmethod
    def with_modifications(
        cls: type[HookResultT],
        *,
        messages: list[Message] | None = None,
        parsed_response: ParsedResponse | None = None,
        force_continue: bool = False,
        tool_output: Any | None = None,
        tool_input: dict[str, Any] | None = None,
        agent_response: str | None = None,
    ) -> HookResultT:
        return cls(
            messages=messages,
            parsed_response=parsed_response,
            force_continue=force_continue,
            tool_output=tool_output,
            tool_input=tool_input,
            agent_response=agent_response,
        )


class BeforeModelHookResult(HookResult):
    """Backward compatible alias for HookResult (before model)."""

    @classmethod
    def with_modifications(cls, messages: list[Message] | None = None) -> BeforeModelHookResult:  # type: ignore[override]
        return cls(messages=messages)


class AfterModelHookResult(HookResult):
    """Backward compatible alias for HookResult (after model)."""

    @classmethod
    def with_modifications(  # type: ignore[override]
        cls,
        parsed_response: ParsedResponse | None = None,
        messages: list[Message] | None = None,
        force_continue: bool = False,
    ) -> AfterModelHookResult:
        return cls(
            parsed_response=parsed_response,
            messages=messages,
            force_continue=force_continue,
        )


@dataclass
class BeforeToolHookInput:
    """Input data passed to before_tool hooks."""

    agent_state: AgentState
    sandbox: BaseSandbox | None
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]
    parallel_execution_id: str | None = None


@dataclass
class AfterToolHookInput(BeforeToolHookInput):
    """Input data passed to after_tool_hooks."""

    tool_output: Any = None


@dataclass
class AfterToolHookResult(HookResult):
    """Backward compatible alias for HookResult (after tool)."""

    @classmethod
    def with_modifications(cls, tool_output: Any) -> AfterToolHookResult:  # type: ignore[override]
        return cls(tool_output=tool_output)


class BeforeModelHook(Protocol):
    def __call__(self, hook_input: BeforeModelHookInput) -> HookResult: ...


class AfterModelHook(Protocol):
    def __call__(self, hook_input: AfterModelHookInput) -> HookResult: ...


class AfterToolHook(Protocol):
    def __call__(self, hook_input: AfterToolHookInput) -> HookResult: ...


class BeforeToolHook(Protocol):
    def __call__(self, hook_input: BeforeToolHookInput) -> HookResult: ...


@dataclass
class ModelCallParams:
    """Context passed to middleware wrapping model calls."""

    messages: list[Message]
    max_tokens: int | None
    force_stop_reason: AgentStopReason | None
    agent_state: AgentState | None
    tool_call_mode: str
    tools: list[ChatCompletionToolParam] | list[ToolParam] | None
    api_params: dict[str, Any]
    openai_client: Any | None = None
    llm_config: Any | None = None
    retry_attempts: int = 5
    shutdown_event: threading.Event | None = None


@dataclass
class ToolCallParams:
    """Context passed to middleware wrapping tool calls."""

    agent_state: AgentState
    sandbox: BaseSandbox | None
    tool_name: str
    parameters: dict[str, Any]
    tool_call_id: str
    execution_params: dict[str, Any]


ModelCallFn = Callable[[ModelCallParams], ModelResponse | None]
ToolCallFn = Callable[[ToolCallParams], Any]


class Middleware:
    """Extensible middleware abstraction for agent execution pipeline."""

    def before_agent(self, hook_input: BeforeAgentHookInput) -> HookResult:
        return HookResult.no_changes()

    def after_agent(self, hook_input: AfterAgentHookInput) -> HookResult:
        return HookResult.no_changes()

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        return HookResult.no_changes()

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        return HookResult.no_changes()

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        return HookResult.no_changes()

    def before_tool(self, hook_input: BeforeToolHookInput) -> HookResult:
        return HookResult.no_changes()

    def wrap_model_call(self, params: ModelCallParams, call_next: ModelCallFn) -> ModelResponse | None:
        """Default implementation simply forwards to the next handler."""

        return call_next(params)

    def wrap_tool_call(self, params: ToolCallParams, call_next: ToolCallFn) -> Any:
        """Default implementation simply forwards to the next handler."""

        return call_next(params)

    def stream_chunk(self, chunk: Any, params: ModelCallParams) -> Any:
        """Inspect or mutate a streaming model chunk before aggregation."""

        return chunk


class FunctionMiddleware(Middleware):
    """Wraps legacy hook callables into middleware instances."""

    def __init__(
        self,
        *,
        before_model_hook: BeforeModelHook | None = None,
        after_model_hook: AfterModelHook | None = None,
        after_tool_hook: AfterToolHook | None = None,
        before_tool_hook: BeforeToolHook | None = None,
        name: str | None = None,
    ) -> None:
        self.before_model_hook = before_model_hook
        self.after_model_hook = after_model_hook
        self.after_tool_hook = after_tool_hook
        self.before_tool_hook = before_tool_hook
        self.name = name or "function_middleware"

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        if not self.before_model_hook:
            return HookResult.no_changes()
        return self.before_model_hook(hook_input)

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        if not self.after_model_hook:
            return HookResult.no_changes()
        return self.after_model_hook(hook_input)

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        if not self.after_tool_hook:
            return HookResult.no_changes()
        return self.after_tool_hook(hook_input)

    def before_tool(self, hook_input: BeforeToolHookInput) -> HookResult:
        if not self.before_tool_hook:
            return HookResult.no_changes()
        return self.before_tool_hook(hook_input)

    def __repr__(self) -> str:  # pragma: no cover - helper for debugging
        hooks: list[str] = []
        if self.before_model_hook:
            hooks.append("before_model")
        if self.after_model_hook:
            hooks.append("after_model")
        if self.after_tool_hook:
            hooks.append("after_tool")
        if self.before_tool_hook:
            hooks.append("before_tool")
        return f"FunctionMiddleware(name={self.name}, hooks={hooks})"


class LoggingMiddleware(Middleware):
    """Middleware that logs after-model and/or after-tool phases."""

    def __init__(
        self,
        *,
        model_logger: str | None = None,
        tool_logger: str | None = None,
        message_preview_chars: int = 120,
        tool_preview_chars: int = 500,
        log_model_calls: bool = False,
    ) -> None:
        self.model_logger = logging.getLogger(model_logger) if model_logger else None
        self.tool_logger = logging.getLogger(tool_logger) if tool_logger else None
        self.message_preview_chars = message_preview_chars
        self.tool_preview_chars = tool_preview_chars
        self.log_model_calls = log_model_calls

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:  # type: ignore[override]
        logger = self.model_logger
        if not logger:
            return HookResult.no_changes()

        logger.info(
            f"before_model hook triggered agent_id: {hook_input.agent_state.agent_id}, agent_name: {hook_input.agent_state.agent_name}"
        )
        return HookResult.no_changes()

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:  # type: ignore[override]
        logger = self.model_logger
        if not logger:
            return HookResult.no_changes()

        parsed = hook_input.parsed_response
        logger.info("🎣 ===== AFTER MODEL HOOK TRIGGERED =====")
        logger.info("Agent: %s (%s)", hook_input.agent_state.agent_name, hook_input.agent_state.agent_id)
        logger.info("Response length: %s characters", len(hook_input.original_response))

        if parsed is None:
            logger.info("No parsed response available")
        else:
            logger.info("Summary: %s", parsed.get_call_summary())
            logger.info("Tool calls: %s", len(parsed.tool_calls))
            logger.info("Sub-agent calls: %s", len(parsed.sub_agent_calls))
            logger.info("Batch agent calls: %s", len(parsed.batch_agent_calls))
            logger.info("Parallel tools: %s", parsed.is_parallel_tools)
            logger.info("Parallel sub-agents: %s", parsed.is_parallel_sub_agents)

        logger.info("Message history: %s items", len(hook_input.messages))
        for idx, msg in enumerate(hook_input.messages[-3:]):
            preview = msg.get_text_content()[: self.message_preview_chars]
            logger.info("Recent message %s: %s -> %s", idx + 1, msg.role.value, preview)
            logger.info(
                f"after_model hook triggered agent_id: {hook_input.agent_state.agent_id}, agent_name: {hook_input.agent_state.agent_name}"
            )

        logger.info("🎣 ===== END AFTER MODEL HOOK =====")
        return HookResult.no_changes()

    def before_tool(self, hook_input: BeforeToolHookInput) -> HookResult:  # type: ignore[override]
        logger = self.tool_logger
        if not logger:
            return HookResult.no_changes()
        logger.info(
            f"before_tool hook triggered "
            f"tool_id: {hook_input.tool_call_id}, "
            f"tool_name: {hook_input.tool_name}, "
            f"agent_name: {hook_input.agent_state.agent_name}, "
            f"agent_id: {hook_input.agent_state.agent_id}"
        )
        return HookResult.no_changes()

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:  # type: ignore[override]
        logger = self.tool_logger
        if not logger:
            return HookResult.no_changes()

        logger.info("🔧 ===== AFTER TOOL HOOK TRIGGERED =====")
        logger.info("Agent: %s (%s)", hook_input.agent_state.agent_name, hook_input.agent_state.agent_id)
        logger.info("Tool: %s", hook_input.tool_name)
        logger.info("Input: %s", hook_input.tool_input)

        output_preview = str(hook_input.tool_output)
        if len(output_preview) > self.tool_preview_chars:
            truncated = output_preview[: self.tool_preview_chars]
            logger.info("🔧 Tool output (truncated): %s...", truncated)
        else:
            logger.info("🔧 Tool output: %s", output_preview)

        logger.info(
            f"after_tool hook triggered "
            f"tool_id: {hook_input.tool_call_id}, "
            f"tool_name: {hook_input.tool_name}, "
            f"agent_name: {hook_input.agent_state.agent_name}, "
            f"agent_id: {hook_input.agent_state.agent_id}"
        )
        logger.info("🔧 ===== END AFTER TOOL HOOK =====")
        return HookResult.no_changes()

    def wrap_model_call(self, params: ModelCallParams, call_next: ModelCallFn) -> ModelResponse | None:  # type: ignore[override]
        if not self.log_model_calls and not self.model_logger:
            return call_next(params)

        self._log_model_call(f"LLM call invoked with {len(params.messages)} messages")
        try:
            response = call_next(params)
            if response is None:
                self._log_model_call("LLM call returned no response")
            else:
                preview = (response.render_text() or response.content or "").strip()
                if preview:
                    preview = preview[: self.message_preview_chars]
                    self._log_model_call(f"LLM response preview: {preview}")
            return response
        except Exception as exc:  # pragma: no cover - logging path
            self._log_model_call(f"LLM call wrapper error: {exc}", error=True)
            raise

    def stream_chunk(self, chunk: Any, params: ModelCallParams) -> Any:
        """Inspect or mutate a streaming model chunk before aggregation."""
        logger = self.model_logger
        if logger:
            logger.info("🎣 Streaming: %s", chunk)

        return chunk

    def _log_model_call(self, message: str, error: bool = False) -> None:
        logger = self.model_logger
        if logger:
            log_fn = logger.error if error else logger.info
            log_fn(message)
        else:
            print(message)


def create_tool_after_approve_hook(tool_name: str) -> AfterModelHook:
    """Compatibility helper used by legacy examples to log approved tool usage."""

    def _hook(hook_input: AfterModelHookInput) -> HookResult:
        parsed = hook_input.parsed_response
        if not parsed or not getattr(parsed, "tool_calls", None):
            return HookResult.no_changes()

        should_log = any(getattr(call, "tool_name", None) == tool_name for call in parsed.tool_calls)
        if should_log:
            logger.info("✅ Tool '%s' auto-approved by create_tool_after_approve_hook", tool_name)
        return HookResult.no_changes()

    return _hook


class MiddlewareManager:
    """Coordinates middleware execution across the agent lifecycle."""

    def __init__(self, middlewares: list[Middleware] | None = None) -> None:
        self.middlewares: list[Middleware] = middlewares or []

    def add(self, middleware: Middleware) -> None:
        self.middlewares.append(middleware)

    def extend(self, middlewares: list[Middleware]) -> None:
        self.middlewares.extend(middlewares)

    def __bool__(self) -> bool:
        return bool(self.middlewares)

    def __len__(self) -> int:
        return len(self.middlewares)

    def run_before_agent(self, hook_input: BeforeAgentHookInput) -> list[Message]:
        for middleware in self.middlewares:
            handler = middleware.before_agent
            try:
                result = handler(hook_input)
                hook_result = self._normalize_result(result)
                if hook_result.messages is not None:
                    hook_input.messages = hook_result.messages
                    logger.info(
                        "🎣 Middleware %s (before_agent) modified messages",
                        middleware.__class__.__name__,
                    )
                else:
                    logger.info(
                        "🎣 Middleware %s (before_agent) made no changes",
                        middleware.__class__.__name__,
                    )
            except Exception as exc:
                logger.warning(f"⚠️ Before-agent middleware {middleware} failed: {exc}")
        return hook_input.messages

    def run_after_agent(
        self,
        hook_input: AfterAgentHookInput,
    ) -> tuple[str, list[Message]]:
        for middleware in reversed(self.middlewares):
            handler = middleware.after_agent
            try:
                result = handler(hook_input)
                hook_result = self._normalize_result(result)
                if hook_result.agent_response is not None:
                    hook_input.agent_response = hook_result.agent_response
                    logger.info(
                        "🎣 Middleware %s (after_agent) modified agent response",
                        middleware.__class__.__name__,
                    )
                if hook_result.messages is not None:
                    hook_input.messages = hook_result.messages
                    logger.info(
                        "🎣 Middleware %s (after_agent) modified messages",
                        middleware.__class__.__name__,
                    )
            except Exception as exc:
                logger.warning(f"⚠️ After-agent middleware {middleware} failed: {exc}")
        return hook_input.agent_response, hook_input.messages

    def run_before_model(self, hook_input: BeforeModelHookInput) -> list[Message]:
        current_messages = hook_input.messages
        for _, middleware in enumerate(self.middlewares):
            handler = getattr(middleware, "before_model", None)
            if handler is None:
                continue
            try:
                hook_input.messages = current_messages
                result = handler(hook_input)
                hook_result = self._normalize_result(result)
                if hook_result.messages is not None:
                    current_messages = hook_result.messages
                    logger.info(f"🎣 Middleware {middleware.__class__.__name__} (before_model) modified messages")
                else:
                    logger.info(f"🎣 Middleware {middleware.__class__.__name__} (before_model) made no changes")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(f"⚠️ Before-model middleware {middleware} failed: {exc}")
        return current_messages

    def run_after_model(
        self,
        hook_input: AfterModelHookInput,
    ) -> tuple[ParsedResponse | None, list[Message], bool]:
        current_parsed = hook_input.parsed_response
        current_messages = hook_input.messages
        force_continue = False
        for middleware in reversed(self.middlewares):
            handler = getattr(middleware, "after_model", None)
            if handler is None:
                continue
            try:
                hook_input.parsed_response = current_parsed
                hook_input.messages = current_messages
                result = handler(hook_input)
                hook_result = self._normalize_result(result)
                if hook_result.parsed_response is not None:
                    current_parsed = hook_result.parsed_response
                    logger.info(f"🎣 Middleware {middleware.__class__.__name__} (after_model) modified parsed response")
                if hook_result.messages is not None:
                    current_messages = hook_result.messages
                    logger.info(f"🎣 Middleware {middleware.__class__.__name__} (after_model) modified messages")
                if hook_result.force_continue:
                    force_continue = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(f"⚠️ After-model middleware {middleware} failed: {exc}")
        return current_parsed, current_messages, force_continue

    def run_after_tool(self, hook_input: AfterToolHookInput, initial_output: Any) -> Any:
        current_output = initial_output
        for middleware in reversed(self.middlewares):
            handler = getattr(middleware, "after_tool", None)
            if handler is None:
                continue
            try:
                hook_input.tool_output = current_output
                result = handler(hook_input)
                hook_result = self._normalize_result(result)
                if hook_result.tool_output is not None:
                    current_output = hook_result.tool_output
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(f"⚠️ After-tool middleware {middleware} failed: {exc}")
        return current_output

    def run_before_tool(self, hook_input: BeforeToolHookInput) -> dict[str, Any]:
        current_input = hook_input.tool_input
        for middleware in self.middlewares:
            handler = getattr(middleware, "before_tool", None)
            if handler is None:
                continue
            try:
                hook_input.tool_input = current_input
                result = handler(hook_input)
                hook_result = self._normalize_result(result)
                if hook_result.tool_input is not None:
                    current_input = hook_result.tool_input
                    logger.info(
                        "🔧 Middleware %s (before_tool) modified tool input",
                        middleware.__class__.__name__,
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"⚠️ Before-tool middleware {middleware} failed: {exc}")
        return current_input

    def wrap_model_call(self, params: ModelCallParams, call_next: ModelCallFn) -> ModelResponse | None:
        def invoke(index: int, current_params: ModelCallParams) -> ModelResponse | None:
            if index >= len(self.middlewares):
                return call_next(current_params)

            middleware = self.middlewares[index]
            wrapper = getattr(middleware, "wrap_model_call", None)
            if wrapper is None:
                return invoke(index + 1, current_params)

            def next_handler(next_params: ModelCallParams) -> ModelResponse | None:
                return invoke(index + 1, next_params)

            return wrapper(current_params, next_handler)

        return invoke(0, params)

    def stream_chunk(self, chunk: Any, params: ModelCallParams) -> Any:
        """Run stream chunks through middleware in call order."""

        current_chunk = chunk
        for middleware in self.middlewares:
            handler = getattr(middleware, "stream_chunk", None)
            if handler is None:
                continue
            try:
                result = handler(current_chunk, params)
                if result is None:
                    logger.info(
                        "🎣 Middleware %s (stream_chunk) dropped a chunk",
                        middleware.__class__.__name__,
                    )
                    return None
                if result is not current_chunk:
                    logger.info(
                        "🎣 Middleware %s (stream_chunk) modified a chunk",
                        middleware.__class__.__name__,
                    )
                current_chunk = result
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(f"⚠️ Streaming middleware {middleware} failed: {exc}")
        return current_chunk

    def wrap_tool_call(self, params: ToolCallParams, call_next: ToolCallFn) -> Any:
        def invoke(index: int, current_params: ToolCallParams) -> Any:
            if index >= len(self.middlewares):
                return call_next(current_params)

            middleware = self.middlewares[index]
            wrapper = getattr(middleware, "wrap_tool_call", None)
            if wrapper is None:
                return invoke(index + 1, current_params)

            def next_handler(next_params: ToolCallParams) -> Any:
                return invoke(index + 1, next_params)

            return wrapper(current_params, next_handler)

        return invoke(0, params)

    @staticmethod
    def _normalize_result(result: HookResult | None) -> HookResult:
        if result is None:
            return HookResult.no_changes()
        return result
