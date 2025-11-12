"""Hook interfaces, middleware abstractions, and utilities for agent execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .model_response import ModelResponse
from .parse_structures import ParsedResponse

if TYPE_CHECKING:
    from ..agent_state import AgentState
    from .executor import AgentStopReason


logger = logging.getLogger(__name__)


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
    messages: list[dict[str, Any]]


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


@dataclass
class HookResult:
    """Unified result object for all middleware hook phases."""

    messages: list[dict[str, Any]] | None = None
    parsed_response: ParsedResponse | None = None
    force_continue: bool = False
    tool_output: Any | None = None
    tool_input: dict[str, Any] | None = None

    def has_messages(self) -> bool:
        return self.messages is not None

    def has_parsed_response(self) -> bool:
        return self.parsed_response is not None

    def has_tool_output(self) -> bool:
        return self.tool_output is not None

    def has_modifications(self) -> bool:
        return (
            self.has_messages()
            or self.has_parsed_response()
            or self.force_continue
            or self.has_tool_output()
            or self.tool_input is not None
        )

    @classmethod
    def no_changes(cls) -> HookResult:
        return cls()

    @classmethod
    def with_modifications(
        cls,
        *,
        messages: list[dict[str, Any]] | None = None,
        parsed_response: ParsedResponse | None = None,
        force_continue: bool = False,
        tool_output: Any | None = None,
        tool_input: dict[str, Any] | None = None,
    ) -> HookResult:
        return cls(
            messages=messages,
            parsed_response=parsed_response,
            force_continue=force_continue,
            tool_output=tool_output,
            tool_input=tool_input,
        )


class BeforeModelHookResult(HookResult):
    """Backward compatible alias for HookResult (before model)."""

    @classmethod
    def with_modifications(cls, messages: list[dict[str, Any]] | None = None) -> BeforeModelHookResult:  # type: ignore[override]
        return cls(messages=messages)


class AfterModelHookResult(HookResult):
    """Backward compatible alias for HookResult (after model)."""

    @classmethod
    def with_modifications(  # type: ignore[override]
        cls,
        parsed_response: ParsedResponse | None = None,
        messages: list[dict[str, Any]] | None = None,
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
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]


@dataclass
class AfterToolHookInput(BeforeToolHookInput):
    """Input data passed to after_tool_hooks."""

    tool_output: Any


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

    messages: list[dict[str, Any]]
    max_tokens: int
    force_stop_reason: AgentStopReason | None
    agent_state: AgentState | None
    tool_call_mode: str
    tools: list[dict[str, Any]] | None
    api_params: dict[str, Any]
    openai_client: Any | None = None
    llm_config: Any | None = None
    retry_attempts: int = 5


@dataclass
class ToolCallParams:
    """Context passed to middleware wrapping tool calls."""

    agent_state: AgentState
    tool_name: str
    parameters: dict[str, Any]
    tool_call_id: str
    execution_params: dict[str, Any]


ModelCallFn = Callable[[ModelCallParams], ModelResponse | None]
ToolCallFn = Callable[[ToolCallParams], Any]


class Middleware:
    """Extensible middleware abstraction for agent execution pipeline."""

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        return HookResult.no_changes()

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        return HookResult.no_changes()

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        return HookResult.no_changes()

    def before_tool(self, hook_input: BeforeToolHookInput) -> HookResult:
        return HookResult.no_changes()

    def wrap_model_call(self, call_next: ModelCallFn) -> ModelCallFn:
        return call_next

    def wrap_tool_call(self, call_next: ToolCallFn) -> ToolCallFn:
        return call_next


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
        hooks = []
        if self.before_model_hook:
            hooks.append("before_model")
        if self.after_model_hook:
            hooks.append("after_model")
        if self.after_tool_hook:
            hooks.append("after_tool")
        if self.before_tool_hook:
            hooks.append("before_tool")
        return f"FunctionMiddleware(name={self.name}, hooks={hooks})"


class CustomLLMGeneratorMiddleware(Middleware):
    """Wraps legacy custom_llm_generator functions as middleware."""

    def __init__(self, generator: Callable[..., Any]) -> None:
        self.generator = generator

    def wrap_model_call(self, call_next: ModelCallFn) -> ModelCallFn:
        generator = self.generator

        def wrapped(params: ModelCallParams) -> ModelResponse | None:
            # Fallback to call_next if generator is not callable
            if callable(generator):
                backoff = 1
                for attempt in range(params.retry_attempts):
                    try:
                        generator_result = generator(
                            params.openai_client,
                            params.llm_config,
                            params.api_params,
                            params.force_stop_reason,
                            params.agent_state,
                        )
                        if generator_result is None:
                            raise ValueError("Custom generator produced no response")
                        if isinstance(generator_result, ModelResponse):
                            response = generator_result
                        else:
                            response = ModelResponse(content=str(generator_result))

                        stop = params.api_params.get("stop", [])
                        stops = stop if isinstance(stop, list) else [stop] if stop else []
                        if stops and response.content:
                            for s in stops:
                                response.content = response.content.split(s)[0]

                        return response
                    except Exception as exc:  # pragma: no cover - error path
                        logger.error(
                            "❌ Custom LLM generator failed (attempt %s/%s): %s",
                            attempt + 1,
                            params.retry_attempts,
                            exc,
                        )
                        if attempt == params.retry_attempts - 1:
                            raise
                        time.sleep(backoff)
                        backoff *= 2
            return call_next(params)

        return wrapped


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
            preview = str(msg.get("content", ""))[: self.message_preview_chars]
            logger.info("Recent message %s: %s -> %s", idx + 1, msg.get("role"), preview)

        logger.info("🎣 ===== END AFTER MODEL HOOK =====")
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
        logger.info("🔧 ===== END AFTER TOOL HOOK =====")
        return HookResult.no_changes()

    def wrap_model_call(self, call_next: ModelCallFn) -> ModelCallFn:  # type: ignore[override]
        if not self.log_model_calls and not self.model_logger:
            return call_next

        def wrapped(params: ModelCallParams) -> ModelResponse | None:
            self._log_model_call(f"Custom LLM Generator called with {len(params.messages)} messages")
            try:
                response = call_next(params)
                if response is None:
                    self._log_model_call("Custom LLM Generator returned no response")
                else:
                    preview = (response.render_text() or response.content or "").strip()
                    if preview:
                        preview = preview[: self.message_preview_chars]
                        self._log_model_call(f"Custom LLM Generator response preview: {preview}")
                return response
            except Exception as exc:  # pragma: no cover - logging path
                self._log_model_call(f"Bypass LLM generator error: {exc}", error=True)
                raise

        return wrapped

    def _log_model_call(self, message: str, error: bool = False) -> None:
        logger = self.model_logger
        if logger:
            log_fn = logger.error if error else logger.info
            log_fn(message)
        else:
            print(message)


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

    def run_before_model(self, hook_input: BeforeModelHookInput) -> list[dict[str, Any]]:
        current_messages = hook_input.messages
        for idx, middleware in enumerate(self.middlewares):
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
    ) -> tuple[ParsedResponse | None, list[dict[str, Any]], bool]:
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

    def wrap_model_call(self, call_next: ModelCallFn) -> ModelCallFn:
        wrapped = call_next
        for middleware in reversed(self.middlewares):
            wrapper = getattr(middleware, "wrap_model_call", None)
            if wrapper:
                wrapped = wrapper(wrapped)
        return wrapped

    def wrap_tool_call(self, call_next: ToolCallFn) -> ToolCallFn:
        wrapped = call_next
        for middleware in reversed(self.middlewares):
            wrapper = getattr(middleware, "wrap_tool_call", None)
            if wrapper:
                wrapped = wrapper(wrapped)
        return wrapped

    @staticmethod
    def _normalize_result(result: HookResult | None) -> HookResult:
        if result is None:
            return HookResult.no_changes()
        if isinstance(result, HookResult):
            return result
        raise TypeError(f"Middleware returned unsupported result type: {type(result)}")


def _coerce_after_model_result(result: HookResult) -> AfterModelHookResult:
    if isinstance(result, AfterModelHookResult):
        return result
    return AfterModelHookResult(
        parsed_response=result.parsed_response,
        messages=result.messages,
        force_continue=result.force_continue,
    )


def _coerce_after_tool_result(result: HookResult) -> AfterToolHookResult:
    if isinstance(result, AfterToolHookResult):
        return result
    return AfterToolHookResult(
        tool_output=result.tool_output,
        tool_input=result.tool_input,
    )


def create_logging_hook(logger_name: str = "after_model_hook") -> AfterModelHook:
    """Legacy-compatible logging hook backed by LoggingMiddleware."""

    middleware = LoggingMiddleware(model_logger=logger_name)

    def hook(hook_input: AfterModelHookInput) -> HookResult:
        return _coerce_after_model_result(middleware.after_model(hook_input))

    return hook


def create_remaining_reminder_hook(logger_name: str = "after_model_hook") -> AfterModelHook:
    """Append a reminder about remaining iterations and log it."""

    import logging

    logger = logging.getLogger(logger_name)

    def hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        remaining = hook_input.max_iterations - hook_input.current_iteration
        logger.info("🎣 Remaining iterations: %s", remaining)
        updated = hook_input.messages + [
            {"role": "user", "content": f"Remaining iterations: {remaining}"},
        ]
        return AfterModelHookResult.with_modifications(messages=updated)

    return hook


def create_tool_after_approve_hook(tool_name: str) -> AfterModelHook:
    """Prompt the operator before running the specified tool."""

    def hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        parsed = hook_input.parsed_response
        if not parsed or not parsed.tool_calls:
            return AfterModelHookResult.no_changes()

        blocked_calls = []
        for call in parsed.tool_calls:
            if call.tool_name != tool_name:
                continue
            print(f"🎣 Tool call: {call.tool_name}")
            print(f"🎣 Tool call parameters: {call.parameters}")
            while True:
                approval = input(f"Approve running {tool_name}? (y/n): ")
                if approval.lower() not in {"y", "n"}:
                    print("🎣 Invalid input. Please enter 'y' or 'n'.")
                    continue
                if approval.lower() == "n":
                    blocked_calls.append(call)
                break

        if not blocked_calls:
            return AfterModelHookResult.no_changes()

        parsed.tool_calls = [call for call in parsed.tool_calls if call not in blocked_calls]
        return AfterModelHookResult.with_modifications(parsed_response=parsed)

    return hook


def create_tool_logging_hook(logger_name: str = "after_tool_hook") -> AfterToolHook:
    """Legacy-compatible tool logging hook backed by LoggingMiddleware."""

    middleware = LoggingMiddleware(tool_logger=logger_name)

    def hook(hook_input: AfterToolHookInput) -> HookResult:
        return _coerce_after_tool_result(middleware.after_tool(hook_input))

    return hook
