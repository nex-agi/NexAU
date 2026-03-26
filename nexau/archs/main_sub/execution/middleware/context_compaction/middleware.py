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

"""Context compaction middleware with customizable trigger and compaction strategies."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

import anthropic
import openai

from nexau.archs.llm.llm_aggregators.events import CompactionFinishedEvent, CompactionStartedEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.tool.tool import StructuredToolDefinitionLike
from nexau.archs.tracer.context import get_current_span
from nexau.archs.tracer.core import BaseTracer, Span, SpanType

from ...hooks import AfterModelHookInput, BeforeModelHookInput, HookResult, Middleware, ModelCallFn, ModelCallParams
from ...llm_caller import LLMCaller
from .config import CompactionConfig
from .factory import create_compaction_strategy, create_emergency_compaction_strategy, create_trigger_strategy
from .llm_config_utils import normalize_summary_llm_overrides, resolve_summary_llm_config

if TYPE_CHECKING:
    from ....utils.token_counter import TokenCounter
    from .compact_stratigies.user_model_full_trace_adaptive import UserModelFullTraceAdaptiveCompaction

from nexau.core.messages import Message, Role, ToolUseBlock

logger = logging.getLogger(__name__)

OpenAI = openai.OpenAI
Anthropic = anthropic.Anthropic

_CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context window",
    "context token",
    "context limit",
    "prompt is too long",
    "prompt too long",
    "input is too long",
    "input token count",
    "token limit exceeded",
    "context_length_exceeded",
)
CompactionPhase = Literal["before_model", "after_model", "wrap_model_call"]
CompactionMode = Literal["regular", "emergency"]


class ContextCompactionMiddleware(Middleware):
    """Middleware for automatic context compaction with customizable strategies.

    This middleware monitors token usage and automatically compacts the message
    history when certain thresholds are exceeded. Both the trigger logic and
    compaction logic can be customized.

    Default behavior:
    - Trigger: When token usage reaches 75% (25% remaining)
    - Compaction: Sliding window - keeps only the most recent N messages (default: 20)
    - Simple and efficient: No summarization overhead
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize context compaction middleware.

        Args:
            token_counter: Token counter instance for fallback counting
            **kwargs: Configuration parameters validated by CompactionConfig.
                     All configuration including strategies should be passed via kwargs.
        """
        # Initialize statistics
        self._compact_count = 0
        self._total_messages_removed = 0
        self._event_emitter: Callable[[Any], None] | None = None
        self._session_id: str | None = None
        self._global_storage: Any | None = None

        # Tracer span state for in-flight compaction (one compaction at a time)
        self._active_compaction_span: Span | None = None
        self._active_compaction_tracer: BaseTracer | None = None

        # Initialize token counter
        if token_counter is None:
            from ....utils.token_counter import TokenCounter

            self.token_counter = TokenCounter()
        else:
            self.token_counter = token_counter

        # Configuration parameters are required
        if not kwargs:
            raise ValueError(
                "Configuration parameters (kwargs) are required. Provide at least the required config fields or use default values."
            )

        # Pydantic validates the flat YAML/dict here
        config = CompactionConfig(**kwargs)

        # None = inherit from AgentConfig at runtime via set_llm_runtime
        self.max_context_tokens: int | None = config.max_context_tokens
        self.auto_compact = config.auto_compact
        self.emergency_compact_enabled = config.emergency_compact_enabled
        self.summary_llm_config_overrides = normalize_summary_llm_overrides(
            config.summary_llm_config,
            summary_model=config.summary_model,
            summary_base_url=config.summary_base_url,
            summary_api_key=config.summary_api_key,
            summary_api_type=config.summary_api_type,
        )
        self.summary_model = self.summary_llm_config_overrides.get("model")
        self.summary_base_url = self.summary_llm_config_overrides.get("base_url")
        self.summary_api_key = self.summary_llm_config_overrides.get("api_key")
        self.summary_api_type = self.summary_llm_config_overrides.get("api_type")

        # Create strategies from config
        self.trigger_strategy = create_trigger_strategy(config)
        self.compaction_strategy = create_compaction_strategy(config)

        # Emergency strategy requires max_context_tokens; defer when None
        if self.max_context_tokens is not None:
            self.emergency_compaction_strategy: UserModelFullTraceAdaptiveCompaction | None = create_emergency_compaction_strategy(
                token_counter=self.token_counter,
                max_context_tokens=self.max_context_tokens,
            )
        else:
            self.emergency_compaction_strategy = None

        emergency_name = (
            self.emergency_compaction_strategy.__class__.__name__ if self.emergency_compaction_strategy is not None else "deferred"
        )
        logger.info(
            f"[ContextCompactionMiddleware] Initialized: "
            f"max_context_tokens={self.max_context_tokens}, "
            f"auto_compact={self.auto_compact}, "
            f"emergency_compact_enabled={self.emergency_compact_enabled}, "
            f"trigger={self.trigger_strategy.__class__.__name__}, "
            f"compaction={self.compaction_strategy.__class__.__name__}, "
            f"emergency_compaction={emergency_name}"
        )

    def set_llm_runtime(
        self,
        llm_config: LLMConfig,
        openai_client: Any | None = None,
        *,
        session_id: str | None = None,
        global_storage: Any | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        """Inject the agent LLM runtime used when no standalone summary config is set."""
        self._session_id = session_id
        self._global_storage = global_storage

        # Resolve max_context_tokens: explicit config > agent-level fallback
        if self.max_context_tokens is None and max_context_tokens is not None:
            self.max_context_tokens = max_context_tokens
            logger.info(
                "[ContextCompactionMiddleware] Inherited max_context_tokens=%d from agent config",
                max_context_tokens,
            )

        # Create emergency strategy now that max_context_tokens is resolved
        if self.emergency_compaction_strategy is None and self.max_context_tokens is not None:
            self.emergency_compaction_strategy = create_emergency_compaction_strategy(
                token_counter=self.token_counter,
                max_context_tokens=self.max_context_tokens,
            )

        configure_runtime = getattr(self.compaction_strategy, "configure_llm_runtime", None)
        if callable(configure_runtime):
            configure_runtime(
                llm_config,
                openai_client,
                session_id=session_id,
                global_storage=global_storage,
                max_context_tokens=self.max_context_tokens,
            )

    def _build_client(self, llm_config: LLMConfig) -> Any | None:
        if llm_config.api_type == "gemini_rest":
            return None
        client_kwargs = llm_config.to_client_kwargs()
        if llm_config.api_type == "anthropic_chat_completion":
            return Anthropic(**client_kwargs)
        if llm_config.api_type in ["openai_responses", "openai_chat_completion"]:
            return OpenAI(**client_kwargs)
        raise ValueError(f"Invalid API type: {llm_config.api_type}")

    def _resolve_summary_runtime(
        self,
        llm_config: LLMConfig,
        openai_client: Any | None,
    ) -> tuple[LLMConfig, Any | None]:
        summary_llm_config = resolve_summary_llm_config(
            base_llm_config=llm_config,
            summary_overrides=self.summary_llm_config_overrides,
        )

        reuse_base_client = (
            openai_client is not None
            and llm_config.api_type == summary_llm_config.api_type
            and llm_config.to_client_kwargs() == summary_llm_config.to_client_kwargs()
        )
        summary_client = openai_client if reuse_base_client else self._build_client(summary_llm_config)
        return summary_llm_config, summary_client

    def set_event_emitter(self, emitter: Callable[[Any], None]) -> None:
        """Inject a unified event emitter (typically from AgentEventsMiddleware)."""
        self._event_emitter = emitter

    @staticmethod
    def _resolve_run_id(agent_state: Any | None) -> str | None:
        if agent_state is None:
            return None
        run_id = getattr(agent_state, "run_id", None)
        return run_id if isinstance(run_id, str) and run_id else None

    @staticmethod
    def _resolve_tracer(agent_state: Any | None) -> BaseTracer | None:
        """Extract tracer from agent_state.global_storage if available."""
        if agent_state is None:
            return None
        gs = getattr(agent_state, "global_storage", None)
        if gs is None:
            return None
        tracer = gs.get("tracer")
        if isinstance(tracer, BaseTracer):
            return tracer
        return None

    def _start_compaction_span(
        self,
        agent_state: Any | None,
        phase: CompactionPhase,
        mode: CompactionMode,
        trigger_reason: str | None,
        original_message_count: int,
        original_token_count: int | None,
    ) -> None:
        """Start a tracer span for the compaction operation (sent to Langfuse)."""
        tracer = self._resolve_tracer(agent_state)
        if tracer is None:
            return
        try:
            parent_span = get_current_span()
            self._active_compaction_span = tracer.start_span(
                name=f"context_compaction.{phase}",
                span_type=SpanType.COMPACTION,
                inputs={
                    "phase": phase,
                    "mode": mode,
                    "trigger_reason": trigger_reason,
                    "original_message_count": original_message_count,
                    "original_token_count": original_token_count,
                    "max_context_tokens": self.max_context_tokens,
                },
                parent_span=parent_span,
                attributes={
                    "compaction.phase": phase,
                    "compaction.mode": mode,
                },
            )
            self._active_compaction_tracer = tracer
        except Exception as exc:
            logger.debug("[ContextCompactionMiddleware] Failed to start compaction tracer span: %s", exc)
            self._active_compaction_span = None
            self._active_compaction_tracer = None

    def _end_compaction_span(
        self,
        phase: CompactionPhase,
        mode: CompactionMode,
        success: bool,
        trigger_reason: str | None,
        original_message_count: int,
        compacted_message_count: int | None,
        original_token_count: int | None,
        compacted_token_count: int | None,
        error: str | None,
    ) -> None:
        """End the active compaction tracer span (sent to Langfuse)."""
        tracer = self._active_compaction_tracer
        span = self._active_compaction_span
        self._active_compaction_span = None
        self._active_compaction_tracer = None
        if tracer is None or span is None:
            return
        try:
            outputs: dict[str, object] = {
                "success": success,
                "compacted_message_count": compacted_message_count,
                "compacted_token_count": compacted_token_count,
            }
            if error:
                outputs["error"] = error
            tracer.end_span(
                span,
                outputs=outputs,
                error=Exception(error) if error else None,
                attributes={
                    "compaction.phase": phase,
                    "compaction.mode": mode,
                    "compaction.success": success,
                    "compaction.trigger_reason": trigger_reason or "",
                    "compaction.original_message_count": original_message_count,
                    "compaction.compacted_message_count": compacted_message_count,
                    "compaction.original_token_count": original_token_count,
                    "compaction.compacted_token_count": compacted_token_count,
                    "compaction.max_context_tokens": self.max_context_tokens,
                },
            )
        except Exception as exc:
            logger.debug("[ContextCompactionMiddleware] Failed to end compaction tracer span: %s", exc)

    def _emit_compaction_started(
        self,
        *,
        agent_state: Any | None,
        phase: CompactionPhase,
        mode: CompactionMode,
        trigger_reason: str | None,
        original_message_count: int,
        original_token_count: int | None,
    ) -> None:
        # 1. 启动 tracer span（独立于 event emitter）
        self._start_compaction_span(
            agent_state,
            phase,
            mode,
            trigger_reason,
            original_message_count,
            original_token_count,
        )

        # 2. 发射事件（可能因缺少 emitter 提前返回）
        if self._event_emitter is None:
            return
        run_id = self._resolve_run_id(agent_state)
        if run_id is None:
            return
        self._event_emitter(
            CompactionStartedEvent(
                run_id=run_id,
                phase=phase,
                mode=mode,
                trigger_reason=trigger_reason,
                original_message_count=original_message_count,
                original_token_count=original_token_count,
                max_context_tokens=self.max_context_tokens,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
        )

    def _emit_compaction_finished(
        self,
        *,
        agent_state: Any | None,
        phase: CompactionPhase,
        mode: CompactionMode,
        success: bool,
        trigger_reason: str | None,
        original_message_count: int,
        compacted_message_count: int | None,
        original_token_count: int | None,
        compacted_token_count: int | None,
        error: str | None = None,
    ) -> None:
        # 1. 结束 tracer span（独立于 event emitter）
        self._end_compaction_span(
            phase,
            mode,
            success,
            trigger_reason,
            original_message_count,
            compacted_message_count,
            original_token_count,
            compacted_token_count,
            error,
        )

        # 2. 发射事件（可能因缺少 emitter 提前返回）
        if self._event_emitter is None:
            return
        run_id = self._resolve_run_id(agent_state)
        if run_id is None:
            return
        self._event_emitter(
            CompactionFinishedEvent(
                run_id=run_id,
                phase=phase,
                mode=mode,
                success=success,
                trigger_reason=trigger_reason,
                original_message_count=original_message_count,
                compacted_message_count=compacted_message_count,
                original_token_count=original_token_count,
                compacted_token_count=compacted_token_count,
                max_context_tokens=self.max_context_tokens,
                error=error,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
        )

    def _get_current_tokens(self, hook_input: AfterModelHookInput) -> int | None:
        """Extract token count from model response usage information.

        Args:
            hook_input: After model hook input containing model response

        Returns:
            Total token count if available, None otherwise
        """
        if not hook_input.model_response or not hook_input.model_response.usage:
            return None

        usage = hook_input.model_response.usage
        current_tokens = usage.context_used_tokens()
        if current_tokens > 0:
            return current_tokens
        return usage.total_tokens if usage.total_tokens > 0 else None

    @staticmethod
    def _is_context_overflow_error(exc: BaseException) -> bool:
        """Best-effort detection for provider errors indicating prompt/context overflow."""
        visited: set[int] = set()
        pending: list[BaseException] = [exc]

        while pending:
            current = pending.pop()
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            text_parts = [str(current)]
            body = getattr(current, "body", None)
            if body is not None:
                text_parts.append(str(body))

            text = " ".join(text_parts).lower()
            if any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS):
                return True

            cause = current.__cause__
            if isinstance(cause, BaseException):
                pending.append(cause)

            context = current.__context__
            if isinstance(context, BaseException):
                pending.append(context)

        return False

    @staticmethod
    def _normalize_tools_for_token_count(
        tools: Sequence[StructuredToolDefinitionLike] | None,
    ) -> list[Mapping[str, object]] | None:
        """Convert tool payloads into dicts for token estimation when possible."""
        if not tools:
            return None

        normalized: list[Mapping[str, object]] = []
        for tool in tools:
            normalized.append(cast(Mapping[str, object], tool))

        return normalized

    def _estimate_tokens(
        self,
        messages: list[Message],
        tools: Sequence[StructuredToolDefinitionLike] | None = None,
    ) -> int | None:
        """Estimate tokens for diagnostics; never raise from middleware paths."""
        normalized_tools = self._normalize_tools_for_token_count(tools)
        try:
            return self.token_counter.count_tokens(messages, tools=normalized_tools)
        except Exception as exc:  # pragma: no cover - diagnostic-only fallback
            logger.debug("[ContextCompactionMiddleware] Token estimation failed: %s", exc)
            return None

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        """Compaction before the model call when budget is near limit."""
        if not self.auto_compact:
            return HookResult.no_changes()

        if self.max_context_tokens is None:
            logger.warning("[ContextCompactionMiddleware] max_context_tokens not resolved, skipping pre-call compaction")
            return HookResult.no_changes()

        messages = hook_input.messages
        current_tokens = self._estimate_tokens(messages)
        if current_tokens is None:
            return HookResult.no_changes()

        usage_ratio = current_tokens / self.max_context_tokens
        logger.info(
            "[ContextCompactionMiddleware] Pre-call compaction check: %d messages, %d/%d tokens (%.1f%%)",
            len(messages),
            current_tokens,
            self.max_context_tokens,
            usage_ratio * 100,
        )

        should_compact, trigger_reason = self.trigger_strategy.should_compact(
            messages,
            current_tokens,
            self.max_context_tokens,
        )
        if not should_compact:
            return HookResult.no_changes()

        logger.info("[ContextCompactionMiddleware] Pre-call compaction triggered: %s", trigger_reason)
        original_count = len(messages)
        self._emit_compaction_started(
            agent_state=hook_input.agent_state,
            phase="before_model",
            mode="regular",
            trigger_reason=trigger_reason,
            original_message_count=original_count,
            original_token_count=current_tokens,
        )
        try:
            self._sync_strategy_global_storage(hook_input.agent_state)
            compacted_messages = self._compact_messages(messages)
        except Exception as exc:
            self._emit_compaction_finished(
                agent_state=hook_input.agent_state,
                phase="before_model",
                mode="regular",
                success=False,
                trigger_reason=trigger_reason,
                original_message_count=original_count,
                compacted_message_count=None,
                original_token_count=current_tokens,
                compacted_token_count=None,
                error=str(exc),
            )
            raise

        self._compact_count += 1
        self._total_messages_removed += original_count - len(compacted_messages)
        after_tokens = self._estimate_tokens(compacted_messages)
        self._emit_compaction_finished(
            agent_state=hook_input.agent_state,
            phase="before_model",
            mode="regular",
            success=True,
            trigger_reason=trigger_reason,
            original_message_count=original_count,
            compacted_message_count=len(compacted_messages),
            original_token_count=current_tokens,
            compacted_token_count=after_tokens,
        )

        logger.info(
            "[ContextCompactionMiddleware] Pre-call compaction complete: %d -> %d messages",
            original_count,
            len(compacted_messages),
        )
        return HookResult.with_modifications(messages=compacted_messages)

    def _build_emergency_summarize_fn(self, params: ModelCallParams) -> Callable[[list[Message], str, int], str]:
        if params.llm_config is None:
            raise RuntimeError("llm_config is required for emergency compaction summarization")

        base_llm_config = cast(LLMConfig, params.llm_config)
        summary_llm_config, summary_client = self._resolve_summary_runtime(base_llm_config, params.openai_client)

        # RFC-0009: 传递 global_storage 以支持 Langfuse 追踪
        gs = getattr(self, "_global_storage", None)
        if gs is None and params.agent_state is not None:
            gs = getattr(params.agent_state, "global_storage", None)
        llm_caller = LLMCaller(
            summary_client,
            summary_llm_config,
            retry_attempts=1,
            middleware_manager=None,
            session_id=self._session_id,
            global_storage=gs,
        )

        def summarize_fn(messages: list[Message], prompt: str, max_tokens: int) -> str:
            summary_messages = list(messages)
            summary_messages.append(Message.user(prompt))
            response = llm_caller.call_llm(
                summary_messages,
                max_tokens=max_tokens,
                force_stop_reason=None,
                agent_state=params.agent_state,
                tool_call_mode=params.tool_call_mode,
                tools=None,
                openai_client=params.openai_client,
                shutdown_event=params.shutdown_event,
            )
            text = (response.content or "").strip() if response else ""
            return text if text else "NONE"

        return summarize_fn

    def wrap_model_call(
        self,
        params: ModelCallParams,
        call_next: ModelCallFn,
    ) -> Any:
        """Fallback path: emergency compact and retry once when provider overflows."""
        try:
            return call_next(params)
        except Exception as exc:
            if not self.auto_compact or not self.emergency_compact_enabled or not self._is_context_overflow_error(exc):
                raise

            if self.emergency_compaction_strategy is None:
                logger.error(
                    "[ContextCompactionMiddleware] Emergency compaction unavailable: "
                    "max_context_tokens was never resolved (set_llm_runtime not called?)"
                )
                raise

            logger.warning(
                "[ContextCompactionMiddleware] Model call failed due to context overflow; "
                "attempting fixed two-segment emergency compaction retry"
            )

            original_messages = params.messages
            summarize_fn = self._build_emergency_summarize_fn(params)
            before_tokens = self._estimate_tokens(original_messages, params.tools)
            self._emit_compaction_started(
                agent_state=params.agent_state,
                phase="wrap_model_call",
                mode="emergency",
                trigger_reason="provider_context_overflow",
                original_message_count=len(original_messages),
                original_token_count=before_tokens,
            )
            try:
                compacted_messages = self.emergency_compaction_strategy.compact(
                    original_messages,
                    summarize_fn=summarize_fn,
                )
            except Exception as compact_exc:
                self._emit_compaction_finished(
                    agent_state=params.agent_state,
                    phase="wrap_model_call",
                    mode="emergency",
                    success=False,
                    trigger_reason="provider_context_overflow",
                    original_message_count=len(original_messages),
                    compacted_message_count=None,
                    original_token_count=before_tokens,
                    compacted_token_count=None,
                    error=str(compact_exc),
                )
                raise

            # Stamp session_id on emergency summary messages for traceability,
            # matching the behaviour of SlidingWindowCompaction (regular path).
            if self._session_id is not None:
                for msg in compacted_messages:
                    md = msg.metadata
                    if md.get("is_compacted") is True or md.get("isSummary") is True:
                        md.setdefault("session_id", self._session_id)

            self._compact_count += 1
            self._total_messages_removed += len(original_messages) - len(compacted_messages)

            after_tokens = self._estimate_tokens(compacted_messages, params.tools)

            if after_tokens is not None and self.max_context_tokens is not None and after_tokens >= self.max_context_tokens:
                logger.error(
                    "[ContextCompactionMiddleware] Emergency compaction result still exceeds context limit: %d/%d",
                    after_tokens,
                    self.max_context_tokens,
                )
                self._emit_compaction_finished(
                    agent_state=params.agent_state,
                    phase="wrap_model_call",
                    mode="emergency",
                    success=False,
                    trigger_reason="provider_context_overflow",
                    original_message_count=len(original_messages),
                    compacted_message_count=len(compacted_messages),
                    original_token_count=before_tokens,
                    compacted_token_count=after_tokens,
                    error=f"maximum context length exceeded after emergency compaction ({after_tokens}/{self.max_context_tokens})",
                )
                raise RuntimeError(
                    f"maximum context length exceeded after emergency compaction ({after_tokens}/{self.max_context_tokens})"
                ) from exc

            token_stats = f", tokens {before_tokens} -> {after_tokens}" if before_tokens is not None and after_tokens is not None else ""

            logger.warning(
                "[ContextCompactionMiddleware] Fallback compaction complete: %d -> %d messages%s. Retrying model call once.",
                len(original_messages),
                len(compacted_messages),
                token_stats,
            )
            self._emit_compaction_finished(
                agent_state=params.agent_state,
                phase="wrap_model_call",
                mode="emergency",
                success=True,
                trigger_reason="provider_context_overflow",
                original_message_count=len(original_messages),
                compacted_message_count=len(compacted_messages),
                original_token_count=before_tokens,
                compacted_token_count=after_tokens,
            )

            compacted_params = replace(params, messages=compacted_messages)
            return call_next(compacted_params)

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        """Check and compact messages after each model call if needed."""
        if not self.auto_compact:
            return HookResult.no_changes()

        if self.max_context_tokens is None:
            logger.warning("[ContextCompactionMiddleware] max_context_tokens not resolved, skipping post-call compaction")
            return HookResult.no_changes()

        messages = hook_input.messages

        # Get token count from model response usage information
        current_tokens = self._get_current_tokens(hook_input)

        # If we couldn't get token count from model response, fall back to token_counter
        if current_tokens is None:
            logger.warning("[ContextCompactionMiddleware] No usage information from model response, falling back to token_counter")
            current_tokens = self.token_counter.count_tokens(messages)
            logger.info(f"[ContextCompactionMiddleware] Local token estimation: {current_tokens}")
        else:
            logger.info(f"[ContextCompactionMiddleware] Using API-reported token usage: {current_tokens}")

        usage_ratio = current_tokens / self.max_context_tokens

        logger.info(
            f"[ContextCompactionMiddleware] Checking compaction: "
            f"{len(messages)} messages, {current_tokens}/{self.max_context_tokens} tokens ({usage_ratio:.1%})"
        )

        # Check if the last assistant message has tool calls
        # If not, skip compaction to preserve the conversation state
        last_assistant_msg: Message | None = None
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT:
                last_assistant_msg = msg
                break

        if last_assistant_msg is not None:
            has_tool_calls = any(isinstance(block, ToolUseBlock) for block in last_assistant_msg.content)

            if not has_tool_calls:
                logger.info(
                    "[ContextCompactionMiddleware] Last assistant message has no tool calls, "
                    "skipping compaction to preserve conversation state"
                )
                return HookResult.no_changes()
            else:
                logger.info("[ContextCompactionMiddleware] Last assistant message has tool calls, proceeding with compaction check")

        # Check if compaction should be triggered
        should_compact, trigger_reason = self.trigger_strategy.should_compact(
            messages,
            current_tokens,
            self.max_context_tokens,
        )

        if not should_compact:
            logger.info("[ContextCompactionMiddleware] No compaction needed")
            return HookResult.no_changes()

        logger.info(f"[ContextCompactionMiddleware] Compaction triggered: {trigger_reason}")

        # Perform compaction
        original_message_count = len(messages)
        self._emit_compaction_started(
            agent_state=hook_input.agent_state,
            phase="after_model",
            mode="regular",
            trigger_reason=trigger_reason,
            original_message_count=original_message_count,
            original_token_count=current_tokens,
        )
        try:
            self._sync_strategy_global_storage(hook_input.agent_state)
            compacted_messages = self._compact_messages(messages)
        except Exception as exc:
            self._emit_compaction_finished(
                agent_state=hook_input.agent_state,
                phase="after_model",
                mode="regular",
                success=False,
                trigger_reason=trigger_reason,
                original_message_count=original_message_count,
                compacted_message_count=None,
                original_token_count=current_tokens,
                compacted_token_count=None,
                error=str(exc),
            )
            raise
        compacted_message_count = len(compacted_messages)
        compacted_tokens = self._estimate_tokens(compacted_messages)

        # Update statistics
        self._compact_count += 1
        self._total_messages_removed += original_message_count - compacted_message_count
        self._emit_compaction_finished(
            agent_state=hook_input.agent_state,
            phase="after_model",
            mode="regular",
            success=True,
            trigger_reason=trigger_reason,
            original_message_count=original_message_count,
            compacted_message_count=compacted_message_count,
            original_token_count=current_tokens,
            compacted_token_count=compacted_tokens,
        )

        logger.info(
            f"[ContextCompactionMiddleware] Compaction complete: "
            f"{original_message_count} -> {compacted_message_count} messages "
            f"({original_message_count - compacted_message_count} removed)"
        )

        return HookResult.with_modifications(messages=compacted_messages)

    def _sync_strategy_global_storage(self, agent_state: Any | None) -> None:
        """Update the compaction strategy's global_storage from agent_state.

        RFC-0009: 在运行时同步 global_storage 到 compaction strategy,
        以支持 LLMCaller 的 Langfuse 追踪。
        """
        gs = self._global_storage
        if gs is None and agent_state is not None:
            gs = getattr(agent_state, "global_storage", None)  # noqa: B009
        if gs is None:
            return

        # 1. 通过 SlidingWindowCompaction 的公开 API 同步
        strategy = self.compaction_strategy
        configure_fn = getattr(strategy, "configure_llm_runtime", None)  # noqa: B009 — duck-typing
        if callable(configure_fn):
            # configure_llm_runtime 会设置 _global_storage 并刷新 LLMCaller
            base_cfg = getattr(strategy, "_base_llm_config", None)  # noqa: B009
            base_client = getattr(strategy, "_base_openai_client", None)  # noqa: B009
            session_id = getattr(strategy, "_session_id", None)  # noqa: B009
            if base_cfg is not None:
                configure_fn(base_cfg, base_client, session_id=session_id, global_storage=gs)
            else:
                # Strategy not yet configured with an LLM config — just store gs directly
                object.__setattr__(strategy, "_global_storage", gs)

    def _compact_messages(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Compact messages using the configured strategy."""
        return self.compaction_strategy.compact(messages)
