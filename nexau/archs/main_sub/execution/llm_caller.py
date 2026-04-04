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

"""LLM request assembly and provider-boundary adapters.

RFC-0006: structured tool calling 的 provider 延迟适配

本模块在真正发送请求前，根据 ``llm_config.api_type`` 把 neutral structured
tool definitions 适配成 OpenAI / Anthropic / Gemini 所需的 provider schema。
"""

import asyncio
import contextvars
import functools
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast

import httpx
import openai
import requests
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.token_trace_session import TokenTraceSession
from nexau.archs.tool.tool import (
    StructuredToolDefinitionLike,
    normalize_structured_tool_definition,
    structured_tool_definition_to_anthropic,
    structured_tool_definition_to_openai,
)
from nexau.archs.tracer.context import TraceContext, get_current_span
from nexau.archs.tracer.core import BaseTracer, SpanType
from nexau.core.messages import Message, Role, ToolResultBlock, ToolUseBlock
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload
from nexau.core.serializers.openai_responses import (
    normalize_openai_responses_api_tools,
    prepare_openai_responses_api_input,
)

from ..agent_state import AgentState
from ..tool_call_modes import (
    STRUCTURED_TOOL_CALL_MODES,
    StructuredProviderTarget,
    normalize_tool_call_mode,
    resolve_structured_provider_target,
)
from .hooks import MiddlewareManager, ModelCallParams
from .model_response import ModelResponse
from .stop_reason import AgentStopReason

logger = logging.getLogger(__name__)

_MISSING_TOOL_RESULT_CONTENT = "no tool result (canceled, compacted or failed)"

# 流式 idle 超时异常，区别于一般网络/API 错误，让重试逻辑可针对性处理
OnRetryCallback = Callable[[int, int, float, str], None]
"""(attempt, max_attempts, backoff_seconds, error_message) → None"""


class StreamIdleTimeoutError(Exception):
    """Raised when no stream chunk is received within the configured idle timeout.

    与 Codex 的 "idle timeout waiting for SSE/websocket" 语义对齐。
    是每帧（per-chunk）超时，不是整个请求的总超时，因此 LLM 长推理
    不会因总时间长而超时，只会在"长时间没有任何数据返回"时才触发。
    """


def _get_stream_idle_timeout_seconds(llm_config: LLMConfig | None) -> float:
    """Return resolved stream idle timeout in seconds."""
    if llm_config is None:
        return LLMConfig.DEFAULT_STREAM_IDLE_TIMEOUT_MS / 1000.0
    return llm_config.get_stream_idle_timeout()


def _is_stream_timeout_exception(exc: Exception) -> bool:
    """Best-effort detection for provider/transport stream read timeouts."""
    if isinstance(exc, (httpx.ReadTimeout, requests.exceptions.ReadTimeout, TimeoutError)):
        return True
    return exc.__class__.__name__ in {"ReadTimeout", "APITimeoutError"}


def _maybe_wrap_stream_idle_timeout(
    exc: Exception,
    *,
    transport_name: str,
    llm_config: LLMConfig | None,
) -> StreamIdleTimeoutError | None:
    """Normalize SDK/provider timeout exceptions into StreamIdleTimeoutError."""
    if not _is_stream_timeout_exception(exc):
        return None
    idle_timeout_seconds = _get_stream_idle_timeout_seconds(llm_config)
    return StreamIdleTimeoutError(
        f"idle timeout waiting for {transport_name} ({idle_timeout_seconds}s): {exc}",
    )


def _normalize_token_ids(value: object, *, context: str) -> list[int]:
    """Normalize token ids to a list of ints."""
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of integers")

    token_ids: list[int] = []
    for item in cast(list[object], value):
        try:
            token_ids.append(int(cast(int | str, item)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} contains a non-integer token: {item!r}") from exc
    return token_ids


def _compact_scalar(value: Any, *, max_chars: int = 256) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated {len(text) - max_chars} chars)"


def _raw_message_metadata(payload: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"raw_type": type(payload).__name__}
    if payload is None:
        return metadata

    safe_fields = ("id", "model", "object", "role", "finish_reason", "stop_reason", "status")

    if isinstance(payload, Mapping):
        payload_mapping = cast(Mapping[str, Any], payload)
        for field in safe_fields:
            value = payload_mapping.get(field)
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[field] = _compact_scalar(value)
        choices_raw = payload_mapping.get("choices")
        if isinstance(choices_raw, list):
            choices_list = cast(list[Any], choices_raw)
            metadata["choices_count"] = len(choices_list)
            if choices_list and isinstance(choices_list[0], Mapping):
                first_choice_mapping = cast(Mapping[str, Any], choices_list[0])
                choice_finish_reason = first_choice_mapping.get("finish_reason")
                if choice_finish_reason is not None:
                    metadata["choice_finish_reason"] = _compact_scalar(choice_finish_reason)
        return metadata

    for field in safe_fields:
        value = getattr(payload, field, None)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = _compact_scalar(value)

    choices_raw = getattr(payload, "choices", None)
    if isinstance(choices_raw, list):
        choices_list = cast(list[Any], choices_raw)
        metadata["choices_count"] = len(choices_list)
        if choices_list:
            first_choice = choices_list[0]
            if isinstance(first_choice, Mapping):
                choice_finish_reason = cast(Mapping[str, Any], first_choice).get("finish_reason")
            else:
                choice_finish_reason = getattr(first_choice, "finish_reason", None)
            if choice_finish_reason is not None:
                metadata["choice_finish_reason"] = _compact_scalar(choice_finish_reason)

    return metadata


def _ensure_tool_results(messages: list[Message]) -> list[Message]:
    """Ensure every ToolUseBlock has a matching ToolResultBlock.

    When a tool execution is interrupted (canceled, compacted, or failed),
    the next LLM call will error with "No tool output found for function call".
    This function detects orphaned tool calls and injects synthetic tool result
    messages so the conversation remains valid.
    """
    # 1. 收集所有 tool_use id、所在消息索引、以及工具名称
    tool_use_ids: dict[str, int] = {}
    tool_use_names: dict[str, str] = {}
    for idx, msg in enumerate(messages):
        if msg.role == Role.ASSISTANT:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    tool_use_ids[block.id] = idx
                    tool_use_names[block.id] = block.name

    if not tool_use_ids:
        return messages

    # 2. 收集所有已有的 tool_result tool_use_id，同时支持前缀匹配
    #    （框架的工具执行系统可能在 tool_call_id 后追加 UUID，如 "tool_call" -> "tool_call_a45a..."）
    matched_tool_use_ids: set[str] = set()
    for msg in messages:
        if msg.role == Role.TOOL:
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    result_id = block.tool_use_id
                    # 精确匹配
                    if result_id in tool_use_ids:
                        matched_tool_use_ids.add(result_id)
                    else:
                        # 前缀匹配：tool_use_id 以某个 ToolUseBlock.id 开头
                        for use_id in tool_use_ids:
                            if result_id.startswith(use_id):
                                matched_tool_use_ids.add(use_id)
                                break

    # 3. 找出缺失的 tool_use_id
    missing_ids = set(tool_use_ids.keys()) - matched_tool_use_ids
    if not missing_ids:
        return messages

    logger.warning(
        "🔧 Found %d tool call(s) without results, injecting synthetic tool results: %s",
        len(missing_ids),
        missing_ids,
    )

    # 4. 按所属 assistant 消息索引分组，以便在正确位置插入
    missing_by_index: dict[int, list[str]] = {}
    for tid in missing_ids:
        assistant_idx = tool_use_ids[tid]
        missing_by_index.setdefault(assistant_idx, []).append(tid)

    # 5. 构建新消息列表，在每个有缺失结果的 assistant 消息后插入合成结果
    result: list[Message] = []
    for idx, msg in enumerate(messages):
        result.append(msg)
        if idx in missing_by_index:
            for tid in missing_by_index[idx]:
                result.append(
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(
                                tool_use_id=tid,
                                content=_MISSING_TOOL_RESULT_CONTENT,
                                is_error=True,
                            )
                        ],
                        # 保存工具名称，供 Gemini REST 转换使用（functionResponse 需要 name）
                        metadata={"tool_name": tool_use_names.get(tid, "")},
                    )
                )

    return result


class LLMCaller:
    """Handles LLM API calls with retry logic."""

    def __init__(
        self,
        openai_client: Any,
        llm_config: LLMConfig,
        retry_attempts: int = 5,
        *,
        retry_backoff_max_seconds: int = 30,
        on_retry: OnRetryCallback | None = None,
        middleware_manager: MiddlewareManager | None = None,
        global_storage: Any = None,
        session_id: str | None = None,
        async_openai_client: Any | None = None,
    ):
        """Initialize LLM caller.

        Args:
            openai_client: OpenAI/Anthropic sync client instance
            llm_config: LLM configuration
            retry_attempts: Number of retry attempts for API calls
            retry_backoff_max_seconds: 指数退避上限（秒），默认 30s。
                避免长时间占用线程池 slot 或让用户等待过久。
            on_retry: 每次重试前的回调，签名见 ``OnRetryCallback``。
                用于 UI 展示「正在重试第 N 次…」等信息。
            middleware_manager: Optional middleware manager for wrapping calls
            global_storage: Optional global storage to retrieve tracer at call time
            session_id: Optional session ID injected into provider payloads
                (OpenAI ``user``, Anthropic ``metadata.user_id``; Gemini skipped)
            async_openai_client: AsyncOpenAI/AsyncAnthropic client for native async calls.
                When provided, ``_call_with_retry_async`` uses direct ``await``
                instead of ``to_thread`` bridging.
        """
        self.openai_client = openai_client
        self.async_openai_client = async_openai_client
        self.llm_config = llm_config
        self.retry_attempts = retry_attempts
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self.on_retry = on_retry
        self.middleware_manager = middleware_manager
        self.global_storage = global_storage
        self.session_id = session_id

        # RFC-0001: 专用线程池执行 sync LLM SDK 调用，避免使用 event loop 的 default executor
        # force stop 时可通过 shutdown(wait=False) 立即释放，不阻塞 event loop 关闭
        self._llm_thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="llm-call",
        )

    def _get_tracer(self) -> BaseTracer | None:
        """Get tracer from global storage at call time."""
        if self.global_storage is not None:
            return self.global_storage.get("tracer")
        return None

    def call_llm(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        force_stop_reason: AgentStopReason | None = None,
        agent_state: AgentState | None = None,
        tool_call_mode: str = "xml",
        tools: Sequence[StructuredToolDefinitionLike] | None = None,
        openai_client: Any | None = None,
        shutdown_event: threading.Event | None = None,
        token_trace_session: TokenTraceSession | None = None,
    ) -> ModelResponse | None:
        """Call LLM with the given messages and return normalized response.

        RFC-0006: structured tool calling 的 provider 延迟适配

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens for the response
            tool_call_mode: Tool calling strategy ('xml' or 'structured')
            tools: Optional neutral structured tool definitions

        Returns:
            A normalized ModelResponse object containing content and tool calls

        Raises:
            RuntimeError: If OpenAI client is not available or API call fails
        """
        runtime_client = openai_client if openai_client is not None else self.openai_client

        if not runtime_client and not self.middleware_manager and self.llm_config.api_type != "gemini_rest":
            raise RuntimeError(
                "OpenAI client is not available. Please check your API configuration.",
            )

        normalized_mode = normalize_tool_call_mode(tool_call_mode)
        use_structured_tools = normalized_mode in STRUCTURED_TOOL_CALL_MODES
        structured_provider_target: StructuredProviderTarget | None = None
        adapted_tools: list[Mapping[str, object]] | None = None
        if use_structured_tools:
            # 1. RFC-0006: structured provider 目标完全由 api_type 决定。
            structured_provider_target = resolve_structured_provider_target(self.llm_config.api_type)

            # 2. 仅在真正组装请求体时，把 neutral definitions 延迟转换为 provider schema。
            adapted_tools = _adapt_structured_tools_for_provider(tools, structured_provider_target)

        # Prepare API parameters
        api_params = self.llm_config.to_openai_params()

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        if adapted_tools and structured_provider_target == "anthropic":
            api_params["tools"] = adapted_tools
            api_params.setdefault("tool_choice", {"type": "auto"})

        if adapted_tools and structured_provider_target == "openai":
            api_params["tools"] = adapted_tools
            api_params.setdefault("tool_choice", "auto")

        if adapted_tools and structured_provider_target == "gemini":
            api_params["tools"] = adapted_tools

        # Add XML stop sequences to prevent malformed XML
        if not use_structured_tools:
            xml_stop_sequences = [
                "</tool_use>",
                "</use_parallel_tool_calls>",
                "</use_batch_agent>",
            ]

            # Merge with existing stop sequences if any
            existing_stop = api_params.get("stop", [])
            if isinstance(existing_stop, str):
                existing_stop = [existing_stop]
            elif existing_stop is None:
                existing_stop = []

            api_params["stop"] = existing_stop + xml_stop_sequences

        # Drop any params the config marks as incompatible
        dropper = getattr(self.llm_config, "apply_param_drops", None)
        if callable(dropper):
            api_params = cast(dict[str, Any], dropper(api_params))

        # Debug logging for LLM messages
        if self.llm_config.debug:
            logger.info("🐛 [DEBUG] LLM Request Messages:")
            for i, msg in enumerate(messages):
                logger.info(
                    f"🐛 [DEBUG] Message {i}: {msg.role.value} -> {msg.get_text_content()}",
                )

        logger.info(f"🧠 Calling LLM with {max_tokens} max tokens...")

        # Ensure all tool calls have corresponding tool results to avoid
        # "No tool output found for function call" errors after interruptions.
        messages = _ensure_tool_results(messages)

        model_call_params = ModelCallParams(
            messages=messages,
            max_tokens=max_tokens,
            force_stop_reason=force_stop_reason,
            agent_state=agent_state,
            tool_call_mode=tool_call_mode,
            tools=tools,
            api_params=api_params,
            openai_client=runtime_client,
            llm_config=self.llm_config,
            retry_attempts=self.retry_attempts,
            shutdown_event=shutdown_event,
            token_trace_session=token_trace_session,
        )

        def base_call(params: ModelCallParams) -> ModelResponse | None:
            return self._call_with_retry(params)

        if self.middleware_manager:
            response_payload = self.middleware_manager.wrap_model_call(model_call_params, base_call)
        else:
            response_payload = base_call(model_call_params)
        if response_payload is None:
            return None

        model_response = response_payload

        if model_response.content:
            from ..utils.xml_utils import XMLUtils

            model_response.content = XMLUtils.restore_closing_tags(model_response.content)

        # Debug logging for LLM response
        if self.llm_config.debug:
            logger.info(f"🐛 [DEBUG] LLM Response: {model_response.render_text()}")

        return model_response

    def _call_once_sync(
        self,
        params: ModelCallParams,
    ) -> ModelResponse | None:
        """Execute a single LLM call attempt (no retry loop).

        P2 async/sync 技术债修复: _call_with_retry_async 的单次调用桥接

        从 _call_with_retry 提取的核心调用逻辑，不含重试循环。
        用于 _call_with_retry_async 的 asyncio.to_thread 桥接，
        外层的 async retry 循环负责重试和退避。

        包含: 参数准备、session_id 注入、message 转换、tracer 传递。
        """
        from .executor import AgentStopReason

        force_stop_reason = params.force_stop_reason
        if force_stop_reason and force_stop_reason != AgentStopReason.SUCCESS:
            return None

        kwargs = dict(params.api_params)

        # session_id → provider-specific user tracking field
        if self.session_id is not None:
            if self.llm_config.api_type == "openai_chat_completion":
                kwargs.setdefault("user", self.session_id)
            elif self.llm_config.api_type == "openai_responses":
                kwargs.setdefault("safety_identifier", self.session_id)
                kwargs.setdefault("prompt_cache_key", self.session_id)
            elif self.llm_config.api_type == "anthropic_chat_completion":
                existing_metadata: dict[str, str] = kwargs.get("metadata") or {}
                existing_metadata.setdefault("user_id", self.session_id)
                kwargs["metadata"] = existing_metadata

        tool_image_policy: Literal["inject_user_message", "embed_in_tool_message"] = "inject_user_message"
        if self.llm_config.api_type in {"openai_chat_completion", "openai_responses"}:
            tool_image_policy = "embed_in_tool_message" if self.llm_config.api_type == "openai_responses" else "inject_user_message"
        if self.llm_config.api_type != "generate_with_token":
            kwargs["messages"] = serialize_ump_to_openai_chat_payload(params.messages, tool_image_policy=tool_image_policy)

        client = params.openai_client if params.openai_client is not None else self.openai_client
        response_content = call_llm_with_different_client(
            client,
            self.llm_config,
            kwargs,
            middleware_manager=self.middleware_manager,
            model_call_params=params,
            tracer=self._get_tracer(),
        )

        stop = kwargs.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        if stop and response_content.content:
            for s in stop:
                response_content.content = response_content.content.split(s)[0]

        if response_content.has_content() or response_content.has_tool_calls():
            return response_content
        else:
            raw_message_meta = _raw_message_metadata(response_content.raw_message)
            finish_reason = raw_message_meta.get("finish_reason") or raw_message_meta.get("choice_finish_reason")
            logger.error(
                "❌ Empty model response received: finish_reason=%s, role=%s, content_len=%d, tool_calls=%d, usage=%s",
                finish_reason if finish_reason is not None else "unknown",
                response_content.role,
                len(response_content.content or ""),
                len(response_content.tool_calls),
                response_content.usage.to_dict(),
            )
            raise RuntimeError("No response content or tool calls from LLM")

    async def _call_once_async(
        self,
        params: ModelCallParams,
    ) -> ModelResponse | None:
        """Execute a single async LLM call attempt (no retry loop).

        async/sync 技术债修复: 使用 AsyncOpenAI / AsyncAnthropic 原生 await，
        彻底消除 to_thread 桥接。force stop 时 asyncio cancellation 直接生效。

        包含: 参数准备、session_id 注入、message 转换、tracer 传递。
        """
        from .executor import AgentStopReason

        force_stop_reason = params.force_stop_reason
        if force_stop_reason and force_stop_reason != AgentStopReason.SUCCESS:
            return None

        kwargs = dict(params.api_params)

        # session_id → provider-specific user tracking field
        if self.session_id is not None:
            if self.llm_config.api_type == "openai_chat_completion":
                kwargs.setdefault("user", self.session_id)
            elif self.llm_config.api_type == "openai_responses":
                kwargs.setdefault("safety_identifier", self.session_id)
                kwargs.setdefault("prompt_cache_key", self.session_id)
            elif self.llm_config.api_type == "anthropic_chat_completion":
                existing_metadata: dict[str, str] = kwargs.get("metadata") or {}
                existing_metadata.setdefault("user_id", self.session_id)
                kwargs["metadata"] = existing_metadata

        tool_image_policy: Literal["inject_user_message", "embed_in_tool_message"] = "inject_user_message"
        if self.llm_config.api_type in {"openai_chat_completion", "openai_responses"}:
            tool_image_policy = "embed_in_tool_message" if self.llm_config.api_type == "openai_responses" else "inject_user_message"
        if self.llm_config.api_type != "generate_with_token":
            kwargs["messages"] = serialize_ump_to_openai_chat_payload(params.messages, tool_image_policy=tool_image_policy)

        async_client = self.async_openai_client
        response_content = await call_llm_with_different_client_async(
            async_client,
            self.llm_config,
            kwargs,
            middleware_manager=self.middleware_manager,
            model_call_params=params,
            tracer=self._get_tracer(),
        )

        stop = kwargs.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        if stop and response_content.content:
            for s in stop:
                response_content.content = response_content.content.split(s)[0]

        if response_content.has_content() or response_content.has_tool_calls():
            return response_content
        else:
            raw_message_meta = _raw_message_metadata(response_content.raw_message)
            finish_reason = raw_message_meta.get("finish_reason") or raw_message_meta.get("choice_finish_reason")
            logger.error(
                "❌ Empty model response received: finish_reason=%s, role=%s, content_len=%d, tool_calls=%d, usage=%s",
                finish_reason if finish_reason is not None else "unknown",
                response_content.role,
                len(response_content.content or ""),
                len(response_content.tool_calls),
                response_content.usage.to_dict(),
            )
            raise RuntimeError("No response content or tool calls from LLM")

    def _call_with_retry(
        self,
        params: ModelCallParams,
    ) -> ModelResponse | None:
        """Call OpenAI client with exponential backoff retry."""
        from .executor import AgentStopReason

        force_stop_reason = params.force_stop_reason

        if force_stop_reason and force_stop_reason != AgentStopReason.SUCCESS:
            reason_name = getattr(force_stop_reason, "name", str(force_stop_reason))
            logger.info(
                f"🛑 LLM call forced to stop due to {reason_name}",
            )
            return None

        backoff = 1
        for i in range(self.retry_attempts):
            try:
                if force_stop_reason and force_stop_reason != AgentStopReason.SUCCESS:
                    return None
                kwargs = dict(params.api_params)

                # session_id → provider-specific user tracking field
                if self.session_id is not None:
                    if self.llm_config.api_type == "openai_chat_completion":
                        kwargs.setdefault("user", self.session_id)
                    elif self.llm_config.api_type == "openai_responses":
                        # 'user' is deprecated and some backends actively reject it.
                        # Only use its official replacements: safety_identifier + prompt_cache_key.
                        kwargs.setdefault("safety_identifier", self.session_id)
                        kwargs.setdefault("prompt_cache_key", self.session_id)
                    elif self.llm_config.api_type == "anthropic_chat_completion":
                        existing_metadata: dict[str, str] = kwargs.get("metadata") or {}
                        existing_metadata.setdefault("user_id", self.session_id)
                        kwargs["metadata"] = existing_metadata

                logger.debug(
                    "🔍 [HISTORY-DEBUG] LLM call: %d Message objects, roles=%s",
                    len(params.messages),
                    [m.role.value for m in params.messages],
                )
                tool_image_policy: Literal["inject_user_message", "embed_in_tool_message"] = "inject_user_message"
                if self.llm_config.api_type in {"openai_chat_completion", "openai_responses"}:
                    tool_image_policy = "embed_in_tool_message" if self.llm_config.api_type == "openai_responses" else "inject_user_message"
                if self.llm_config.api_type != "generate_with_token":
                    kwargs["messages"] = serialize_ump_to_openai_chat_payload(params.messages, tool_image_policy=tool_image_policy)
                    logger.debug(
                        "🔍 [HISTORY-DEBUG] After legacy conversion: %d dicts, roles=%s",
                        len(kwargs["messages"]),
                        [m.get("role") for m in kwargs["messages"]],
                    )
                client = params.openai_client if params.openai_client is not None else self.openai_client
                response_content = call_llm_with_different_client(
                    client,
                    self.llm_config,
                    kwargs,
                    middleware_manager=self.middleware_manager,
                    model_call_params=params,
                    tracer=self._get_tracer(),
                )

                stop = kwargs.get("stop", [])
                if isinstance(stop, str):
                    stop = [stop]
                if stop and response_content.content:
                    for s in stop:
                        response_content.content = response_content.content.split(s)[0]

                if response_content.has_content() or response_content.has_tool_calls():
                    return response_content
                else:
                    raw_message_meta = _raw_message_metadata(response_content.raw_message)
                    finish_reason = raw_message_meta.get("finish_reason") or raw_message_meta.get("choice_finish_reason")
                    logger.error(
                        "❌ Empty model response received: finish_reason=%s, role=%s, content_len=%d, tool_calls=%d, usage=%s",
                        finish_reason if finish_reason is not None else "unknown",
                        response_content.role,
                        len(response_content.content or ""),
                        len(response_content.tool_calls),
                        response_content.usage.to_dict(),
                    )
                    logger.error("❌ Empty model raw_message_meta=%s", raw_message_meta)
                    raise Exception("No response content or tool calls")

            except Exception as e:
                # RFC-0001: shutdown_event 已设置时不重试，直接返回 None
                # 让 execute() 在下一次迭代边界检测 stop_signal
                if params.shutdown_event and params.shutdown_event.is_set():
                    logger.info("🛑 LLM call interrupted by shutdown_event, skipping retry")
                    return None

                logger.error(
                    f"❌ LLM call failed (attempt {i + 1}/{self.retry_attempts}): {e}",
                    exc_info=True,
                )
                if i == self.retry_attempts - 1:
                    raise e
                capped_backoff = min(backoff, self.retry_backoff_max_seconds)
                if self.on_retry is not None:
                    self.on_retry(i + 1, self.retry_attempts, capped_backoff, str(e))
                time.sleep(capped_backoff)
                backoff = min(backoff * 2, self.retry_backoff_max_seconds)
        return None

    async def call_llm_async(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        force_stop_reason: AgentStopReason | None = None,
        agent_state: AgentState | None = None,
        tool_call_mode: str = "xml",
        tools: Sequence[StructuredToolDefinitionLike] | None = None,
        openai_client: Any | None = None,
        shutdown_event: threading.Event | None = None,
        token_trace_session: TokenTraceSession | None = None,
    ) -> ModelResponse | None:
        """Async version of call_llm.

        P2 async/sync 技术债修复: 异步 LLM 调用路径

        使用 asyncio.sleep 替代 time.sleep 做退避重试，
        Gemini REST 使用 httpx.AsyncClient 做真正的异步 HTTP 调用，
        其他 provider (OpenAI/Anthropic) 通过 asyncio.to_thread 桥接 sync SDK。
        """
        # 参数准备逻辑与 call_llm 相同
        runtime_client = openai_client if openai_client is not None else self.openai_client

        if not runtime_client and not self.middleware_manager and self.llm_config.api_type != "gemini_rest":
            raise RuntimeError(
                "OpenAI client is not available. Please check your API configuration.",
            )

        normalized_mode = normalize_tool_call_mode(tool_call_mode)
        use_structured_tools = normalized_mode in STRUCTURED_TOOL_CALL_MODES
        adapted_tools: list[Mapping[str, object]] | None = None
        if use_structured_tools:
            structured_provider_target = resolve_structured_provider_target(self.llm_config.api_type)
            adapted_tools = _adapt_structured_tools_for_provider(tools, structured_provider_target)

        api_params = self.llm_config.to_openai_params()
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        if use_structured_tools:
            sp_target = resolve_structured_provider_target(self.llm_config.api_type)
            if adapted_tools and sp_target == "anthropic":
                api_params["tools"] = adapted_tools
                api_params.setdefault("tool_choice", {"type": "auto"})
            elif adapted_tools and sp_target == "openai":
                api_params["tools"] = adapted_tools
                api_params.setdefault("tool_choice", "auto")
            elif adapted_tools and sp_target == "gemini":
                api_params["tools"] = adapted_tools

        if not use_structured_tools:
            xml_stop_sequences = ["</tool_use>", "</use_parallel_tool_calls>", "</use_batch_agent>"]
            existing_stop = api_params.get("stop", [])
            if isinstance(existing_stop, str):
                existing_stop = [existing_stop]
            elif existing_stop is None:
                existing_stop = []
            api_params["stop"] = existing_stop + xml_stop_sequences

        dropper = getattr(self.llm_config, "apply_param_drops", None)
        if callable(dropper):
            api_params = cast(dict[str, Any], dropper(api_params))

        messages = _ensure_tool_results(messages)

        model_call_params = ModelCallParams(
            messages=messages,
            max_tokens=max_tokens,
            force_stop_reason=force_stop_reason,
            agent_state=agent_state,
            tool_call_mode=tool_call_mode,
            tools=tools,
            api_params=api_params,
            openai_client=runtime_client,
            llm_config=self.llm_config,
            retry_attempts=self.retry_attempts,
            shutdown_event=shutdown_event,
            token_trace_session=token_trace_session,
        )

        # 使用 async retry wrapper
        response_payload: ModelResponse | None
        if self.middleware_manager:
            # Middleware wrapping 仍走 sync (在 to_thread 中)
            # 使用 _call_once_sync 而非 _call_with_retry，避免嵌套重试
            def _wrapped(params: ModelCallParams) -> ModelResponse | None:
                return self.middleware_manager.wrap_model_call(params, lambda p: self._call_once_sync(p))  # type: ignore[union-attr]

            response_payload = await self._call_with_retry_async(model_call_params, _wrapped)
        else:
            response_payload = await self._call_with_retry_async(model_call_params)

        if response_payload is None:
            return None

        if response_payload.content:
            from ..utils.xml_utils import XMLUtils

            response_payload.content = XMLUtils.restore_closing_tags(response_payload.content)
        return response_payload

    async def _call_with_retry_async(
        self,
        params: ModelCallParams,
        sync_call_fn: Any | None = None,
    ) -> ModelResponse | None:
        """Async retry wrapper with asyncio.sleep for backoff.

        async/sync 技术债修复: 原生 async SDK 调用 + asyncio.sleep 退避

        - Gemini REST: httpx.AsyncClient 原生异步
        - OpenAI / Anthropic (无 middleware): AsyncOpenAI / AsyncAnthropic
          原生 await，asyncio cancellation 直接生效
        - Middleware-wrapped: sync hook 需通过 _llm_thread_pool 桥接，
          cleanup() 可 shutdown(wait=False) 释放

        重要: 对非 Gemini provider，直接桥接 _call_once_sync() (单次调用)，
        而不是 _call_with_retry() (自带重试循环)，避免 retry_attempts² 的
        嵌套重试和重复 tracing span。
        """
        from .executor import AgentStopReason

        force_stop_reason = params.force_stop_reason
        if force_stop_reason and force_stop_reason != AgentStopReason.SUCCESS:
            return None

        backoff = 1
        for i in range(self.retry_attempts):
            try:
                if params.shutdown_event and params.shutdown_event.is_set():
                    logger.info("🛑 LLM call interrupted by shutdown_event (async), skipping retry")
                    return None

                if sync_call_fn is not None:
                    # Middleware-wrapped path: sync hook 需线程桥接
                    # 使用专用 _llm_thread_pool 而非 default executor，
                    # 避免 force stop 时阻塞 event loop shutdown
                    return await self._run_sync_in_llm_pool(sync_call_fn, params)

                # Gemini REST: native async path (httpx.AsyncClient)
                if self.llm_config.api_type == "gemini_rest":
                    kwargs = dict(params.api_params)
                    return await call_llm_with_gemini_rest_async(
                        kwargs,
                        middleware_manager=self.middleware_manager,
                        model_call_params=params,
                        llm_config=self.llm_config,
                        tracer=self._get_tracer(),
                    )

                # OpenAI / Anthropic: 原生 async SDK，直接 await
                # 通过 shutdown_event 监控实现可取消：stop() 设置 event 后
                # 立即 cancel 正在 await 的 LLM 调用
                if self.async_openai_client is not None:
                    return await self._call_once_async_cancellable(params)

                # Fallback: 无 async client 时仍走线程桥接
                return await self._run_sync_in_llm_pool(self._call_once_sync, params)

            except Exception as e:
                if params.shutdown_event and params.shutdown_event.is_set():
                    logger.info("🛑 LLM call interrupted by shutdown_event (async), skipping retry")
                    return None

                logger.error(
                    f"❌ LLM call failed (attempt {i + 1}/{self.retry_attempts}, async): {e}",
                    exc_info=True,
                )
                if i == self.retry_attempts - 1:
                    raise e
                capped_backoff = min(backoff, self.retry_backoff_max_seconds)
                if self.on_retry is not None:
                    self.on_retry(i + 1, self.retry_attempts, capped_backoff, str(e))
                await asyncio.sleep(capped_backoff)
                backoff = min(backoff * 2, self.retry_backoff_max_seconds)
        return None

    async def _run_sync_in_llm_pool(
        self,
        func: Any,
        *args: Any,
    ) -> ModelResponse | None:
        """Run a sync function in the dedicated LLM thread pool.

        RFC-0001: 替代 asyncio.to_thread 以使用专用线程池

        与 asyncio.to_thread 相同语义（含 contextvars 传播），
        但使用 _llm_thread_pool 而非 event loop 的 default executor，
        确保 force stop 时 cleanup() 可通过 shutdown(wait=False) 释放。
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        func_call = functools.partial(ctx.run, func, *args)
        return await loop.run_in_executor(self._llm_thread_pool, func_call)

    def shutdown_thread_pool(self) -> None:
        """Shut down the dedicated LLM thread pool without waiting.

        Called by Executor.cleanup() during force stop to release
        middleware-path worker threads immediately.
        """
        self._llm_thread_pool.shutdown(wait=False, cancel_futures=True)

    async def _call_once_async_cancellable(
        self,
        params: ModelCallParams,
    ) -> ModelResponse | None:
        """Run _call_once_async but cancel if shutdown_event is set.

        async/sync 技术债修复: 让 async LLM 调用可被 graceful stop 取消。
        stop() 设置 shutdown_event 后，此方法 cancel 正在 await 的 HTTP 请求，
        使 execute_async 的主循环能及时退出，asyncio.gather 不再无限等待。
        """
        shutdown_ev = params.shutdown_event
        if shutdown_ev is not None and shutdown_ev.is_set():
            return None

        llm_task = asyncio.ensure_future(self._call_once_async(params))

        if shutdown_ev is None:
            return await llm_task

        # 轮询 shutdown_event（不使用 to_thread 避免 default executor 问题）
        async def _poll_shutdown() -> None:
            while not shutdown_ev.is_set():
                await asyncio.sleep(0.1)

        shutdown_task = asyncio.ensure_future(_poll_shutdown())

        done, pending = await asyncio.wait(
            {llm_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()
            try:
                await p
            except asyncio.CancelledError:
                pass

        if llm_task in done:
            return llm_task.result()

        logger.info("🛑 Async LLM call cancelled by shutdown_event")
        return None


def call_llm_with_different_client(
    client: Any,
    llm_config: LLMConfig,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call LLM with the given messages and return response content."""
    if llm_config.api_type == "anthropic_chat_completion":
        return call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
            cache_control_ttl=llm_config.cache_control_ttl,
        )
    elif llm_config.api_type == "openai_responses":
        return call_llm_with_openai_responses(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    elif llm_config.api_type == "openai_chat_completion":
        return call_llm_with_openai_chat_completion(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    elif llm_config.api_type == "gemini_rest":
        return call_llm_with_gemini_rest(
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    elif llm_config.api_type == "generate_with_token":
        return call_llm_with_generate_with_token(
            client,
            kwargs,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    else:
        raise ValueError(f"Invalid API type: {llm_config.api_type}")


def _safe_int(value: Any) -> int:
    """Best-effort integer coercion for provider usage metadata."""
    try:
        return int(cast(int | str, value))
    except (TypeError, ValueError):
        return 0


def _normalize_generate_with_token_finish_reason(finish_reason: Any) -> str | None:
    """Normalize finish reasons from generate-with-token responses."""
    if finish_reason is None:
        return None

    if isinstance(finish_reason, Mapping):
        finish_reason_mapping = cast(Mapping[str, Any], finish_reason)
        finish_type = finish_reason_mapping.get("type")
        if finish_type is not None:
            return str(finish_type)
        return json.dumps(dict(finish_reason_mapping), ensure_ascii=False)

    return str(finish_reason)


def _get_generate_with_token_meta_info(response_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return `meta_info` mapping from generate-with-token responses when available."""
    meta_info_raw = response_payload.get("meta_info")
    if isinstance(meta_info_raw, Mapping):
        return cast(Mapping[str, Any], meta_info_raw)
    return {}


def _extract_generate_with_token_message(response_payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Extract an assistant message from either OpenAI-like or raw token payloads."""
    choices_payload = response_payload.get("choices")
    if isinstance(choices_payload, list) and choices_payload:
        choices_list = cast(list[object], choices_payload)
        first_choice_raw = choices_list[0]
        if not isinstance(first_choice_raw, Mapping):
            raise ValueError("generate_with_token response choice must be a mapping")
        choice_mapping = cast(Mapping[str, Any], first_choice_raw)
        choice_message_payload = choice_mapping.get("message")
        if not isinstance(choice_message_payload, Mapping):
            raise ValueError("generate_with_token response missing assistant message")
        return dict(cast(Mapping[str, Any], choice_message_payload)), choice_mapping.get("finish_reason")

    fallback_message_payload: dict[str, Any] = {
        "role": "assistant",
        "content": response_payload.get("text"),
    }
    tool_calls_raw = response_payload.get("tool_calls")
    if isinstance(tool_calls_raw, list):
        fallback_message_payload["tool_calls"] = list(cast(list[object], tool_calls_raw))

    finish_reason_raw = response_payload.get("finish_reason")
    if finish_reason_raw is None:
        finish_reason_raw = _get_generate_with_token_meta_info(response_payload).get("finish_reason")
    return fallback_message_payload, finish_reason_raw


def _extract_generate_with_token_nexrl_train(
    response_payload: dict[str, Any],
    *,
    request_tokens: list[int],
) -> dict[str, Any]:
    """Extract or synthesize NexRL train metadata from generate-with-token responses."""
    nexrl_train_payload = response_payload.get("nexrl_train")
    if isinstance(nexrl_train_payload, Mapping):
        return dict(cast(Mapping[str, Any], nexrl_train_payload))

    output_token_ids = _normalize_token_ids(
        response_payload.get("output_token_ids", response_payload.get("output_ids", [])),
        context="generate_with_token output token ids",
    )
    meta_info = _get_generate_with_token_meta_info(response_payload)
    response_logprobs: list[float] = []
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if isinstance(output_token_logprobs, list):
        for entry in cast(list[object], output_token_logprobs):
            if isinstance(entry, (list, tuple)) and entry:
                entry_items = cast(Sequence[object], entry)
                first_logprob = entry_items[0]
                if isinstance(first_logprob, (int, float)):
                    response_logprobs.append(float(first_logprob))
            elif isinstance(entry, (int, float)):
                response_logprobs.append(float(entry))

    return {
        "prompt_tokens": list(request_tokens),
        "response_tokens": output_token_ids,
        "response_logprobs": response_logprobs,
    }


def _normalize_generate_with_token_usage(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize token usage from generate-with-token responses."""
    usage_payload = response_payload.get("usage")
    if isinstance(usage_payload, Mapping):
        usage_dict: dict[str, Any] = dict(cast(Mapping[str, Any], usage_payload))
    else:
        usage_dict = {}

    meta_info = dict(_get_generate_with_token_meta_info(response_payload))
    nexrl_train_payload = response_payload.get("nexrl_train")
    nexrl_train = dict(cast(Mapping[str, Any], nexrl_train_payload)) if isinstance(nexrl_train_payload, Mapping) else {}

    prompt_tokens = _safe_int(
        usage_dict.get(
            "prompt_tokens",
            usage_dict.get(
                "input_tokens",
                meta_info.get("prompt_tokens", len(cast(list[Any], nexrl_train.get("prompt_tokens", [])))),
            ),
        )
    )
    completion_tokens = _safe_int(
        usage_dict.get(
            "completion_tokens",
            usage_dict.get(
                "output_tokens",
                meta_info.get("completion_tokens", len(cast(list[Any], nexrl_train.get("response_tokens", [])))),
            ),
        )
    )

    if prompt_tokens > 0 or "prompt_tokens" in usage_dict or "input_tokens" in usage_dict or "prompt_tokens" in nexrl_train:
        usage_dict.setdefault("prompt_tokens", prompt_tokens)
        usage_dict.setdefault("input_tokens", prompt_tokens)

    if completion_tokens > 0 or "completion_tokens" in usage_dict or "output_tokens" in usage_dict or "response_tokens" in nexrl_train:
        usage_dict.setdefault("completion_tokens", completion_tokens)

    if "total_tokens" not in usage_dict and (prompt_tokens or completion_tokens):
        usage_dict["total_tokens"] = prompt_tokens + completion_tokens

    cached_tokens = _safe_int(usage_dict.get("cached_tokens", meta_info.get("cached_tokens")))
    if cached_tokens > 0 or "cached_tokens" in usage_dict or "cached_tokens" in meta_info:
        usage_dict.setdefault("cached_tokens", cached_tokens)

    finish_reason_raw = response_payload.get("finish_reason", meta_info.get("finish_reason"))
    if finish_reason_raw is None:
        choices_payload = response_payload.get("choices")
        if isinstance(choices_payload, list) and choices_payload:
            choices_list = cast(list[object], choices_payload)
            first_choice_raw = choices_list[0]
            if isinstance(first_choice_raw, Mapping):
                finish_reason_raw = cast(Mapping[str, Any], first_choice_raw).get("finish_reason")
    finish_reason = _normalize_generate_with_token_finish_reason(finish_reason_raw)
    if finish_reason is not None:
        usage_dict["finish_reason"] = finish_reason
        if isinstance(finish_reason_raw, Mapping):
            usage_dict["finish_reason_details"] = dict(cast(Mapping[str, Any], finish_reason_raw))

    return usage_dict


def call_llm_with_generate_with_token(
    client: Any,
    kwargs: dict[str, Any],
    *,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call a client-backed token generate API and normalize the response."""
    if llm_config is None:
        raise ValueError("llm_config is required for generate_with_token call")
    if model_call_params is None or model_call_params.token_trace_session is None:
        raise ValueError("token_trace_session is required for generate_with_token call")
    if client is None:
        raise ValueError("client is required for generate_with_token call")

    token_trace_session = model_call_params.token_trace_session
    token_trace_session.sync_external_messages(model_call_params.messages)

    request_tokens = list(token_trace_session.token_ids)
    request_payload = kwargs.copy()
    request_payload.pop("messages", None)
    stream_requested = bool(request_payload.pop("stream", False) or getattr(llm_config, "stream", False))
    if stream_requested:
        logger.warning("Streaming is not supported for generate_with_token; falling back to non-stream mode")

    def _invoke_generate() -> dict[str, Any]:
        request_kwargs = token_trace_session.build_generate_with_token_kwargs(
            max_output_tokens=cast(int | None, request_payload.pop("max_tokens", None)),
            request_params=request_payload,
        )
        if model_call_params.tools is not None:
            request_kwargs["tools"] = [
                structured_tool_definition_to_openai(tool) for tool in cast(list[dict[str, Any]], model_call_params.tools)
            ]
        response_payload = client.generate_with_token(**request_kwargs)
        if not isinstance(response_payload, Mapping):
            raise ValueError("generate_with_token response must be a mapping")
        return dict(cast(Mapping[str, Any], response_payload))

    if tracer is not None and get_current_span() is not None:
        trace_ctx = TraceContext(
            tracer,
            "generate_with_token",
            SpanType.LLM,
            inputs={
                "model": llm_config.model,
                "input_token_count": len(request_tokens),
            },
        )
        with trace_ctx:
            response_payload = _invoke_generate()
            trace_ctx.set_outputs(response_payload)
    else:
        response_payload = _invoke_generate()

    message_payload, finish_reason_raw = _extract_generate_with_token_message(response_payload)
    nexrl_train = _extract_generate_with_token_nexrl_train(
        response_payload,
        request_tokens=request_tokens,
    )

    normalized_output_token_ids = _normalize_token_ids(
        nexrl_train.get("response_tokens", []),
        context="generate_with_token nexrl_train.response_tokens",
    )

    usage_dict = _normalize_generate_with_token_usage(response_payload)
    if "finish_reason" not in usage_dict:
        finish_reason = _normalize_generate_with_token_finish_reason(finish_reason_raw)
        if finish_reason is not None:
            usage_dict["finish_reason"] = finish_reason
            if isinstance(finish_reason_raw, Mapping):
                usage_dict["finish_reason_details"] = dict(cast(Mapping[str, Any], finish_reason_raw))
    model_response = ModelResponse.from_openai_message(
        message_payload,
        usage=usage_dict,
    )
    model_response.raw_message = response_payload
    model_response.output_token_ids = normalized_output_token_ids

    output_text = model_response.content
    if output_text is None and normalized_output_token_ids:
        model_response.content = token_trace_session.detokenize(normalized_output_token_ids)
        output_text = model_response.content

    token_trace_session.record_round(
        request_tokens=request_tokens,
        response_tokens=normalized_output_token_ids,
        response_text=output_text,
        tool_calls=[call.to_openai_dict() for call in model_response.tool_calls],
        usage=usage_dict,
    )

    return model_response


def _adapt_structured_tools_for_provider(
    tools: Sequence[StructuredToolDefinitionLike] | None,
    provider_target: StructuredProviderTarget,
) -> list[Mapping[str, object]] | None:
    """Adapt neutral structured tools for the selected provider.

    RFC-0006: Provider 延迟适配

    输入保持 neutral / compatibility definition，输出在请求边界收敛到目标
    provider 所需 schema；Gemini 路径继续保留 neutral definition 并走原生 adapter。
    """

    if not tools:
        return None

    adapted_tools: list[Mapping[str, object]] = []
    for tool in tools:
        normalized = normalize_structured_tool_definition(tool)
        if provider_target == "openai":
            adapted_tools.append(structured_tool_definition_to_openai(normalized))
        elif provider_target == "anthropic":
            adapted_tools.append(structured_tool_definition_to_anthropic(normalized))
        elif provider_target == "gemini":
            adapted_tools.append(normalized)
        else:  # pragma: no cover - guarded by provider target resolution
            raise ValueError(f"Unsupported structured provider target: {provider_target}")

    return adapted_tools


def _strip_responses_api_artifacts(messages: list[Any]) -> list[Any]:
    """Remove Responses API-only artifacts from generic chat messages."""

    sanitized: list[Any] = []

    for message in messages or []:
        if not isinstance(message, Mapping):
            sanitized.append(message)
            continue

        message_mapping = cast(Mapping[str, Any], message)
        cleaned: dict[str, Any] = dict(message_mapping)
        cleaned.pop("response_items", None)
        cleaned.pop("reasoning", None)
        sanitized.append(cleaned)

    return sanitized


def call_llm_with_anthropic_chat_completion(
    client: Any,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
    cache_control_ttl: str | None = None,
) -> ModelResponse:
    """Call Anthropic chat completion with the given messages and return response content."""
    stream_requested = bool(kwargs.pop("stream", False))

    # Check if tracing is active (there's a current span and we have a tracer)
    should_trace = tracer is not None and get_current_span() is not None

    def _build_cache_control() -> dict[str, str]:
        cc: dict[str, str] = {"type": "ephemeral"}
        if cache_control_ttl:
            cc["ttl"] = cache_control_ttl
        return cc

    def _apply_cache_control(
        system_messages: list[dict[str, Any]],
        user_messages: list[dict[str, Any]],
    ) -> None:
        """Apply Anthropic cache_control to system and user message blocks.

        System blocks carry a ``_cache`` flag from SystemPromptBlock
        configuration; when absent the default is to cache.
        """
        for sys_block in system_messages:
            should_cache = sys_block.pop("_cache", True)
            if should_cache:
                sys_block["cache_control"] = _build_cache_control()

        if user_messages and user_messages[-1].get("content"):
            content = cast(list[dict[str, Any]] | str | None, user_messages[-1].get("content"))
            if isinstance(content, list) and content:
                content[0]["cache_control"] = _build_cache_control()

    def _build_anthropic_messages() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if type(model_call_params) is not ModelCallParams:
            raise ValueError("Anthropic calls require explicit ModelCallParams with UMP messages")

        from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter

        return AnthropicMessagesAdapter().to_vendor_format(model_call_params.messages)

    def llm_call() -> Any:
        # 组装 Anthropic 参数
        system_messages, user_messages = _build_anthropic_messages()
        _apply_cache_control(system_messages, user_messages)

        new_kwargs = kwargs.copy()
        new_kwargs.pop("messages", None)
        new_kwargs.pop("anthropic_cache_control_ttl", None)

        # Build the exact kwargs for tracing
        api_kwargs: dict[str, Any] = {"system": system_messages, "messages": user_messages, **new_kwargs}

        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "Anthropic messages.create", SpanType.LLM, inputs=api_kwargs)
            with trace_ctx:
                resp = client.messages.create(**api_kwargs)
                trace_ctx.set_outputs(_to_serializable_dict(resp))
                return resp
        else:
            resp = client.messages.create(**api_kwargs)
            return resp

    if not stream_requested:
        response = llm_call()
        return ModelResponse.from_anthropic_message(response)

    def llm_stream_call() -> tuple[dict[str, Any], str | None]:
        system_messages, user_messages = _build_anthropic_messages()
        _apply_cache_control(system_messages, user_messages)

        new_kwargs: dict[str, Any] = kwargs.copy()
        new_kwargs.pop("messages", None)
        new_kwargs.pop("anthropic_cache_control_ttl", None)

        # Build the exact kwargs for tracing
        api_kwargs: dict[str, Any] = {"system": system_messages, "messages": user_messages, **new_kwargs}

        aggregator = AnthropicStreamAggregator()

        try:
            if should_trace and tracer is not None:
                trace_ctx = TraceContext(tracer, "Anthropic messages.stream", SpanType.LLM, inputs=api_kwargs)
                with trace_ctx:
                    start_time = time.time()
                    first_token_time = None
                    # RFC-0001: shutdown_event 检测
                    _shutdown_ev = model_call_params.shutdown_event if model_call_params else None
                    with client.messages.create(**api_kwargs, stream=True) as stream:
                        for event in stream:
                            if _shutdown_ev is not None and _shutdown_ev.is_set():
                                logger.info("🛑 Shutdown event detected during Anthropic streaming, finalizing partial response")
                                break
                            if first_token_time is None:
                                first_token_time = time.time()
                            processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                            if processed_event is None:
                                continue
                            aggregator.consume(processed_event)
                    message_payload = aggregator.finalize()
                    trace_ctx.set_outputs(message_payload)
                    if first_token_time is not None:
                        trace_ctx.set_attributes(
                            {
                                "time_to_first_token_ms": (first_token_time - start_time) * 1000,
                            }
                        )
                    return message_payload, aggregator.model_name

            # RFC-0001: shutdown_event 检测
            _shutdown_ev = model_call_params.shutdown_event if model_call_params else None
            with client.messages.create(**api_kwargs, stream=True) as stream:
                for event in stream:
                    if _shutdown_ev is not None and _shutdown_ev.is_set():
                        logger.info("🛑 Shutdown event detected during Anthropic streaming, finalizing partial response")
                        break
                    processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                    if processed_event is None:
                        continue
                    aggregator.consume(processed_event)
            message_payload = aggregator.finalize()
            return message_payload, aggregator.model_name
        except Exception as exc:
            wrapped_error = _maybe_wrap_stream_idle_timeout(
                exc,
                transport_name="anthropic stream",
                llm_config=llm_config,
            )
            if wrapped_error is not None:
                raise wrapped_error from exc
            raise

    response_payload, _ = llm_stream_call()

    return ModelResponse.from_anthropic_message(response_payload)


def call_llm_with_openai_chat_completion(
    client: openai.OpenAI,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call OpenAI chat completion with the given messages and return response content."""

    messages = _strip_responses_api_artifacts(kwargs.get("messages", []))
    # Some providers (eg. AWS Bedrock) reject assistant messages where content is an empty string.
    # Only strip content from assistant messages with tool_calls, where content is optional.
    for msg in messages:
        if isinstance(msg, dict):
            typed_msg = cast(dict[str, object], msg)
            if typed_msg.get("role") == "assistant" and typed_msg.get("content") == "" and typed_msg.get("tool_calls"):
                del typed_msg["content"]
    kwargs["messages"] = messages
    stream_requested = bool(kwargs.pop("stream", False) or getattr(llm_config, "stream", False))

    # Check if tracing is active (there's a current span and we have a tracer)
    should_trace = tracer is not None and get_current_span() is not None

    if stream_requested:

        def call_llm_stream(payload: dict[str, Any]) -> tuple[dict[str, Any], ChatCompletionChunk | None, str | None]:
            payload = payload.copy()
            payload.pop("stream", None)
            stream_options = {"include_usage": True}
            payload["stream_options"] = stream_options
            aggregator = OpenAIChatStreamAggregator()
            last_chunk: ChatCompletionChunk | None = None

            try:
                if should_trace and tracer is not None:
                    trace_ctx: TraceContext = TraceContext(tracer, "OpenAI chat.completions.create (stream)", SpanType.LLM, inputs=payload)
                    with trace_ctx:
                        start_time = time.time()
                        first_token_time = None
                        stream_ctx: Stream[ChatCompletionChunk] = client.chat.completions.create(
                            stream=True,
                            **payload,
                        )
                        # RFC-0001: shutdown_event 检测，流式中断时提前终止
                        _shutdown_ev = model_call_params.shutdown_event if model_call_params else None
                        with stream_ctx:
                            for chunk in stream_ctx:
                                if _shutdown_ev is not None and _shutdown_ev.is_set():
                                    logger.info("🛑 Shutdown event detected during OpenAI streaming, finalizing partial response")
                                    break
                                if first_token_time is None:
                                    first_token_time = time.time()
                                last_chunk = chunk
                                processed_chunk = _process_stream_chunk(chunk, middleware_manager, model_call_params)
                                if processed_chunk is None:
                                    continue
                                aggregator.consume(processed_chunk)
                        message_payload = aggregator.finalize()
                        trace_ctx.set_outputs(message_payload)
                        if first_token_time is not None:
                            trace_ctx.set_attributes(
                                {
                                    "time_to_first_token_ms": (first_token_time - start_time) * 1000,
                                }
                            )
                        return message_payload, last_chunk, aggregator.model_name

                stream_ctx = client.chat.completions.create(
                    stream=True,
                    **payload,
                )
                # RFC-0001: shutdown_event 检测
                _shutdown_ev = model_call_params.shutdown_event if model_call_params else None
                with stream_ctx:
                    for chunk in stream_ctx:
                        if _shutdown_ev is not None and _shutdown_ev.is_set():
                            logger.info("🛑 Shutdown event detected during OpenAI streaming, finalizing partial response")
                            break
                        last_chunk = chunk
                        processed_chunk = _process_stream_chunk(chunk, middleware_manager, model_call_params)
                        if processed_chunk is None:
                            continue
                        aggregator.consume(processed_chunk)
                message_payload_untraced = aggregator.finalize()
                return message_payload_untraced, last_chunk, aggregator.model_name
            except Exception as exc:
                wrapped_error = _maybe_wrap_stream_idle_timeout(
                    exc,
                    transport_name="openai chat stream",
                    llm_config=llm_config,
                )
                if wrapped_error is not None:
                    raise wrapped_error from exc
                raise

        message, _, _ = call_llm_stream(kwargs)

        return ModelResponse.from_openai_message(message)

    def _ensure_chat_completion(payload: object) -> ChatCompletion:
        if isinstance(payload, ChatCompletion):
            return payload

        # Allow duck-typed mocks in unit tests while still guarding against
        # obviously invalid payloads.
        if hasattr(payload, "choices"):
            return cast(ChatCompletion, payload)

        raise TypeError("Unexpected OpenAI response type")

    def _invoke_chat_completion(api_kwargs: dict[str, Any]) -> ChatCompletion:
        unvalidated_result: object = cast(object, client.chat.completions.create(**api_kwargs))
        return _ensure_chat_completion(unvalidated_result)

    def call_llm(api_kwargs: dict[str, Any]) -> ChatCompletion:
        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "OpenAI chat.completions.create", SpanType.LLM, inputs=api_kwargs)
            with trace_ctx:
                chat_response = _invoke_chat_completion(api_kwargs)
                trace_ctx.set_outputs(_to_serializable_dict(chat_response))
                return chat_response

        return _invoke_chat_completion(api_kwargs)

    response_payload: ChatCompletion = call_llm(kwargs)
    response_message: Any = response_payload.choices[0].message

    # Extract usage information from response
    usage = None
    if hasattr(response_payload, "usage") and response_payload.usage is not None:
        usage = _to_serializable_dict(response_payload.usage)
    return ModelResponse.from_openai_message(response_message, usage=usage)


def call_llm_with_openai_responses(
    client: Any,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call OpenAI Responses API and normalize the outcome."""

    request_payload = kwargs.copy()

    messages = request_payload.pop("messages", None)
    if messages is not None:
        response_items, instructions = prepare_openai_responses_api_input(messages)
        if response_items:
            request_payload.setdefault("input", response_items)
        if instructions:
            existing_instructions = request_payload.get("instructions")
            if existing_instructions:
                combined_instructions = f"{existing_instructions.rstrip()}\n\n{instructions}"
            else:
                combined_instructions = instructions
            request_payload["instructions"] = combined_instructions.strip()

    # Responses API uses max_output_tokens instead of max_tokens
    max_tokens = request_payload.pop("max_tokens", None)
    if max_tokens is not None:
        request_payload.setdefault("max_output_tokens", max_tokens)

    tools = request_payload.get("tools")
    if tools:
        request_payload["tools"] = normalize_openai_responses_api_tools(tools)

    stream_requested = bool(request_payload.pop("stream", False) or getattr(llm_config, "stream", False))

    request_payload.pop("store", None)

    # 默认使用 detailed reasoning summary，使得 reasoning item 包含可读摘要。
    # 如果调用方未指定 summary，自动注入 "detailed"。
    reasoning_param = request_payload.get("reasoning")
    if isinstance(reasoning_param, dict) and "summary" not in reasoning_param:
        reasoning_param["summary"] = "detailed"

    # Always request encrypted reasoning content so that reasoning items can be
    # passed back in subsequent conversation turns (required for stateless / ZDR mode).
    include_value = request_payload.get("include")
    include_list: list[str] = []
    if isinstance(include_value, (list, tuple)):
        include_items = cast(list[object] | tuple[object, ...], include_value)
        include_list = [item for item in include_items if isinstance(item, str)]
    request_payload["include"] = include_list
    if "reasoning.encrypted_content" not in include_list:
        include_list.append("reasoning.encrypted_content")

    # 将代理专用参数（如 prompt_cache_key）移入 extra_body，
    # 因为 OpenAI SDK v2+ 不接受非标准 kwargs。
    extra_body: dict[str, Any] = request_payload.pop("extra_body", None) or {}
    prompt_cache_key = request_payload.pop("prompt_cache_key", None)
    if prompt_cache_key is not None:
        extra_body["prompt_cache_key"] = prompt_cache_key
    if extra_body:
        request_payload["extra_body"] = extra_body

    # Check if tracing is active (there's a current span and we have a tracer)
    should_trace = tracer is not None and get_current_span() is not None

    def call_llm(api_payload: dict[str, Any]) -> Any:
        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "OpenAI responses.create", SpanType.LLM, inputs=api_payload)
            with trace_ctx:
                response = client.responses.create(**api_payload)
                trace_ctx.set_outputs(_to_serializable_dict(response))
                return response
        else:
            response = client.responses.create(**api_payload)
            return response

    if not stream_requested:
        return ModelResponse.from_openai_response(call_llm(request_payload))

    def call_llm_stream(payload: dict[str, Any]) -> dict[str, Any]:
        aggregator = OpenAIResponsesStreamAggregator()
        # RFC-0001: shutdown_event 检测
        _shutdown_ev = model_call_params.shutdown_event if model_call_params else None

        try:
            if should_trace and tracer is not None:
                trace_ctx = TraceContext(tracer, "OpenAI responses.stream", SpanType.LLM, inputs=payload)
                start_time = time.time()
                first_token_time = None
                with trace_ctx:
                    with client.responses.stream(**payload) as stream:
                        for event in stream:
                            if _shutdown_ev is not None and _shutdown_ev.is_set():
                                logger.info("🛑 Shutdown event detected during OpenAI Responses streaming, finalizing partial response")
                                break
                            if first_token_time is None:
                                first_token_time = time.time()
                            processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                            if processed_event is None:
                                continue
                            aggregator.consume(processed_event)
                    response_payload = aggregator.finalize()
                    trace_ctx.set_outputs(response_payload)
                    if first_token_time is not None:
                        trace_ctx.set_attributes(
                            {
                                "time_to_first_token_ms": (first_token_time - start_time) * 1000,
                            }
                        )
                    return response_payload

            with client.responses.stream(**payload) as stream:
                for event in stream:
                    if _shutdown_ev is not None and _shutdown_ev.is_set():
                        logger.info("🛑 Shutdown event detected during OpenAI Responses streaming, finalizing partial response")
                        break
                    processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                    if processed_event is None:
                        continue
                    aggregator.consume(processed_event)
            return aggregator.finalize()
        except Exception as exc:
            wrapped_error = _maybe_wrap_stream_idle_timeout(
                exc,
                transport_name="openai responses stream",
                llm_config=llm_config,
            )
            if wrapped_error is not None:
                raise wrapped_error from exc
            raise

    response_payload = call_llm_stream(request_payload)
    return ModelResponse.from_openai_response(response_payload)


# ── Async LLM call functions ────────────────────────────────────────
#
# async/sync 技术债修复: 使用 AsyncOpenAI / AsyncAnthropic / httpx.AsyncClient
# 实现原生 async 调用，消除 to_thread 桥接。
# force stop 时 asyncio cancellation 直接传播，不再阻塞线程。


async def call_llm_with_different_client_async(
    client: Any,
    llm_config: LLMConfig,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Async dispatcher — routes to the correct async provider function."""
    if llm_config.api_type == "anthropic_chat_completion":
        return await call_llm_with_anthropic_chat_completion_async(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
            cache_control_ttl=llm_config.cache_control_ttl,
        )
    elif llm_config.api_type == "openai_responses":
        return await call_llm_with_openai_responses_async(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    elif llm_config.api_type == "openai_chat_completion":
        return await call_llm_with_openai_chat_completion_async(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    elif llm_config.api_type == "gemini_rest":
        return await call_llm_with_gemini_rest_async(
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
            tracer=tracer,
        )
    else:
        raise ValueError(f"Invalid API type for async call: {llm_config.api_type}")


async def call_llm_with_openai_chat_completion_async(
    client: openai.AsyncOpenAI,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Async OpenAI chat completion — mirrors sync version with await."""

    messages = _strip_responses_api_artifacts(kwargs.get("messages", []))
    for msg in messages:
        if isinstance(msg, dict):
            typed_msg = cast(dict[str, object], msg)
            if typed_msg.get("role") == "assistant" and typed_msg.get("content") == "" and typed_msg.get("tool_calls"):
                del typed_msg["content"]
    kwargs["messages"] = messages
    stream_requested = bool(kwargs.pop("stream", False) or getattr(llm_config, "stream", False))

    should_trace = tracer is not None and get_current_span() is not None

    if stream_requested:
        # 1. 异步流式路径
        payload = kwargs.copy()
        payload.pop("stream", None)
        stream_options = {"include_usage": True}
        payload["stream_options"] = stream_options
        aggregator = OpenAIChatStreamAggregator()

        _shutdown_ev = model_call_params.shutdown_event if model_call_params else None

        try:
            if should_trace and tracer is not None:
                trace_ctx = TraceContext(tracer, "OpenAI chat.completions.create (async stream)", SpanType.LLM, inputs=payload)
                with trace_ctx:
                    start_time = time.time()
                    first_token_time = None
                    stream_ctx: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
                        stream=True,
                        **payload,
                    )
                    async with stream_ctx:
                        async for chunk in stream_ctx:
                            if _shutdown_ev is not None and _shutdown_ev.is_set():
                                logger.info("🛑 Shutdown event detected during async OpenAI streaming")
                                break
                            if first_token_time is None:
                                first_token_time = time.time()
                            processed_chunk = _process_stream_chunk(chunk, middleware_manager, model_call_params)
                            if processed_chunk is None:
                                continue
                            aggregator.consume(processed_chunk)
                    message_payload = aggregator.finalize()
                    trace_ctx.set_outputs(message_payload)
                    if first_token_time is not None:
                        trace_ctx.set_attributes({"time_to_first_token_ms": (first_token_time - start_time) * 1000})
            else:
                stream_ctx = await client.chat.completions.create(stream=True, **payload)
                async with stream_ctx:
                    async for chunk in stream_ctx:
                        if _shutdown_ev is not None and _shutdown_ev.is_set():
                            logger.info("🛑 Shutdown event detected during async OpenAI streaming")
                            break
                        processed_chunk = _process_stream_chunk(chunk, middleware_manager, model_call_params)
                        if processed_chunk is None:
                            continue
                        aggregator.consume(processed_chunk)
                message_payload = aggregator.finalize()
        except Exception as exc:
            wrapped_error = _maybe_wrap_stream_idle_timeout(
                exc,
                transport_name="openai chat stream",
                llm_config=llm_config,
            )
            if wrapped_error is not None:
                raise wrapped_error from exc
            raise

        return ModelResponse.from_openai_message(message_payload)

    # 2. 非流式路径
    async def _invoke(api_kwargs: dict[str, Any]) -> ChatCompletion:
        return cast(ChatCompletion, await client.chat.completions.create(**api_kwargs))

    if should_trace and tracer is not None:
        trace_ctx_nr = TraceContext(tracer, "OpenAI chat.completions.create (async)", SpanType.LLM, inputs=kwargs)
        with trace_ctx_nr:
            chat_response = await _invoke(kwargs)
            trace_ctx_nr.set_outputs(_to_serializable_dict(chat_response))
    else:
        chat_response = await _invoke(kwargs)

    response_message: Any = chat_response.choices[0].message
    usage = None
    if chat_response.usage is not None:
        usage = _to_serializable_dict(chat_response.usage)
    return ModelResponse.from_openai_message(response_message, usage=usage)


async def call_llm_with_anthropic_chat_completion_async(
    client: Any,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
    cache_control_ttl: str | None = None,
) -> ModelResponse:
    """Async Anthropic chat completion — mirrors sync version with await."""
    stream_requested = bool(kwargs.pop("stream", False))
    should_trace = tracer is not None and get_current_span() is not None

    def _build_cache_control() -> dict[str, str]:
        cc: dict[str, str] = {"type": "ephemeral"}
        if cache_control_ttl:
            cc["ttl"] = cache_control_ttl
        return cc

    def _apply_cache_control(
        system_messages: list[dict[str, Any]],
        user_messages: list[dict[str, Any]],
    ) -> None:
        for sys_block in system_messages:
            should_cache = sys_block.pop("_cache", True)
            if should_cache:
                sys_block["cache_control"] = _build_cache_control()
        if user_messages and user_messages[-1].get("content"):
            content = cast(list[dict[str, Any]] | str | None, user_messages[-1].get("content"))
            if isinstance(content, list) and content:
                content[0]["cache_control"] = _build_cache_control()

    def _build_anthropic_messages() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if type(model_call_params) is not ModelCallParams:
            raise ValueError("Anthropic calls require explicit ModelCallParams with UMP messages")

        from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter

        return AnthropicMessagesAdapter().to_vendor_format(model_call_params.messages)

    # 1. 组装参数（与 sync 版完全相同）
    system_messages, user_messages = _build_anthropic_messages()
    _apply_cache_control(system_messages, user_messages)

    new_kwargs = kwargs.copy()
    new_kwargs.pop("messages", None)
    new_kwargs.pop("anthropic_cache_control_ttl", None)
    api_kwargs: dict[str, Any] = {"system": system_messages, "messages": user_messages, **new_kwargs}

    if not stream_requested:
        # 2. 非流式路径
        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "Anthropic messages.create (async)", SpanType.LLM, inputs=api_kwargs)
            with trace_ctx:
                resp = await client.messages.create(**api_kwargs)
                trace_ctx.set_outputs(_to_serializable_dict(resp))
        else:
            resp = await client.messages.create(**api_kwargs)
        return ModelResponse.from_anthropic_message(resp)

    # 3. 流式路径
    aggregator = AnthropicStreamAggregator()
    _shutdown_ev = model_call_params.shutdown_event if model_call_params else None

    try:
        if should_trace and tracer is not None:
            trace_ctx_s = TraceContext(tracer, "Anthropic messages.stream (async)", SpanType.LLM, inputs=api_kwargs)
            with trace_ctx_s:
                start_time = time.time()
                first_token_time = None
                async with client.messages.stream(**api_kwargs) as stream:
                    async for event in stream:
                        if _shutdown_ev is not None and _shutdown_ev.is_set():
                            logger.info("🛑 Shutdown event detected during async Anthropic streaming")
                            break
                        if first_token_time is None:
                            first_token_time = time.time()
                        processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                        if processed_event is None:
                            continue
                        aggregator.consume(processed_event)
                message_payload = aggregator.finalize()
                trace_ctx_s.set_outputs(message_payload)
                if first_token_time is not None:
                    trace_ctx_s.set_attributes({"time_to_first_token_ms": (first_token_time - start_time) * 1000})
        else:
            async with client.messages.stream(**api_kwargs) as stream:
                async for event in stream:
                    if _shutdown_ev is not None and _shutdown_ev.is_set():
                        logger.info("🛑 Shutdown event detected during async Anthropic streaming")
                        break
                    processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                    if processed_event is None:
                        continue
                    aggregator.consume(processed_event)
            message_payload = aggregator.finalize()
    except Exception as exc:
        wrapped_error = _maybe_wrap_stream_idle_timeout(
            exc,
            transport_name="anthropic stream",
            llm_config=llm_config,
        )
        if wrapped_error is not None:
            raise wrapped_error from exc
        raise

    return ModelResponse.from_anthropic_message(message_payload)


async def call_llm_with_openai_responses_async(
    client: Any,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Async OpenAI Responses API — mirrors sync version with await."""

    request_payload = kwargs.copy()

    messages = request_payload.pop("messages", None)
    if messages is not None:
        response_items, instructions = prepare_openai_responses_api_input(messages)
        if response_items:
            request_payload.setdefault("input", response_items)
        if instructions:
            existing_instructions = request_payload.get("instructions")
            if existing_instructions:
                combined_instructions = f"{existing_instructions.rstrip()}\n\n{instructions}"
            else:
                combined_instructions = instructions
            request_payload["instructions"] = combined_instructions.strip()

    max_tokens = request_payload.pop("max_tokens", None)
    if max_tokens is not None:
        request_payload.setdefault("max_output_tokens", max_tokens)

    tools = request_payload.get("tools")
    if tools:
        request_payload["tools"] = normalize_openai_responses_api_tools(tools)

    stream_requested = bool(request_payload.pop("stream", False) or getattr(llm_config, "stream", False))

    request_payload.pop("store", None)

    # 默认使用 detailed reasoning summary，使得 reasoning item 包含可读摘要。
    reasoning_param = request_payload.get("reasoning")
    if isinstance(reasoning_param, dict) and "summary" not in reasoning_param:
        reasoning_param["summary"] = "detailed"

    include_value = request_payload.get("include")
    include_list: list[str] = []
    if isinstance(include_value, (list, tuple)):
        include_items = cast(list[object] | tuple[object, ...], include_value)
        include_list = [item for item in include_items if isinstance(item, str)]
    request_payload["include"] = include_list
    if "reasoning.encrypted_content" not in include_list:
        include_list.append("reasoning.encrypted_content")

    extra_body: dict[str, Any] = request_payload.pop("extra_body", None) or {}
    prompt_cache_key = request_payload.pop("prompt_cache_key", None)
    if prompt_cache_key is not None:
        extra_body["prompt_cache_key"] = prompt_cache_key
    if extra_body:
        request_payload["extra_body"] = extra_body

    should_trace = tracer is not None and get_current_span() is not None

    if not stream_requested:
        # 非流式路径
        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "OpenAI responses.create (async)", SpanType.LLM, inputs=request_payload)
            with trace_ctx:
                response = await client.responses.create(**request_payload)
                trace_ctx.set_outputs(_to_serializable_dict(response))
        else:
            response = await client.responses.create(**request_payload)
        return ModelResponse.from_openai_response(response)

    # 流式路径
    aggregator = OpenAIResponsesStreamAggregator()
    _shutdown_ev = model_call_params.shutdown_event if model_call_params else None

    try:
        if should_trace and tracer is not None:
            trace_ctx_s = TraceContext(tracer, "OpenAI responses.stream (async)", SpanType.LLM, inputs=request_payload)
            start_time = time.time()
            first_token_time = None
            with trace_ctx_s:
                async with client.responses.stream(**request_payload) as stream:
                    async for event in stream:
                        if _shutdown_ev is not None and _shutdown_ev.is_set():
                            logger.info("🛑 Shutdown event detected during async OpenAI Responses streaming")
                            break
                        if first_token_time is None:
                            first_token_time = time.time()
                        processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                        if processed_event is None:
                            continue
                        aggregator.consume(processed_event)
                response_payload_s = aggregator.finalize()
                trace_ctx_s.set_outputs(response_payload_s)
                if first_token_time is not None:
                    trace_ctx_s.set_attributes({"time_to_first_token_ms": (first_token_time - start_time) * 1000})
        else:
            async with client.responses.stream(**request_payload) as stream:
                async for event in stream:
                    if _shutdown_ev is not None and _shutdown_ev.is_set():
                        logger.info("🛑 Shutdown event detected during async OpenAI Responses streaming")
                        break
                    processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                    if processed_event is None:
                        continue
                    aggregator.consume(processed_event)
            response_payload_s = aggregator.finalize()
    except Exception as exc:
        wrapped_error = _maybe_wrap_stream_idle_timeout(
            exc,
            transport_name="openai responses stream",
            llm_config=llm_config,
        )
        if wrapped_error is not None:
            raise wrapped_error from exc
        raise

    return ModelResponse.from_openai_response(response_payload_s)


def _process_stream_chunk(
    chunk: Any,
    middleware_manager: MiddlewareManager | None,
    model_call_params: ModelCallParams | None,
) -> Any:
    """Run a raw stream chunk through middleware pipeline."""

    if middleware_manager is None or model_call_params is None:
        return chunk
    return middleware_manager.stream_chunk(chunk, model_call_params)


def _safe_get(item: Any, key: str, default: Any = None) -> Any:
    """Generic attribute/dict getter."""

    if isinstance(item, Mapping):
        mapping_item = cast(Mapping[str, Any], item)
        return mapping_item.get(key, default)
    return getattr(item, key, default)


def _to_serializable_dict(payload: Any) -> dict[str, Any]:
    """Convert SDK models into plain dictionaries."""

    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            # Pydantic v2 defaults `warnings="warn"` which can be noisy for some
            # third-party SDK models (e.g., OpenAI Responses typed generics).
            # Prefer a JSON-ready dump and silence serializer warnings.
            try:
                return cast(dict[str, Any], model_dump(mode="json", warnings=False))
            except TypeError:
                # Older/newer pydantic versions may not support all kwargs.
                try:
                    return cast(dict[str, Any], model_dump(warnings=False))
                except TypeError:
                    return cast(dict[str, Any], model_dump())
        except Exception:  # pragma: no cover - defensive
            pass

    result: dict[str, Any] = {}
    for attr in dir(payload):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(payload, attr)
        except Exception:
            continue
        if callable(value):
            continue
        result[attr] = value
    return result


def _enrich_gemini_trace_outputs(
    output: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Enrich Gemini REST trace output with model and usage for Langfuse.

    Gemini REST 响应使用 modelVersion / usageMetadata，而 Langfuse tracer
    的 end_span 依赖 output dict 中的 model / usage 来填充 generation 的
    model 标签和 token 用量。此函数做字段映射注入。
    """
    enriched = dict(output)
    # 1. 注入 model（Langfuse 用来标记 generation 的模型名）
    enriched["model"] = model_name
    # 2. 将 usageMetadata 映射为 Langfuse 期望的 usage 格式
    #    Langfuse _sanitize_usage 只保留 int 值，所以这里全部转 int。
    usage_meta = output.get("usageMetadata")
    if isinstance(usage_meta, dict):
        meta: dict[str, object] = cast(dict[str, object], usage_meta)

        def _int_field(key: str) -> int:
            val = meta.get(key, 0)
            return int(val) if isinstance(val, int) else 0

        usage: dict[str, int] = {
            "input_tokens": _int_field("promptTokenCount"),
            "output_tokens": _int_field("candidatesTokenCount"),
            "total_tokens": _int_field("totalTokenCount"),
        }
        # 缓存和推理 token — 仅在实际存在时注入，保持 Langfuse 用量面板简洁
        cached = _int_field("cachedContentTokenCount")
        if cached > 0:
            usage["cached_tokens"] = cached
        thoughts = _int_field("thoughtsTokenCount")
        if thoughts > 0:
            usage["reasoning_tokens"] = thoughts
        enriched["usage"] = usage
    return enriched


class OpenAIChatStreamAggregator:
    """Aggregate OpenAI chat completion stream chunks into a final message dict."""

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._reasoning_parts: list[str] = []
        self.role: str = "assistant"
        self.model_name: str | None = None
        self.usage: Any | None = None

    def consume(self, chunk: Any) -> None:
        payload = _to_serializable_dict(chunk)
        model = payload.get("model")
        if isinstance(model, str):
            self.model_name = model

        usage_payload = payload.get("usage")
        if usage_payload is not None:
            self.usage = _to_serializable_dict(usage_payload)

        choices: list[Any] = payload.get("choices") or []
        for choice in choices:
            choice_dict: dict[str, Any] = _to_serializable_dict(choice)
            delta: dict[str, Any] = _to_serializable_dict(choice_dict.get("delta", {}))

            role = delta.get("role")
            if role:
                self.role = role

            content_delta = delta.get("content")
            if isinstance(content_delta, str):
                self._content_parts.append(content_delta)
            elif isinstance(content_delta, list):
                content_delta_list: list[Any] = cast(list[Any], content_delta)
                for entry in content_delta_list:
                    entry_text = _safe_get(entry, "text")
                    if entry_text:
                        self._content_parts.append(str(entry_text))

            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str):
                self._reasoning_parts.append(reasoning)
            elif isinstance(reasoning, list):
                reasoning_list: list[Any] = cast(list[Any], reasoning)
                for entry in reasoning_list:
                    text = _safe_get(entry, "text")
                    if text:
                        self._reasoning_parts.append(str(text))

            tool_calls: list[Any] = delta.get("tool_calls") or []
            for tool_delta in tool_calls:
                tool_dict: dict[str, Any] = _to_serializable_dict(tool_delta)
                index = int(tool_dict.get("index", 0))
                builder = self._tool_calls.setdefault(
                    index,
                    {
                        "id": tool_dict.get("id"),
                        "type": tool_dict.get("type", "function"),
                        "function": {"name": None, "arguments": ""},
                    },
                )
                if tool_dict.get("id"):
                    builder["id"] = tool_dict["id"]
                if tool_dict.get("type"):
                    builder["type"] = tool_dict["type"]

                function_delta: dict[str, Any] = _to_serializable_dict(tool_dict.get("function", {}))
                if function_delta.get("name"):
                    builder.setdefault("function", {})["name"] = function_delta["name"]
                arguments = function_delta.get("arguments")
                if arguments:
                    current = builder.setdefault("function", {}).get("arguments") or ""
                    builder["function"]["arguments"] = f"{current}{arguments}"

    def finalize(self) -> dict[str, Any]:
        if not self._content_parts and not self._tool_calls and not self._reasoning_parts:
            raise RuntimeError("No stream chunks were received from OpenAI chat completion")

        message: dict[str, Any] = {
            "role": self.role or "assistant",
            "content": "".join(self._content_parts) if self._content_parts else "",
        }

        if self._tool_calls:
            ordered_calls: list[dict[str, Any]] = []
            for index in sorted(self._tool_calls):
                call = self._tool_calls[index]
                function_payload: dict[str, Any] = cast(dict[str, Any], call.get("function") or {})
                arguments = function_payload.get("arguments") or ""
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                ordered_calls.append(
                    {
                        "id": call.get("id"),
                        "type": call.get("type", "function"),
                        "function": {
                            "name": function_payload.get("name"),
                            "arguments": arguments,
                        },
                    },
                )
            message["tool_calls"] = ordered_calls

        if self._reasoning_parts:
            message["reasoning_content"] = "".join(self._reasoning_parts)

        if self.model_name:
            message["model"] = self.model_name

        if self.usage is not None:
            message["usage"] = self.usage

        return message


class AnthropicStreamAggregator:
    """Aggregate Anthropic streaming events into a final message payload."""

    def __init__(self) -> None:
        self.role: str = "assistant"
        self.model_name: str | None = None
        self.usage: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self._active_blocks: dict[int, dict[str, Any]] = {}
        self._completed_blocks: list[dict[str, Any]] = []

    def consume(self, event: Any) -> None:
        payload = _to_serializable_dict(event)
        event_type = payload.get("type")

        if event_type == "message_start":
            message = _to_serializable_dict(payload.get("message", {}))
            role = message.get("role")
            if role:
                self.role = role
            if message.get("model"):
                self.model_name = message["model"]
            # Extract initial usage information if available
            usage_data = message.get("usage")
            if usage_data:
                self.usage = _to_serializable_dict(usage_data)
        elif event_type == "message_delta":
            # Update usage and stop_reason from delta events
            delta = _to_serializable_dict(payload.get("delta", {}))
            stop_reason = delta.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                self.stop_reason = stop_reason
            usage_data = delta.get("usage") or payload.get("usage")
            if usage_data:
                merged = _to_serializable_dict(usage_data)
                if self.usage is not None:
                    self.usage.update(merged)
                else:
                    self.usage = merged
        elif event_type == "content_block_start":
            index = payload.get("index")
            block = _to_serializable_dict(payload.get("content_block", {}))
            if block and isinstance(index, int):
                # Some Anthropic SDK stream traces can surface duplicate content_block_start events
                # for the same index (sometimes with missing fields like name/id). Never overwrite a
                # previously-seen block with a more empty one; merge only meaningful fields.
                existing = self._active_blocks.get(index)
                if existing is not None:
                    existing_input_buffer = existing.get("_input_buffer", "")
                    for key, value in block.items():
                        if key == "_input_buffer":
                            continue
                        # Only merge "meaningful" values; don't clobber with None/empty.
                        if value in (None, "", {}, []):
                            continue
                        if existing.get(key) in (None, "", {}, []):
                            existing[key] = value
                    existing["_input_buffer"] = existing_input_buffer
                else:
                    block["_input_buffer"] = ""
                    self._active_blocks[index] = block
        elif event_type == "content_block_delta":
            self._apply_block_delta(payload)
        elif event_type == "content_block_stop":
            self._finalize_block(payload.get("index"))
        elif event_type == "message_stop":
            # Extract final usage information from message_stop event
            message = _to_serializable_dict(payload.get("message", {}))
            usage_data = message.get("usage")
            if usage_data:
                self.usage = _to_serializable_dict(usage_data)
            self._flush_active_blocks()

    def finalize(self) -> dict[str, Any]:
        self._flush_active_blocks()
        if not self._completed_blocks:
            raise RuntimeError("No stream chunks were received from Anthropic messages stream")
        message: dict[str, Any] = {
            "role": self.role or "assistant",
            "content": self._completed_blocks,
        }
        if self.model_name:
            message["model"] = self.model_name
        if self.stop_reason:
            message["stop_reason"] = self.stop_reason
        if self.usage is not None:
            message["usage"] = self.usage
        return message

    def _apply_block_delta(self, payload: dict[str, Any]) -> None:
        index = payload.get("index")
        if not isinstance(index, int):
            return
        delta = _to_serializable_dict(payload.get("delta", {}))
        # Avoid defaulting to text: in rare malformed streams we might see deltas before start,
        # and for tool blocks we can infer the type from the delta itself.
        block = self._active_blocks.setdefault(index, {"_input_buffer": ""})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            block["type"] = "text"
            # content_block_start 可能将 text 初始化为 None，.get() 的默认值不会覆盖已有的 None
            block["text"] = (block.get("text") or "") + (delta.get("text") or "")
        elif delta_type == "thinking_delta":
            # Anthropic streams thinking in fragments; append like text.
            block["type"] = "thinking"
            block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
        elif delta_type == "signature_delta":
            # signature 只会到达一次（与官方 SDK 行为一致），直接赋值
            block["signature"] = delta.get("signature") or ""
        elif delta_type == "input_json_delta":
            block.setdefault("type", "tool_use")
            fragment = delta.get("partial_json") or ""
            block["_input_buffer"] = (block.get("_input_buffer") or "") + fragment
        else:
            # For other delta types, merge raw structure
            for key, value in delta.items():
                if key == "type":
                    continue
                block[key] = value

    def _finalize_block(self, index: int | None) -> None:
        if index is None:
            return
        block = self._active_blocks.pop(index, None)
        if not block:
            return
        input_buffer = block.pop("_input_buffer", None)
        if input_buffer:
            try:
                block["input"] = json.loads(input_buffer)
            except json.JSONDecodeError:
                # eager_input_streaming 跳过 JSON 验证，可能产生拼接或截断的 JSON；
                # 用 raw_decode 提取第一个合法 JSON 对象。
                try:
                    first_obj, _ = json.JSONDecoder().raw_decode(input_buffer.lstrip())
                    block["input"] = first_obj
                    logger.warning(
                        "⚠️ Anthropic tool input contained extra data after first JSON object "
                        "(eager_input_streaming); used first object only. raw length=%d",
                        len(input_buffer),
                    )
                except (json.JSONDecodeError, ValueError):
                    block["input"] = input_buffer
        self._completed_blocks.append(block)

    def _flush_active_blocks(self) -> None:
        remaining = list(self._active_blocks.keys())
        for idx in remaining:
            self._finalize_block(idx)


class ResponseMessageBuilder:
    """Helper for assembling streamed Responses API assistant messages."""

    def __init__(self, item_id: str, role: str = "assistant", phase: str | None = None) -> None:
        self.item_id = item_id
        self.role = role
        self.phase = phase
        self._parts: dict[int, dict[str, Any]] = {}

    def add_part(self, index: int, part: dict[str, Any]) -> None:
        self._parts[index] = part

    def update_from_item(self, item: Mapping[str, Any]) -> None:
        role = item.get("role")
        if isinstance(role, str) and role:
            self.role = role
        phase = item.get("phase")
        if isinstance(phase, str) and phase:
            self.phase = phase

    def append_text(self, index: int, delta_text: str) -> None:
        part = self._parts.setdefault(index, {"type": "output_text", "text": ""})
        part["type"] = part.get("type") or "output_text"
        part["text"] = (part.get("text") or "") + delta_text

    def to_output_item(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for index in sorted(self._parts):
            part = self._parts[index]
            content.append(part)
        item: dict[str, Any] = {
            "type": "message",
            "role": self.role or "assistant",
            "id": self.item_id,
            "content": content,
        }
        if self.phase is not None:
            item["phase"] = self.phase
        return item


class ResponseToolCallBuilder:
    """Helper for assembling streamed Responses API tool calls."""

    def __init__(self, item_id: str) -> None:
        self.item_id = item_id
        self.call_id = item_id
        self.name: str | None = None
        self.arguments: str = ""

    def update_from_item(self, item: dict[str, Any]) -> None:
        if item.get("call_id"):
            self.call_id = item["call_id"]
        if item.get("name"):
            self.name = item["name"]
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            self.arguments = arguments

    def append_arguments(self, delta: str) -> None:
        if not delta:
            return
        self.arguments = f"{self.arguments}{delta}"

    def set_arguments(self, arguments: str) -> None:
        if not arguments:
            return
        self.arguments = arguments

    def to_output_item(self) -> dict[str, Any]:
        return {
            "type": "function_call",
            "id": self.item_id,
            "call_id": self.call_id,
            "name": self.name or "unknown",
            "arguments": self.arguments,
        }


class ReasoningSummaryBuilder:
    """Helper for assembling streamed reasoning summary output."""

    def __init__(
        self,
        item_id: str,
        *,
        content: list[dict[str, Any]] | None = None,
        summary: list[dict[str, Any]] | None = None,
        encrypted_content: str | None = None,
    ) -> None:
        self.item_id = item_id
        self.content: list[dict[str, Any]] = list(content or [])
        self.encrypted_content: str | None = encrypted_content
        self._summary_parts: dict[int, dict[str, Any]] = {}
        self._summary_order: list[int] = []
        self._seed_initial_summary(summary)

    def update_from_item(self, item: dict[str, Any]) -> None:
        content = item.get("content")
        if isinstance(content, list):
            content_dicts: list[dict[str, Any]] = []
            content_list: list[Mapping[str, Any]] = cast(list[Mapping[str, Any]], content)
            for content_part in content_list:
                content_dicts.append(dict(content_part))
            self.content = content_dicts
        encrypted = item.get("encrypted_content")
        if isinstance(encrypted, str):
            self.encrypted_content = encrypted
        summary = item.get("summary")
        if isinstance(summary, list):
            summary_parts: list[dict[str, Any]] = []
            summary_list: list[Mapping[str, Any]] = cast(list[Mapping[str, Any]], summary)
            for summary_part in summary_list:
                summary_parts.append(dict(summary_part))
            self._seed_initial_summary(summary_parts)

    def _seed_initial_summary(self, summary: list[dict[str, Any]] | None) -> None:
        if not summary or self._summary_parts:
            return
        for idx, part in enumerate(summary):
            entry = dict(part)
            self._summary_parts[idx] = entry
            if idx not in self._summary_order:
                self._summary_order.append(idx)

    def append_summary_delta(self, index: int, delta_text: str) -> None:
        entry = self._summary_parts.setdefault(index, {"type": "summary_text", "text": ""})
        entry["text"] = f"{entry.get('text', '')}{delta_text}"
        if index not in self._summary_order:
            self._summary_order.append(index)

    def to_output_item(self) -> dict[str, Any]:
        summary_parts = [self._summary_parts[idx] for idx in sorted(self._summary_order)]
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": self.item_id,
        }
        if self.encrypted_content is not None:
            item["encrypted_content"] = self.encrypted_content
        if self.content:
            item["content"] = self.content
        # Always include summary — API requires it even as [] for encrypted reasoning items
        item["summary"] = summary_parts
        return item


class OpenAIResponsesStreamAggregator:
    """Aggregate Responses API streaming events into a final Response payload."""

    def __init__(self) -> None:
        self._message_builders: dict[str, ResponseMessageBuilder] = {}
        self._tool_builders: dict[str, ResponseToolCallBuilder] = {}
        self._reasoning_builders: dict[str, ReasoningSummaryBuilder] = {}
        self._output_order: list[str] = []
        self._completed_response: dict[str, Any] | None = None
        self.response_id: str | None = None
        self.model_name: str | None = None
        self.usage: Any | None = None

    def consume(self, event: Any) -> None:
        payload = _to_serializable_dict(event)
        event_type = payload.get("type")

        if event_type == "response.output_item.added":
            self._handle_output_item_added(payload)
        elif event_type == "response.output_item.done":
            self._handle_output_item_done(payload)
        elif event_type == "response.content_part.added":
            self._handle_content_part_added(payload)
        elif event_type == "response.output_text.delta":
            self._handle_text_delta(payload)
        elif event_type == "response.function_call_arguments.delta":
            self._handle_function_arguments_delta(payload)
        elif event_type == "response.function_call_arguments.done":
            self._handle_function_arguments_done(payload)
        elif event_type == "response.reasoning_summary_text.delta":
            self._handle_reasoning_delta(payload)
        elif event_type == "response.completed":
            response_data = _to_serializable_dict(payload.get("response", {}))
            self._completed_response = response_data
            self.response_id = response_data.get("id") or self.response_id
            self.model_name = response_data.get("model") or self.model_name
            usage = response_data.get("usage")
            if usage:
                self.usage = usage
        elif event_type == "response.created":
            response_data = _to_serializable_dict(payload.get("response", {}))
            self.response_id = response_data.get("id") or self.response_id
            self.model_name = response_data.get("model") or self.model_name

    def finalize(self) -> dict[str, Any]:
        output_items: list[dict[str, Any]] = []

        for item_id in self._output_order:
            if item_id in self._message_builders:
                output_items.append(self._message_builders[item_id].to_output_item())
            elif item_id in self._tool_builders:
                output_items.append(self._tool_builders[item_id].to_output_item())
            elif item_id in self._reasoning_builders:
                output_items.append(self._reasoning_builders[item_id].to_output_item())

        # Append any builders that were not in the initial output order
        for item_id, message_builder in self._message_builders.items():
            if item_id not in self._output_order:
                output_items.append(message_builder.to_output_item())
        for item_id, tool_builder in self._tool_builders.items():
            if item_id not in self._output_order:
                output_items.append(tool_builder.to_output_item())
        for item_id, reasoning_builder in self._reasoning_builders.items():
            if item_id not in self._output_order:
                output_items.append(reasoning_builder.to_output_item())

        if not output_items and self._completed_response is not None:
            return self._completed_response

        response_payload: dict[str, Any] = {
            "id": self.response_id,
            "model": self.model_name,
            "output": output_items,
        }
        if self.usage is not None:
            response_payload["usage"] = self.usage
        return response_payload

    def _handle_output_item_added(self, payload: dict[str, Any]) -> None:
        item = _to_serializable_dict(payload.get("item", {}))
        item_id = item.get("id")
        if not item_id:
            return
        if item_id not in self._output_order:
            self._output_order.append(item_id)

        item_type = item.get("type")
        if item_type == "message":
            message_builder = self._message_builders.setdefault(
                item_id,
                ResponseMessageBuilder(item_id, item.get("role", "assistant"), item.get("phase")),
            )
            message_builder.update_from_item(item)
            content_list: list[Any] = item.get("content") or []
            for idx, part in enumerate(content_list):
                message_builder.add_part(idx, _to_serializable_dict(part))
        elif item_type == "function_call":
            tool_builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
            tool_builder.update_from_item(item)
        elif item_type == "reasoning":
            reasoning_builder = self._reasoning_builders.setdefault(
                item_id,
                ReasoningSummaryBuilder(
                    item_id,
                    content=item.get("content"),
                    summary=item.get("summary"),
                    encrypted_content=item.get("encrypted_content"),
                ),
            )
            reasoning_builder.update_from_item(item)

    def _handle_output_item_done(self, payload: dict[str, Any]) -> None:
        """Handle response.output_item.done to capture final encrypted_content for reasoning items."""
        item = _to_serializable_dict(payload.get("item", {}))
        item_id = item.get("id")
        if not item_id:
            return
        item_type = item.get("type")
        if item_type == "message":
            message_builder = self._message_builders.get(item_id)
            if message_builder is not None:
                message_builder.update_from_item(item)
        elif item_type == "reasoning":
            reasoning_builder = self._reasoning_builders.get(item_id)
            if reasoning_builder is not None:
                reasoning_builder.update_from_item(item)

    def _handle_content_part_added(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        message_builder = self._message_builders.setdefault(item_id, ResponseMessageBuilder(item_id))
        message_builder.add_part(payload.get("content_index", 0), _to_serializable_dict(payload.get("part", {})))

    def _handle_text_delta(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        message_builder = self._message_builders.setdefault(item_id, ResponseMessageBuilder(item_id))
        delta_text = payload.get("delta", "")
        if not isinstance(delta_text, str):
            delta_text = str(delta_text)
        message_builder.append_text(payload.get("content_index", 0), delta_text)

    def _handle_function_arguments_delta(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        tool_builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
        tool_builder.append_arguments(payload.get("delta", ""))

    def _handle_function_arguments_done(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        tool_builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
        tool_builder.set_arguments(payload.get("arguments", ""))

    def _handle_reasoning_delta(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id") or f"_reasoning_{payload.get('output_index', 0)}"
        summary_index = payload.get("summary_index", 0)
        delta = payload.get("delta", "")
        if not isinstance(delta, str):
            delta = str(delta)
        reasoning_builder = self._reasoning_builders.setdefault(item_id, ReasoningSummaryBuilder(item_id))
        reasoning_builder.append_summary_delta(summary_index, delta)


class GeminiRestStreamAggregator:
    """Aggregate Gemini REST API SSE stream chunks into a final response dict.

    RFC-0003: Gemini REST 流式响应聚合器

    Each SSE chunk from Gemini streamGenerateContent has the same structure
    as a non-streaming generateContent response.  This aggregator merges
    text deltas, thinking content, thought signatures, function calls,
    and usage metadata across all chunks into a single dict compatible
    with ModelResponse.from_gemini_rest().
    """

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._thought_signature: str | None = None
        self._tool_calls: list[dict[str, Any]] = []
        self._usage_metadata: dict[str, Any] = {}
        self.model_name: str | None = None

    def consume(self, chunk_json: dict[str, Any]) -> None:
        """Process a single SSE chunk (parsed JSON dict).

        RFC-0003: 处理单个 SSE 数据块
        """
        # 1. 提取 candidates
        candidates_raw = chunk_json.get("candidates")
        if not isinstance(candidates_raw, Sequence) or isinstance(candidates_raw, str | bytes | bytearray):
            return

        candidates: list[Mapping[str, object]] = []
        for candidate_item in cast(Sequence[object], candidates_raw):
            if isinstance(candidate_item, Mapping):
                candidates.append(cast(Mapping[str, object], candidate_item))
        if not candidates:
            return

        candidate = candidates[0]
        content_raw = candidate.get("content")
        content_obj: Mapping[str, object] = cast(Mapping[str, object], content_raw) if isinstance(content_raw, Mapping) else {}

        parts_raw = content_obj.get("parts")
        if not isinstance(parts_raw, Sequence) or isinstance(parts_raw, str | bytes | bytearray):
            parts: list[Mapping[str, object]] = []
        else:
            parts = []
            for part_item in cast(Sequence[object], parts_raw):
                if isinstance(part_item, Mapping):
                    parts.append(cast(Mapping[str, object], part_item))

        # 2. 遍历 parts，分类聚合
        for part in parts:
            if part.get("thought"):
                text = part.get("text")
                if isinstance(text, str) and text:
                    self._reasoning_parts.append(text)

            if "thoughtSignature" in part:
                thought_signature = part["thoughtSignature"]
                if isinstance(thought_signature, str):
                    self._thought_signature = thought_signature

            if "text" in part and not part.get("thought"):
                content_text = part.get("text")
                if isinstance(content_text, str):
                    self._content_parts.append(content_text)

            if "functionCall" in part:
                self._tool_calls.append(dict(part))

        # 3. 提取 usage metadata（通常在最后一个 chunk）
        usage_meta = chunk_json.get("usageMetadata")
        if usage_meta:
            self._usage_metadata = usage_meta

        # 4. 提取 model name（如果存在）
        model = chunk_json.get("modelVersion")
        if isinstance(model, str):
            self.model_name = model

    def finalize(self) -> dict[str, Any]:
        """Return aggregated response dict compatible with ModelResponse.from_gemini_rest().

        RFC-0003: 返回与 ModelResponse.from_gemini_rest() 兼容的聚合响应
        """
        if not self._content_parts and not self._tool_calls and not self._reasoning_parts:
            raise RuntimeError("No stream chunks were received from Gemini REST API")

        # Build parts list matching Gemini response structure
        parts: list[dict[str, Any]] = []

        if self._reasoning_parts:
            parts.append({"text": "".join(self._reasoning_parts), "thought": True})

        if self._thought_signature is not None:
            parts.append({"thoughtSignature": self._thought_signature})

        if self._content_parts:
            parts.append({"text": "".join(self._content_parts)})

        for tc_part in self._tool_calls:
            parts.append(tc_part)

        result: dict[str, Any] = {
            "candidates": [{"content": {"parts": parts, "role": "model"}}],
        }
        if self._usage_metadata:
            result["usageMetadata"] = self._usage_metadata

        return result


def _iter_gemini_sse_chunks(response: requests.Response) -> Iterator[dict[str, Any]]:
    """Parse streaming chunks from a Gemini streaming response.

    RFC-0003: 解析 Gemini 流式响应

    Supports two response formats:
    1. SSE format (``alt=sse``): lines prefixed with ``data:`` contain JSON payloads.
    2. JSON array format (default ``streamGenerateContent``): the response body is
       a JSON array of candidate objects, streamed incrementally.

    The parser first attempts SSE parsing.  If no ``data:`` lines are found it
    falls back to accumulating the raw bytes and parsing them as a JSON array.
    """
    raw_lines: list[str] = []
    yielded_any = False

    for line_bytes in response.iter_lines():
        if not line_bytes:
            continue
        line = line_bytes.decode() if isinstance(line_bytes, bytes) else line_bytes

        # 1. SSE 格式: 处理 "data:" 前缀行（兼容有无空格）
        if line.startswith("data:"):
            json_str = line[5:].lstrip()  # strip "data:" and optional leading space
            if not json_str:
                continue
            try:
                chunk = json.loads(json_str)
                yielded_any = True
                yield chunk
            except json.JSONDecodeError:
                logger.warning("Failed to parse Gemini SSE chunk: %s", json_str[:200])
            continue

        # 2. 非 SSE 行: 收集用于 JSON 数组回退解析
        raw_lines.append(line)

    # 3. 回退: 如果没有 SSE 数据，尝试将收集的行解析为 JSON 数组
    if not yielded_any and raw_lines:
        body = "\n".join(raw_lines)
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list):
                for item_obj in cast(list[object], parsed):
                    if isinstance(item_obj, dict):
                        yield cast(dict[str, object], item_obj)
            elif isinstance(parsed, dict):
                yield parsed
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini streaming response as JSON: %s", body[:500])


def _gemini_sanitize_parameters(params: dict[str, object]) -> dict[str, object]:
    """Recursively sanitize schema for Gemini (strip $schema, additionalProperties, etc.)."""
    allowed = {"type", "properties", "required", "description", "enum", "items", "format", "nullable"}
    sanitized: dict[str, object] = {}
    for k, v in params.items():
        if k not in allowed:
            continue
        if k == "properties" and isinstance(v, dict):
            v_dict = cast(dict[str, object], v)
            sanitized[k] = {
                pk: _gemini_sanitize_parameters(cast(dict[str, object], pv)) for pk, pv in v_dict.items() if isinstance(pv, dict)
            }
        elif k == "items" and isinstance(v, dict):
            sanitized[k] = _gemini_sanitize_parameters(cast(dict[str, object], v))
        else:
            sanitized[k] = v
    return sanitized


def convert_tools_to_gemini(
    tools: Sequence[StructuredToolDefinitionLike],
) -> list[dict[str, Any]]:
    """Convert structured tool definitions to Gemini function declarations.

    RFC-0006: Gemini 原生 structured tool adapter

    Gemini 直接从 neutral structured definition 生成
    ``functionDeclarations``，不再以 OpenAI schema 作为主中转形状。
    """

    gemini_tools: list[dict[str, Any]] = []
    for tool in tools:
        try:
            normalized = normalize_structured_tool_definition(tool)
        except ValueError:
            if tool.get("type") != "function":
                continue
            raise

        gemini_tools.append(
            {
                "name": normalized["name"],
                "description": normalized["description"],
                "parameters": _gemini_sanitize_parameters(
                    cast(dict[str, object], normalized["input_schema"]),
                ),
            }
        )
    return gemini_tools


def call_llm_with_gemini_rest(
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call Gemini API directly via REST.

    RFC-0006: Gemini 原生 structured tool adapter

    Gemini 请求体直接从统一消息表示与 neutral structured tool definitions
    生成，不再把 OpenAI schema 作为 structured tool calling 的主中转格式。
    """
    stream_requested = bool(kwargs.pop("stream", False) or getattr(llm_config, "stream", False))
    if not llm_config:
        raise ValueError("llm_config is required for gemini_rest call")

    tools = kwargs.get("tools")

    # Convert messages
    if model_call_params is None:
        raise ValueError("Gemini REST calls require explicit ModelCallParams with UMP messages")

    from nexau.core.adapters.gemini_messages import GeminiMessagesAdapter

    contents, system_instruction = GeminiMessagesAdapter().to_vendor_format(model_call_params.messages)

    # Base URL handling
    base_url = llm_config.base_url.rstrip("/") if llm_config.base_url else ""
    model_name = llm_config.model
    api_key = llm_config.api_key

    # RFC-0003: 根据是否流式选择不同的 endpoint
    endpoint = "streamGenerateContent" if stream_requested else "generateContent"

    if not base_url or "generativelanguage.googleapis.com" in base_url:
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com"
        url = f"{base_url}/v1beta/models/{model_name}:{endpoint}?key={api_key}"
    else:
        url = f"{base_url}/models/{model_name}:{endpoint}?key={api_key}"

    if stream_requested:
        url += "&alt=sse"

    # Construct request body - only include non-None values
    generation_config: dict[str, Any] = {
        "temperature": llm_config.temperature if llm_config.temperature is not None else 0.7,
    }
    if llm_config.max_tokens is not None:
        generation_config["maxOutputTokens"] = llm_config.max_tokens
    if llm_config.top_p is not None:
        generation_config["topP"] = llm_config.top_p

    # Optional: Gemini's topK sampling parameter, taken from extra_params["top_k"]
    top_k = llm_config.extra_params.get("top_k")
    if top_k is not None:
        try:
            generation_config["topK"] = int(top_k)
        except (TypeError, ValueError):
            pass

    request_body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if system_instruction:
        request_body["systemInstruction"] = system_instruction

    if tools:
        gemini_tools = convert_tools_to_gemini(tools)
        if gemini_tools:
            request_body["tools"] = [{"functionDeclarations": gemini_tools}]

    thinking_config = llm_config.extra_params.get("thinkingConfig")
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config

    # Check if tracing is active (there's a current span and we have a tracer)
    should_trace = tracer is not None and get_current_span() is not None

    # RFC-0003: 流式请求路径
    if stream_requested:

        def do_stream_request() -> dict[str, Any]:
            """Execute streaming request and return aggregated response dict.

            RFC-0003: 执行 Gemini REST 流式请求
            """
            aggregator = GeminiRestStreamAggregator()
            _shutdown_ev = model_call_params.shutdown_event if model_call_params else None

            if should_trace and tracer is not None:
                trace_ctx = TraceContext(
                    tracer,
                    "Gemini REST streamGenerateContent",
                    SpanType.LLM,
                    inputs=request_body,
                )
                with trace_ctx:
                    start_time = time.time()
                    first_token_time = None
                    # stream_idle_timeout → requests read timeout (每帧超时)
                    _read_timeout = llm_config.get_stream_idle_timeout()
                    _connect_timeout = llm_config.get_connect_timeout()
                    try:
                        resp = requests.post(
                            url,
                            json=request_body,
                            timeout=(_connect_timeout, _read_timeout),
                            stream=True,
                        )
                        resp.raise_for_status()
                        for chunk_json in _iter_gemini_sse_chunks(resp):
                            if _shutdown_ev is not None and _shutdown_ev.is_set():
                                logger.info(
                                    "🛑 Shutdown event detected during Gemini REST streaming, finalizing partial response",
                                )
                                break
                            if first_token_time is None:
                                first_token_time = time.time()
                            processed_chunk = _process_stream_chunk(
                                chunk_json,
                                middleware_manager,
                                model_call_params,
                            )
                            if processed_chunk is None:
                                continue
                            aggregator.consume(processed_chunk)
                    except requests.exceptions.ReadTimeout as exc:
                        raise StreamIdleTimeoutError(
                            f"Gemini REST stream idle timeout ({_read_timeout}s): {exc}",
                        ) from exc
                    result = aggregator.finalize()
                    trace_ctx.set_outputs(_enrich_gemini_trace_outputs(result, model_name))
                    if first_token_time is not None:
                        trace_ctx.set_attributes(
                            {"time_to_first_token_ms": (first_token_time - start_time) * 1000},
                        )
                    return result
            else:
                _read_timeout = llm_config.get_stream_idle_timeout()
                _connect_timeout = llm_config.get_connect_timeout()
                try:
                    resp = requests.post(
                        url,
                        json=request_body,
                        timeout=(_connect_timeout, _read_timeout),
                        stream=True,
                    )
                    resp.raise_for_status()
                    for chunk_json in _iter_gemini_sse_chunks(resp):
                        if _shutdown_ev is not None and _shutdown_ev.is_set():
                            logger.info(
                                "🛑 Shutdown event detected during Gemini REST streaming, finalizing partial response",
                            )
                            break
                        processed_chunk = _process_stream_chunk(
                            chunk_json,
                            middleware_manager,
                            model_call_params,
                        )
                        if processed_chunk is None:
                            continue
                        aggregator.consume(processed_chunk)
                except requests.exceptions.ReadTimeout as exc:
                    raise StreamIdleTimeoutError(
                        f"Gemini REST stream idle timeout ({_read_timeout}s): {exc}",
                    ) from exc
                return aggregator.finalize()

        try:
            response_json = do_stream_request()
            return ModelResponse.from_gemini_rest(response_json)
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini REST API streaming call failed: {e}")
            if e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Gemini REST API streaming call failed: {e}")
            raise

    # Non-streaming request path
    def do_request() -> dict[str, Any]:
        response = requests.post(
            url,
            json=request_body,
            timeout=(llm_config.get_connect_timeout(), float(llm_config.timeout or 120)),
        )
        response.raise_for_status()
        return response.json()

    # Perform request
    try:
        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "Gemini REST generateContent", SpanType.LLM, inputs=request_body)
            with trace_ctx:
                response_json = do_request()
                trace_ctx.set_outputs(_enrich_gemini_trace_outputs(_to_serializable_dict(response_json), model_name))
        else:
            response_json = do_request()

        return ModelResponse.from_gemini_rest(response_json)
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini REST API call failed: {e}")
        if e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Gemini REST API call failed: {e}")
        raise


async def _iter_gemini_sse_chunks_async(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Async version of _iter_gemini_sse_chunks using httpx.Response.

    P2 async/sync 技术债修复: 异步解析 Gemini 流式响应

    与 sync 版本相同的解析逻辑（SSE 格式 + JSON 数组回退），
    但使用 httpx.Response.aiter_lines() 异步迭代。
    """
    raw_lines: list[str] = []
    yielded_any = False

    async for line in response.aiter_lines():
        if not line:
            continue

        # 1. SSE 格式: 处理 "data:" 前缀行
        if line.startswith("data:"):
            json_str = line[5:].lstrip()
            if not json_str:
                continue
            try:
                chunk = json.loads(json_str)
                yielded_any = True
                yield chunk
            except json.JSONDecodeError:
                logger.warning("Failed to parse Gemini SSE chunk (async): %s", json_str[:200])
            continue

        # 2. 非 SSE 行: 收集用于 JSON 数组回退解析
        raw_lines.append(line)

    # 3. 回退: 如果没有 SSE 数据，尝试将收集的行解析为 JSON 数组
    if not yielded_any and raw_lines:
        body = "\n".join(raw_lines)
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list):
                for item_obj in cast(list[object], parsed):
                    if isinstance(item_obj, dict):
                        yield cast(dict[str, object], item_obj)
            elif isinstance(parsed, dict):
                yield parsed
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini streaming response as JSON (async): %s", body[:500])


async def call_llm_with_gemini_rest_async(
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Async version of call_llm_with_gemini_rest using httpx.AsyncClient.

    P2 async/sync 技术债修复: 异步 Gemini REST API 调用

    使用 httpx.AsyncClient 替代 requests.post，在主事件循环上执行
    Gemini REST API 调用（含流式和非流式），避免阻塞 event loop。
    """
    stream_requested = bool(kwargs.pop("stream", False) or getattr(llm_config, "stream", False))
    tools = kwargs.get("tools")

    # 消息转换
    if model_call_params is None:
        raise ValueError("Gemini REST calls require explicit ModelCallParams with UMP messages")

    from nexau.core.adapters.gemini_messages import GeminiMessagesAdapter

    contents, system_instruction = GeminiMessagesAdapter().to_vendor_format(model_call_params.messages)

    # URL 构建
    base_url = llm_config.base_url.rstrip("/") if llm_config.base_url else ""
    model_name = llm_config.model
    api_key = llm_config.api_key

    if not api_key:
        raise ValueError("API key is required for Gemini REST API")

    endpoint = "streamGenerateContent" if stream_requested else "generateContent"

    if not base_url or "generativelanguage.googleapis.com" in base_url:
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com"
        url = f"{base_url}/v1beta/models/{model_name}:{endpoint}?key={api_key}"
    else:
        url = f"{base_url}/models/{model_name}:{endpoint}?key={api_key}"

    if stream_requested:
        url += "&alt=sse"

    # 请求体构建
    generation_config: dict[str, Any] = {
        "temperature": llm_config.temperature if llm_config.temperature is not None else 0.7,
    }
    if llm_config.max_tokens is not None:
        generation_config["maxOutputTokens"] = llm_config.max_tokens
    if llm_config.top_p is not None:
        generation_config["topP"] = llm_config.top_p

    top_k = llm_config.extra_params.get("top_k")
    if top_k is not None:
        try:
            generation_config["topK"] = int(top_k)
        except (TypeError, ValueError):
            pass

    request_body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_instruction:
        request_body["systemInstruction"] = system_instruction
    if tools:
        gemini_tools = convert_tools_to_gemini(tools)
        if gemini_tools:
            request_body["tools"] = [{"functionDeclarations": gemini_tools}]

    thinking_config = llm_config.extra_params.get("thinkingConfig")
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config

    should_trace = tracer is not None and get_current_span() is not None
    # stream_idle_timeout → httpx read timeout (每帧超时)
    _read_timeout = llm_config.get_stream_idle_timeout()
    _connect_timeout = llm_config.get_connect_timeout()

    if stream_requested:
        aggregator = GeminiRestStreamAggregator()
        _shutdown_ev = model_call_params.shutdown_event if model_call_params else None

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=None, connect=_connect_timeout, read=_read_timeout)) as client:
                if should_trace and tracer is not None:
                    trace_ctx = TraceContext(
                        tracer,
                        "Gemini REST streamGenerateContent (async)",
                        SpanType.LLM,
                        inputs=request_body,
                    )
                    with trace_ctx:
                        start_time = time.time()
                        first_token_time = None
                        async with client.stream("POST", url, json=request_body) as resp:
                            resp.raise_for_status()
                            async for chunk_json in _iter_gemini_sse_chunks_async(resp):
                                if _shutdown_ev is not None and _shutdown_ev.is_set():
                                    logger.info(
                                        "🛑 Shutdown event detected during Gemini REST streaming (async), finalizing partial response",
                                    )
                                    break
                                if first_token_time is None:
                                    first_token_time = time.time()
                                processed_chunk = _process_stream_chunk(
                                    chunk_json,
                                    middleware_manager,
                                    model_call_params,
                                )
                                if processed_chunk is None:
                                    continue
                                aggregator.consume(processed_chunk)
                        result = aggregator.finalize()
                        trace_ctx.set_outputs(_enrich_gemini_trace_outputs(result, model_name))
                        if first_token_time is not None:
                            trace_ctx.set_attributes(
                                {"time_to_first_token_ms": (first_token_time - start_time) * 1000},
                            )
                        return ModelResponse.from_gemini_rest(result)
                else:
                    async with client.stream("POST", url, json=request_body) as resp:
                        resp.raise_for_status()
                        async for chunk_json in _iter_gemini_sse_chunks_async(resp):
                            if _shutdown_ev is not None and _shutdown_ev.is_set():
                                logger.info(
                                    "🛑 Shutdown event detected during Gemini REST streaming (async), finalizing partial response",
                                )
                                break
                            processed_chunk = _process_stream_chunk(
                                chunk_json,
                                middleware_manager,
                                model_call_params,
                            )
                            if processed_chunk is None:
                                continue
                            aggregator.consume(processed_chunk)
                    return ModelResponse.from_gemini_rest(aggregator.finalize())
        except httpx.ReadTimeout as exc:
            raise StreamIdleTimeoutError(
                f"Gemini REST stream idle timeout ({_read_timeout}s, async): {exc}",
            ) from exc

    # Non-streaming async request
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=float(llm_config.timeout or 120), connect=_connect_timeout),
    ) as client:
        try:
            if should_trace and tracer is not None:
                trace_ctx = TraceContext(tracer, "Gemini REST generateContent (async)", SpanType.LLM, inputs=request_body)
                with trace_ctx:
                    response = await client.post(url, json=request_body)
                    response.raise_for_status()
                    response_json = response.json()
                    trace_ctx.set_outputs(_enrich_gemini_trace_outputs(_to_serializable_dict(response_json), model_name))
            else:
                response = await client.post(url, json=request_body)
                response.raise_for_status()
                response_json = response.json()

            return ModelResponse.from_gemini_rest(response_json)
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini REST API call failed (async): {e}")
            logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Gemini REST API call failed (async): {e}")
            raise
