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

"""Simple LLM API caller component."""

import json
import logging
import time
from collections.abc import Mapping
from typing import Any, Literal, cast

import openai
import requests
from anthropic.types import ToolParam
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.tracer.context import TraceContext, get_current_span
from nexau.archs.tracer.core import BaseTracer, SpanType
from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
from nexau.core.messages import Message

from ..agent_state import AgentState
from ..tool_call_modes import STRUCTURED_TOOL_CALL_MODES, normalize_tool_call_mode
from .hooks import MiddlewareManager, ModelCallParams
from .model_response import ModelResponse
from .stop_reason import AgentStopReason

logger = logging.getLogger(__name__)


class LLMCaller:
    """Handles LLM API calls with retry logic."""

    def __init__(
        self,
        openai_client: Any,
        llm_config: LLMConfig,
        retry_attempts: int = 5,
        middleware_manager: MiddlewareManager | None = None,
        global_storage: Any = None,
    ):
        """Initialize LLM caller.

        Args:
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            retry_attempts: Number of retry attempts for API calls
            middleware_manager: Optional middleware manager for wrapping calls
            global_storage: Optional global storage to retrieve tracer at call time
        """
        self.openai_client = openai_client
        self.llm_config = llm_config
        self.retry_attempts = retry_attempts
        self.middleware_manager = middleware_manager
        self.global_storage = global_storage

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
        tools: list[ChatCompletionToolParam] | list[ToolParam] | None = None,
        openai_client: Any | None = None,
    ) -> ModelResponse | None:
        """Call LLM with the given messages and return normalized response.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens for the response
            tool_call_mode: Tool calling strategy ('xml', 'openai', or 'anthropic')
            tools: Optional structured tool definitions for the selected mode

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

        # Prepare API parameters
        api_params = self.llm_config.to_openai_params()

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        if normalized_mode == "anthropic":
            api_params["tools"] = tools
            api_params.setdefault("tool_choice", {"type": "auto"})

        if tools and normalized_mode == "openai":
            api_params["tools"] = tools
            api_params.setdefault("tool_choice", "auto")

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
                tool_image_policy: Literal["inject_user_message", "embed_in_tool_message"] = (
                    "embed_in_tool_message" if self.llm_config.api_type == "openai_responses" else "inject_user_message"
                )
                kwargs["messages"] = messages_to_legacy_openai_chat(params.messages, tool_image_policy=tool_image_policy)
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
                    raise Exception("No response content or tool calls")

            except Exception as e:
                logger.error(
                    f"❌ LLM call failed (attempt {i + 1}/{self.retry_attempts}): {e}",
                )
                if i == self.retry_attempts - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2
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
            tracer=tracer,
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
    else:
        raise ValueError(f"Invalid API type: {llm_config.api_type}")


def openai_to_anthropic_message(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert NexAU legacy OpenAI-shaped chat messages to Anthropic Messages payload.

    Internally this now uses the vendor-agnostic UMP model:
      legacy dicts -> UMP -> Anthropic content blocks
    """

    from nexau.core.adapters.anthropic_messages import anthropic_payload_from_legacy_openai_chat

    system_messages, user_messages = anthropic_payload_from_legacy_openai_chat(messages)
    return system_messages, user_messages


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
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call Anthropic chat completion with the given messages and return response content."""
    messages = _strip_responses_api_artifacts(kwargs.get("messages", []))
    stream_requested = bool(kwargs.pop("stream", False))

    # Check if tracing is active (there's a current span and we have a tracer)
    should_trace = tracer is not None and get_current_span() is not None

    def llm_call(messages: list[dict[str, Any]]):
        # 组装 Anthropic 参数
        system_messages, user_messages = openai_to_anthropic_message(messages)
        # set cache control ttl
        if user_messages and user_messages[-1].get("content"):
            content = cast(list[dict[str, Any]] | str | None, user_messages[-1].get("content"))
            if isinstance(content, list) and content:
                content[0]["cache_control"] = {
                    "type": "ephemeral",
                    # remove ttl due to litellm incompatibility
                    # "ttl": kwargs.get("anthropic_cache_control_ttl", "5m"),
                }

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
        response = llm_call(messages)
        return ModelResponse.from_anthropic_message(response)

    def llm_stream_call(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
        system_messages, user_messages = openai_to_anthropic_message(messages)
        if user_messages and user_messages[-1].get("content"):
            content = cast(list[dict[str, Any]] | str | None, user_messages[-1].get("content"))
            if isinstance(content, list) and content:
                content[0]["cache_control"] = {
                    "type": "ephemeral",
                    # remove ttl due to litellm incompatibility
                    # "ttl": kwargs.get("anthropic_cache_control_ttl", "5m"),
                }

        new_kwargs: dict[str, Any] = kwargs.copy()
        new_kwargs.pop("messages", None)
        new_kwargs.pop("anthropic_cache_control_ttl", None)

        # Build the exact kwargs for tracing
        api_kwargs: dict[str, Any] = {"system": system_messages, "messages": user_messages, **new_kwargs}

        aggregator = AnthropicStreamAggregator()

        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "Anthropic messages.stream", SpanType.LLM, inputs=api_kwargs)
            with trace_ctx:
                start_time = time.time()
                first_token_time = None
                with client.messages.stream(**api_kwargs) as stream:
                    for event in stream:
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
        else:
            with client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                    if processed_event is None:
                        continue
                    aggregator.consume(processed_event)
            message_payload = aggregator.finalize()
            return message_payload, aggregator.model_name

    response_payload, _ = llm_stream_call(messages)

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

            if should_trace and tracer is not None:
                trace_ctx: TraceContext = TraceContext(tracer, "OpenAI chat.completions.create (stream)", SpanType.LLM, inputs=payload)
                with trace_ctx:
                    start_time = time.time()
                    first_token_time = None
                    stream_ctx: Stream[ChatCompletionChunk] = client.chat.completions.create(
                        stream=True,
                        **payload,
                    )
                    with stream_ctx:
                        for chunk in stream_ctx:
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
            else:
                stream_ctx = client.chat.completions.create(
                    stream=True,
                    **payload,
                )
                with stream_ctx:
                    for chunk in stream_ctx:
                        last_chunk = chunk
                        processed_chunk = _process_stream_chunk(chunk, middleware_manager, model_call_params)
                        if processed_chunk is None:
                            continue
                        aggregator.consume(processed_chunk)
                message_payload_untraced = aggregator.finalize()
                return message_payload_untraced, last_chunk, aggregator.model_name

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
        response_items, instructions = _prepare_responses_api_input(messages)
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
        request_payload["tools"] = _normalize_responses_api_tools(tools)

    stream_requested = bool(request_payload.pop("stream", False) or getattr(llm_config, "stream", False))

    request_payload.pop("store", None)

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

        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "OpenAI responses.stream", SpanType.LLM, inputs=payload)
            start_time = time.time()
            first_token_time = None
            with trace_ctx:
                with client.responses.stream(**payload) as stream:
                    for event in stream:
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
        else:
            with client.responses.stream(**payload) as stream:
                for event in stream:
                    processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                    if processed_event is None:
                        continue
                    aggregator.consume(processed_event)
            return aggregator.finalize()

    response_payload = call_llm_stream(request_payload)
    return ModelResponse.from_openai_response(response_payload)


def _prepare_responses_api_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Convert internal message representation into Responses API input items and instructions."""

    prepared: list[dict[str, Any]] = []
    instructions: list[str] = []

    for message in messages:
        # If the message already carries raw response items, reuse them directly
        response_items = message.get("response_items")
        if response_items:
            prepared.extend(_sanitize_response_items_for_input(response_items, drop_ephemeral_ids=True))
            continue

        role = message.get("role", "user")
        content = message.get("content", "") or ""

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                output_value: Any = None
                if isinstance(content, list):
                    out_items: list[dict[str, Any]] = []
                    for part in cast(list[Any], content):
                        if not isinstance(part, Mapping):
                            continue
                        part_map = cast(Mapping[str, Any], part)
                        part_type = str(part_map.get("type") or "")

                        if part_type in {"text", "output_text", "input_text"}:
                            text_val: Any = part_map.get("text") or part_map.get("content")
                            out_items.append({"type": "input_text", "text": str(text_val or "")})
                            continue

                        if part_type == "image_url":
                            image_url_any = part_map.get("image_url")
                            url: str | None = None
                            detail: str | None = None
                            if isinstance(image_url_any, Mapping):
                                image_url_map = cast(Mapping[str, Any], image_url_any)
                                url_any = image_url_map.get("url")
                                if isinstance(url_any, str) and url_any.strip():
                                    url = url_any.strip()
                                detail_any = image_url_map.get("detail")
                                if isinstance(detail_any, str) and detail_any in {"low", "high", "auto"}:
                                    detail = detail_any
                            elif isinstance(image_url_any, str) and image_url_any.strip():
                                url = image_url_any.strip()

                            if url:
                                payload: dict[str, Any] = {"type": "input_image", "image_url": url}
                                if detail and detail != "auto":
                                    payload["detail"] = detail
                                out_items.append(payload)
                            continue

                    # Use array output if it contains any images; otherwise keep legacy string coercion.
                    if any(item.get("type") == "input_image" for item in out_items):
                        output_value = out_items

                if output_value is None:
                    output_value = _coerce_tool_output_text(content)

                prepared.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": output_value,
                    },
                )
            continue

        if role == "system":
            instruction_text = _collapse_message_content_to_text(content)
            if instruction_text:
                instructions.append(instruction_text)
            continue

        # Build message item for standard roles
        if role == "user":
            text_type = "input_text"
        else:
            text_type = "output_text"

        content_parts: list[dict[str, Any]] = []
        if content:
            content_parts.append({"type": text_type, "text": str(content)})

        message_item: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": content_parts,
        }

        prepared.append(message_item)

        # Reconstruct tool calls if present on assistant messages
        tool_calls_payload_raw = message.get("tool_calls")
        if isinstance(tool_calls_payload_raw, list):
            tool_calls_payload_list: list[Any] = cast(list[Any], tool_calls_payload_raw)
            for tc in tool_calls_payload_list:
                if not isinstance(tc, Mapping):
                    continue
                typed_tool_call: Mapping[str, Any] = cast(Mapping[str, Any], tc)
                tool_call_dict: dict[str, Any] = {}
                for key_any, value_any in typed_tool_call.items():
                    key: str = str(key_any)
                    value: Any = value_any
                    tool_call_dict[key] = value

                assert tool_call_dict.get("function") is not None, "Tool call dict must contain a function"
                assert isinstance(tool_call_dict.get("function"), dict), "Function must be a dict"
                function = cast(dict[str, Any], tool_call_dict.get("function"))
                assert function.get("name") is not None, "Function must contain a name"
                assert function.get("arguments") is not None, "Function must contain arguments"
                assert isinstance(function.get("name"), str), "Name must be a string"
                assert isinstance(function.get("arguments"), dict), "Arguments must be a dict"

                prepared.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call_dict.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                    },
                )

        # Include stored reasoning items if available
        reasoning_items = message.get("reasoning")
        if reasoning_items:
            if isinstance(reasoning_items, list):
                prepared.extend(_sanitize_response_items_for_input(cast(list[Any], reasoning_items)))

    joined_instructions = "\n\n".join(part.strip() for part in instructions if part.strip()) or None
    return prepared, joined_instructions


def _normalize_responses_api_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Ensure tool definitions align with the Responses API schema."""

    normalized: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, Mapping):
            normalized.append(tool)
            continue

        tool_mapping = cast(Mapping[str, Any], tool)
        tool_dict: dict[str, Any] = dict(tool_mapping)
        tool_type = tool_dict.get("type")

        if tool_type == "function":
            function_spec = cast(dict[str, Any] | None, tool_dict.get("function"))

            if isinstance(function_spec, dict):
                name = function_spec.get("name")
                description = function_spec.get("description")
                parameters = function_spec.get("parameters")
                strict = function_spec.get("strict")

                if name and not tool_dict.get("name"):
                    tool_dict["name"] = name
                if description and not tool_dict.get("description"):
                    tool_dict["description"] = description
                if parameters is not None and "parameters" not in tool_dict:
                    tool_dict["parameters"] = parameters
                if strict is not None and "strict" not in tool_dict:
                    tool_dict["strict"] = strict

            # The Responses API expects function tools to specify the name at the top level
            # and uses the Chat Completions style schema for parameters/description.
            if tool_dict.get("name"):
                tool_dict.pop("function", None)

        normalized.append(tool_dict)

    return normalized


def _sanitize_response_items_for_input(items: list[Any], *, drop_ephemeral_ids: bool = False) -> list[dict[str, Any]]:
    """Strip response-only fields that the Responses API rejects on input."""

    sanitized: list[dict[str, Any]] = []

    def _is_multimodal_output_list(value: Any) -> bool:
        if not isinstance(value, list):
            return False
        for item in cast(list[Any], value):
            if not isinstance(item, Mapping):
                continue
            item_dict = cast(Mapping[str, Any], item)
            item_type = str(item_dict.get("type") or "")
            if item_type in {"input_image", "image", "image_url", "input_file"}:
                return True
        return False

    for item in items:
        if isinstance(item, Mapping):
            item_mapping = cast(Mapping[str, Any], item)
            item_copy: dict[str, Any] = dict(item_mapping)
            item_copy.pop("status", None)
            if drop_ephemeral_ids:
                item_copy.pop("id", None)
            item_type = item_copy.get("type")
            if item_type == "message":
                item_copy.pop("status", None)
            elif item_type == "function_call_output":
                output_value = item_copy.get("output")
                # Preserve multimodal tool outputs as arrays; otherwise coerce to string for legacy compatibility.
                if _is_multimodal_output_list(output_value):
                    cleaned: list[dict[str, Any]] = []
                    for out_item in cast(list[Any], output_value):
                        if not isinstance(out_item, Mapping):
                            continue
                        out_dict = dict(cast(Mapping[str, Any], out_item))
                        out_type = str(out_dict.get("type") or "")
                        if out_type == "output_text":
                            cleaned.append({"type": "input_text", "text": str(out_dict.get("text") or "")})
                            continue
                        if out_type == "image_url":
                            image_url_any = out_dict.get("image_url")
                            url: str | None = None
                            detail: str | None = None
                            if isinstance(image_url_any, Mapping):
                                image_url_map = cast(Mapping[str, Any], image_url_any)
                                url_any = image_url_map.get("url")
                                if isinstance(url_any, str) and url_any.strip():
                                    url = url_any.strip()
                                detail_any = image_url_map.get("detail")
                                if isinstance(detail_any, str) and detail_any in {"low", "high", "auto"}:
                                    detail = detail_any
                            elif isinstance(image_url_any, str) and image_url_any.strip():
                                url = image_url_any.strip()
                            if url:
                                payload: dict[str, Any] = {"type": "input_image", "image_url": url}
                                if detail and detail != "auto":
                                    payload["detail"] = detail
                                cleaned.append(payload)
                            continue
                        cleaned.append(out_dict)
                    item_copy["output"] = cleaned
                else:
                    item_copy["output"] = _coerce_tool_output_text(output_value)
            elif item_type == "reasoning":
                item_copy.pop("id", None)
                item_copy["summary"] = _ensure_reasoning_summary(item_copy)
        else:
            sanitized.append(item)
            continue

        sanitized.append(item_copy)

    return sanitized


def _coerce_tool_output_text(output: Any) -> str:
    """Convert arbitrary tool output into Responses-compatible string."""

    if output is None:
        return ""

    if isinstance(output, list):
        # Responses API rejects nested output_text objects; join textual parts instead
        parts: list[str] = []
        output_list: list[Any] = cast(list[Any], output)
        for item in output_list:
            if isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                item_type = str(item_dict.get("type") or "")
                if item_type in {"image_url", "input_image", "image"}:
                    parts.append("<image>")
                    continue
                list_text_value: Any = item_dict.get("text") or item_dict.get("content")
                if list_text_value is not None:
                    parts.append(str(list_text_value))
            else:
                parts.append(str(item))
        # Ensure non-empty output so tool results aren't silently dropped.
        rendered = "\n".join(parts)
        return rendered if rendered.strip() else "<tool_output>"

    if isinstance(output, dict):
        output_dict = cast(dict[str, Any], output)
        dict_text_value: Any = output_dict.get("text") or output_dict.get("content")
        if dict_text_value is not None:
            return str(dict_text_value)
        return json.dumps(output, ensure_ascii=False)

    return str(output)


def _ensure_reasoning_summary(reasoning_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee reasoning items include a summary list."""

    summary_entries = reasoning_item.get("summary")
    sanitized: list[dict[str, Any]] = []

    if isinstance(summary_entries, list):
        summary_entries_list: list[Any] = cast(list[Any], summary_entries)
        for entry in summary_entries_list:
            if isinstance(entry, dict):
                entry_dict = cast(dict[str, Any], entry)
                summary_type: str = cast(str, entry_dict.get("type") or "summary_text")
                text = entry_dict.get("text")
                sanitized.append({"type": summary_type, "text": str(text or "")})
            elif entry is not None:
                sanitized.append({"type": "summary_text", "text": str(entry)})

    if not sanitized:
        fallback_text = _collapse_message_content_to_text(reasoning_item.get("content"))
        sanitized.append({"type": "summary_text", "text": fallback_text or ""})

    return sanitized


def _collapse_message_content_to_text(content: Any) -> str:
    """Render potential structured message content into a plain-text instruction."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        content_list: list[Any] = cast(list[Any], content)
        for item in content_list:
            if isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                list_text_value: Any = item_dict.get("text") or item_dict.get("content")
                if list_text_value is not None:
                    parts.append(str(list_text_value))
            elif item:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        content_dict = cast(dict[str, Any], content)
        dict_text_value: Any = content_dict.get("text")
        if dict_text_value is not None:
            return str(dict_text_value)
        nested = content_dict.get("content")
        if nested:
            return _collapse_message_content_to_text(nested)
        return json.dumps(content_dict)

    return str(content)


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
            # Update usage information from delta events
            delta = _to_serializable_dict(payload.get("delta", {}))
            usage_data = delta.get("usage") or payload.get("usage")
            if usage_data:
                self.usage = _to_serializable_dict(usage_data)
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
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif delta_type == "input_json_delta":
            block.setdefault("type", "tool_use")
            fragment = delta.get("partial_json", "")
            block["_input_buffer"] = block.get("_input_buffer", "") + fragment
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
                block["input"] = input_buffer
        self._completed_blocks.append(block)

    def _flush_active_blocks(self) -> None:
        remaining = list(self._active_blocks.keys())
        for idx in remaining:
            self._finalize_block(idx)


class ResponseMessageBuilder:
    """Helper for assembling streamed Responses API assistant messages."""

    def __init__(self, item_id: str, role: str = "assistant") -> None:
        self.item_id = item_id
        self.role = role
        self._parts: dict[int, dict[str, Any]] = {}

    def add_part(self, index: int, part: dict[str, Any]) -> None:
        self._parts[index] = part

    def append_text(self, index: int, delta_text: str) -> None:
        part = self._parts.setdefault(index, {"type": "output_text", "text": ""})
        part["type"] = part.get("type") or "output_text"
        part["text"] = (part.get("text") or "") + delta_text

    def to_output_item(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for index in sorted(self._parts):
            part = self._parts[index]
            content.append(part)
        return {
            "type": "message",
            "role": self.role or "assistant",
            "id": self.item_id,
            "content": content,
        }


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
    ) -> None:
        self.item_id = item_id
        self.content: list[dict[str, Any]] = list(content or [])
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
        if self.content:
            item["content"] = self.content
        if summary_parts:
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
            message_builder = self._message_builders.setdefault(item_id, ResponseMessageBuilder(item_id, item.get("role", "assistant")))
            content_list: list[Any] = item.get("content") or []
            for idx, part in enumerate(content_list):
                message_builder.add_part(idx, _to_serializable_dict(part))
        elif item_type == "function_call":
            tool_builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
            tool_builder.update_from_item(item)
        elif item_type == "reasoning":
            reasoning_builder = self._reasoning_builders.setdefault(
                item_id,
                ReasoningSummaryBuilder(item_id, content=item.get("content"), summary=item.get("summary")),
            )
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


def _gemini_sanitize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Keep only type, properties, required for Gemini (strip $schema, additionalProperties, etc.)."""
    allowed = {"type", "properties", "required"}
    return {k: v for k, v in params.items() if k in allowed}


def convert_tools_to_gemini(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Gemini function declarations."""
    gemini_tools: list[dict[str, Any]] = []
    for tool in tools:
        tool_dict = cast(dict[str, Any], tool)
        if tool_dict.get("type") == "function":
            func = cast(dict[str, Any], tool_dict.get("function", {}))
            raw_params = func.get("parameters", {"type": "object", "properties": {}})
            gemini_tools.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "parameters": _gemini_sanitize_parameters(raw_params),
                }
            )
    return gemini_tools


def openai_to_gemini_rest_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Convert OpenAI-style messages to Gemini REST API format."""
    gemini_contents: list[dict[str, Any]] = []
    system_instruction: dict[str, Any] | None = None
    # Track function names from last assistant tool_calls for filling functionResponse.name (tool messages have no name)
    last_model_function_names: list[str] = []
    tool_result_index = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "system":
            system_instruction = {"parts": [{"text": content}]}
            continue

        parts: list[dict[str, Any]] = []
        if content:
            parts.append({"text": content})

        if role == "assistant":
            if tool_calls:
                tool_calls_list = cast(list[dict[str, Any]], tool_calls)
                last_model_function_names = [str(cast(dict[str, Any], tc.get("function", {})).get("name", "")) for tc in tool_calls_list]
                tool_result_index = 0
                # Gemini 3 Requirement: The first functionCall in a turn needs the thought_signature
                # Check for top-level signature or OpenAI-compatible nested signature
                thought_sig: str | None = cast(str | None, msg.get("thought_signature"))
                if not thought_sig:
                    # Fallback: check first tool_call for nested extra_content.google.thought_signature
                    first_tc: dict[str, Any] = tool_calls_list[0] if tool_calls_list else {}
                    extra_content = cast(dict[str, Any], first_tc.get("extra_content", {}))
                    google_content = cast(dict[str, Any], extra_content.get("google", {}))
                    thought_sig = cast(str | None, google_content.get("thought_signature"))

                for i, tc in enumerate(tool_calls_list):
                    func = cast(dict[str, Any], tc.get("function", {}))
                    args: dict[str, Any] | str = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = cast(dict[str, Any], json.loads(args))
                        except json.JSONDecodeError:
                            args = {}

                    fc_part: dict[str, Any] = {"functionCall": {"name": func.get("name"), "args": args}}
                    if i == 0 and thought_sig:
                        fc_part["thoughtSignature"] = thought_sig
                    parts.append(fc_part)
            elif parts and msg.get("thought_signature"):
                # Handle cases where thought_signature is present but no tool_calls (rare for Gemini function calling flow)
                parts[-1]["thoughtSignature"] = msg.get("thought_signature")

            gemini_contents.append({"role": "model", "parts": parts})

        elif role == "user":
            gemini_contents.append({"role": "user", "parts": parts})

        elif role == "tool":
            # Fill name from last assistant's tool_calls by order (tool messages from legacy have no name)
            func_name: str | None = None
            if tool_result_index < len(last_model_function_names):
                func_name = last_model_function_names[tool_result_index] or None
            tool_result_index += 1
            if func_name is None:
                func_name = msg.get("name")
                if func_name is None:
                    raise ValueError("Function name is required for tool result messages")
            resp_part = {"functionResponse": {"name": func_name, "response": {"result": content}}}
            if gemini_contents and gemini_contents[-1]["role"] == "user":
                # Check if it's already a tool response block
                if any("functionResponse" in p for p in gemini_contents[-1]["parts"]):
                    gemini_contents[-1]["parts"].append(resp_part)
                else:
                    gemini_contents.append({"role": "user", "parts": [resp_part]})
            else:
                gemini_contents.append({"role": "user", "parts": [resp_part]})

    return gemini_contents, system_instruction


def call_llm_with_gemini_rest(
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
    tracer: BaseTracer | None = None,
) -> ModelResponse:
    """Call Gemini API directly via REST."""
    from nexau.core.messages import ImageBlock

    # TODO: Support image input for Gemini REST API
    if model_call_params is not None:
        for msg in model_call_params.messages:
            for block in msg.content:
                if isinstance(block, ImageBlock):
                    raise ValueError("Image input is not supported for Gemini REST API")
    # TODO: Support stream for Gemini REST API
    stream_requested = bool(kwargs.pop("stream", False) or getattr(llm_config, "stream", False))
    if stream_requested:
        raise ValueError("Stream is not supported for Gemini REST API")
    if not llm_config:
        raise ValueError("llm_config is required for gemini_rest call")

    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools")

    # Convert messages
    contents, system_instruction = openai_to_gemini_rest_messages(messages)

    # Base URL handling
    base_url = llm_config.base_url.rstrip("/") if llm_config.base_url else ""
    model_name = llm_config.model
    api_key = llm_config.api_key

    if not base_url or "generativelanguage.googleapis.com" in base_url:
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com"
        url = f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
    else:
        url = f"{base_url}/models/{model_name}:generateContent?key={api_key}"

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

    def do_request() -> dict[str, Any]:
        response = requests.post(url, json=request_body, timeout=llm_config.timeout or 120)
        response.raise_for_status()
        return response.json()

    # Perform request
    try:
        if should_trace and tracer is not None:
            trace_ctx = TraceContext(tracer, "Gemini REST generateContent", SpanType.LLM, inputs=request_body)
            with trace_ctx:
                response_json = do_request()
                trace_ctx.set_outputs(_to_serializable_dict(response_json))
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
