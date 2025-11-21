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
from typing import Any

import openai
from langfuse import get_client, observe

from nexau.archs.llm.llm_config import LLMConfig

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
        llm_config: Any,
        retry_attempts: int = 5,
        middleware_manager: MiddlewareManager | None = None,
    ):
        """Initialize LLM caller.

        Args:
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            retry_attempts: Number of retry attempts for API calls
            middleware_manager: Optional middleware manager for wrapping calls
        """
        self.openai_client = openai_client
        self.llm_config = llm_config
        self.retry_attempts = retry_attempts
        self.middleware_manager = middleware_manager

    def call_llm(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        force_stop_reason: AgentStopReason | None = None,
        agent_state: AgentState | None = None,
        tool_call_mode: str = "xml",
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse | None:
        """Call LLM with the given messages and return normalized response.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens for the response
            tool_call_mode: Tool calling strategy ('xml', 'openai', or 'anthorpic')
            tools: Optional structured tool definitions for the selected mode

        Returns:
            A normalized ModelResponse object containing content and tool calls

        Raises:
            RuntimeError: If OpenAI client is not available or API call fails
        """
        if not self.openai_client and not self.middleware_manager:
            raise RuntimeError(
                "OpenAI client is not available. Please check your API configuration.",
            )

        normalized_mode = normalize_tool_call_mode(tool_call_mode)
        use_structured_tools = normalized_mode in STRUCTURED_TOOL_CALL_MODES

        # Prepare API parameters
        api_params = self.llm_config.to_openai_params()
        api_params["messages"] = messages

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        if normalized_mode == "anthorpic":
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
            api_params = dropper(api_params)

        # Debug logging for LLM messages
        if self.llm_config.debug:
            logger.info("🐛 [DEBUG] LLM Request Messages:")
            for i, msg in enumerate(messages):
                logger.info(
                    f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content']}",
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
            openai_client=self.openai_client,
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

        model_response = response_payload if isinstance(response_payload, ModelResponse) else ModelResponse(content=response_payload)

        if model_response.content:
            from ..utils.xml_utils import XMLUtils

            model_response.content = XMLUtils.restore_closing_tags(model_response.content)

        # Debug logging for LLM response
        if self.llm_config.debug:
            logger.info(f"🐛 [DEBUG] LLM Response: {model_response.render_text()}")

        logger.info(f"💬 LLM Response: {model_response.render_text()}")

        return model_response

    def _call_with_retry(
        self,
        params: ModelCallParams,
    ) -> ModelResponse | str | None:
        """Call OpenAI client with exponential backoff retry."""
        from .executor import AgentStopReason

        force_stop_reason = params.force_stop_reason

        if force_stop_reason != AgentStopReason.SUCCESS:
            logger.info(
                f"🛑 LLM call forced to stop due to {force_stop_reason.name}",
            )
            return None

        backoff = 1
        for i in range(self.retry_attempts):
            try:
                if force_stop_reason != AgentStopReason.SUCCESS:
                    return None
                kwargs = dict(params.api_params)
                response_content = call_llm_with_different_client(
                    self.openai_client,
                    self.llm_config,
                    kwargs,
                    middleware_manager=self.middleware_manager,
                    model_call_params=params,
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


def call_llm_with_different_client(
    client: Any,
    llm_config: LLMConfig,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
) -> ModelResponse:
    """Call LLM with the given messages and return response content."""
    if llm_config.api_type == "anthropic_chat_completion":
        return call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
        )
    elif llm_config.api_type == "openai_responses":
        return call_llm_with_openai_responses(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
        )
    elif llm_config.api_type == "openai_chat_completion":
        return call_llm_with_openai_chat_completion(
            client,
            kwargs,
            middleware_manager=middleware_manager,
            model_call_params=model_call_params,
            llm_config=llm_config,
        )
    else:
        raise ValueError(f"Invalid API type: {llm_config.api_type}")


def openai_to_anthropic_message(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert an OpenAI message to an Anthropic message."""
    system_messages = []
    user_messages = []
    for message in messages:
        if message.get("role") == "system":
            system_messages.append({"type": "text", "text": message.get("content", "")})
        elif message.get("role") == "user":
            user_messages.append({"role": "user", "content": [{"type": "text", "text": message.get("content", "")}]})
        elif message.get("role") == "tool":
            user_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""), "content": message.get("content", "")}
                    ],
                }
            )
        elif message.get("role") == "assistant":
            new_message = {"role": "assistant", "content": []}
            if message.get("content", ""):
                # if content is not empty, add it to the content list
                new_message["content"].append({"type": "text", "text": message.get("content", "")})
            tool_calls = message.get("tool_calls", [])
            new_tool_calls = []
            for tool_call in tool_calls:
                new_tool_calls.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": json.loads(tool_call.get("function", {}).get("arguments", {})),
                    }
                )
            new_message["content"].extend(new_tool_calls)
            user_messages.append(new_message)
        else:
            raise ValueError(f"Invalid message role: {message.get('role')}")
    return system_messages, user_messages


def _strip_responses_api_artifacts(messages: list[Any]) -> list[Any]:
    """Remove Responses API-only artifacts from generic chat messages."""

    sanitized: list[Any] = []

    for message in messages or []:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue

        cleaned = dict(message)
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
) -> ModelResponse:
    """Call Anthropic chat completion with the given messages and return response content."""
    messages = _strip_responses_api_artifacts(kwargs.get("messages", []))
    stream_requested = bool(kwargs.pop("stream", False))

    @observe(name="anthropic_run", as_type="generation")
    def llm_call(messages: list[dict[str, Any]]):
        # 组装 Anthropic 参数
        system_messages, user_messages = openai_to_anthropic_message(messages)
        # set cache control ttl
        if user_messages and user_messages[-1].get("content"):
            content = user_messages[-1]["content"]
            if isinstance(content, list) and content:
                content[0]["cache_control"] = {
                    "type": "ephemeral",
                    "ttl": kwargs.get("anthropic_cache_control_ttl", "5m"),
                }

        new_kwargs = kwargs.copy()
        new_kwargs.pop("messages", None)
        new_kwargs.pop("anthropic_cache_control_ttl", None)

        # 调用 Anthropic
        resp = client.messages.create(system=system_messages, messages=user_messages, **new_kwargs)

        # 获取 usage 详情
        usage_details = None
        if getattr(resp, "usage", None):
            cache_creation_input_tokens = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            cache_read_input_tokens = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            input_tokens = getattr(resp.usage, "input_tokens", 0) or 0
            output_tokens = getattr(resp.usage, "output_tokens", 0) or 0
            total_output_tokens = output_tokens
            usage_details = {
                "input_tokens": input_tokens,
                "output_tokens": total_output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }

        model_name = new_kwargs.get("model") or getattr(resp, "model", None)

        # 更新当前 generation（让前端显示成 LLM 卡片）
        langfuse_client = get_client()
        langfuse_client.update_current_generation(
            model=model_name,
            input=messages,
            usage_details=usage_details,
            metadata={"provider": "anthropic", "api": "messages.create"},
        )
        return resp

    if not stream_requested:
        response = llm_call(messages)
        return ModelResponse.from_anthropic_message(response)

    @observe(name="anthropic_run_stream", as_type="generation")
    def llm_stream_call(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
        system_messages, user_messages = openai_to_anthropic_message(messages)
        if user_messages and user_messages[-1].get("content"):
            content = user_messages[-1]["content"]
            if isinstance(content, list) and content:
                content[0]["cache_control"] = {
                    "type": "ephemeral",
                    "ttl": kwargs.get("anthropic_cache_control_ttl", "5m"),
                }

        new_kwargs = kwargs.copy()
        new_kwargs.pop("messages", None)
        new_kwargs.pop("anthropic_cache_control_ttl", None)

        aggregator = AnthropicStreamAggregator()
        with client.messages.stream(system=system_messages, messages=user_messages, **new_kwargs) as stream:
            for event in stream:
                processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                if processed_event is None:
                    continue
                aggregator.consume(processed_event)
        message_payload = aggregator.finalize()
        return message_payload, aggregator.model_name

    response_payload, agg_model = llm_stream_call(messages)
    langfuse_client = get_client()
    langfuse_client.update_current_generation(
        model=agg_model or kwargs.get("model"),
        input=messages,
        usage_details=None,
        metadata={"provider": "anthropic", "api": "messages.stream"},
    )
    return ModelResponse.from_anthropic_message(response_payload)


def call_llm_with_openai_chat_completion(
    client: openai.OpenAI,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
) -> ModelResponse:
    """Call OpenAI chat completion with the given messages and return response content."""

    messages = _strip_responses_api_artifacts(kwargs.get("messages", []))
    kwargs["messages"] = messages
    stream_requested = bool(kwargs.pop("stream", False) or getattr(llm_config, "stream", False))

    if stream_requested:

        @observe(name="OpenAI-Chat Completion (stream)", as_type="generation")
        def call_llm(payload: dict[str, Any]) -> tuple[dict[str, Any], Any | None, str | None]:
            payload = payload.copy()
            payload["stream"] = True
            aggregator = OpenAIChatStreamAggregator()
            last_chunk: Any | None = None
            with client.chat.completions.create(**payload) as stream:
                for chunk in stream:
                    last_chunk = chunk
                    processed_chunk = _process_stream_chunk(chunk, middleware_manager, model_call_params)
                    if processed_chunk is None:
                        continue
                    aggregator.consume(processed_chunk)
            message_payload = aggregator.finalize()
            return message_payload, last_chunk, aggregator.model_name

        message, last_chunk, agg_model = call_llm(kwargs)
        langfuse_client = get_client()
        model_name = agg_model or (message.get("model") if isinstance(message, dict) else None) or getattr(last_chunk, "model", None)
        langfuse_client.update_current_generation(model=model_name, usage_details=getattr(last_chunk, "usage", None))
        return ModelResponse.from_openai_message(message)

    @observe(name="OpenAI-Chat Completion", as_type="generation")
    def call_llm(kwargs):
        response = client.chat.completions.create(**kwargs)
        langfuse_client = get_client()
        langfuse_client.update_current_generation(model=response.model, usage_details=response.usage)
        return response

    response = call_llm(kwargs)
    message = response.choices[0].message
    return ModelResponse.from_openai_message(message)


def call_llm_with_openai_responses(
    client: Any,
    kwargs: dict[str, Any],
    *,
    middleware_manager: MiddlewareManager | None = None,
    model_call_params: ModelCallParams | None = None,
    llm_config: LLMConfig | None = None,
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

    @observe(name="OpenAI-Responses API", as_type="generation")
    def call_llm(request_payload):
        response = client.responses.create(**request_payload)
        langfuse_client = get_client()
        langfuse_client.update_current_generation(model=response.model, usage_details=response.usage)
        return response

    if not stream_requested:
        return ModelResponse.from_openai_response(call_llm(request_payload))

    @observe(name="OpenAI-Responses API (stream)", as_type="generation")
    def call_llm_stream(payload: dict[str, Any]) -> dict[str, Any]:
        aggregator = OpenAIResponsesStreamAggregator()
        with client.responses.stream(**payload) as stream:
            for event in stream:
                processed_event = _process_stream_chunk(event, middleware_manager, model_call_params)
                if processed_event is None:
                    continue
                aggregator.consume(processed_event)
        return aggregator.finalize()

    response_payload = call_llm_stream(request_payload)
    langfuse_client = get_client()
    langfuse_client.update_current_generation(
        model=response_payload.get("model"),
        usage_details=response_payload.get("usage"),
        metadata={"provider": "openai", "api": "responses.stream"},
    )
    return ModelResponse.from_openai_response(response_payload)


def _prepare_responses_api_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Convert internal message representation into Responses API input items and instructions."""

    prepared: list[dict[str, Any]] = []
    instructions: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

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
                prepared.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": _coerce_tool_output_text(content),
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
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls or []:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            prepared.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                },
            )

        # Include stored reasoning items if available
        reasoning_items = message.get("reasoning")
        if reasoning_items:
            if isinstance(reasoning_items, list):
                prepared.extend(_sanitize_response_items_for_input(reasoning_items))

    joined_instructions = "\n\n".join(part.strip() for part in instructions if part.strip()) or None
    return prepared, joined_instructions


def _normalize_responses_api_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Ensure tool definitions align with the Responses API schema."""

    normalized: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue

        tool_dict = dict(tool)
        tool_type = tool_dict.get("type")

        if tool_type == "function":
            function_spec = tool_dict.get("function")

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

    for item in items:
        if isinstance(item, dict):
            item_copy = dict(item)
            item_copy.pop("status", None)
            if drop_ephemeral_ids:
                item_copy.pop("id", None)
            item_type = item_copy.get("type")
            if item_type == "message":
                item_copy.pop("status", None)
            elif item_type == "function_call_output":
                item_copy["output"] = _coerce_tool_output_text(item_copy.get("output"))
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
        for item in output:
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if text_value:
                    parts.append(str(text_value))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(output, dict):
        text_value = output.get("text") or output.get("content")
        if text_value:
            return str(text_value)
        return json.dumps(output, ensure_ascii=False)

    return str(output)


def _ensure_reasoning_summary(reasoning_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee reasoning items include a summary list."""

    summary_entries = reasoning_item.get("summary")
    sanitized: list[dict[str, Any]] = []

    if isinstance(summary_entries, list):
        for entry in summary_entries:
            if isinstance(entry, dict):
                summary_type = entry.get("type") or "summary_text"
                text = entry.get("text")
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
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if text_value:
                    parts.append(str(text_value))
            elif item:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        text_value = content.get("text")
        if text_value:
            return str(text_value)
        nested = content.get("content")
        if nested:
            return _collapse_message_content_to_text(nested)

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

    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_serializable_dict(payload: Any) -> dict[str, Any]:
    """Convert SDK models into plain dictionaries."""

    if isinstance(payload, dict):
        return payload

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
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

    def consume(self, chunk: Any) -> None:
        payload = _to_serializable_dict(chunk)
        model = payload.get("model")
        if isinstance(model, str):
            self.model_name = model

        choices = payload.get("choices") or []
        for choice in choices:
            choice_dict = _to_serializable_dict(choice)
            delta = _to_serializable_dict(choice_dict.get("delta", {}))

            role = delta.get("role")
            if role:
                self.role = role

            content_delta = delta.get("content")
            if isinstance(content_delta, str):
                self._content_parts.append(content_delta)
            elif isinstance(content_delta, list):
                for entry in content_delta:
                    entry_text = _safe_get(entry, "text")
                    if entry_text:
                        self._content_parts.append(str(entry_text))

            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str):
                self._reasoning_parts.append(reasoning)
            elif isinstance(reasoning, list):
                for entry in reasoning:
                    text = _safe_get(entry, "text")
                    if text:
                        self._reasoning_parts.append(str(text))

            tool_calls = delta.get("tool_calls") or []
            for tool_delta in tool_calls:
                tool_dict = _to_serializable_dict(tool_delta)
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

                function_delta = _to_serializable_dict(tool_dict.get("function", {}))
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
                function_payload = call.get("function") or {}
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

        return message


class AnthropicStreamAggregator:
    """Aggregate Anthropic streaming events into a final message payload."""

    def __init__(self) -> None:
        self.role: str = "assistant"
        self.model_name: str | None = None
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
        elif event_type == "content_block_start":
            index = payload.get("index")
            block = _to_serializable_dict(payload.get("content_block", {}))
            if block:
                block["_input_buffer"] = ""
                self._active_blocks[index] = block
        elif event_type == "content_block_delta":
            self._apply_block_delta(payload)
        elif event_type == "content_block_stop":
            self._finalize_block(payload.get("index"))
        elif event_type == "message_stop":
            self._flush_active_blocks()

    def finalize(self) -> dict[str, Any]:
        self._flush_active_blocks()
        if not self._completed_blocks:
            raise RuntimeError("No stream chunks were received from Anthropic messages stream")
        message = {
            "role": self.role or "assistant",
            "content": self._completed_blocks,
        }
        if self.model_name:
            message["model"] = self.model_name
        return message

    def _apply_block_delta(self, payload: dict[str, Any]) -> None:
        index = payload.get("index")
        delta = _to_serializable_dict(payload.get("delta", {}))
        block = self._active_blocks.setdefault(index, {"type": "text", "text": "", "_input_buffer": ""})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            block["type"] = "text"
            block["text"] = block.get("text", "") + delta.get("text", "")
        elif delta_type == "input_json_delta":
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
        if not isinstance(delta_text, str):
            delta_text = str(delta_text)
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
        if not isinstance(delta, str):
            delta = str(delta)
        self.arguments = f"{self.arguments}{delta}"

    def set_arguments(self, arguments: str) -> None:
        if not arguments:
            return
        if not isinstance(arguments, str):
            arguments = str(arguments)
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
        self.content = list(content or [])
        self._summary_parts: dict[int, dict[str, Any]] = {}
        self._summary_order: list[int] = []
        self._seed_initial_summary(summary)

    def update_from_item(self, item: dict[str, Any]) -> None:
        content = item.get("content")
        if isinstance(content, list):
            self.content = list(content)
        summary = item.get("summary")
        self._seed_initial_summary(summary)

    def _seed_initial_summary(self, summary: list[dict[str, Any]] | None) -> None:
        if not summary or self._summary_parts:
            return
        for idx, part in enumerate(summary):
            entry = dict(part)
            self._summary_parts[idx] = entry
            if idx not in self._summary_order:
                self._summary_order.append(idx)

    def append_summary_delta(self, index: int, delta_text: str) -> None:
        if not isinstance(delta_text, str):
            delta_text = str(delta_text)
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
        for item_id, builder in self._message_builders.items():
            if item_id not in self._output_order:
                output_items.append(builder.to_output_item())
        for item_id, builder in self._tool_builders.items():
            if item_id not in self._output_order:
                output_items.append(builder.to_output_item())
        for item_id, builder in self._reasoning_builders.items():
            if item_id not in self._output_order:
                output_items.append(builder.to_output_item())

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
            builder = self._message_builders.setdefault(item_id, ResponseMessageBuilder(item_id, item.get("role", "assistant")))
            content = item.get("content") or []
            for idx, part in enumerate(content):
                builder.add_part(idx, _to_serializable_dict(part))
        elif item_type == "function_call":
            builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
            builder.update_from_item(item)
        elif item_type == "reasoning":
            builder = self._reasoning_builders.setdefault(
                item_id,
                ReasoningSummaryBuilder(item_id, content=item.get("content"), summary=item.get("summary")),
            )
            builder.update_from_item(item)

    def _handle_content_part_added(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        builder = self._message_builders.setdefault(item_id, ResponseMessageBuilder(item_id))
        builder.add_part(payload.get("content_index", 0), _to_serializable_dict(payload.get("part", {})))

    def _handle_text_delta(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        builder = self._message_builders.setdefault(item_id, ResponseMessageBuilder(item_id))
        delta_text = payload.get("delta", "")
        if not isinstance(delta_text, str):
            delta_text = str(delta_text)
        builder.append_text(payload.get("content_index", 0), delta_text)

    def _handle_function_arguments_delta(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
        builder.append_arguments(payload.get("delta", ""))

    def _handle_function_arguments_done(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id")
        if not item_id:
            return
        builder = self._tool_builders.setdefault(item_id, ResponseToolCallBuilder(item_id))
        builder.set_arguments(payload.get("arguments", ""))

    def _handle_reasoning_delta(self, payload: dict[str, Any]) -> None:
        item_id = payload.get("item_id") or f"_reasoning_{payload.get('output_index', 0)}"
        summary_index = payload.get("summary_index", 0)
        delta = payload.get("delta", "")
        if not isinstance(delta, str):
            delta = str(delta)
        builder = self._reasoning_builders.setdefault(item_id, ReasoningSummaryBuilder(item_id))
        builder.append_summary_delta(summary_index, delta)
