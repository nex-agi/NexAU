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
        max_tokens: int,
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
            return self._call_with_retry(
                force_stop_reason=params.force_stop_reason,
                agent_state=params.agent_state,
                **params.api_params,
            )

        call_fn = base_call
        if self.middleware_manager:
            call_fn = self.middleware_manager.wrap_model_call(call_fn)

        response_payload = call_fn(model_call_params)
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
        force_stop_reason: AgentStopReason | None = None,
        agent_state: AgentState | None = None,
        **kwargs: Any,
    ) -> ModelResponse | str | None:
        """Call OpenAI client with exponential backoff retry."""
        from .executor import AgentStopReason

        if force_stop_reason != AgentStopReason.SUCCESS:
            logger.info(
                f"🛑 LLM call forced to stop due to {force_stop_reason.name}",
            )

        backoff = 1
        for i in range(self.retry_attempts):
            try:
                if force_stop_reason != AgentStopReason.SUCCESS:
                    return None
                response_content = call_llm_with_different_client(self.openai_client, self.llm_config, kwargs)

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
) -> ModelResponse:
    """Call LLM with the given messages and return response content."""
    if llm_config.api_type == "anthropic_chat_completion":
        return call_llm_with_anthropic_chat_completion(client, kwargs)
    elif llm_config.api_type == "openai_responses":
        return call_llm_with_openai_responses(client, kwargs)
    elif llm_config.api_type == "openai_chat_completion":
        return call_llm_with_openai_chat_completion(client, kwargs)
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


def call_llm_with_anthropic_chat_completion(
    client: Any,
    kwargs: dict[str, Any],
) -> ModelResponse:
    """Call Anthropic chat completion with the given messages and return response content."""
    messages = kwargs.get("messages", [])

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

    response = llm_call(messages)
    return ModelResponse.from_anthropic_message(response)


def call_llm_with_openai_chat_completion(
    client: openai.OpenAI,
    kwargs: dict[str, Any],
) -> ModelResponse:
    """Call OpenAI chat completion with the given messages and return response content."""

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
) -> ModelResponse:
    """Call OpenAI Responses API and normalize the outcome."""

    request_payload = kwargs.copy()

    messages = request_payload.pop("messages", None)
    if messages is not None:
        response_items = _prepare_responses_api_input(messages)
        request_payload.setdefault("input", response_items)

    # Responses API uses max_output_tokens instead of max_tokens
    max_tokens = request_payload.pop("max_tokens", None)
    if max_tokens is not None:
        request_payload.setdefault("max_output_tokens", max_tokens)

    tools = request_payload.get("tools")
    if tools:
        request_payload["tools"] = _normalize_responses_api_tools(tools)

    @observe(name="OpenAI-Responses API", as_type="generation")
    def call_llm(request_payload):
        response = client.responses.create(**request_payload)
        langfuse_client = get_client()
        langfuse_client.update_current_generation(model=response.model, usage_details=response.usage)
        return response

    return ModelResponse.from_openai_response(call_llm(request_payload))


def _prepare_responses_api_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal message representation into Responses API input items."""

    prepared: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        # If the message already carries raw response items, reuse them directly
        response_items = message.get("response_items")
        if response_items:
            prepared.extend(_sanitize_response_items_for_input(response_items))
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

        # Build message item for standard roles
        if role in {"user", "system"}:
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
                prepared.extend(reasoning_items)

    return prepared


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


def _sanitize_response_items_for_input(items: list[Any]) -> list[dict[str, Any]]:
    """Strip response-only fields that the Responses API rejects on input."""

    sanitized: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            item_copy = dict(item)
            item_copy.pop("status", None)
            item_type = item_copy.get("type")
            if item_type == "message":
                item_copy.pop("status", None)
            elif item_type == "function_call_output":
                item_copy["output"] = _coerce_tool_output_text(item_copy.get("output"))
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


def bypass_llm_generator(
    openai_client: Any,
    llm_config: LLMConfig,
    kwargs: dict[str, Any],
    force_stop_reason: AgentStopReason | None,
    agent_state: AgentState | None,
) -> ModelResponse | None:
    """Legacy helper that directly proxies an OpenAI request with extra logging."""

    message_count = len(kwargs.get("messages", []) or [])
    print(f"Custom LLM Generator called with {message_count} messages")

    if force_stop_reason and force_stop_reason != AgentStopReason.SUCCESS:
        print(f"Bypass LLM generator aborted due to {force_stop_reason.name}")
        return None

    try:
        response = call_llm_with_different_client(openai_client, llm_config, kwargs)
    except Exception as exc:
        print(f"Bypass LLM generator error: {exc}")
        raise

    return response
