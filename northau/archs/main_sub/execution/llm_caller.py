"""Simple LLM API caller component."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from langfuse import get_client, observe

from northau.archs.llm.llm_config import LLMConfig

from ..agent_state import AgentState
from ..tool_call_modes import STRUCTURED_TOOL_CALL_MODES, normalize_tool_call_mode
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
        custom_llm_generator: (
            Callable[
                [
                    Any,
                    dict[str, Any],
                ],
                Any,
            ]
            | None
        ) = None,
    ):
        """Initialize LLM caller.

        Args:
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            retry_attempts: Number of retry attempts for API calls
            custom_llm_generator: Optional custom LLM generator function
        """
        self.openai_client = openai_client
        self.llm_config = llm_config
        self.retry_attempts = retry_attempts
        self.custom_llm_generator = custom_llm_generator

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
        if not self.openai_client and not self.custom_llm_generator:
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

        # Call LLM with retry
        response_payload = self._call_with_retry(
            force_stop_reason=force_stop_reason,
            agent_state=agent_state,
            **api_params,
        )
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
        """Call OpenAI client or custom LLM generator with exponential backoff retry."""
        from .executor import AgentStopReason

        if force_stop_reason != AgentStopReason.SUCCESS:
            logger.info(
                f"🛑 LLM call forced to stop due to {force_stop_reason.name}",
            )

        backoff = 1
        for i in range(self.retry_attempts):
            try:
                # Use custom LLM generator if provided, otherwise use OpenAI client
                if self.custom_llm_generator:
                    generator_result = self.custom_llm_generator(
                        self.openai_client,
                        self.llm_config,
                        kwargs,
                        force_stop_reason,
                        agent_state,
                    )
                    if generator_result is None:
                        raise Exception("Custom generator produced no response")
                    if isinstance(generator_result, ModelResponse):
                        response_content = generator_result
                    else:
                        response_content = ModelResponse(content=str(generator_result))
                else:
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
    client: Any,
    kwargs: dict[str, Any],
) -> ModelResponse:
    """Call OpenAI chat completion with the given messages and return response content."""
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    return ModelResponse.from_openai_message(message)


def bypass_llm_generator(
    openai_client: Any,
    llm_config: LLMConfig,
    kwargs: dict[str, Any],
    force_stop_reason: str,
    agent_state: AgentState,
) -> ModelResponse:
    """
    Custom LLM generator that does nothing.

    Args:
        openai_client: The OpenAI client instance (can be used or ignored)
        kwargs: The parameters that would be passed to openai_client.chat.completions.create()

    Returns:
        ModelResponse with the generated content and tool calls
    """
    print(
        f"🔧 Custom LLM Generator called with {len(kwargs.get('messages', []))} messages",
    )

    try:
        return call_llm_with_different_client(openai_client, llm_config, kwargs)

    except Exception as e:
        print(f"❌ Bypass LLM generator error: {e}")
        # You could implement custom fallback logic here
        raise
