"""Token trace session for token-generate providers.

RFC-0009: Token Generate Provider And Trace Memory

Maintains a growing token buffer across model/tool turns and a same-length
response mask where model-generated tokens are marked as 1 and all other
tokens are marked as 0.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import requests

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.tool.tool import StructuredToolDefinitionLike, structured_tool_definition_to_openai
from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
from nexau.core.messages import Message, Role

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class _ChatTemplateTokenizer(Protocol):
    """Protocol for tokenizers exposing apply_chat_template."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool,
        return_dict: bool,
        add_generation_prompt: bool = False,
        tools: list[dict[str, object]] | None = None,
    ) -> object: ...


def _coerce_text(value: object) -> str | None:
    """Normalize a detokenized text payload."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-safe copy of tool call summaries."""
    try:
        payload = json.loads(json.dumps(tool_calls, ensure_ascii=False))
    except (TypeError, ValueError):
        payload = []
    return cast(list[dict[str, Any]], payload) if isinstance(payload, list) else []


def _copy_mapping(value: object) -> dict[str, Any]:
    """Return a shallow dict copy when the input is mapping-like."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in cast(Mapping[object, Any], value).items()}
    return {}


def _empty_int_list() -> list[int]:
    return []


def _empty_dict_list() -> list[dict[str, Any]]:
    return []


def _empty_message_list() -> list[Message]:
    return []


class TokenTraceContextOverflowError(RuntimeError):
    """Token trace session 的 token 总数超过了 max_context_tokens。

    Token trace sessions do not support context compaction because
    reconstructing token lists from compacted message text may be inaccurate.
    """


@dataclass
class TokenTraceSession:
    """Stateful token buffer used by `generate_with_token`."""

    llm_config: LLMConfig
    max_context_tokens: int | None = None
    token_ids: list[int] = field(default_factory=_empty_int_list)
    response_mask: list[int] = field(default_factory=_empty_int_list)
    round_traces: list[dict[str, Any]] = field(default_factory=_empty_dict_list)
    token_provider_usage: list[dict[str, Any]] = field(default_factory=_empty_dict_list)
    synced_message_count: int = 0
    _pending_tool_messages: list[Message] = field(default_factory=_empty_message_list, init=False, repr=False)
    _hf_tokenizer: PreTrainedTokenizerBase | None = field(default=None, init=False, repr=False)

    @property
    def model(self) -> str:
        if not self.llm_config.model:
            raise ValueError("llm_config.model is required for token trace session")
        return self.llm_config.model

    @property
    def timeout(self) -> float:
        raw_timeout = self.llm_config.timeout
        return float(raw_timeout) if raw_timeout is not None else 60.0

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        raw_extra_headers = self.llm_config.get_param("extra_headers", {})
        if isinstance(raw_extra_headers, dict):
            for key, value in cast(dict[str, object], raw_extra_headers).items():
                headers[str(key)] = str(value)
        if self.llm_config.api_key:
            headers.setdefault("Authorization", f"Bearer {self.llm_config.api_key}")
        return headers

    def _build_url(self, param_name: str, default_path: str) -> str:
        base_url = (self.llm_config.base_url or "").rstrip("/")
        if not base_url:
            raise ValueError("llm_config.base_url is required for token trace session")

        custom_path = self.llm_config.get_param(param_name)
        if isinstance(custom_path, str) and custom_path.strip():
            if custom_path.startswith("http://") or custom_path.startswith("https://"):
                return custom_path
            path = custom_path
        else:
            path = default_path

        return f"{base_url}/{path.lstrip('/')}"

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            url,
            json=payload,
            headers=self._build_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object from {url}, got {type(data).__name__}")
        return cast(dict[str, Any], data)

    def _messages_to_legacy(self, messages: list[Message]) -> list[dict[str, Any]]:
        return messages_to_legacy_openai_chat(messages, tool_image_policy="inject_user_message")

    def _get_hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """Lazily load HuggingFace tokenizer from llm_config.tokenizer_path."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer

        tokenizer_path = self.llm_config.tokenizer_path
        if not tokenizer_path:
            raise ValueError(
                "llm_config.tokenizer_path is required for api_type='generate_with_token'. "
                "Pass a HuggingFace model path, e.g. tokenizer_path='meta-llama/Llama-3.1-8B-Instruct'"
            )

        from transformers import AutoTokenizer

        logger.info("Loading HuggingFace tokenizer from '%s'", tokenizer_path)
        auto_tokenizer_cls: Any = AutoTokenizer
        loaded_tokenizer = auto_tokenizer_cls.from_pretrained(tokenizer_path)
        if loaded_tokenizer is None:
            raise ValueError(f"Failed to load tokenizer from '{tokenizer_path}'")
        self._hf_tokenizer = cast("PreTrainedTokenizerBase", loaded_tokenizer)
        return self._hf_tokenizer

    def tokenize_messages(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = False,
        tools: Sequence[StructuredToolDefinitionLike] | None = None,
    ) -> list[int]:
        """Encode messages into token ids via HuggingFace AutoTokenizer.

        使用 llm_config.tokenizer_path 指定的 HF 模型路径加载 tokenizer，
        通过 apply_chat_template 将 Message 列表编码为 token id 列表。
        """
        if not messages:
            return []

        tokenizer = cast(_ChatTemplateTokenizer, self._get_hf_tokenizer())
        legacy_messages = self._messages_to_legacy(messages)
        openai_tools = [structured_tool_definition_to_openai(tool) for tool in tools] if tools is not None else None
        result = tokenizer.apply_chat_template(
            conversation=cast(list[dict[str, object]], legacy_messages),
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            tools=cast(list[dict[str, object]] | None, openai_tools),
        )
        if not isinstance(result, Mapping):
            raise TypeError(f"Expected mapping from apply_chat_template, got {type(result).__name__}")
        result_mapping = cast(Mapping[str, object], result)
        input_ids = result_mapping.get("input_ids")
        if not isinstance(input_ids, list):
            raise TypeError(f"Expected list[int] from apply_chat_template, got {type(input_ids).__name__}")
        return cast(list[int], input_ids)

    def detokenize(self, token_ids: list[int]) -> str:
        """Decode token ids with the backend detokenizer."""
        if not token_ids:
            return ""

        url = self._build_url("detokenize_path", "/detokenize")
        payload = {
            "model": self.model,
            "token_ids": token_ids,
        }
        data = self._post_json(url, payload)
        text = _coerce_text(data.get("text"))
        if text is None:
            raise ValueError("detokenize response missing text")
        return text

    def build_generate_with_token_kwargs(
        self,
        *,
        max_output_tokens: int | None,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs for `client.generate_with_token(...)`."""
        self._flush_pending_tool_messages()
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input_ids": list(self.token_ids),
        }

        sampling_params = _copy_mapping(request_params.get("sampling_params"))
        if max_output_tokens is not None:
            sampling_params["max_new_tokens"] = max_output_tokens

        sampling_param_keys = (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "min_new_tokens",
            "stop",
            "stop_token_ids",
            "stop_regex",
            "n",
            "ignore_eos",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "no_stop_trim",
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "custom_params",
        )
        for key in sampling_param_keys:
            if key in request_params and request_params[key] is not None:
                sampling_params[key] = request_params[key]

        if sampling_params:
            request_kwargs["sampling_params"] = sampling_params

        top_level_keys = (
            "stream",
            "rid",
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "token_ids_logprob",
            "return_text_in_logprobs",
            "lora_path",
            "custom_logit_processor",
            "return_hidden_states",
            "return_routed_experts",
        )
        for key in top_level_keys:
            if key in request_params and request_params[key] is not None:
                request_kwargs[key] = request_params[key]

        return request_kwargs

    def initialize_from_messages(
        self,
        messages: list[Message],
        *,
        tools: Sequence[StructuredToolDefinitionLike] | None = None,
    ) -> None:
        """Initialize the session from the first full message trace."""
        if self.synced_message_count > 0:
            return
        initial_tokens = self.tokenize_messages(
            messages,
            add_generation_prompt=True,
            tools=tools,
        )
        self._append_tokens(initial_tokens, mask_value=0)
        self.synced_message_count = len(messages)

    def sync_external_messages(self, messages: list[Message]) -> None:
        """Append newly-added non-model messages before the next model call."""
        if len(messages) <= self.synced_message_count:
            self._flush_pending_tool_messages()
            return
        trailing_messages = messages[self.synced_message_count :]
        assistant_messages = [message for message in trailing_messages if message.role == Role.ASSISTANT]
        if assistant_messages:
            raise ValueError("sync_external_messages does not accept assistant messages")
        external_messages = trailing_messages
        if external_messages:
            self.append_messages(external_messages, mask_value=0)
        self._flush_pending_tool_messages()

    def append_messages(self, messages: list[Message], *, mask_value: int) -> list[int]:
        """Tokenize a message fragment and append it to the session."""
        if not messages:
            return []

        token_ids: list[int]
        if mask_value != 0:
            self._flush_pending_tool_messages()
            token_ids = self.tokenize_messages(messages, add_generation_prompt=True)
            self._append_tokens(token_ids, mask_value=mask_value)
            self.synced_message_count += len(messages)
            return token_ids

        combined_messages = [*self._pending_tool_messages, *messages]
        processable_messages, trailing_tool_messages = self._split_trailing_tool_messages(combined_messages)
        token_ids = []
        if processable_messages:
            token_ids = self.tokenize_messages(processable_messages, add_generation_prompt=True)
            self._append_tokens(token_ids, mask_value=0)
        elif trailing_tool_messages and self.max_context_tokens is not None:
            trailing_token_ids = self.tokenize_messages(
                trailing_tool_messages,
                add_generation_prompt=True,
            )
            self._assert_within_context_limit(len(trailing_token_ids))
        self._pending_tool_messages = list(trailing_tool_messages)
        self.synced_message_count += len(messages)
        return token_ids

    def append_model_response(
        self,
        *,
        output_token_ids: list[int],
        fallback_messages: list[Message],
    ) -> list[int]:
        """Append model output, falling back to message tokenization if needed."""
        self._flush_pending_tool_messages()
        if output_token_ids:
            self._append_tokens(output_token_ids, mask_value=1)
            self.synced_message_count += len(fallback_messages)
            return output_token_ids

        fallback_token_ids = self.tokenize_messages(fallback_messages)
        self._append_tokens(fallback_token_ids, mask_value=1)
        self.synced_message_count += len(fallback_messages)
        return fallback_token_ids

    def _append_tokens(self, token_ids: list[int], *, mask_value: int) -> None:
        self._assert_within_context_limit(len(token_ids))
        self.token_ids.extend(token_ids)
        self.response_mask.extend([mask_value] * len(token_ids))

    def _assert_within_context_limit(self, additional_token_count: int) -> None:
        """Raise when adding more tokens would exceed the context limit."""
        new_total = len(self.token_ids) + additional_token_count
        if self.max_context_tokens is not None and new_total > self.max_context_tokens:
            raise TokenTraceContextOverflowError(
                f"Token trace session overflow: appending {additional_token_count} tokens would "
                f"bring total to {new_total}, exceeding max_context_tokens "
                f"({self.max_context_tokens}). Context compaction is not supported "
                f"for token trace sessions because message-to-token mapping would "
                f"become inconsistent after compaction."
            )

    def _split_trailing_tool_messages(self, messages: list[Message]) -> tuple[list[Message], list[Message]]:
        """Keep the trailing tool run buffered until its boundary is known."""
        split_index = len(messages)
        while split_index > 0 and messages[split_index - 1].role == Role.TOOL:
            split_index -= 1
        return messages[:split_index], messages[split_index:]

    def _flush_pending_tool_messages(self) -> list[int]:
        """Append any buffered trailing tool messages as one template fragment."""
        if not self._pending_tool_messages:
            return []
        token_ids = self.tokenize_messages(self._pending_tool_messages, add_generation_prompt=True)
        self._append_tokens(token_ids, mask_value=0)
        self._pending_tool_messages = []
        return token_ids

    def record_round(
        self,
        *,
        request_tokens: list[int],
        response_tokens: list[int],
        response_text: str | None,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, Any] | None,
    ) -> None:
        """Record a single model round for debugging/training analysis."""
        self.round_traces.append(
            {
                "request_tokens": list(request_tokens),
                "response_tokens": list(response_tokens),
                "response_text": response_text or "",
                "tool_calls": _serialize_tool_calls(tool_calls),
            }
        )
        if usage:
            try:
                usage_copy = cast(dict[str, Any], json.loads(json.dumps(usage, ensure_ascii=False)))
            except (TypeError, ValueError):
                usage_copy = {}
            self.token_provider_usage.append(usage_copy)

    def export_trace(self) -> dict[str, Any]:
        """Return the final token trace payload."""
        self._flush_pending_tool_messages()
        return {
            "final_token_list": list(self.token_ids),
            "response_mask": list(self.response_mask),
            "round_traces": list(self.round_traces),
            "token_provider_usage": list(self.token_provider_usage),
        }
