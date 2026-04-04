"""Gemini REST adapters built from neutral UMP messages.

RFC-0006: Gemini 原生 structured adapter 路径

Gemini 请求体直接从统一消息表示生成 ``contents`` / ``systemInstruction``，
避免 structured tool calling 再经由 OpenAI 形状中转。
"""

from __future__ import annotations

from typing import Any

from nexau.core.adapters.base import LLMAdapter
from nexau.core.messages import Message
from nexau.core.serializers.gemini_messages import serialize_ump_to_gemini_messages_payload


class GeminiMessagesAdapter(LLMAdapter):
    """Convert UMP messages into Gemini REST payloads.

    RFC-0006: Gemini 原生消息适配路径

    输入为统一的 UMP Message 列表，输出为 Gemini REST 所需的
    ``contents`` / ``systemInstruction`` 结构，不依赖 OpenAI message 作为主链路。
    """

    def to_vendor_format(self, messages: list[Message]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        return serialize_ump_to_gemini_messages_payload(messages)

    def from_vendor_response(self, response: Any) -> Message:  # pragma: no cover - not wired yet
        raise NotImplementedError
