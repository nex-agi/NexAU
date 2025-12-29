from __future__ import annotations

from typing import Any

from nexau.core.adapters.base import LLMAdapter
from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
from nexau.core.messages import Message


class OpenAIChatAdapter(LLMAdapter):
    """Adapter for OpenAI Chat Completions-style payloads.

    Note: NexAU historically uses this shape even when calling non-OpenAI vendors,
    then converts internally (e.g. Anthropic Messages API).
    """

    def to_vendor_format(self, messages: list[Message]) -> list[dict[str, Any]]:
        return messages_to_legacy_openai_chat(messages)

    def from_vendor_response(self, response: Any) -> Message:  # pragma: no cover (not wired yet)
        raise NotImplementedError
