from __future__ import annotations

import logging
from typing import Any

from nexau.core.adapters.base import LLMAdapter
from nexau.core.messages import Message
from nexau.core.serializers.anthropic_messages import serialize_ump_to_anthropic_messages_payload

_logger = logging.getLogger(__name__)


class AnthropicMessagesAdapter(LLMAdapter):
    """Adapter for Anthropic Messages API payloads.

    Returns a tuple of:
    - system: list[{"type": "text", "text": "..."}]
    - messages: list[{"role": "...", "content": [blocks...]}]
    """

    def to_vendor_format(self, messages: list[Message]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return serialize_ump_to_anthropic_messages_payload(messages)

    def from_vendor_response(self, response: Any) -> Message:  # pragma: no cover (not wired yet)
        raise NotImplementedError
