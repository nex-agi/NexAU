from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexau.core.messages import Message


class LLMAdapter(ABC):
    """Convert between UMP messages and vendor-specific payloads."""

    @abstractmethod
    def to_vendor_format(self, messages: list[Message]) -> Any: ...

    @abstractmethod
    def from_vendor_response(self, response: Any) -> Message: ...
