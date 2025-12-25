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

"""Core tracer classes and definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _empty_str_any_dict() -> dict[str, Any]:
    """Return a fresh dict for span data to keep types known."""
    return {}


class SpanType(str, Enum):
    """Types of spans that can be traced."""

    AGENT = "AGENT"
    SUB_AGENT = "SUB_AGENT"
    TOOL = "TOOL"
    LLM = "LLM"


@dataclass
class Span:
    """Represents a traced span in the execution flow.

    Attributes:
        id: Unique identifier for this span
        name: Human-readable name for the span
        type: Type of span (AGENT, TOOL, LLM, etc.)
        parent_id: ID of the parent span, if any
        start_time: Unix timestamp when span started
        end_time: Unix timestamp when span ended (None if still running)
        inputs: Input data for this span
        outputs: Output data from this span
        attributes: Additional metadata/attributes
        error: Error information if span failed
        vendor_obj: Holds vendor-specific span objects (e.g., Langfuse trace/span/generation)
    """

    id: str
    name: str
    type: SpanType
    parent_id: str | None = None
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    end_time: float | None = None
    inputs: dict[str, Any] = field(default_factory=_empty_str_any_dict)
    outputs: dict[str, Any] = field(default_factory=_empty_str_any_dict)
    attributes: dict[str, Any] = field(default_factory=_empty_str_any_dict)
    error: str | None = None
    vendor_obj: dict[int, Any] | object | None = None

    def duration_ms(self) -> float | None:
        """Calculate span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class BaseTracer(ABC):
    """Abstract base class for all tracer implementations.

    Implementations should handle the specifics of sending trace data
    to their respective backends (Langfuse, OpenTelemetry, etc.).
    """

    @abstractmethod
    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Human-readable name for the span
            span_type: Type of span being created
            inputs: Input data for this span
            parent_span: Optional parent span for creating hierarchy
            attributes: Optional metadata/attributes

        Returns:
            A new Span object representing this traced operation
        """

    @abstractmethod
    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End an existing span.

        Args:
            span: The span to end
            outputs: Output data from the operation
            error: Optional exception if the operation failed
            attributes: Optional additional attributes to add
        """

    def flush(self) -> None:
        """Flush any pending trace data to the backend.

        Override this method in implementations that buffer data.
        """

    def shutdown(self) -> None:
        """Shutdown the tracer and release resources.

        Override this method in implementations that need cleanup.
        """

    # ---- Optional vendor context propagation hooks ----
    #
    # Some tracer backends (e.g., Langfuse's OpenAI auto-instrumentation) rely on
    # their own notion of an "active span" (often via OpenTelemetry contextvars).
    #
    # Nexau's TraceContext manages its own contextvar (`nexau.archs.tracer.context`),
    # so we provide optional hooks for tracer implementations to also set / restore
    # vendor-specific "current span" state when a span becomes active.
    #
    # Implementations should return an opaque token that can later be passed to
    # `deactivate_span` to restore the previous vendor context.
    def activate_span(self, span: Span) -> Any | None:  # noqa: ANN401
        """Activate the vendor-specific span context for this span (optional).

        Args:
            span: The span that is becoming the current active span.

        Returns:
            An opaque token to be passed back to `deactivate_span`, or None.
        """
        return None

    def deactivate_span(self, token: Any | None) -> None:  # noqa: ANN401
        """Deactivate previously-activated vendor context (optional)."""
        return None
