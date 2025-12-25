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

"""Context management for tracer spans using contextvars.

This module enables automatic parent-child span relationships across threads.
The design leverages Python's contextvars module, which works correctly with
copy_context() when submitting tasks to ThreadPoolExecutor.
"""

from contextvars import ContextVar, Token
from typing import Any

from nexau.archs.tracer.core import BaseTracer, Span, SpanType

# The current active span for this thread/context
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


def get_current_span() -> Span | None:
    """Get the currently active span in this context.

    Returns:
        The current Span or None if no span is active
    """
    return _current_span.get()


def set_current_span(span: Span) -> Token[Span | None]:
    """Set the current span for this context.

    Args:
        span: The span to set as current

    Returns:
        A token that can be used to reset the context
    """
    return _current_span.set(span)


def reset_current_span(token: Token[Span | None]) -> None:
    """Reset the current span to its previous value.

    Args:
        token: The token returned from set_current_span
    """
    _current_span.reset(token)


class TraceContext:
    """Context manager for automatic span lifecycle management.

    This context manager handles:
    - Creating a new span with the correct parent
    - Setting the span as current in the context
    - Ending the span when exiting (with error handling)
    - Restoring the previous span as current

    Example:
        ```python
        tracer = get_tracer()
        with TraceContext(tracer, "my_operation", SpanType.TOOL, {"input": "value"}) as span:
            # Do work here
            result = perform_operation()
            # Span will be ended automatically when exiting
        ```

    Note:
        Because this uses contextvars, it works correctly with ThreadPoolExecutor
        when using copy_context() before submitting tasks.
    """

    def __init__(
        self,
        tracer: BaseTracer,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        """Initialize the trace context.

        Args:
            tracer: The tracer implementation to use
            name: Human-readable name for the span
            span_type: Type of span being created
            inputs: Optional input data for the span
            attributes: Optional metadata/attributes
        """
        self.tracer = tracer
        self.name = name
        self.span_type = span_type
        self.inputs = inputs or {}
        self.attributes = attributes or {}
        self.span: Span | None = None
        self.token: Token[Span | None] | None = None
        self._outputs: Any = None
        self._vendor_ctx_token: Any | None = None

    def __enter__(self) -> Span:
        """Enter the context and start a new span.

        Returns:
            The newly created span
        """
        # Get the current parent span (if any)
        parent = get_current_span()

        # Create a new span with the parent relationship
        self.span = self.tracer.start_span(
            name=self.name,
            span_type=self.span_type,
            inputs=self.inputs,
            parent_span=parent,
            attributes=self.attributes,
        )

        # Set this new span as the current context
        self.token = set_current_span(self.span)

        # Also activate vendor-specific "current span" context (optional).
        # This enables auto-instrumentations (e.g., Langfuse OpenAI) to attach
        # their spans as children of this span.
        try:
            self._vendor_ctx_token = self.tracer.activate_span(self.span)
        except Exception:
            # Never break core execution due to vendor context propagation issues.
            self._vendor_ctx_token = None
        return self.span

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Exit the context and end the span.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred

        """
        if self.span is not None:
            # End the span with error info if exception occurred
            error = exc_val if isinstance(exc_val, Exception) else None
            self.tracer.end_span(
                self.span,
                outputs=self._outputs,
                error=error,
            )

        # Restore vendor-specific context first so outer spans (if any) become active again.
        try:
            self.tracer.deactivate_span(self._vendor_ctx_token)
        except Exception:
            pass
        self._vendor_ctx_token = None

        # Restore the previous parent span
        if self.token is not None:
            reset_current_span(self.token)

    def set_outputs(self, outputs: Any) -> None:
        """Set the outputs to be recorded when the span ends.

        Args:
            outputs: The output data to record
        """
        self._outputs = outputs
