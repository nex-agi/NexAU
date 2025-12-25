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
"""Composite tracer for sending data to multiple backends simultaneously."""

from __future__ import annotations

import logging
import uuid
from typing import Any, TypeGuard

from nexau.archs.tracer.core import BaseTracer, Span, SpanType

logger = logging.getLogger(__name__)


VendorObjMap = dict[int, Any]


def _is_vendor_map(obj: object | None) -> TypeGuard[VendorObjMap]:
    return isinstance(obj, dict)


class CompositeTracer(BaseTracer):
    """Tracer that forwards calls to multiple underlying tracers.

    This allows sending trace data to multiple backends (e.g., Langfuse + OpenTelemetry)
    simultaneously from a single trace point in the code.

    Example:
        ```python
        langfuse_tracer = LangfuseTracer(...)
        otel_tracer = OpenTelemetryTracer(...)
        tracer = CompositeTracer([langfuse_tracer, otel_tracer])

        with TraceContext(tracer, "operation", SpanType.AGENT) as span:
            # Both backends receive this span
            do_work()
        ```
    """

    def __init__(self, tracers: list[BaseTracer]):
        """Initialize composite tracer with list of backend tracers.

        Args:
            tracers: List of tracer implementations to forward calls to
        """
        self.tracers = tracers

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span across all registered tracers.

        Creates a unified Span that tracks vendor-specific span objects
        for each registered tracer.

        Args:
            name: Human-readable name for the span
            span_type: Type of span being created
            inputs: Input data for this span
            parent_span: Optional parent span for hierarchy
            attributes: Optional metadata/attributes

        Returns:
            A unified Span containing vendor objects for each tracer
        """
        span_id = str(uuid.uuid4())

        # Create a unified span that holds vendor-specific objects
        vendor_obj_map: VendorObjMap = {}
        unified_span = Span(
            id=span_id,
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
            vendor_obj=vendor_obj_map,  # Dictionary mapping tracer index to vendor object
        )

        # Start span in each underlying tracer
        for idx, tracer in enumerate(self.tracers):
            try:
                # Get the corresponding vendor parent object if it exists
                vendor_parent = None
                if parent_span and _is_vendor_map(parent_span.vendor_obj):
                    parent_vendor_map = parent_span.vendor_obj
                    vendor_parent_obj = parent_vendor_map.get(idx)
                    if vendor_parent_obj is not None:
                        # Create a wrapper span with just the vendor object
                        vendor_parent = Span(
                            id=parent_span.id,
                            name=parent_span.name,
                            type=parent_span.type,
                            vendor_obj=vendor_parent_obj,
                        )

                # Start span in this tracer
                internal_span = tracer.start_span(
                    name=name,
                    span_type=span_type,
                    inputs=inputs,
                    parent_span=vendor_parent,
                    attributes=attributes,
                )

                # Store the vendor-specific span object
                if _is_vendor_map(unified_span.vendor_obj):
                    vendor_obj_dict = unified_span.vendor_obj
                    vendor_value: dict[int, Any] | object | None = internal_span.vendor_obj if internal_span else None
                    vendor_obj_dict[idx] = vendor_value

            except Exception as e:
                logger.warning(f"Failed to start span in tracer {idx}: {e}")

        return unified_span

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End span in all registered tracers.

        Args:
            span: The unified span to end
            outputs: Output data from the operation
            error: Optional exception if operation failed
            attributes: Optional additional attributes
        """
        if not _is_vendor_map(span.vendor_obj):
            return

        vendor_obj_dict = span.vendor_obj

        for idx, tracer in enumerate(self.tracers):
            vendor_handle = vendor_obj_dict.get(idx)
            if vendor_handle is None:
                continue

            try:
                # Create a wrapper span with the vendor handle
                vendor_span = Span(
                    id=span.id,
                    name=span.name,
                    type=span.type,
                    vendor_obj=vendor_handle,
                )
                tracer.end_span(vendor_span, outputs, error, attributes)
            except Exception as e:
                logger.warning(f"Failed to end span in tracer {idx}: {e}")

    def flush(self) -> None:
        """Flush data in all registered tracers."""
        for idx, tracer in enumerate(self.tracers):
            try:
                tracer.flush()
            except Exception as e:
                logger.warning(f"Failed to flush tracer {idx}: {e}")

    def activate_span(self, span: Span) -> Any | None:  # noqa: ANN401
        """Activate vendor context in all underlying tracers (best-effort)."""
        if not _is_vendor_map(span.vendor_obj):
            return None

        vendor_obj_dict = span.vendor_obj
        tokens: VendorObjMap = {}

        for idx, tracer in enumerate(self.tracers):
            vendor_handle = vendor_obj_dict.get(idx)
            if vendor_handle is None:
                continue
            try:
                vendor_span = Span(
                    id=span.id,
                    name=span.name,
                    type=span.type,
                    vendor_obj=vendor_handle,
                )
                token = tracer.activate_span(vendor_span)
                if token is not None:
                    tokens[idx] = token
            except Exception:
                continue

        return tokens or None

    def deactivate_span(self, token: Any | None) -> None:  # noqa: ANN401
        if not _is_vendor_map(token):
            return

        tokens = token
        for idx, tracer in enumerate(self.tracers):
            t = tokens.get(idx)
            if t is None:
                continue
            try:
                tracer.deactivate_span(t)
            except Exception:
                continue

    def shutdown(self) -> None:
        """Shutdown all registered tracers."""
        for idx, tracer in enumerate(self.tracers):
            try:
                tracer.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown tracer {idx}: {e}")
