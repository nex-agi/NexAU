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

"""Langfuse tracer adapter for agent observability."""

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from langfuse import Langfuse, LangfuseSpan

from nexau.archs.tracer.core import BaseTracer, Span, SpanType

logger = logging.getLogger(__name__)


class LangfuseTracer(BaseTracer):
    """Tracer adapter for Langfuse observability platform.

    This adapter sends trace data to Langfuse, which provides:
    - LLM generation tracking with token usage and latency
    - Tool/function call tracing
    - Agent execution hierarchies
    - Cost analytics

    Langfuse concepts mapping:
    - Agent/Sub-Agent spans → Langfuse Traces (root) or Spans (nested)
    - LLM spans → Langfuse Generations
    - Tool spans → Langfuse Spans

    Example:
        ```python
        tracer = LangfuseTracer(public_key="pk-...", secret_key="sk-...", host="https://cloud.langfuse.com")

        with TraceContext(tracer, "my_agent", SpanType.AGENT) as span:
            response = agent.run("Hello")
        ```

    Environment variables can also be used:
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_HOST
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        debug: bool = False,
        enabled: bool = True,
    ):
        """Initialize Langfuse tracer.

        Args:
            public_key: Langfuse public key (or use LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or use LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (or use LANGFUSE_HOST env var)
            debug: Enable debug logging
            enabled: Whether tracing is enabled (can be disabled for testing)

        Raises:
            ImportError: If langfuse package is not installed
        """
        self.enabled = enabled
        self.debug = debug

        if not self.enabled:
            self.client = None
            logger.info("Langfuse tracer disabled")
            return

        # Initialize Langfuse client
        # It will fall back to environment variables if not provided
        client_kwargs: dict[str, Any] = {}
        if public_key:
            client_kwargs["public_key"] = public_key
        if secret_key:
            client_kwargs["secret_key"] = secret_key
        if host:
            client_kwargs["host"] = host
        client_kwargs["debug"] = debug

        self.session_id = str(uuid.uuid4()) if session_id is None else session_id
        self.user_id = user_id
        self.tags = tags

        self.client = Langfuse(**client_kwargs)
        logger.info(f"Langfuse tracer initialized (host: {host or 'default'})")

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span and create corresponding Langfuse object.

        The mapping to Langfuse objects:
        - No parent → Create a new Trace
        - LLM span type → Create a Generation
        - Other types → Create a Span

        Args:
            name: Human-readable name for the span
            span_type: Type of span (AGENT, LLM, TOOL, etc.)
            inputs: Input data for the span
            parent_span: Optional parent span
            attributes: Optional metadata/attributes

        Returns:
            Span with vendor_obj containing the Langfuse object
        """
        span_id = str(uuid.uuid4())
        now = datetime.now()

        # Create our internal span representation
        span = Span(
            id=span_id,
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            start_time=now.timestamp(),
            inputs=inputs or {},
            attributes=attributes or {},
        )

        if not self.enabled or self.client is None:
            return span

        # Prepare common parameters
        langfuse_params: dict[str, Any] = {
            "name": name,
            "metadata": {
                "span_type": span_type.value,
                **(attributes or {}),
            },
        }

        if self.session_id:
            langfuse_params["metadata"]["langfuse_session_id"] = self.session_id
        if self.user_id:
            langfuse_params["metadata"]["langfuse_user_id"] = self.user_id
        if self.tags:
            langfuse_params["metadata"]["langfuse_tags"] = self.tags

        # Serialize inputs properly
        if inputs:
            langfuse_params["input"] = self._serialize_for_langfuse(inputs)

        try:
            if parent_span is None or parent_span.vendor_obj is None:
                # Root level: Create a Trace
                langfuse_span = self.client.start_span(**langfuse_params)
                span.vendor_obj = langfuse_span

            elif span_type == SpanType.LLM:
                # LLM call: Create a Generation
                parent_obj = cast(LangfuseSpan, parent_span.vendor_obj)
                # Workaround for Langfuse, as_type="generation" may lead to LLM event lost when using new api.
                generation = parent_obj.start_observation(**langfuse_params, as_type="span")
                span.vendor_obj = generation
                if self.debug:
                    logger.debug(f"Created Langfuse generation: {name}")
            elif span_type == SpanType.TOOL:
                # Tool call: Create a Span
                parent_obj = cast(LangfuseSpan, parent_span.vendor_obj)
                langfuse_span = parent_obj.start_span(**langfuse_params)
                span.vendor_obj = langfuse_span
                if self.debug:
                    logger.debug(f"Created Langfuse span: {name}")
            else:
                # Other types: Create a Span
                parent_obj = cast(LangfuseSpan, parent_span.vendor_obj)
                langfuse_span = parent_obj.start_span(**langfuse_params)
                span.vendor_obj = langfuse_span
                if self.debug:
                    logger.debug(f"Created Langfuse span: {name}")

        except Exception as e:
            logger.warning(f"Failed to create Langfuse span '{name}': {e}")

        return span

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End a span and update the Langfuse object.

        Args:
            span: The span to end
            outputs: Output data from the operation
            error: Optional exception if operation failed
            attributes: Optional additional attributes
        """
        span.end_time = datetime.now().timestamp()

        if outputs is not None:
            span.outputs = outputs if isinstance(outputs, dict) else {"result": outputs}

        if error is not None:
            span.error = str(error)

        if not self.enabled or span.vendor_obj is None:
            return

        try:
            langfuse_span = cast(LangfuseSpan, span.vendor_obj)

            # Prepare update parameters
            update_params: dict[str, Any] = {}

            if outputs is not None:
                update_params["output"] = self._serialize_for_langfuse(outputs)
                if "model" in outputs and "usage" in outputs:
                    langfuse_span.update(model=outputs["model"], usage_details=outputs["usage"])  # type: ignore

            if error is not None:
                update_params["level"] = "ERROR"
                update_params["status_message"] = str(error)

            if attributes:
                # Merge with existing metadata
                existing_metadata = getattr(langfuse_span, "metadata", {}) or {}
                update_params["metadata"] = {**existing_metadata, **attributes}

            # Update the Langfuse object
            if update_params:
                langfuse_span.update(**update_params)

            # End the span (for timing)
            if hasattr(langfuse_span, "end"):
                langfuse_span.end()

            if self.debug:
                duration = span.duration_ms()
                logger.debug(f"Ended Langfuse span: {span.name} (duration={duration:.2f}ms)")
            if self.client is not None:
                self.client.flush()

        except Exception as e:
            logger.warning(f"Failed to end Langfuse span '{span.name}': {e}")

    def flush(self) -> None:
        """Flush pending data to Langfuse."""
        if self.enabled and self.client is not None:
            try:
                self.client.flush()
                if self.debug:
                    logger.debug("Flushed Langfuse data")
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse data: {e}")

    def shutdown(self) -> None:
        """Shutdown the Langfuse client."""
        if self.enabled and self.client is not None:
            try:
                self.client.shutdown()
                logger.info("Langfuse tracer shutdown")
            except Exception as e:
                logger.warning(f"Failed to shutdown Langfuse client: {e}")

    @staticmethod
    def _serialize_for_langfuse(data: Any) -> Any:
        """Serialize data for Langfuse API.

        Langfuse accepts strings, dicts, and lists. Complex objects
        need to be converted to JSON strings.

        Args:
            data: Data to serialize

        Returns:
            Langfuse-compatible representation
        """
        if data is None:
            return None

        if isinstance(data, (str, int, float, bool)):
            return data

        if isinstance(data, Mapping):
            # Recursively serialize mapping values with typed keys
            mapping_data = cast(Mapping[str, Any], data)
            return {str(k): LangfuseTracer._serialize_for_langfuse(v) for k, v in mapping_data.items()}

        if isinstance(data, (list, tuple)):
            sequence_data = cast(Sequence[Any], data)
            return [LangfuseTracer._serialize_for_langfuse(item) for item in sequence_data]

        # For other types, convert to JSON string
        try:
            return json.dumps(data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(data)
