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
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from langfuse import Langfuse, LangfuseSpan
from opentelemetry import trace as otel_trace_api

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
        trace_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        debug: bool = False,
        enabled: bool = True,
    ):
        """Initialize Langfuse tracer.

        Args:
            public_key: Langfuse public key (or use LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or use LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (or use LANGFUSE_HOST env var)
            session_id: Langfuse session ID
            user_id: Langfuse user ID
            trace_id: Langfuse trace ID
            tags: Langfuse tags
            metadata: Langfuse metadata
            debug: Enable debug logging
            enabled: Whether tracing is enabled (can be disabled for testing)

        Raises:
            ImportError: If langfuse package is not installed
        """
        # IMPORTANT:
        # - This tracer is created during server warmup, before per-run configs/envs may be ready.
        # - We must not "lock in" a Langfuse client too early, otherwise different projects/keys
        #   in the same process can leak across runs.
        # Therefore we ALWAYS initialize attributes and lazily create (or rotate) the client
        # on first real span when keys are available.
        self.enabled = enabled
        self.debug = debug

        # Always define attributes to avoid AttributeError in start_span/end_span.
        self.client: Langfuse | None = None
        self.session_id = str(uuid.uuid4()) if session_id is None else session_id
        self.user_id = user_id
        self.tags = tags
        self.metadata = metadata
        self.trace_id = trace_id
        # Store config passed at construction time; actual keys may be injected later via env.
        self._init_public_key = public_key
        self._init_secret_key = secret_key
        self._init_host = host

        # Track which credentials the current client was created with (to support rotation).
        self._client_identity: tuple[str, str, str | None] | None = None
        self._missing_keys_warned = False

        if not self.enabled:
            logger.info("Langfuse tracer disabled")

    def _current_credentials(self) -> tuple[str | None, str | None, str | None]:
        """Resolve Langfuse credentials, preferring explicit args then environment variables."""
        public_key = self._init_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = self._init_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = self._init_host or os.getenv("LANGFUSE_HOST")
        return public_key, secret_key, host

    def _ensure_client(self) -> Langfuse | None:
        """Create or rotate the Langfuse client if credentials are available.

        This is intentionally lazy so server warmup doesn't initialize a client before
        per-run configuration (e.g. keys injected by prepare_env) is ready.
        """
        if not self.enabled:
            return None

        public_key, secret_key, host = self._current_credentials()
        if not public_key or not secret_key:
            # Keys not ready yet (common during warmup). Don't crash; just no-op tracing.
            if not self._missing_keys_warned:
                logger.warning("Langfuse tracer not initialized yet (public_key/secret_key missing)")
                self._missing_keys_warned = True
            return None

        identity: tuple[str, str, str | None] = (public_key, secret_key, host)
        if self.client is not None and self._client_identity == identity:
            return self.client

        # Credentials changed (multi-project in one process) OR client not created yet.
        # Flush/shutdown old client best-effort to avoid dropping buffered events.
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                pass
            try:
                self.client.shutdown()
            except Exception:
                pass

        client_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
            "debug": self.debug,
        }
        if host:
            client_kwargs["host"] = host

        try:
            self.client = Langfuse(**client_kwargs)
            self._client_identity = identity
            self._missing_keys_warned = False
            logger.info(f"Langfuse tracer initialized (host: {host or 'default'})")
        except Exception as e:
            self.client = None
            self._client_identity = None
            logger.warning(f"Langfuse tracer failed to initialize: {e}")
        return self.client

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

        client = self._ensure_client()
        if not self.enabled or client is None:
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
        if self.metadata:
            langfuse_params["metadata"].update(self.metadata)
        if self.trace_id:
            if langfuse_params.get("trace_context") is None:
                langfuse_params["trace_context"] = {}
            langfuse_params["trace_context"]["trace_id"] = self.trace_id
        # Serialize inputs properly
        if inputs:
            langfuse_params["input"] = self._serialize_for_langfuse(inputs)
        try:
            if parent_span is None or parent_span.vendor_obj is None:
                # Root level: Create a Trace
                langfuse_span = client.start_span(**langfuse_params)
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

    def activate_span(self, span: Span) -> Any | None:  # noqa: ANN401
        """Activate this span in OpenTelemetry context so Langfuse auto-instrumentations can parent correctly."""
        if not self.enabled:
            return None

        vendor_obj = span.vendor_obj
        if vendor_obj is None:
            return None

        # Langfuse spans wrap an OTEL span; activating that OTEL span makes downstream
        # auto-instrumentation (e.g., Langfuse's OpenAI patch) attach as children.
        otel_span = getattr(vendor_obj, "_otel_span", None)
        if otel_span is None:
            return None

        try:
            ctx_manager = otel_trace_api.use_span(otel_span, end_on_exit=False)
            ctx_manager.__enter__()
            return ctx_manager
        except Exception:
            return None

    def deactivate_span(self, token: Any | None) -> None:  # noqa: ANN401
        if token is None:
            return
        try:
            token.__exit__(None, None, None)
        except Exception:
            return

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

            # Trace-level fields are optional depending on the Langfuse SDK object type.
            # Keep this best-effort so missing methods (e.g., in tests/mocks) don't prevent `.end()`/flush.
            if hasattr(langfuse_span, "update_trace"):
                if self.metadata:
                    langfuse_span.update_trace(metadata=self.metadata)
                if self.user_id:
                    langfuse_span.update_trace(user_id=self.user_id)
                if self.session_id:
                    langfuse_span.update_trace(session_id=self.session_id)
                if self.tags:
                    langfuse_span.update_trace(tags=self.tags)

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

    def set_trace_id(self, trace_id: str) -> None:
        """Set the trace ID for the current session.

        Args:
            trace_id: The trace ID to set
        """
        self.trace_id = trace_id
        logger.info(f"Langfuse trace ID set: {trace_id}")

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
