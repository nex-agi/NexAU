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

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan  # type: ignore
from opentelemetry import trace as otel_trace_api

from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.core.usage import TokenUsage

logger = logging.getLogger(__name__)

_TTFT_ATTRIBUTE_KEY = "time_to_first_token_ms"


# 已知字段名 → Langfuse 标准字段名（覆盖 TokenUsage 内部名 + 各 provider 原始名）
_TO_LANGFUSE_FIELD: dict[str, str | None] = {
    # Provider aliases
    "prompt_tokens": "input_tokens",
    "output_tokens": "completion_tokens",
    "cached_tokens": "cache_read_input_tokens",
    # TokenUsage 内部名
    "cache_read_tokens": "cache_read_input_tokens",
    "cache_creation_tokens": "cache_creation_input_tokens",
    "input_tokens_uncached": None,
    # Anthropic 嵌套 details 子桶。_flatten_usage_dict 会把它们从 `cache_creation` /
    # `output_tokens_details` 子 dict 提升到顶层，此时与顶层聚合字段**重叠**：
    # ephemeral_5m/1h 之和 == cache_creation_input_tokens，thinking_tokens ⊆
    # output_tokens。Anthropic usage 无字面量 total，Langfuse ingestion 对
    # usage_details 全部值求和当 total，于是这些子桶被二次累加、total 虚高
    # （实测 cache_creation_input_tokens=37112 与其明细 ephemeral_5m=37112 同时入账，
    # total 多算 37112，接近 2×）。映射为 None 丢弃——顶层聚合字段已承载其值，
    # 与上面 `input_tokens_uncached` 同理。
    "ephemeral_5m_input_tokens": None,
    "ephemeral_1h_input_tokens": None,
    "thinking_tokens": None,
    # provider/会话预聚合的总数必须用 Langfuse 规范的字面量 `total` key 上报。
    # Langfuse ingestion（worker IngestionService）的逻辑是：usage_details 里若不存在
    # 字面量 `total` key，就把 map 里**所有值求和**当作 total。`total_tokens`（带后缀）
    # 不被识别为 total，于是会和 input/completion/reasoning 等拆分项一起被二次累加，
    # 导致 UI 上 total ≈ 2×（拆分项之和 + total_tokens）。映射成 `total` 后 Langfuse
    # 直接采用该权威值，不再求和。
    "total_tokens": "total",
    # Gemini REST 原始 camelCase（与 TokenUsage._resolve_total_tokens 的识别保持一致）。
    # nexau 自身 pipeline 在 _enrich_gemini_trace_outputs 已归一化为 total_tokens，
    # 此别名保护直接使用 SDK tracer、传入原始 usageMetadata 的第三方调用方。
    "totalTokenCount": "total",
}


def _flatten_usage_dict(usage: Mapping[str, object]) -> dict[str, int]:
    """展开 provider usage dict 中的嵌套 details 字段。

    OpenAI 将 cache/reasoning 数放在 prompt_tokens_details 等子 dict 中。
    此函数提升嵌套 int 字段到顶层，顶层字段优先。
    """
    flat: dict[str, int] = {}
    for k, v in usage.items():
        if isinstance(v, int):
            flat[k] = v
        elif isinstance(v, Mapping):
            nested = cast(Mapping[str, object], v)
            for nested_k, nested_v in nested.items():
                if isinstance(nested_v, int) and nested_k not in flat:
                    flat[nested_k] = nested_v
    return flat


def _sanitize_usage(usage: Mapping[str, object] | TokenUsage) -> dict[str, int]:
    """Sanitize and map usage data for Langfuse SDK compatibility.

    1. 只保留 int 值字段，避免 pydantic 校验失败。
       参见: https://github.com/langfuse/langfuse/issues/4961
    2. 展开 provider 嵌套的 details 字段（OpenAI prompt_tokens_details 等）。
    3. 统一映射字段名为 Langfuse 标准名，确保所有 provider 的
       cache 命中率在 Langfuse UI 正确显示；并把预聚合的 `total_tokens`
       映射为字面量 `total`，避免被 Langfuse 当成额外拆分项二次累加。
    """
    if isinstance(usage, TokenUsage):
        raw: dict[str, int] = usage.to_dict()
    else:
        raw = _flatten_usage_dict(usage)

    result: dict[str, int] = {}
    for key, value in raw.items():
        mapped_key = _TO_LANGFUSE_FIELD.get(key, key)
        if mapped_key is None or mapped_key in result:
            continue
        # 严格 int 过滤（排除 bool 子类）：raw-dict 路径 _flatten_usage_dict 已过滤，
        # 但 TokenUsage 路径是裸信任 to_dict()——动态运行时仍可能被构造进 None/str，
        # 在此统一兜底，确保 docstring 的「只保留 int」承诺对两条路径都成立。
        if type(value) is not int:
            continue
        # total<=0（例如直接构造、未填总数的 TokenUsage）时不写 `total`，
        # 让 Langfuse 回落到对拆分项求和，避免把 total 钉死成 0 而盖掉真实用量。
        if mapped_key == "total" and value <= 0:
            continue
        result[mapped_key] = value
    return result


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
        version: str | None = None,
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
            version: Langfuse trace version (native trace field, shows as the
                Langfuse "version" column — not a tag). Note: Langfuse v3 has no
                per-trace ``release`` field on ``update_trace``; ``release`` is a
                client-level setting (``LANGFUSE_RELEASE`` / ``Langfuse(release=...)``).
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
        self.version = version
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

        # #495: Credential rotation — async cleanup of old client.
        # Detach old client immediately so new spans use the new client.
        # Offload flush+shutdown to background thread with timeout to avoid
        # blocking the event loop. Falls back to sync cleanup when no event
        # loop is running (e.g. plain script context).
        if self.client is not None:
            old_client = self.client
            self.client = None
            self._client_identity = None
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_cleanup_old_client(old_client))
            except RuntimeError:
                # No running event loop — sync context, best-effort
                self._sync_cleanup_old_client(old_client)

        client_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
            "debug": self.debug,
        }
        if host:
            client_kwargs["host"] = host

        # IMPORTANT: Use an isolated TracerProvider to prevent Langfuse SDK from
        # overwriting the global TracerProvider. Without this, Langfuse v3+ (which
        # is built on OpenTelemetry) would register its span processor on the global
        # provider, causing ALL spans (including Jaeger-only spans like sandbox
        # operations) to be sent to Langfuse.
        # See: https://langfuse.com/faq/all/existing-otel-setup
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

        isolated_provider = SdkTracerProvider()
        client_kwargs["tracer_provider"] = isolated_provider

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

        # Serialize inputs properly
        if inputs:
            langfuse_params["input"] = self._serialize_for_langfuse(inputs)
        try:
            if parent_span is None or parent_span.vendor_obj is None:
                if self.trace_id:
                    if langfuse_params.get("trace_context") is None:
                        langfuse_params["trace_context"] = {}
                    langfuse_params["trace_context"]["trace_id"] = self.trace_id
                langfuse_span = client.start_span(**langfuse_params)
                span.vendor_obj = langfuse_span

            elif span_type == SpanType.LLM:
                # LLM call: Create a Generation
                parent_obj = cast(LangfuseSpan, parent_span.vendor_obj)
                # Preferred path: record LLM observations as Langfuse generations so
                # usage_details/model/completion_start_time are indexed correctly.
                vendor_obj: LangfuseSpan | LangfuseGeneration
                try:
                    vendor_obj = parent_obj.start_observation(**langfuse_params, as_type="generation")
                except Exception as generation_error:
                    logger.warning(
                        "Failed to create Langfuse generation for '%s': %s; falling back to span",
                        name,
                        generation_error,
                    )
                    vendor_obj = parent_obj.start_observation(**langfuse_params, as_type="span")
                span.vendor_obj = vendor_obj
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
                if isinstance(outputs, Mapping):
                    outputs_map: Mapping[str, object] = cast(Mapping[str, object], outputs)
                    if "model" in outputs_map:
                        update_params["model"] = outputs_map["model"]
                    usage = outputs_map.get("usage")
                    if isinstance(usage, (Mapping, TokenUsage)):
                        update_params["usage_details"] = _sanitize_usage(cast(Mapping[str, object] | TokenUsage, usage))

            ttft_ms: float | None = None
            for source in (attributes, span.attributes):
                if not isinstance(source, Mapping):
                    continue
                raw_ttft = source.get(_TTFT_ATTRIBUTE_KEY)
                if raw_ttft is None:
                    continue
                try:
                    parsed_ttft = float(raw_ttft)
                except (TypeError, ValueError):
                    continue
                if parsed_ttft >= 0:
                    ttft_ms = parsed_ttft
                    break
            if ttft_ms is not None:
                completion_start_ts = span.start_time + (ttft_ms / 1000.0)
                update_params["completion_start_time"] = datetime.fromtimestamp(completion_start_ts, tz=UTC)

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
                # For root spans (no parent), update trace name, input, and output.
                # This is defensive programming: when using trace_context.trace_id,
                # Langfuse SDK creates a trace with empty name. We must explicitly
                # call update_trace to ensure the trace has meaningful data.
                # Without this, users who enable auto-instrumentation (FastAPI, httpx)
                # may see unnamed traces in Langfuse UI.
                if span.parent_id is None:
                    trace_update: dict[str, Any] = {"name": span.name}
                    if span.inputs:
                        trace_update["input"] = self._serialize_for_langfuse(span.inputs)
                    if outputs is not None:
                        trace_update["output"] = self._serialize_for_langfuse(outputs)
                    langfuse_span.update_trace(**trace_update)
                if self.metadata:
                    langfuse_span.update_trace(metadata=self.metadata)
                if self.user_id:
                    langfuse_span.update_trace(user_id=self.user_id)
                if self.session_id:
                    langfuse_span.update_trace(session_id=self.session_id)
                if self.tags:
                    langfuse_span.update_trace(tags=self.tags)
                # Native trace field — shows up as the Langfuse "version" column,
                # not in the free-form tag list.
                if self.version:
                    langfuse_span.update_trace(version=self.version)

            # End the span (for timing)
            if hasattr(langfuse_span, "end"):
                langfuse_span.end()

            if self.debug:
                duration = span.duration_ms()
                logger.debug(f"Ended Langfuse span: {span.name} (duration={duration:.2f}ms)")

            # #495: Async flush on root span completion (fire-and-forget).
            # Offload flush to thread pool so the SDK
            # uploads trace data promptly without blocking the agent main path.
            # If no event loop is running, rely on SDK background batch worker.
            if span.parent_id is None and self.client is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self._async_flush_best_effort(),
                        name="langfuse-root-span-flush",
                    )
                except RuntimeError:
                    pass  # No event loop — SDK batch worker handles delivery

        except Exception as e:
            logger.warning(f"Failed to end Langfuse span '{span.name}': {e}")

    # ------------------------------------------------------------------
    # Old-client lifecycle helpers (#495)
    # ------------------------------------------------------------------

    _ASYNC_OP_TIMEOUT: float = 5.0
    """Timeout in seconds for async flush / old-client cleanup operations."""

    async def _async_cleanup_old_client(self, old_client: Langfuse) -> None:
        """Offload old-client flush+shutdown to a thread with timeout.

        Runs flush()+shutdown() via asyncio.to_thread with a wait_for timeout
        to prevent Langfuse SDK Queue.join() from blocking indefinitely.
        """
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._sync_cleanup_old_client, old_client),
                timeout=self._ASYNC_OP_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Old Langfuse client cleanup timed out after %.1fs, skipping",
                self._ASYNC_OP_TIMEOUT,
            )
        except Exception as e:
            logger.warning("Old Langfuse client async cleanup failed: %s", e)

    @staticmethod
    def _sync_cleanup_old_client(old_client: Langfuse) -> None:
        """Best-effort synchronous flush+shutdown of an old Langfuse client."""
        try:
            old_client.flush()
        except Exception:
            pass
        try:
            old_client.shutdown()
        except Exception:
            pass

    async def _async_flush_best_effort(self) -> None:
        """Fire-and-forget async flush for root span completion.

        Offloads the blocking Langfuse SDK flush() to a thread pool with a
        timeout guard. Timeout or failure is silently handled so the agent
        main path is never affected.
        """
        if self.client is None:
            return
        client = self.client  # capture reference in case of rotation
        try:
            await asyncio.wait_for(
                asyncio.to_thread(client.flush),
                timeout=self._ASYNC_OP_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Langfuse async flush timed out after %.1fs",
                self._ASYNC_OP_TIMEOUT,
            )
        except Exception as e:
            logger.debug("Langfuse async flush best-effort failed: %s", e)

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
        """Shutdown the Langfuse client.

        Called on agent/process exit via CleanupManager -> agent.sync_cleanup().
        Langfuse SDK shutdown() internally calls flush() then stops worker
        threads. Uses best-effort error handling to never raise.
        """
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

    def set_session_id(self, session_id: str) -> None:
        """Set the canonical session ID for Langfuse traces.

        Called by Agent to replace the default random UUID with the
        framework's actual session_id, ensuring Langfuse traces are
        grouped under the correct session.

        Args:
            session_id: The canonical session ID from Agent
        """
        self.session_id = session_id

    @staticmethod
    def _serialize_for_langfuse(data: Any) -> Any:
        """Serialize data for Langfuse API.

        Langfuse accepts strings, dicts, and lists. Complex objects
        need to be converted to JSON strings.

        base64 图片数据不做截断 — Langfuse SDK 内置 MediaManager 会自动检测
        Anthropic/OpenAI/Vertex 格式的 base64 图片，异步上传到对象存储后替换
        为 media reference，保证 trace 中能看到完整图片。

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
