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

"""Tool implementation plus neutral structured-tool helpers.

RFC-0006: 中性 Structured Tool Definitions

在 structured tool calling 路径中，Tool/SubAgent 先统一归一化为中性
structured definition，再由边界 adapter 延迟转换为 OpenAI / Anthropic /
Gemini 所需的 provider schema。
"""

import asyncio
import dataclasses
import inspect
import logging
import traceback
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, TypedDict, Union, cast, get_args, get_origin, get_type_hints

import jsonschema
import yaml
from anthropic.types import ToolParam
from jsonschema.validators import validator_for
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, Field

from .formatters import ToolFormatter, ToolFormatterContext, resolve_tool_formatter

logger = logging.getLogger(__name__)
_UNRESOLVED_ANNOTATION = object()


StructuredToolKind = Literal["tool", "sub_agent"]


class StructuredToolDefinition(TypedDict):
    """Vendor-neutral structured tool definition used by the runtime."""

    name: str
    description: str
    input_schema: dict[str, Any]
    kind: StructuredToolKind


StructuredToolDefinitionLike = StructuredToolDefinition | Mapping[str, object]


def normalize_input_schema(input_schema: Mapping[str, object] | None) -> dict[str, Any]:
    """Normalize a tool schema to an object-shaped JSON Schema."""

    params: dict[str, Any] = deepcopy(dict(input_schema or {}))
    if not params:
        return {"type": "object", "properties": {}}
    if "type" not in params:
        return {"type": "object", "properties": params.get("properties", {})}
    return params


def build_structured_tool_definition(
    *,
    name: str,
    description: str | None,
    input_schema: Mapping[str, object] | None,
    kind: StructuredToolKind = "tool",
) -> StructuredToolDefinition:
    """Build a vendor-neutral structured tool definition.

    RFC-0006: 中性 Structured Tool Definitions

    统一 Tool 与 SubAgent 在 structured 模式下的上游表示，避免在 Agent /
    Executor 主状态中提前持有 vendor-specific schema。
    """

    return {
        "name": name,
        "description": description or "",
        "input_schema": normalize_input_schema(input_schema),
        "kind": kind,
    }


def normalize_structured_tool_definition(
    tool_definition: StructuredToolDefinitionLike,
) -> StructuredToolDefinition:
    """Normalize a neutral or provider-specific tool definition.

    RFC-0006: 中性 Structured Tool Definitions

    接受中性 definition，以及 OpenAI / Anthropic 兼容形状，并统一收敛到
    runtime 使用的 neutral structured definition。
    """

    if tool_definition.get("type") == "function":
        function_raw = tool_definition.get("function", {})
        function_block: Mapping[str, object] = cast(Mapping[str, object], function_raw) if isinstance(function_raw, Mapping) else {}
        name = function_block.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("OpenAI tool definition missing function.name")
        description = function_block.get("description")
        parameters = function_block.get("parameters")
        parameters_mapping: Mapping[str, object] | None = (
            cast(Mapping[str, object], parameters) if isinstance(parameters, Mapping) else None
        )
        kind_raw = tool_definition.get("kind", "tool")
        openai_compatible_kind: StructuredToolKind = "sub_agent" if kind_raw == "sub_agent" else "tool"
        return build_structured_tool_definition(
            name=name,
            description=str(description) if description is not None else "",
            input_schema=parameters_mapping,
            kind=openai_compatible_kind,
        )

    if "name" in tool_definition and "input_schema" in tool_definition:
        name = tool_definition.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Structured tool definition missing name")
        description = tool_definition.get("description")
        input_schema = tool_definition.get("input_schema")
        input_schema_mapping: Mapping[str, object] | None = (
            cast(Mapping[str, object], input_schema) if isinstance(input_schema, Mapping) else None
        )
        kind_raw = tool_definition.get("kind", "tool")
        neutral_kind: StructuredToolKind = "sub_agent" if kind_raw == "sub_agent" else "tool"
        return build_structured_tool_definition(
            name=name,
            description=str(description) if description is not None else "",
            input_schema=input_schema_mapping,
            kind=neutral_kind,
        )

    raise ValueError(f"Unsupported structured tool definition: {tool_definition}")


def structured_tool_definition_to_openai(
    tool_definition: StructuredToolDefinitionLike,
) -> ChatCompletionToolParam:
    """Convert a neutral or compatible tool definition into OpenAI schema.

    RFC-0006: Provider 延迟适配

    OpenAI family schema 只在真正发请求前生成；上游继续传递 neutral
    definition 以避免 provider 耦合泄漏到 runtime 主状态。
    """

    normalized = normalize_structured_tool_definition(tool_definition)
    return {
        "type": "function",
        "function": {
            "name": normalized["name"],
            "description": normalized["description"],
            "parameters": normalize_input_schema(normalized["input_schema"]),
        },
    }


def structured_tool_definition_to_anthropic(
    tool_definition: StructuredToolDefinitionLike,
) -> ToolParam:
    """Convert a neutral or compatible tool definition into Anthropic schema.

    RFC-0006: Provider 延迟适配

    Anthropic tool schema 仅在 provider 边界生成，保持 Agent / Executor 内部
    仍以 neutral structured definition 作为唯一上游表示。
    """

    normalized = normalize_structured_tool_definition(tool_definition)
    return {
        "name": normalized["name"],
        "description": normalized["description"],
        "input_schema": normalize_input_schema(normalized["input_schema"]),
        # 启用 fine-grained tool streaming，减少大参数（如写入长文件）的首 token 延迟，
        # 避免 SSE 超时断联。
        "eager_input_streaming": True,
    }


class ToolYamlSchema(BaseModel):
    """Schema describing a tool YAML definition."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool"] | None = Field(default=None)
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    skill_description: str | None = None
    disable_parallel: bool = False
    lazy: bool = False
    defer_loading: bool = False
    search_hint: str | None = None
    template_override: str | None = None
    builtin: str | None = None
    binding: str | None = None
    formatter: str | None = None


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class Tool:
    """Tool class that represents a callable function with schema validation."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        implementation: Callable[..., Any] | str | None,
        skill_description: str | None = None,
        as_skill: bool = False,
        disable_parallel: bool = False,
        lazy: bool = False,
        defer_loading: bool = False,
        search_hint: str | None = None,
        template_override: str | None = None,
        formatter: str | ToolFormatter | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        source_name: str | None = None,
    ):
        """Initialize a tool with schema and implementation."""
        self.name = name
        self.source_name = source_name
        self.description = description
        self.skill_description = skill_description
        self.as_skill = as_skill
        self.input_schema = input_schema
        self.lazy = lazy
        self.implementation = None
        self.implementation_import_path = None
        if isinstance(implementation, str):
            self.implementation_import_path = implementation
            if lazy:
                self.implementation = None  # lazy import and bind at runtime
            else:
                from ..main_sub.utils import import_from_string

                func = import_from_string(implementation)
                self.implementation = func
        else:
            self.implementation = implementation
        self.defer_loading = defer_loading
        self.search_hint = search_hint
        self.template_override = template_override
        self.formatter = formatter
        self._resolved_formatter: ToolFormatter | None = None
        self.disable_parallel = disable_parallel
        reserved_keys = {"agent_state", "global_storage", "ctx"}
        extra_kwargs = extra_kwargs or {}
        conflict_keys = set(extra_kwargs) & reserved_keys
        if conflict_keys:
            raise ConfigError(
                f"Tool '{self.name}' extra_kwargs contains reserved keys that cannot be overridden: {sorted(conflict_keys)}",
            )
        self.extra_kwargs = extra_kwargs

        # 子类（如 MCPTool）可重写此属性为 True，表示其 execute_async()
        # 有独立的原生 async 实现，executor 应直接 await 而非走
        # to_thread → sync execute → asyncio.run 的间接路径。
        self._has_native_async_execute: bool = False

        # Validate schema
        self._validate_schema()
        self._validate_reserved_param_annotations()

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        binding: Callable[..., Any] | str | None = None,
        *,
        as_skill: bool = False,
        extra_kwargs: dict[str, Any] | None = None,
        lazy: bool | None = None,
        name: str | None = None,
        description: str | None = None,
        description_suffix: str = "",
        defer_loading: bool | None = None,
    ) -> "Tool":
        """Load tool definition from YAML file and bind to implementation."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool YAML file not found: {yaml_path}")

        with open(path) as f:
            tool_def_model = ToolYamlSchema.model_validate(yaml.safe_load(f))
        tool_def = tool_def_model.model_dump()

        # Extract required fields
        source_name = tool_def["name"]
        base_description = tool_def["description"]
        skill_description = tool_def.get("skill_description", "")
        input_schema = tool_def.get("input_schema", {})
        disable_parallel = tool_def.get("disable_parallel", False)
        yaml_defer_loading = tool_def.get("defer_loading", False)
        search_hint_val = tool_def.get("search_hint")
        yaml_lazy = tool_def.get("lazy", False)
        effective_name = source_name if name is None else name
        effective_source_name = source_name if effective_name != source_name else None
        effective_description = base_description if description is None else description
        if description_suffix:
            effective_description += description_suffix
        effective_defer_loading = yaml_defer_loading if defer_loading is None else defer_loading
        effective_lazy = yaml_lazy if lazy is None else lazy

        if binding is None and "binding" in tool_def:
            binding = tool_def.get("binding")

        if "global_storage" in input_schema:
            raise ValueError(
                f"Tool definition of `{source_name}` contains 'global_storage' field in {yaml_path}, "
                "which will be injected by the framework, please remove it from the tool definition.",
            )

        if "agent_state" in input_schema:
            raise ValueError(
                f"Tool definition of `{source_name}` contains 'agent_state' field in {yaml_path}, "
                "which will be injected by the framework, please remove it from the tool definition."
            )

        if "ctx" in input_schema:
            raise ValueError(
                f"Tool definition of `{source_name}` contains 'ctx' field in {yaml_path}, "
                "which will be injected by the framework, please remove it from the tool definition.",
            )

        template_override = tool_def.get("template_override")
        formatter = tool_def.get("formatter")

        # Create tool instance
        return cls(
            name=effective_name,
            source_name=effective_source_name,
            description=effective_description,
            skill_description=skill_description,
            input_schema=input_schema,
            implementation=binding,
            as_skill=as_skill,
            disable_parallel=disable_parallel,
            lazy=effective_lazy,
            defer_loading=effective_defer_loading,
            search_hint=search_hint_val,
            template_override=template_override,
            formatter=formatter,
            extra_kwargs=extra_kwargs,
        )

    def resolve_formatter(self) -> ToolFormatter:
        """Resolve and cache the formatter configured for this tool.

        RFC-0017: formatter resolver

        未显式配置时自动回退到 builtin `xml` formatter。
        """

        if self._resolved_formatter is None:
            self._resolved_formatter = resolve_tool_formatter(self.formatter)
        return self._resolved_formatter

    def format_output_for_llm(
        self,
        *,
        tool_input: dict[str, Any],
        tool_output: object,
        tool_call_id: str | None,
        is_error: bool,
    ) -> object:
        """Format raw tool output into the LLM-facing representation.

        RFC-0017: tool output flattening

        先执行 formatter，再进入 after_tool middleware，保证后续截断与最终
        模型输入始终作用于同一份 llm-facing output。
        """

        formatter_context = ToolFormatterContext(
            tool_name=self.name,
            tool_input=cast(dict[str, object], dict(tool_input)),
            tool_output=tool_output,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )

        formatter = self.resolve_formatter()
        try:
            return formatter(formatter_context)
        except Exception:
            logger.exception("Tool '%s' formatter failed; falling back to builtin XML formatter", self.name)
            fallback_formatter = resolve_tool_formatter("xml")
            try:
                return fallback_formatter(formatter_context)
            except Exception:
                logger.exception("Tool '%s' builtin XML formatter failed; falling back to raw tool output", self.name)
                return tool_output

    def _validate_reserved_param_annotations(self) -> None:
        """Validate reserved framework parameter annotations for tool implementations."""
        impl = self.implementation
        if impl is None:
            return

        signature = inspect.signature(impl)
        ctx_param = signature.parameters.get("ctx")
        if ctx_param is None:
            return

        if ctx_param.annotation is inspect.Signature.empty:
            logger.warning(
                "Tool '%s' declares 'ctx' without a FrameworkContext annotation; annotate it as FrameworkContext.",
                self.name,
            )
            return

        annotation = self._resolve_reserved_param_annotation(impl, "ctx")
        if annotation is _UNRESOLVED_ANNOTATION:
            return

        if annotation is Any:
            logger.warning(
                "Tool '%s' declares 'ctx' as Any; annotate it as FrameworkContext for stricter validation.",
                self.name,
            )
            return

        if not self._is_framework_context_annotation(annotation):
            raise ConfigError(
                f"Tool '{self.name}' declares 'ctx' with incompatible type {annotation!r}; use FrameworkContext.",
            )

    def _resolve_reserved_param_annotation(self, impl: Callable[..., Any], param_name: str) -> Any:
        """Resolve a reserved parameter annotation, including forward references."""
        from nexau.archs.main_sub.framework_context import FrameworkContext

        globalns = dict(getattr(impl, "__globals__", {}))
        globalns.setdefault("FrameworkContext", FrameworkContext)
        try:
            hints = get_type_hints(impl, globalns=globalns, localns={"FrameworkContext": FrameworkContext}, include_extras=True)
        except Exception as exc:
            logger.warning(
                "Tool '%s' declares '%s' but its annotation could not be resolved (%s); skipping strict type validation.",
                self.name,
                param_name,
                exc,
            )
            return _UNRESOLVED_ANNOTATION

        return hints.get(param_name, _UNRESOLVED_ANNOTATION)

    @staticmethod
    def _is_framework_context_annotation(annotation: Any) -> bool:
        """Return whether an annotation represents FrameworkContext."""
        from nexau.archs.main_sub.framework_context import FrameworkContext

        if annotation is FrameworkContext:
            return True

        origin = get_origin(annotation)
        if origin is None:
            return False

        if origin is Annotated:
            args = get_args(annotation)
            return bool(args) and Tool._is_framework_context_annotation(args[0])

        if origin in (UnionType, Union):
            union_args = [arg for arg in get_args(annotation) if arg is not type(None)]
            return len(union_args) == 1 and Tool._is_framework_context_annotation(union_args[0])

        return False

    @property
    def has_native_async_execute(self) -> bool:
        """Whether this tool has a native async execute_async() override.

        子类（如 MCPTool）重写 execute_async() 为原生 async 实现时，
        应在 __init__ 中设置 ``_has_native_async_execute = True``。
        executor 据此决定直接 await execute_async() 还是走
        to_thread → sync execute 路径。
        """
        return self._has_native_async_execute

    def execute(self, **params: Any) -> dict[str, Any]:
        """Execute the tool with given parameters."""

        if self.implementation is None:
            if self.implementation_import_path:
                logger.info(f"Dynamic importing tool implementation '{self.name}': {self.implementation_import_path}")

                from ..main_sub.utils import import_from_string

                func = import_from_string(str(self.implementation_import_path))
                self.implementation = func
                self._validate_reserved_param_annotations()
            else:
                raise ValueError(f"Tool '{self.name}' has no implementation")

        # RFC-0006: 参数注入 — ctx 优先，agent_state 向后兼容
        merged_params = {**self.extra_kwargs, **params}
        filtered_params = merged_params.copy()

        impl = self.implementation
        if impl is not None:
            sig = inspect.signature(impl)

            # RFC-0006: ctx (FrameworkContext) 注入
            if "ctx" not in sig.parameters:
                filtered_params.pop("ctx", None)

            # 向后兼容: agent_state 注入
            if "agent_state" in merged_params:
                if "agent_state" not in sig.parameters:
                    filtered_params.pop("agent_state", None)

                    # For backwards compatibility, check if function accepts global_storage
                    if "global_storage" in sig.parameters:
                        agent_state = merged_params["agent_state"]
                        filtered_params["global_storage"] = agent_state.global_storage
                if "sandbox" not in sig.parameters:
                    filtered_params.pop("sandbox", None)

        # Validate parameters (excluding framework-injected params for schema validation)
        _injected_keys = {"agent_state", "global_storage", "sandbox", "ctx"}
        validation_params = {k: v for k, v in filtered_params.items() if k not in _injected_keys}
        self.validate_params(validation_params)

        try:
            if impl is None:
                raise ValueError(f"Tool '{self.name}' has no implementation")

            raw_result: Any = impl(**filtered_params)

            # Support async tool implementations.
            # 当 impl 返回 coroutine 时，根据调用上下文分发：
            # - 无 running loop（ThreadPoolExecutor worker / CLI / 脚本）→ asyncio.run() 安全
            # - 有 running loop（async context 误调 execute()）→ 报错，引导用 execute_async()
            if inspect.iscoroutine(raw_result):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    # 无 running loop — sync 入口（ThreadPoolExecutor worker / CLI）
                    # 警告开发者: async tool 将在隔离的 event loop 中执行，
                    # 无法访问主 loop 的共享 async 状态（如 task group、shared lock 等）。
                    # 如需复用主 loop，应通过 executor async 路径调用 execute_async()。
                    logger.warning(
                        "Tool '%s' is async but invoked via sync execute() — running in an "
                        "isolated event loop. Use execute_async() to run on the main loop.",
                        self.name,
                    )
                    raw_result = asyncio.run(raw_result)
                else:
                    # 关闭未 await 的 coroutine 以避免 RuntimeWarning
                    raw_result.close()
                    raise RuntimeError(
                        f"Tool '{self.name}' returned a coroutine but execute() was called "
                        "from an async context. Use `await tool.execute_async(...)` instead."
                    )

            # Ensure result is a dictionary
            final_result: dict[str, Any]
            if isinstance(raw_result, dict):
                final_result = cast(dict[str, Any], raw_result)
            elif dataclasses.is_dataclass(raw_result) and not isinstance(raw_result, type):
                final_result = dataclasses.asdict(raw_result)
            elif isinstance(raw_result, list):
                result_list = cast(list[object], raw_result)
                final_result = {
                    "result": [
                        dataclasses.asdict(item) if (dataclasses.is_dataclass(item) and not isinstance(item, type)) else item
                        for item in result_list
                    ]
                }
            else:
                final_result = {"result": raw_result}

            return final_result

        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "tool_name": self.name,
            }

    async def execute_async(self, **params: Any) -> dict[str, Any]:
        """Execute the tool asynchronously.

        P1 async/sync 技术债修复: 消除 asyncio.run() 嵌套

        对 async tool 实现直接 await（避免 asyncio.run() 创建嵌套 event loop），
        对 sync tool 实现通过 asyncio.to_thread() 在线程池中执行（避免阻塞 event loop）。
        现有 execute() 方法保持不变，供 sync 调用方（如 ThreadPoolExecutor workers）使用。
        """
        if self.implementation is None:
            if self.implementation_import_path:
                logger.info(f"Dynamic importing tool implementation '{self.name}': {self.implementation_import_path}")

                from ..main_sub.utils import import_from_string

                func = import_from_string(str(self.implementation_import_path))
                self.implementation = func
                self._validate_reserved_param_annotations()
            else:
                raise ValueError(f"Tool '{self.name}' has no implementation")

        # RFC-0006: 参数注入 — ctx 优先，agent_state 向后兼容
        merged_params = {**self.extra_kwargs, **params}
        filtered_params = merged_params.copy()

        impl = self.implementation
        if impl is not None:
            sig = inspect.signature(impl)

            # RFC-0006: ctx (FrameworkContext) 注入
            if "ctx" not in sig.parameters:
                filtered_params.pop("ctx", None)

            # 向后兼容: agent_state 注入
            if "agent_state" in merged_params:
                if "agent_state" not in sig.parameters:
                    filtered_params.pop("agent_state", None)

                    # For backwards compatibility, check if function accepts global_storage
                    if "global_storage" in sig.parameters:
                        agent_state = merged_params["agent_state"]
                        filtered_params["global_storage"] = agent_state.global_storage
                if "sandbox" not in sig.parameters:
                    filtered_params.pop("sandbox", None)

        # Validate parameters (excluding framework-injected params for schema validation)
        _injected_keys = {"agent_state", "global_storage", "sandbox", "ctx"}
        validation_params = {k: v for k, v in filtered_params.items() if k not in _injected_keys}
        self.validate_params(validation_params)

        try:
            if impl is None:
                raise ValueError(f"Tool '{self.name}' has no implementation")

            # 根据实现类型分发：async 直接 await，sync 走 to_thread
            if inspect.iscoroutinefunction(impl):
                raw_result: Any = await impl(**filtered_params)
            else:
                # asyncio.to_thread (Python 3.12+) 会自动将当前 contextvars
                # 复制到 worker 线程，因此 TraceContext 等 contextvar 能正确传播。
                # 无需手动 copy_context()。
                raw_result = await asyncio.to_thread(impl, **filtered_params)

            # Ensure result is a dictionary
            final_result: dict[str, Any]
            if isinstance(raw_result, dict):
                final_result = cast(dict[str, Any], raw_result)
            elif dataclasses.is_dataclass(raw_result) and not isinstance(raw_result, type):
                final_result = dataclasses.asdict(raw_result)
            elif isinstance(raw_result, list):
                result_list = cast(list[object], raw_result)
                final_result = {
                    "result": [
                        dataclasses.asdict(item) if (dataclasses.is_dataclass(item) and not isinstance(item, type)) else item
                        for item in result_list
                    ]
                }
            else:
                final_result = {"result": raw_result}

            return final_result

        except Exception as e:
            # Return error information
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "tool_name": self.name,
            }

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate parameters against schema.

        Only validates parameters that are defined in the schema.
        Extra parameters (injected by hooks or with default values) are ignored.

        Raises:
            ValueError: If parameters fail schema validation, with detailed error message.
        """
        # Extract only the parameters that are defined in the schema
        schema_properties = self.input_schema.get("properties", {})
        schema_params = {k: v for k, v in params.items() if k in schema_properties}

        try:
            # Validate only the schema-defined parameters
            jsonschema.validate(schema_params, self.input_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"Invalid parameters for tool '{self.name}': {e.message}. params={schema_params}",
            ) from e

    def _validate_schema(self):
        """Validate that the input schema is valid JSON Schema."""
        try:
            # Check if it's a valid JSON Schema
            validator_for(self.input_schema).check_schema(self.input_schema)
        except jsonschema.SchemaError as e:
            raise ValueError(
                f"Invalid JSON Schema for tool '{self.name}': {e}",
            )

    def get_schema(self) -> dict[str, Any]:
        """Get the tool's input schema."""
        return self.input_schema.copy()

    def get_info(self) -> dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "template_override": self.template_override,
            "description": self.description,
            "skill_description": self.skill_description,
            "input_schema": self.input_schema,
            "formatter": self.formatter if isinstance(self.formatter, str) else None,
        }

    def __repr__(self) -> str:
        impl = self.implementation
        impl_name = getattr(impl, "__name__", repr(impl)) if impl is not None else "None"
        formatter_name = self.formatter if isinstance(self.formatter, str) else getattr(self.formatter, "__name__", "xml")
        return f"Tool(name='{self.name}', implementation={impl_name}, formatter={formatter_name})"

    def __str__(self) -> str:
        tool_str = f"Tool '{self.name}': {self.description[:50]}{'...' if len(self.description) > 50 else ''}"
        if self.skill_description:
            tool_str += f"\nSkill description: {self.skill_description}"
        return tool_str

    def get_structured_description(self) -> str:
        """Return the description exposed to structured tool-calling models."""

        if self.as_skill:
            if not self.skill_description:
                raise ValueError(
                    f"Tool {self.name} is marked as a skill but has no skill_description",
                )
            return self.skill_description
        return self.description or ""

    def to_structured_definition(
        self,
        *,
        description: str | None = None,
        kind: StructuredToolKind = "tool",
    ) -> StructuredToolDefinition:
        """Return the vendor-neutral structured tool definition.

        RFC-0006: 中性 Structured Tool Definitions

        structured 模式下，Tool 先产出 neutral definition；provider-specific
        schema 在后续 LLM adapter 中按 ``api_type`` 再做延迟转换。
        """

        return build_structured_tool_definition(
            name=self.name,
            description=self.description if description is None else description,
            input_schema=self.get_schema(),
            kind=kind,
        )

    def to_openai(self) -> ChatCompletionToolParam:
        """Return the OpenAI-compatible function tool schema.

        RFC-0006: Provider 延迟适配兼容包装

        该方法保留为兼容入口，内部委托 neutral structured definition →
        OpenAI adapter，而不是把 OpenAI schema 当作 runtime 主表示。
        """

        return structured_tool_definition_to_openai(self.to_structured_definition())

    def to_anthropic(self) -> ToolParam:
        """Return the Anthropic-compatible tool schema.

        RFC-0006: Provider 延迟适配兼容包装

        该方法保留为兼容入口，内部委托 neutral structured definition →
        Anthropic adapter，而不是在 Tool 层直接持有 Anthropic 主形状。
        """

        return structured_tool_definition_to_anthropic(self.to_structured_definition())
