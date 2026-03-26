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

"""Coverage improvement tests for nexau/archs/tool/tool.py.

Targets uncovered paths: execute_async, to_openai, to_anthropic,
get_structured_description, get_info, __repr__, __str__,
validate_params, _validate_schema, execute result coercion.
"""

import dataclasses
from typing import Annotated
from unittest.mock import patch

import pytest

from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.tool import (
    ConfigError,
    Tool,
    normalize_input_schema,
    normalize_structured_tool_definition,
    structured_tool_definition_to_anthropic,
    structured_tool_definition_to_openai,
)

# ---------------------------------------------------------------------------
# normalize_input_schema
# ---------------------------------------------------------------------------


class TestNormalizeInputSchema:
    def test_none_returns_empty_object(self):
        result = normalize_input_schema(None)
        assert result == {"type": "object", "properties": {}}

    def test_empty_dict(self):
        result = normalize_input_schema({})
        assert result == {"type": "object", "properties": {}}

    def test_no_type_key(self):
        result = normalize_input_schema({"properties": {"x": {"type": "string"}}})
        assert result["type"] == "object"

    def test_already_has_type(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = normalize_input_schema(schema)
        assert result == schema


# ---------------------------------------------------------------------------
# normalize_structured_tool_definition
# ---------------------------------------------------------------------------


class TestNormalizeStructuredToolDefinition:
    def test_openai_format(self):
        defn = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        result = normalize_structured_tool_definition(defn)
        assert result["name"] == "my_tool"

    def test_openai_format_missing_name_raises(self):
        defn = {"type": "function", "function": {}}
        with pytest.raises(ValueError, match="missing function.name"):
            normalize_structured_tool_definition(defn)

    def test_neutral_format(self):
        defn = {
            "name": "my_tool",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {}},
            "kind": "tool",
        }
        result = normalize_structured_tool_definition(defn)
        assert result["name"] == "my_tool"

    def test_neutral_format_missing_name_raises(self):
        defn = {"name": "", "input_schema": {}, "description": "d"}
        with pytest.raises(ValueError, match="missing name"):
            normalize_structured_tool_definition(defn)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            normalize_structured_tool_definition({"random_key": True})


# ---------------------------------------------------------------------------
# structured_tool_definition_to_openai / to_anthropic
# ---------------------------------------------------------------------------


class TestStructuredToolDefinitionConverters:
    def test_to_openai(self):
        defn = {
            "name": "my_tool",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {}},
            "kind": "tool",
        }
        result = structured_tool_definition_to_openai(defn)
        assert result["type"] == "function"
        assert result["function"]["name"] == "my_tool"

    def test_to_anthropic(self):
        defn = {
            "name": "my_tool",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {}},
            "kind": "tool",
        }
        result = structured_tool_definition_to_anthropic(defn)
        assert result["name"] == "my_tool"
        assert "eager_input_streaming" in result


# ---------------------------------------------------------------------------
# Tool.execute — result coercion paths
# ---------------------------------------------------------------------------


class TestToolExecuteResultCoercion:
    def test_dict_result_passthrough(self):
        tool = Tool(
            name="dict_tool",
            description="returns dict",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: {"key": "value"},
        )
        result = tool.execute()
        assert result == {"key": "value"}

    def test_list_result_wrapped(self):
        tool = Tool(
            name="list_tool",
            description="returns list",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: [1, 2, 3],
        )
        result = tool.execute()
        assert result == {"result": [1, 2, 3]}

    def test_scalar_result_wrapped(self):
        tool = Tool(
            name="scalar_tool",
            description="returns scalar",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: "hello",
        )
        result = tool.execute()
        assert result == {"result": "hello"}

    def test_dataclass_result_converted(self):
        @dataclasses.dataclass
        class MyResult:
            x: int = 42
            y: str = "ok"

        tool = Tool(
            name="dc_tool",
            description="returns dataclass",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: MyResult(),
        )
        result = tool.execute()
        assert result == {"x": 42, "y": "ok"}

    def test_exception_returns_error_dict(self):
        def failing_impl():
            raise ValueError("Something went wrong")

        tool = Tool(
            name="fail_tool",
            description="fails",
            input_schema={"type": "object", "properties": {}},
            implementation=failing_impl,
        )
        result = tool.execute()
        assert "error" in result
        assert result["error_type"] == "ValueError"

    def test_dataclass_in_list_result(self):
        @dataclasses.dataclass
        class Item:
            name: str = "a"

        tool = Tool(
            name="dc_list_tool",
            description="returns list of dataclasses",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: [Item("x"), Item("y")],
        )
        result = tool.execute()
        assert result == {"result": [{"name": "x"}, {"name": "y"}]}


# ---------------------------------------------------------------------------
# Tool.execute_async
# ---------------------------------------------------------------------------


class TestToolExecuteAsync:
    @pytest.mark.anyio
    async def test_async_impl_directly_awaited(self):
        async def async_impl():
            return {"async": True}

        tool = Tool(
            name="async_tool",
            description="async tool",
            input_schema={"type": "object", "properties": {}},
            implementation=async_impl,
        )
        result = await tool.execute_async()
        assert result == {"async": True}

    @pytest.mark.anyio
    async def test_sync_impl_runs_in_thread(self):
        def sync_impl():
            return {"sync": True}

        tool = Tool(
            name="sync_tool",
            description="sync tool in async",
            input_schema={"type": "object", "properties": {}},
            implementation=sync_impl,
        )
        result = await tool.execute_async()
        assert result == {"sync": True}

    @pytest.mark.anyio
    async def test_execute_async_no_impl_raises(self):
        tool = Tool(
            name="no_impl",
            description="desc",
            input_schema={},
            implementation=None,
        )
        with pytest.raises(ValueError, match="no implementation"):
            await tool.execute_async()

    @pytest.mark.anyio
    async def test_execute_async_lazy_import(self):
        tool = Tool(
            name="lazy_tool",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            implementation="builtins:len",
            lazy=True,
        )
        # The lazy tool tries to import and call builtins:len
        with patch("nexau.archs.main_sub.utils.import_from_string", return_value=lambda: {"ok": True}):
            result = await tool.execute_async()
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# Tool string representations
# ---------------------------------------------------------------------------


class TestToolStringRepresentations:
    def test_repr(self):
        tool = Tool(
            name="test_tool",
            description="desc",
            input_schema={},
            implementation=lambda: None,
        )
        r = repr(tool)
        assert "test_tool" in r

    def test_str_short_description(self):
        tool = Tool(
            name="test_tool",
            description="Short desc",
            input_schema={},
            implementation=lambda: None,
        )
        s = str(tool)
        assert "test_tool" in s
        assert "Short desc" in s

    def test_str_long_description(self):
        tool = Tool(
            name="test_tool",
            description="x" * 100,
            input_schema={},
            implementation=lambda: None,
        )
        s = str(tool)
        assert "..." in s

    def test_str_with_skill_description(self):
        tool = Tool(
            name="skill_tool",
            description="desc",
            input_schema={},
            implementation=lambda: None,
            skill_description="A useful skill",
        )
        s = str(tool)
        assert "Skill description" in s


# ---------------------------------------------------------------------------
# Tool.get_structured_description
# ---------------------------------------------------------------------------


class TestGetStructuredDescription:
    def test_as_skill_with_skill_description(self):
        tool = Tool(
            name="t",
            description="normal",
            input_schema={},
            implementation=lambda: None,
            skill_description="skill desc",
            as_skill=True,
        )
        assert tool.get_structured_description() == "skill desc"

    def test_as_skill_without_skill_description_raises(self):
        tool = Tool(
            name="t",
            description="normal",
            input_schema={},
            implementation=lambda: None,
            as_skill=True,
        )
        with pytest.raises(ValueError, match="no skill_description"):
            tool.get_structured_description()

    def test_normal_tool_returns_description(self):
        tool = Tool(
            name="t",
            description="my description",
            input_schema={},
            implementation=lambda: None,
        )
        assert tool.get_structured_description() == "my description"


# ---------------------------------------------------------------------------
# Tool.get_info / get_schema
# ---------------------------------------------------------------------------


class TestToolInfo:
    def test_get_info(self):
        tool = Tool(
            name="t",
            description="desc",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            implementation=lambda: None,
        )
        info = tool.get_info()
        assert info["name"] == "t"
        assert info["description"] == "desc"

    def test_get_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tool = Tool(name="t", description="d", input_schema=schema, implementation=lambda: None)
        result = tool.get_schema()
        assert result == schema
        assert result is not schema  # should be a copy


# ---------------------------------------------------------------------------
# Tool.to_openai / to_anthropic
# ---------------------------------------------------------------------------


class TestToolProviderConversion:
    def test_to_openai(self):
        tool = Tool(
            name="t",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: None,
        )
        result = tool.to_openai()
        assert result["type"] == "function"
        assert result["function"]["name"] == "t"

    def test_to_anthropic(self):
        tool = Tool(
            name="t",
            description="desc",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda: None,
        )
        result = tool.to_anthropic()
        assert result["name"] == "t"


# ---------------------------------------------------------------------------
# Tool._validate_reserved_param_annotations
# ---------------------------------------------------------------------------


class TestValidateReservedParamAnnotations:
    def test_ctx_as_framework_context_is_valid(self):
        def impl(ctx: FrameworkContext):
            pass

        tool = Tool(
            name="t",
            description="d",
            input_schema={},
            implementation=impl,
        )
        # Should not raise
        assert tool is not None

    def test_ctx_incompatible_type_raises(self):
        def impl(ctx: int):
            pass

        with pytest.raises(ConfigError, match="incompatible type"):
            Tool(name="t", description="d", input_schema={}, implementation=impl)

    def test_ctx_annotated_framework_context_valid(self):
        def impl(ctx: Annotated[FrameworkContext, "some metadata"]):
            pass

        tool = Tool(name="t", description="d", input_schema={}, implementation=impl)
        assert tool is not None

    def test_ctx_optional_framework_context_valid(self):
        def impl(ctx: FrameworkContext | None = None):
            pass

        tool = Tool(name="t", description="d", input_schema={}, implementation=impl)
        assert tool is not None


# ---------------------------------------------------------------------------
# Tool — extra_kwargs conflicts
# ---------------------------------------------------------------------------


class TestExtraKwargsConflicts:
    def test_reserved_keys_raise(self):
        with pytest.raises(ConfigError, match="reserved keys"):
            Tool(
                name="t",
                description="d",
                input_schema={},
                implementation=lambda: None,
                extra_kwargs={"agent_state": "bad"},
            )

    def test_non_reserved_extra_kwargs_ok(self):
        tool = Tool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            implementation=lambda custom_param="val": {"result": custom_param},
            extra_kwargs={"custom_param": "injected"},
        )
        result = tool.execute()
        assert result == {"result": "injected"}
