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

"""
Unit tests for ToolExecutor class.

Tests cover:
- Initialization with various configurations
- Tool execution (success and failure cases)
- Tool hooks execution
- Stop tool handling
- Parameter type conversion for all schema types
- Error handling and edge cases
"""

import logging
from unittest.mock import Mock

import pytest

from nexau.archs.main_sub.execution.hooks import (
    AfterToolHookInput,
    FunctionMiddleware,
    HookResult,
    MiddlewareManager,
)
from nexau.archs.main_sub.execution.tool_executor import ToolExecutor
from nexau.archs.tool.tool import ConfigError as ToolConfigError
from nexau.archs.tool.tool import Tool


class TestToolExecutorInitialization:
    """Test ToolExecutor initialization."""

    def test_init_basic(self):
        """Test basic initialization without optional parameters."""
        tool_registry = {"tool1": Mock()}
        stop_tools = {"stop_tool"}

        executor = ToolExecutor(
            tool_registry=tool_registry,
            stop_tools=stop_tools,
        )

        assert executor.tool_registry == tool_registry
        assert executor.stop_tools == stop_tools
        assert executor.middleware_manager is None
        assert executor.xml_parser is not None

    def test_init_with_hook_manager(self):
        """Test initialization with tool hook manager."""
        mock_hook_manager = MiddlewareManager()
        tool_registry = {}
        stop_tools = set()

        executor = ToolExecutor(
            tool_registry=tool_registry,
            stop_tools=stop_tools,
            middleware_manager=mock_hook_manager,
        )

        assert executor.middleware_manager == mock_hook_manager

    def test_init_with_all_parameters(self):
        """Test initialization with all optional parameters."""
        tool_registry = {"tool1": Mock()}
        stop_tools = {"stop_tool"}
        mock_hook_manager = MiddlewareManager()

        executor = ToolExecutor(
            tool_registry=tool_registry,
            stop_tools=stop_tools,
            middleware_manager=mock_hook_manager,
        )

        assert executor.tool_registry == tool_registry
        assert executor.stop_tools == stop_tools
        assert executor.middleware_manager == mock_hook_manager


class TestToolExecutorExecution:
    """Test tool execution functionality."""

    def test_execute_tool_success(self, agent_state):
        """Test successful tool execution."""

        def simple_tool(message: str, agent_state=None) -> dict:
            return {"result": f"Processed: {message}"}

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
            implementation=simple_tool,
        )

        executor = ToolExecutor(
            tool_registry={"simple_tool": tool},
            stop_tools=set(),
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="simple_tool",
            parameters={"message": "test"},
            tool_call_id="call_123",
        )

        assert "result" in result
        assert "Processed: test" in str(result)

    def test_execute_tool_not_found(self, agent_state):
        """Test execution with non-existent tool."""
        executor = ToolExecutor(
            tool_registry={},
            stop_tools=set(),
        )

        with pytest.raises(ValueError) as exc_info:
            executor.execute_tool(
                agent_state=agent_state,
                tool_name="non_existent",
                parameters={},
                tool_call_id="call_123",
            )

        assert "not found" in str(exc_info.value)
        assert "non_existent" in str(exc_info.value)

    def test_execute_tool_with_error(self, agent_state):
        """Test tool execution when tool raises an error."""

        def error_tool(agent_state=None) -> dict:
            raise ValueError("Tool execution failed")

        tool = Tool(
            name="error_tool",
            description="A tool that fails",
            input_schema={"type": "object", "properties": {}},
            implementation=error_tool,
        )

        executor = ToolExecutor(
            tool_registry={"error_tool": tool},
            stop_tools=set(),
        )

        # Tool.execute wraps errors, so it should not raise but return error dict
        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="error_tool",
            parameters={},
            tool_call_id="call_123",
        )

        # Verify error information is in the result
        assert "error" in result
        assert "Tool execution failed" in result["error"]
        assert result.get("error_type") == "ValueError"

    def test_execute_tool_with_agent_state(self, agent_state):
        """Test that agent_state is properly passed to tool."""
        received_state = None

        def state_aware_tool(value: int, agent_state=None) -> dict:
            nonlocal received_state
            received_state = agent_state
            return {"result": value * 2}

        tool = Tool(
            name="state_tool",
            description="A tool that uses agent state",
            input_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
            implementation=state_aware_tool,
        )

        executor = ToolExecutor(
            tool_registry={"state_tool": tool},
            stop_tools=set(),
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="state_tool",
            parameters={"value": 5},
            tool_call_id="call_123",
        )

        assert received_state is agent_state
        assert result["result"] == 10


class TestToolExecutorExtraKwargs:
    """Test how ToolExecutor handles extra_kwargs and unknown params."""

    def test_merge_and_override_extra_kwargs(self, agent_state):
        """extra_kwargs merge and call-time override should survive executor flow."""

        def impl(a=None, b=None, c=None, agent_state=None):
            return {"a": a, "b": b, "c": c}

        tool = Tool(
            name="with_extra",
            description="test extra kwargs",
            input_schema={"type": "object", "properties": {"b": {"type": "number"}, "c": {"type": "number"}}},
            implementation=impl,
            extra_kwargs={"a": 1, "b": 2},
        )

        executor = ToolExecutor(tool_registry={"with_extra": tool}, stop_tools=set())

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="with_extra",
            parameters={"b": 99, "c": 3},
            tool_call_id="call_extra",
        )

        assert result["a"] == 1  # from extra_kwargs
        assert result["b"] == 99  # overridden by call-time parameter
        assert result["c"] == 3

    def test_reserved_keys_in_extra_kwargs_fail(self):
        """Reserved keys should be rejected when constructing the tool."""
        with pytest.raises(ToolConfigError, match="reserved keys"):
            Tool(
                name="bad_extra",
                description="bad",
                input_schema={"type": "object", "properties": {}},
                implementation=lambda **_: {},
                extra_kwargs={"agent_state": "x"},
            )

    def test_unknown_field_passes_through_and_errors(self, agent_state):
        """Unknown field (not in schema) bypasses validation and reaches the function, causing TypeError if not accepted."""

        def impl(a, agent_state=None):
            return {"a": a}

        tool = Tool(
            name="unknown_field",
            description="unknown passthrough",
            input_schema={"type": "object", "properties": {"a": {"type": "number"}}, "additionalProperties": False},
            implementation=impl,
            extra_kwargs={"x": 1},
        )

        executor = ToolExecutor(tool_registry={"unknown_field": tool}, stop_tools=set())

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="unknown_field",
            parameters={"a": 5},
            tool_call_id="call_unknown",
        )

        assert result["error_type"] == "TypeError"
        assert "unexpected keyword argument 'x'" in result["error"]

    def test_extra_kwargs_satisfy_required_fields(self, agent_state):
        """extra_kwargs alone can satisfy required schema fields."""

        def impl(a=None, agent_state=None):
            return {"a": a}

        tool = Tool(
            name="required_from_extra",
            description="required filled by extra",
            input_schema={"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]},
            implementation=impl,
            extra_kwargs={"a": 10},
        )

        executor = ToolExecutor(tool_registry={"required_from_extra": tool}, stop_tools=set())

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="required_from_extra",
            parameters={},
            tool_call_id="call_required",
        )

        assert result["a"] == 10

    def test_unknown_field_with_kwargs_accepted(self, agent_state):
        """Unknown fields should succeed when implementation accepts **kwargs."""

        def impl(a, **kwargs):
            return {"a": a, "extras": kwargs}

        tool = Tool(
            name="unknown_ok",
            description="accept unknown via kwargs",
            input_schema={"type": "object", "properties": {"a": {"type": "number"}}, "additionalProperties": False},
            implementation=impl,
            extra_kwargs={"x": 1},
        )

        executor = ToolExecutor(tool_registry={"unknown_ok": tool}, stop_tools=set())

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="unknown_ok",
            parameters={"a": 5},
            tool_call_id="call_unknown_ok",
        )

        assert result["a"] == 5
        assert result["extras"]["x"] == 1


class TestToolExecutorHooks:
    """Test tool hook execution."""

    def test_execute_tool_with_hook(self, agent_state):
        """Test tool execution with after-tool middleware."""

        def simple_tool(x: int, agent_state=None) -> dict:
            return {"result": x * 2}

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=simple_tool,
        )

        hook = Mock(
            return_value=HookResult.with_modifications(
                tool_output={"result": 10, "modified": True},
            ),
        )
        middleware_manager = MiddlewareManager(
            [FunctionMiddleware(after_tool_hook=hook)],
        )

        executor = ToolExecutor(
            tool_registry={"simple_tool": tool},
            stop_tools=set(),
            middleware_manager=middleware_manager,
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="simple_tool",
            parameters={"x": 5},
            tool_call_id="call_123",
        )

        hook.assert_called_once()
        call_args = hook.call_args[0][0]
        assert isinstance(call_args, AfterToolHookInput)
        assert call_args.tool_name == "simple_tool"
        assert call_args.tool_input == {"x": 5}
        assert result["modified"] is True

    def test_execute_tool_without_hook(self, agent_state):
        """Test tool execution without hooks."""

        def simple_tool(x: int, agent_state=None) -> dict:
            return {"result": x * 2}

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=simple_tool,
        )

        executor = ToolExecutor(
            tool_registry={"simple_tool": tool},
            stop_tools=set(),
            middleware_manager=None,
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="simple_tool",
            parameters={"x": 5},
            tool_call_id="call_123",
        )

        # Result should not be modified
        assert result["result"] == 10
        assert "modified" not in result

    def test_execute_tool_with_before_tool_middleware(self, agent_state):
        """Test that before-tool middleware can modify parameters."""

        def multiplier_tool(x: int, agent_state=None) -> dict:
            return {"result": x * 2}

        tool = Tool(
            name="multiplier_tool",
            description="Multiplies input",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=multiplier_tool,
        )

        def before_hook(hook_input):
            updated = dict(hook_input.tool_input)
            updated["x"] = 10
            return HookResult.with_modifications(tool_input=updated)

        middleware_manager = MiddlewareManager(
            [FunctionMiddleware(before_tool_hook=before_hook)],
        )

        executor = ToolExecutor(
            tool_registry={"multiplier_tool": tool},
            stop_tools=set(),
            middleware_manager=middleware_manager,
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="multiplier_tool",
            parameters={"x": 1},
            tool_call_id="call_321",
        )

        assert result["result"] == 20


class TestToolExecutorStopTools:
    """Test stop tool handling."""

    def test_stop_tool_dict_result(self, agent_state):
        """Test stop tool with dict result."""

        def stop_tool(agent_state=None) -> dict:
            return {"status": "completed"}

        tool = Tool(
            name="stop_tool",
            description="A stop tool",
            input_schema={"type": "object", "properties": {}},
            implementation=stop_tool,
        )

        executor = ToolExecutor(
            tool_registry={"stop_tool": tool},
            stop_tools={"stop_tool"},
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="stop_tool",
            parameters={},
            tool_call_id="call_123",
        )

        assert result["_is_stop_tool"] is True
        assert result["status"] == "completed"

    def test_stop_tool_non_dict_result(self, agent_state):
        """Test stop tool with non-dict result."""

        def stop_tool(agent_state=None) -> str:
            return "Task completed"

        tool = Tool(
            name="stop_tool",
            description="A stop tool",
            input_schema={"type": "object", "properties": {}},
            implementation=stop_tool,
        )

        executor = ToolExecutor(
            tool_registry={"stop_tool": tool},
            stop_tools={"stop_tool"},
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="stop_tool",
            parameters={},
            tool_call_id="call_123",
        )

        # Non-dict results should be wrapped
        assert result["_is_stop_tool"] is True
        assert "result" in result

    def test_stop_tool_with_error_status(self, agent_state):
        """Test stop tool that returns error status."""

        def stop_tool(agent_state=None) -> dict:
            return {"status": "error", "message": "Failed"}

        tool = Tool(
            name="stop_tool",
            description="A stop tool",
            input_schema={"type": "object", "properties": {}},
            implementation=stop_tool,
        )

        executor = ToolExecutor(
            tool_registry={"stop_tool": tool},
            stop_tools={"stop_tool"},
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="stop_tool",
            parameters={},
            tool_call_id="call_123",
        )

        # Error status should clear the stop tool flag
        assert result["_is_stop_tool"] is False
        assert result["status"] == "error"

    def test_non_stop_tool(self, agent_state):
        """Test normal tool (not a stop tool)."""

        def normal_tool(agent_state=None) -> dict:
            return {"result": "success"}

        tool = Tool(
            name="normal_tool",
            description="A normal tool",
            input_schema={"type": "object", "properties": {}},
            implementation=normal_tool,
        )

        executor = ToolExecutor(
            tool_registry={"normal_tool": tool},
            stop_tools=set(),  # Empty stop tools
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="normal_tool",
            parameters={},
            tool_call_id="call_123",
        )

        # Should not have stop tool marker
        assert "_is_stop_tool" not in result


class TestToolExecutorParameterConversion:
    """Test parameter type conversion functionality."""

    def test_convert_boolean_true_values(self):
        """Test conversion of various true boolean values."""

        def bool_tool(flag: bool, agent_state=None) -> dict:
            return {"flag": flag}

        tool = Tool(
            name="bool_tool",
            description="A boolean tool",
            input_schema={"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]},
            implementation=bool_tool,
        )

        executor = ToolExecutor(
            tool_registry={"bool_tool": tool},
            stop_tools=set(),
        )

        # Test various true values
        for true_value in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "On"]:
            result = executor._convert_parameter_type("bool_tool", "flag", true_value)
            assert result is True, f"Failed for value: {true_value}"

    def test_convert_boolean_false_values(self):
        """Test conversion of various false boolean values."""

        def bool_tool(flag: bool, agent_state=None) -> dict:
            return {"flag": flag}

        tool = Tool(
            name="bool_tool",
            description="A boolean tool",
            input_schema={"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]},
            implementation=bool_tool,
        )

        executor = ToolExecutor(
            tool_registry={"bool_tool": tool},
            stop_tools=set(),
        )

        # Test various false values
        for false_value in ["false", "False", "FALSE", "0", "no", "No", "off", "Off"]:
            result = executor._convert_parameter_type("bool_tool", "flag", false_value)
            assert result is False, f"Failed for value: {false_value}"

    def test_convert_boolean_already_bool(self):
        """Test that boolean values are preserved."""
        executor = ToolExecutor(
            tool_registry={},
            stop_tools=set(),
        )

        assert executor._convert_parameter_type("tool", "param", True) is True
        assert executor._convert_parameter_type("tool", "param", False) is False

    def test_convert_integer(self):
        """Test integer conversion."""

        def int_tool(count: int, agent_state=None) -> dict:
            return {"count": count}

        tool = Tool(
            name="int_tool",
            description="An integer tool",
            input_schema={"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]},
            implementation=int_tool,
        )

        executor = ToolExecutor(
            tool_registry={"int_tool": tool},
            stop_tools=set(),
        )

        result = executor._convert_parameter_type("int_tool", "count", "42")
        assert result == 42
        assert isinstance(result, int)

    def test_convert_number_float(self):
        """Test float/number conversion."""

        def num_tool(value: float, agent_state=None) -> dict:
            return {"value": value}

        tool = Tool(
            name="num_tool",
            description="A number tool",
            input_schema={"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
            implementation=num_tool,
        )

        executor = ToolExecutor(
            tool_registry={"num_tool": tool},
            stop_tools=set(),
        )

        result = executor._convert_parameter_type("num_tool", "value", "3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_convert_array_json(self):
        """Test array conversion from JSON."""

        def array_tool(items: list, agent_state=None) -> dict:
            return {"items": items}

        tool = Tool(
            name="array_tool",
            description="An array tool",
            input_schema={"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]},
            implementation=array_tool,
        )

        executor = ToolExecutor(
            tool_registry={"array_tool": tool},
            stop_tools=set(),
        )

        result = executor._convert_parameter_type("array_tool", "items", '["a", "b", "c"]')
        assert result == ["a", "b", "c"]
        assert isinstance(result, list)

    def test_convert_array_comma_separated_fallback(self):
        """Test array conversion fallback to comma-separated."""

        def array_tool(items: list, agent_state=None) -> dict:
            return {"items": items}

        tool = Tool(
            name="array_tool",
            description="An array tool",
            input_schema={"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]},
            implementation=array_tool,
        )

        executor = ToolExecutor(
            tool_registry={"array_tool": tool},
            stop_tools=set(),
        )

        # Should fallback to comma-separated if JSON parsing fails
        result = executor._convert_parameter_type("array_tool", "items", "a, b, c")
        assert result == ["a", "b", "c"]
        assert isinstance(result, list)

    def test_convert_object_json(self):
        """Test object conversion from JSON."""

        def obj_tool(config: dict, agent_state=None) -> dict:
            return {"config": config}

        tool = Tool(
            name="obj_tool",
            description="An object tool",
            input_schema={"type": "object", "properties": {"config": {"type": "object"}}, "required": ["config"]},
            implementation=obj_tool,
        )

        executor = ToolExecutor(
            tool_registry={"obj_tool": tool},
            stop_tools=set(),
        )

        result = executor._convert_parameter_type("obj_tool", "config", '{"key": "value"}')
        assert result == {"key": "value"}
        assert isinstance(result, dict)

    def test_convert_object_invalid_json_fallback(self):
        """Test that invalid JSON for object falls back to returning string."""

        def obj_tool(config: dict, agent_state=None) -> dict:
            return {"config": config}

        tool = Tool(
            name="obj_tool",
            description="An object tool",
            input_schema={"type": "object", "properties": {"config": {"type": "object"}}, "required": ["config"]},
            implementation=obj_tool,
        )

        executor = ToolExecutor(
            tool_registry={"obj_tool": tool},
            stop_tools=set(),
        )

        # Should fallback to string if JSON parsing fails (logs warning but doesn't raise)
        result = executor._convert_parameter_type("obj_tool", "config", "invalid json")
        assert result == "invalid json"

    def test_convert_string_unchanged(self):
        """Test that string values remain unchanged."""

        def str_tool(text: str, agent_state=None) -> dict:
            return {"text": text}

        tool = Tool(
            name="str_tool",
            description="A string tool",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            implementation=str_tool,
        )

        executor = ToolExecutor(
            tool_registry={"str_tool": tool},
            stop_tools=set(),
        )

        result = executor._convert_parameter_type("str_tool", "text", "hello world")
        assert result == "hello world"

    def test_convert_already_dict(self):
        """Test that dict values are returned as-is."""
        executor = ToolExecutor(
            tool_registry={},
            stop_tools=set(),
        )

        input_dict = {"key": "value"}
        result = executor._convert_parameter_type("tool", "param", input_dict)
        assert result is input_dict

    def test_convert_already_list(self):
        """Test that list values are returned as-is."""
        executor = ToolExecutor(
            tool_registry={},
            stop_tools=set(),
        )

        input_list = ["a", "b", "c"]
        result = executor._convert_parameter_type("tool", "param", input_list)
        assert result is input_list

    def test_convert_non_string_unchanged(self):
        """Test that non-string values are returned as-is."""
        executor = ToolExecutor(
            tool_registry={},
            stop_tools=set(),
        )

        assert executor._convert_parameter_type("tool", "param", 42) == 42
        assert executor._convert_parameter_type("tool", "param", 3.14) == 3.14
        assert executor._convert_parameter_type("tool", "param", True) is True

    def test_convert_tool_not_found(self):
        """Test conversion when tool is not in registry."""
        executor = ToolExecutor(
            tool_registry={},
            stop_tools=set(),
        )

        # Should return value as-is if tool not found
        result = executor._convert_parameter_type("non_existent", "param", "value")
        assert result == "value"

    def test_convert_param_not_in_schema(self):
        """Test conversion when parameter is not in schema."""

        def simple_tool(x: int, agent_state=None) -> dict:
            return {"x": x}

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=simple_tool,
        )

        executor = ToolExecutor(
            tool_registry={"simple_tool": tool},
            stop_tools=set(),
        )

        # Should return value as-is if parameter not in schema
        result = executor._convert_parameter_type("simple_tool", "unknown_param", "value")
        assert result == "value"

    def test_convert_invalid_integer_fallback(self):
        """Test that invalid integer conversion falls back to string."""

        def int_tool(count: int, agent_state=None) -> dict:
            return {"count": count}

        tool = Tool(
            name="int_tool",
            description="An integer tool",
            input_schema={"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]},
            implementation=int_tool,
        )

        executor = ToolExecutor(
            tool_registry={"int_tool": tool},
            stop_tools=set(),
        )

        # Should fallback to string if conversion fails
        result = executor._convert_parameter_type("int_tool", "count", "not_a_number")
        assert result == "not_a_number"

    def test_convert_invalid_float_fallback(self):
        """Test that invalid float conversion falls back to string."""

        def num_tool(value: float, agent_state=None) -> dict:
            return {"value": value}

        tool = Tool(
            name="num_tool",
            description="A number tool",
            input_schema={"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
            implementation=num_tool,
        )

        executor = ToolExecutor(
            tool_registry={"num_tool": tool},
            stop_tools=set(),
        )

        # Should fallback to string if conversion fails
        result = executor._convert_parameter_type("num_tool", "value", "not_a_number")
        assert result == "not_a_number"


class TestToolExecutorEdgeCases:
    """Test edge cases and error conditions."""

    def test_execute_tool_with_complex_parameters(self, agent_state):
        """Test tool execution with complex nested parameters."""

        def complex_tool(config: dict, items: list, agent_state=None) -> dict:
            return {"config": config, "items": items}

        tool = Tool(
            name="complex_tool",
            description="A complex tool",
            input_schema={
                "type": "object",
                "properties": {"config": {"type": "object"}, "items": {"type": "array"}},
                "required": ["config", "items"],
            },
            implementation=complex_tool,
        )

        executor = ToolExecutor(
            tool_registry={"complex_tool": tool},
            stop_tools=set(),
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="complex_tool",
            parameters={"config": {"nested": {"key": "value"}}, "items": [1, 2, 3]},
            tool_call_id="call_123",
        )

        assert result["config"] == {"nested": {"key": "value"}}
        assert result["items"] == [1, 2, 3]

    def test_execute_tool_with_empty_parameters(self, agent_state):
        """Test tool execution with no parameters."""

        def no_param_tool(agent_state=None) -> dict:
            return {"result": "success"}

        tool = Tool(
            name="no_param_tool",
            description="A tool with no parameters",
            input_schema={"type": "object", "properties": {}},
            implementation=no_param_tool,
        )

        executor = ToolExecutor(
            tool_registry={"no_param_tool": tool},
            stop_tools=set(),
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="no_param_tool",
            parameters={},
            tool_call_id="call_123",
        )

        assert result["result"] == "success"

    def test_execute_tool_logs_correctly(self, agent_state, caplog):
        """Test that tool execution produces appropriate logs."""

        def simple_tool(x: int, agent_state=None) -> dict:
            return {"result": x}

        tool = Tool(
            name="simple_tool",
            description="A simple tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=simple_tool,
        )

        executor = ToolExecutor(
            tool_registry={"simple_tool": tool},
            stop_tools=set(),
        )

        with caplog.at_level(logging.INFO):
            executor.execute_tool(
                agent_state=agent_state,
                tool_name="simple_tool",
                parameters={"x": 5},
                tool_call_id="call_123",
            )

        # Check for expected log messages
        assert any("Executing tool 'simple_tool'" in record.message for record in caplog.records)
        assert any("executed successfully" in record.message for record in caplog.records)

    def test_execute_stop_tool_logs_correctly(self, agent_state, caplog):
        """Test that stop tool execution produces appropriate logs."""

        def stop_tool(agent_state=None) -> dict:
            return {"result": "done"}

        tool = Tool(
            name="stop_tool",
            description="A stop tool",
            input_schema={"type": "object", "properties": {}},
            implementation=stop_tool,
        )

        executor = ToolExecutor(
            tool_registry={"stop_tool": tool},
            stop_tools={"stop_tool"},
        )

        with caplog.at_level(logging.INFO):
            executor.execute_tool(
                agent_state=agent_state,
                tool_name="stop_tool",
                parameters={},
                tool_call_id="call_123",
            )

        # Check for stop tool log message
        assert any("Stop tool" in record.message and "executed" in record.message for record in caplog.records)


class TestToolExecutorIntegration:
    """Integration tests with multiple components."""

    def test_execute_tool_with_all_features(self, agent_state):
        """Test tool execution with hooks, and stop tool combined."""

        def stop_tool(x: int, agent_state=None) -> dict:
            return {"result": x * 2}

        tool = Tool(
            name="stop_tool",
            description="A stop tool with all features",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=stop_tool,
        )

        hook = Mock(
            return_value=HookResult.with_modifications(
                tool_output={"result": 10, "_is_stop_tool": True, "hooked": True},
            ),
        )
        middleware_manager = MiddlewareManager([FunctionMiddleware(after_tool_hook=hook)])

        executor = ToolExecutor(
            tool_registry={"stop_tool": tool},
            stop_tools={"stop_tool"},
            middleware_manager=middleware_manager,
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="stop_tool",
            parameters={"x": 5},
            tool_call_id="call_123",
        )

        # Verify all features worked
        assert result["_is_stop_tool"] is True
        assert result["hooked"] is True
        hook.assert_called_once()


class TestToolExecutorReturnDisplayStripping:
    """Test that returnDisplay is stripped from tool results before returning to LLM."""

    def test_return_display_stripped_from_result(self, agent_state):
        """returnDisplay should be removed from the result dict."""

        def tool_with_display(file_path: str, agent_state=None) -> dict:
            return {
                "content": "file contents here",
                "returnDisplay": "Read 42 lines",
            }

        tool = Tool(
            name="read_tool",
            description="A tool that returns returnDisplay",
            input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            implementation=tool_with_display,
        )

        executor = ToolExecutor(
            tool_registry={"read_tool": tool},
            stop_tools=set(),
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="read_tool",
            parameters={"file_path": "test.txt"},
            tool_call_id="call_123",
        )

        assert "returnDisplay" not in result
        assert result["content"] == "file contents here"

    def test_return_display_streamed_to_middleware_before_stripping(self, agent_state):
        """Middleware (e.g. AgentEventsMiddleware) should still see returnDisplay."""
        captured_return_display = None

        def after_hook(hook_input: AfterToolHookInput) -> HookResult:
            nonlocal captured_return_display
            captured_return_display = hook_input.tool_output.get("returnDisplay")
            return HookResult.no_changes()

        def tool_with_display(agent_state=None) -> dict:
            return {
                "content": "data",
                "returnDisplay": "Summary for UI",
            }

        tool = Tool(
            name="display_tool",
            description="Tool with display",
            input_schema={"type": "object", "properties": {}},
            implementation=tool_with_display,
        )

        middleware_manager = MiddlewareManager(
            [FunctionMiddleware(after_tool_hook=after_hook)],
        )

        executor = ToolExecutor(
            tool_registry={"display_tool": tool},
            stop_tools=set(),
            middleware_manager=middleware_manager,
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="display_tool",
            parameters={},
            tool_call_id="call_456",
        )

        # Middleware saw returnDisplay (for streaming to frontend)
        assert captured_return_display == "Summary for UI"
        # But final result does not contain it (for LLM)
        assert "returnDisplay" not in result
        assert result["content"] == "data"

    def test_no_return_display_is_noop(self, agent_state):
        """Tools without returnDisplay should work unchanged."""

        def plain_tool(agent_state=None) -> dict:
            return {"result": "plain output"}

        tool = Tool(
            name="plain_tool",
            description="Plain tool",
            input_schema={"type": "object", "properties": {}},
            implementation=plain_tool,
        )

        executor = ToolExecutor(
            tool_registry={"plain_tool": tool},
            stop_tools=set(),
        )

        result = executor.execute_tool(
            agent_state=agent_state,
            tool_name="plain_tool",
            parameters={},
            tool_call_id="call_789",
        )

        assert result == {"result": "plain output"}


class TestToolExecutorParallelExecutionId:
    """Test parallel_execution_id propagation in ToolExecutor."""

    def test_execute_tool_passes_parallel_execution_id_to_before_hook(self, agent_state):
        """Test that parallel_execution_id is passed to before_tool hook."""
        captured_parallel_execution_id = None

        def mock_tool(x: int) -> dict:
            return {"result": x * 2}

        def before_hook(hook_input):
            nonlocal captured_parallel_execution_id
            captured_parallel_execution_id = hook_input.parallel_execution_id
            return HookResult()

        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=mock_tool,
        )

        middleware_manager = MiddlewareManager([FunctionMiddleware(before_tool_hook=before_hook)])

        executor = ToolExecutor(
            tool_registry={"test_tool": tool},
            stop_tools=set(),
            middleware_manager=middleware_manager,
        )

        # Execute tool with parallel_execution_id
        executor.execute_tool(
            agent_state=agent_state,
            tool_name="test_tool",
            parameters={"x": 5},
            tool_call_id="call_123",
            parallel_execution_id="test-exec-id-999",
        )

        # Verify hook received parallel_execution_id
        assert captured_parallel_execution_id == "test-exec-id-999"

    def test_execute_tool_without_parallel_execution_id(self, agent_state):
        """Test that tool execution works without parallel_execution_id (backward compatibility)."""
        captured_parallel_execution_id = "not_set"

        def mock_tool(x: int) -> dict:
            return {"result": x * 2}

        def before_hook(hook_input):
            nonlocal captured_parallel_execution_id
            captured_parallel_execution_id = hook_input.parallel_execution_id
            return HookResult()

        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation=mock_tool,
        )

        middleware_manager = MiddlewareManager([FunctionMiddleware(before_tool_hook=before_hook)])

        executor = ToolExecutor(
            tool_registry={"test_tool": tool},
            stop_tools=set(),
            middleware_manager=middleware_manager,
        )

        # Execute tool without parallel_execution_id
        executor.execute_tool(
            agent_state=agent_state,
            tool_name="test_tool",
            parameters={"x": 5},
            tool_call_id="call_123",
        )

        # Verify hook received None for parallel_execution_id
        assert captured_parallel_execution_id is None
