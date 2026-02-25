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

"""Tool execution management with XML parsing and parallel execution."""

import _thread
import dataclasses
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..agent_state import AgentState

from nexau.archs.sandbox.base_sandbox import BaseSandbox
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import BaseTracer, SpanType

from ..utils.xml_utils import XMLParser
from .hooks import AfterToolHookInput, BeforeToolHookInput, MiddlewareManager, ToolCallParams

JsonDict = dict[str, Any]

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles tool execution with XML parsing and type conversion."""

    def __init__(
        self,
        *,
        tool_registry: dict[str, Tool],
        stop_tools: set[str],
        middleware_manager: MiddlewareManager | None = None,
        registry_lock: _thread.RLock | None = None,
    ):
        """Initialize tool executor.

        Args:
            tool_registry: Dictionary mapping tool names to tool objects
            stop_tools: Set of tool names that should trigger execution stop
            middleware_manager: Optional middleware manager
            registry_lock: Optional shared lock protecting tool_registry access
        """
        self.tool_registry: dict[str, Any] = tool_registry
        self.stop_tools: set[str] = stop_tools
        self.xml_parser = XMLParser()
        self.middleware_manager = middleware_manager
        # Re-entrant lock is shared with Executor to synchronize add/read and allow nested acquisitions in hooks.
        self._registry_lock: _thread.RLock = registry_lock or threading.RLock()

    def execute_tool(
        self,
        agent_state: "AgentState",
        tool_name: str,
        parameters: dict[str, Any],
        tool_call_id: str,
        parallel_execution_id: str | None = None,
    ) -> JsonDict:
        """Execute a tool with given parameters.

        Args:
            agent_state: AgentState containing agent context and global storage
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Tool execution result (possibly modified by hooks)

        Raises:
            ValueError: If tool is not found
        """
        with self._registry_lock:
            if tool_name not in self.tool_registry:
                error_msg = f"Tool '{tool_name}' for agent '{agent_state.agent_id}' not found"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            tool = self.tool_registry[tool_name]
            # Fetch tool while holding the lock to avoid TOCTOU races with concurrent registry updates.

        sandbox: BaseSandbox | None = agent_state.get_sandbox()

        tool_parameters = parameters.copy()
        if self.middleware_manager:
            before_input = BeforeToolHookInput(
                agent_state=agent_state,
                sandbox=sandbox,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_parameters,
                parallel_execution_id=parallel_execution_id,
            )
            tool_parameters = self.middleware_manager.run_before_tool(before_input)

        logger.info(
            f"🔧 Executing tool '{tool_name}' for agent '{agent_state.agent_id}' with parameters: {tool_parameters}",
        )

        # Get tracer from global storage
        tracer: BaseTracer | None = agent_state.get_global_value("tracer")

        if tracer:
            return self._execute_tool_with_tracing(
                tracer=tracer,
                agent_state=agent_state,
                sandbox=sandbox,
                tool=tool,
                tool_name=tool_name,
                tool_parameters=tool_parameters,
                tool_call_id=tool_call_id,
            )
        else:
            return self._execute_tool_inner(
                agent_state=agent_state,
                sandbox=sandbox,
                tool=tool,
                tool_name=tool_name,
                tool_parameters=tool_parameters,
                tool_call_id=tool_call_id,
            )

    def _execute_tool_with_tracing(
        self,
        tracer: BaseTracer,
        agent_state: "AgentState",
        sandbox: BaseSandbox | None,
        tool: Any,
        tool_name: str,
        tool_parameters: dict[str, Any],
        tool_call_id: str,
    ) -> JsonDict:
        """Execute tool with tracing enabled.

        Args:
            tracer: The tracer instance
            agent_state: Agent state instance
            tool: The tool object to execute
            tool_name: Name of the tool
            tool_parameters: Parameters for the tool
            tool_call_id: Unique ID for this tool call

        Returns:
            Tool execution result
        """
        span_name = f"Tool: {tool_name}"
        inputs: JsonDict = {
            "parameters": tool_parameters,
            "tool_call_id": tool_call_id,
        }
        attributes: JsonDict = {
            "agent_name": agent_state.agent_name,
            "agent_id": agent_state.agent_id,
        }

        trace_ctx = TraceContext(tracer, span_name, SpanType.TOOL, inputs, attributes)
        with trace_ctx:
            try:
                result = self._execute_tool_inner(
                    agent_state=agent_state,
                    sandbox=sandbox,
                    tool=tool,
                    tool_name=tool_name,
                    tool_parameters=tool_parameters,
                    tool_call_id=tool_call_id,
                )
                trace_ctx.set_outputs({"result": result})
                return result
            except Exception as e:
                logger.error(f"❌ Tool '{tool_name}' execution failed: {e}")
                trace_ctx.set_outputs({"result": {"status": "error", "error": str(e), "error_type": type(e).__name__}})
                raise

    def _execute_tool_inner(
        self,
        agent_state: "AgentState",
        sandbox: BaseSandbox | None,
        tool: Any,
        tool_name: str,
        tool_parameters: dict[str, Any],
        tool_call_id: str,
    ) -> JsonDict:
        """Inner tool execution logic without tracing wrapper.

        Args:
            agent_state: Agent state instance
            tool: The tool object to execute
            tool_name: Name of the tool
            tool_parameters: Parameters for the tool
            tool_call_id: Unique ID for this tool call

        Returns:
            Tool execution result
        """
        execution_params: JsonDict = dict(tool_parameters)
        execution_params["agent_state"] = agent_state
        execution_params["sandbox"] = sandbox

        call_params = ToolCallParams(
            agent_state=agent_state,
            sandbox=sandbox,
            tool_name=tool_name,
            parameters=tool_parameters,
            tool_call_id=tool_call_id,
            execution_params=execution_params,
        )

        def _execute_tool_call(params: ToolCallParams) -> JsonDict | Any:
            exec_params = params.execution_params

            return tool.execute(**exec_params)

        execution_error: Exception | None = None
        try:
            if self.middleware_manager:
                result = self.middleware_manager.wrap_tool_call(call_params, _execute_tool_call)
            else:
                result = _execute_tool_call(call_params)
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' execution failed: {e}")
            execution_error = e
            result = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

        # Normalize result to a dict for downstream processing
        def _make_jsonable(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {str(k): _make_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict):
                obj_dict = cast(dict[Any, Any], obj)
                return {str(k): _make_jsonable(v) for k, v in obj_dict.items()}
            if isinstance(obj, list):
                obj_list = cast(list[Any], obj)
                return [_make_jsonable(v) for v in obj_list]
            return obj

        result = _make_jsonable(result)
        if isinstance(result, dict):
            result_dict: JsonDict = cast(JsonDict, result)
        else:
            result_dict = {"result": result}

        if tool_name in self.stop_tools:
            logger.info(
                f"🛑 Stop tool '{tool_name}' executed, marking for early termination",
            )
            result_dict["_is_stop_tool"] = True

        if self.middleware_manager:
            try:
                hook_input = AfterToolHookInput(
                    agent_state=agent_state,
                    sandbox=sandbox,
                    tool_name=tool_name,
                    tool_input=tool_parameters,
                    tool_output=result_dict,
                    tool_call_id=tool_call_id,
                )
                after_result = self.middleware_manager.run_after_tool(hook_input, result_dict)
                result_dict = cast(JsonDict, after_result) if isinstance(after_result, dict) else {"result": after_result}
            except Exception as hook_error:
                logger.error(f"❌ After-tool middleware execution failed for '{tool_name}': {hook_error}")

        if execution_error:
            raise execution_error

        if result_dict.get("status") == "error" and result_dict.get("_is_stop_tool", False):
            logger.error(
                f"❌ Finish Tool '{tool_name}' execution failed, will continue.",
            )
            result_dict["_is_stop_tool"] = False
        return result_dict

    def convert_parameter_type(
        self,
        tool_name: str,
        param_name: str,
        param_value: Any,
    ) -> Any:
        """Public helper to convert a tool parameter to its declared type."""
        return self._convert_parameter_type(tool_name, param_name, param_value)

    def _convert_parameter_type(
        self,
        tool_name: str,
        param_name: str,
        param_value: Any,
    ) -> Any:
        """Convert parameter value to the correct type based on tool schema.

        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter
            param_value: Raw parameter value

        Returns:
            Converted parameter value
        """
        # If param_value is already a dict (from nested XML parsing), return it as-is for object types
        if isinstance(param_value, dict):
            return cast(JsonDict, param_value)

        # If param_value is already a list, return it as-is for array types
        if isinstance(param_value, list):
            return cast(list[Any], param_value)

        # If param_value is not a string, return as-is (already converted)
        if not isinstance(param_value, str):
            return param_value

        with self._registry_lock:
            tool = self.tool_registry.get(tool_name)
        if tool is None:
            return param_value
        schema = getattr(tool, "input_schema", {})
        properties = schema.get("properties", {})

        if param_name not in properties:
            return param_value  # Return as string if parameter not in schema

        param_info = properties[param_name]
        param_type = param_info.get("type", "string")

        try:
            if param_type == "boolean":
                return param_value.lower() in ("true", "1", "yes", "on")
            elif param_type == "integer":
                return int(param_value)
            elif param_type == "number":
                return float(param_value)
            elif param_type == "array":
                # Try to parse as JSON array, fallback to comma-separated
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError as json_err:
                    logger.debug(
                        f"JSON parsing failed for array parameter '{param_name}': {json_err}. Trying comma-separated fallback.",
                    )
                    return [item.strip() for item in param_value.split(",")]
            elif param_type == "object":
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"Failed to parse JSON object for parameter '{param_name}': {json_err}",
                    )
                    logger.error(f"Parameter value was: {repr(param_value)}")
                    raise ValueError(
                        f"Invalid JSON for parameter '{param_name}': {json_err}",
                    )
            else:  # string or unknown type
                return param_value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                f"Failed to convert parameter '{param_name}' of type '{param_type}': {e}",
            )
            logger.warning(f"Parameter value was: {repr(param_value)}")
            return param_value  # Fallback to string value
