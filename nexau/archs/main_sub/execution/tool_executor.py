"""Tool execution management with XML parsing and parallel execution."""

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent_state import AgentState

from ..utils.xml_utils import XMLParser
from .hooks import AfterToolHookInput, BeforeToolHookInput, MiddlewareManager, ToolCallParams

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None


class ToolExecutor:
    """Handles tool execution with XML parsing and type conversion."""

    def __init__(
        self,
        tool_registry: dict[str, Any],
        stop_tools: set[str],
        langfuse_client=None,
        middleware_manager: MiddlewareManager | None = None,
    ):
        """Initialize tool executor.

        Args:
            tool_registry: Dictionary mapping tool names to tool objects
            stop_tools: Set of tool names that should trigger execution stop
            langfuse_client: Optional Langfuse client for tracing
            middleware_manager: Optional middleware manager
        """
        self.tool_registry = tool_registry
        self.stop_tools = stop_tools
        self.langfuse_client = langfuse_client
        self.xml_parser = XMLParser()
        self.middleware_manager = middleware_manager

    def execute_tool(self, agent_state: "AgentState", tool_name: str, parameters: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
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
        if tool_name not in self.tool_registry:
            error_msg = f"Tool '{tool_name}' for agent '{agent_state.agent_id}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        tool = self.tool_registry[tool_name]

        tool_parameters = parameters.copy()
        if self.middleware_manager:
            before_input = BeforeToolHookInput(
                agent_state=agent_state,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_parameters,
            )
            tool_parameters = self.middleware_manager.run_before_tool(before_input)

        logger.info(
            f"🔧 Executing tool '{tool_name}' for agent '{agent_state.agent_id}' with parameters: {tool_parameters}",
        )

        execution_params = tool_parameters.copy()
        execution_params["agent_state"] = agent_state

        call_params = ToolCallParams(
            agent_state=agent_state,
            tool_name=tool_name,
            parameters=tool_parameters,
            tool_call_id=tool_call_id,
            execution_params=execution_params,
        )

        def _execute_tool_call(params: ToolCallParams) -> dict[str, Any]:
            exec_params = params.execution_params
            tool_input = params.parameters

            if self.langfuse_client:
                try:
                    with self.langfuse_client.start_as_current_generation(
                        name=f"tool_{params.tool_name}",
                        input=tool_input,
                        metadata={
                            "tool_name": params.tool_name,
                            "type": "tool_execution",
                        },
                    ):
                        tool_result = tool.execute(**exec_params)
                        self.langfuse_client.update_current_generation(output=tool_result)
                    self.langfuse_client.flush()
                    return tool_result
                except Exception as langfuse_error:
                    logger.warning(
                        f"⚠️ Langfuse tool tracing failed: {langfuse_error}",
                    )
            return tool.execute(**exec_params)

        wrapped_call: Callable[[ToolCallParams], dict[str, Any]] = _execute_tool_call
        if self.middleware_manager:
            wrapped_call = self.middleware_manager.wrap_tool_call(wrapped_call)

        execution_error = None
        try:
            result = wrapped_call(call_params)
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' execution failed: {e}")
            execution_error = e
            result = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

        if self.middleware_manager and result is not None:
            try:
                hook_input = AfterToolHookInput(
                    agent_state=agent_state,
                    tool_name=tool_name,
                    tool_input=tool_parameters,
                    tool_output=result,
                    tool_call_id=tool_call_id,
                )
                result = self.middleware_manager.run_after_tool(hook_input, result)
            except Exception as hook_error:
                logger.error(f"❌ After-tool middleware execution failed for '{tool_name}': {hook_error}")

        if execution_error:
            raise execution_error

        # Check if this is a stop tool
        if tool_name in self.stop_tools:
            logger.info(
                f"🛑 Stop tool '{tool_name}' executed, marking for early termination",
            )
            if isinstance(result, dict):
                result["_is_stop_tool"] = True
            else:
                # Wrap non-dict results to include the marker
                result = {"result": result, "_is_stop_tool": True}

        if isinstance(result, dict) and result.get("status") == "error" and result.get("_is_stop_tool", False):
            logger.error(
                f"❌ Finish Tool '{tool_name}' execution failed, will continue.",
            )
            result["_is_stop_tool"] = False

        return result

    def _convert_parameter_type(self, tool_name: str, param_name: str, param_value):
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
            return param_value

        # If param_value is already a list, return it as-is for array types
        if isinstance(param_value, list):
            return param_value

        # If param_value is not a string, return as-is (already converted)
        if not isinstance(param_value, str):
            return param_value

        if tool_name not in self.tool_registry:
            return param_value  # Return as string if tool not found

        tool = self.tool_registry[tool_name]
        schema = getattr(tool, "input_schema", {})
        properties = schema.get("properties", {})

        if param_name not in properties:
            return param_value  # Return as string if parameter not in schema

        param_info = properties[param_name]
        param_type = param_info.get("type", "string")

        try:
            if param_type == "boolean":
                if isinstance(param_value, bool):
                    return param_value
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
