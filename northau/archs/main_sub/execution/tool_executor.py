"""Tool execution management with XML parsing and parallel execution."""

import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import xml.etree.ElementTree as ET
import html

from ..utils.xml_utils import XMLParser, XMLUtils
from ..tracing.tracer import Tracer

logger = logging.getLogger(__name__)

try:
    from langfuse.client import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None


class ToolExecutor:
    """Handles tool execution with XML parsing and type conversion."""
    
    def __init__(self, tool_registry: Dict[str, Any], stop_tools: set, langfuse_client=None):
        """Initialize tool executor.
        
        Args:
            tool_registry: Dictionary mapping tool names to tool objects
            stop_tools: Set of tool names that should trigger execution stop
            langfuse_client: Optional Langfuse client for tracing
        """
        self.tool_registry = tool_registry
        self.stop_tools = stop_tools
        self.langfuse_client = langfuse_client
        self.xml_parser = XMLParser()
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool is not found
        """
        logger.info(f"🔧 Executing tool '{tool_name}' with parameters: {parameters}")
        
        if tool_name not in self.tool_registry:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        tool = self.tool_registry[tool_name]
        try:
            if self.langfuse_client:
                try:
                    with self.langfuse_client.start_as_current_generation(
                        name=f"tool_{tool_name}",
                        input=parameters,
                        metadata={"tool_name": tool_name, "type": "tool_execution"}
                    ) as generation:
                        result = tool.execute(**parameters)
                        self.langfuse_client.update_current_generation(output=result)
                    self.langfuse_client.flush()
                except Exception as langfuse_error:
                    logger.warning(f"⚠️ Langfuse tool tracing failed: {langfuse_error}")
                    result = tool.execute(**parameters)
            else:
                result = tool.execute(**parameters)
            
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
            
            # Check if this is a stop tool
            if tool_name in self.stop_tools:
                logger.info(f"🛑 Stop tool '{tool_name}' executed, marking for early termination")
                # Add special marker to indicate this is a stop tool result
                if isinstance(result, dict):
                    result['_is_stop_tool'] = True
                else:
                    # Wrap non-dict results to include the marker
                    result = {'result': result, '_is_stop_tool': True}
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' execution failed: {e}")
            raise
    
    def execute_tool_from_xml(self, xml_content: str, tracer: Optional[Tracer] = None) -> Tuple[str, str]:
        """Execute a tool from XML content.
        
        Args:
            xml_content: XML content describing the tool call
            tracer: Optional tracer for logging
            
        Returns:
            Tuple of (tool_name, result_json)
        """
        try:
            # Parse XML using robust parsing
            root = self.xml_parser.parse_xml_content(xml_content)
            
            # Get tool name
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is None:
                raise ValueError("Missing tool_name in tool_use XML")
            
            tool_name = (tool_name_elem.text or "").strip()
            
            # Get parameters
            parameters = {}
            params_elem = root.find('parameter')
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    
                    # Check if parameter has child elements (nested XML structure)
                    if len(param) > 0:
                        # Parameter has child elements - parse as nested XML into dictionary
                        param_value = self.xml_parser.parse_nested_xml_to_dict(param)
                    else:
                        # Handle both regular text and CDATA content, preserving whitespace for JSON
                        if param.text is not None:
                            param_value = param.text
                            # If there's tail text, include it
                            if param.tail:
                                param_value = ''.join(param.itertext())
                        else:
                            # Handle case where content is in CDATA or mixed content
                            param_value = ''.join(param.itertext()) or ""
                        
                        # Unescape HTML entities in parameter values
                        param_value = html.unescape(param_value)
                        
                        # Don't strip whitespace for JSON parameters as it might be significant
                        if param_value.strip().startswith(('{', '[')):
                            # Likely JSON, preserve formatting but clean up excessive whitespace
                            param_value = param_value.strip()
                        else:
                            param_value = param_value.strip()
                    
                    parameters[param_name] = self._convert_parameter_type(
                        tool_name, param_name, param_value
                    )
            
            # Log tool request to trace if enabled
            if tracer:
                tracer.add_tool_request(tool_name, parameters)
            
            # Execute tool
            result = self.execute_tool(tool_name, parameters)
            
            # Log tool response to trace if enabled
            if tracer:
                tracer.add_tool_response(tool_name, result)
            
            return tool_name, json.dumps(result, indent=2, ensure_ascii=False)
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
    
    def execute_tool_from_xml_safe(self, xml_content: str, tracer: Optional[Tracer] = None) -> Tuple[str, str, bool]:
        """Safe wrapper for execute_tool_from_xml that handles exceptions.
        
        Args:
            xml_content: XML content describing the tool call
            tracer: Optional tracer for logging
            
        Returns:
            Tuple of (tool_name, result, is_error)
        """
        try:
            tool_name, result = self.execute_tool_from_xml(xml_content, tracer)
            return tool_name, result, False
        except Exception as e:
            # Extract tool name for error reporting using more robust parsing
            tool_name = XMLUtils.extract_tool_name_from_xml(xml_content)
            return tool_name, str(e), True
    
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
        schema = getattr(tool, 'input_schema', {})
        properties = schema.get('properties', {})
        
        if param_name not in properties:
            return param_value  # Return as string if parameter not in schema
        
        param_info = properties[param_name]
        param_type = param_info.get('type', 'string')
        
        try:
            if param_type == 'boolean':
                if isinstance(param_value, bool):
                    return param_value
                return param_value.lower() in ('true', '1', 'yes', 'on')
            elif param_type == 'integer':
                return int(param_value)
            elif param_type == 'number':
                return float(param_value)
            elif param_type == 'array':
                # Try to parse as JSON array, fallback to comma-separated
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError as json_err:
                    logger.debug(f"JSON parsing failed for array parameter '{param_name}': {json_err}. Trying comma-separated fallback.")
                    return [item.strip() for item in param_value.split(',')]
            elif param_type == 'object':
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse JSON object for parameter '{param_name}': {json_err}")
                    logger.error(f"Parameter value was: {repr(param_value)}")
                    raise ValueError(f"Invalid JSON for parameter '{param_name}': {json_err}")
            else:  # string or unknown type
                return param_value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to convert parameter '{param_name}' of type '{param_type}': {e}")
            logger.warning(f"Parameter value was: {repr(param_value)}")
            return param_value  # Fallback to string value