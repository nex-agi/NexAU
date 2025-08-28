"""Unified parser for LLM responses containing tool calls, sub-agent calls, and batch operations."""

import re
import logging
from typing import List
import xml.etree.ElementTree as ET

from .parse_structures import ParsedResponse, ToolCall, SubAgentCall, BatchAgentCall
from ..utils.xml_utils import XMLParser

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses LLM responses to extract all executable calls."""
    
    def __init__(self):
        self.xml_parser = XMLParser()
    
    def parse_response(self, response: str) -> ParsedResponse:
        """Parse LLM response to extract all tool calls, sub-agent calls, and batch operations.
        
        Args:
            response: The LLM response containing XML calls
            
        Returns:
            ParsedResponse containing all parsed calls
        """
        logger.info("🔍 Parsing LLM response for executable calls")
        
        tool_calls = []
        sub_agent_calls = []
        batch_agent_calls = []
        is_parallel_tools = False
        is_parallel_sub_agents = False
        
        # Check for batch agent calls first (they take priority)
        batch_agent_pattern = r'<use_batch_agent>(.*?)</use_batch_agent>'
        batch_agent_match = re.search(batch_agent_pattern, response, re.DOTALL)
        
        if batch_agent_match:
            logger.info("📊 Found batch agent call")
            batch_call = self._parse_batch_agent_call(batch_agent_match.group(1))
            if batch_call:
                batch_agent_calls.append(batch_call)
        
        # Check for parallel execution formats
        parallel_tool_calls_pattern = r'<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>'
        parallel_tool_calls_match = re.search(parallel_tool_calls_pattern, response, re.DOTALL)
        
        parallel_sub_agents_pattern = r'<use_parallel_sub_agents>(.*?)</use_parallel_sub_agents>'
        parallel_sub_agents_match = re.search(parallel_sub_agents_pattern, response, re.DOTALL)
        
        if parallel_tool_calls_match:
            logger.info("🔧⚡ Found parallel tool calls")
            is_parallel_tools = True
            parallel_content = parallel_tool_calls_match.group(1)
            tool_pattern = r'<parallel_tool>(.*?)</parallel_tool>'
            tool_matches = re.findall(tool_pattern, parallel_content, re.DOTALL)
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                if tool_call:
                    tool_calls.append(tool_call)
        
        elif parallel_sub_agents_match:
            logger.info("🤖⚡ Found parallel sub-agents")
            is_parallel_sub_agents = True
            parallel_content = parallel_sub_agents_match.group(1)
            
            # Extract tool calls within parallel sub-agents
            tool_pattern = r'<parallel_tool>(.*?)</parallel_tool>'
            tool_matches = re.findall(tool_pattern, parallel_content, re.DOTALL)
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                if tool_call:
                    tool_calls.append(tool_call)
            
            # Extract sub-agent calls within parallel sub-agents
            sub_agent_pattern = r'<parallel_agent>(.*?)</parallel_agent>'
            sub_agent_matches = re.findall(sub_agent_pattern, parallel_content, re.DOTALL)
            for sub_agent_xml in sub_agent_matches:
                sub_agent_call = self._parse_sub_agent_call(sub_agent_xml)
                if sub_agent_call:
                    sub_agent_calls.append(sub_agent_call)
        
        else:
            # Fall back to individual tool calls and sub-agent calls
            logger.info("🔧 Looking for individual tool and sub-agent calls")
            
            # Find individual tool calls
            tool_pattern = r'<tool_use>(.*?)</tool_use>'
            tool_matches = re.findall(tool_pattern, response, re.DOTALL)
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                if tool_call:
                    tool_calls.append(tool_call)
            
            # Find individual sub-agent calls
            sub_agent_pattern = r'<sub_agent>(.*?)</sub_agent>'
            sub_agent_matches = re.findall(sub_agent_pattern, response, re.DOTALL)
            for sub_agent_xml in sub_agent_matches:
                sub_agent_call = self._parse_sub_agent_call(sub_agent_xml)
                if sub_agent_call:
                    sub_agent_calls.append(sub_agent_call)
        
        parsed_response = ParsedResponse(
            original_response=response,
            tool_calls=tool_calls,
            sub_agent_calls=sub_agent_calls,
            batch_agent_calls=batch_agent_calls,
            is_parallel_tools=is_parallel_tools,
            is_parallel_sub_agents=is_parallel_sub_agents
        )
        
        logger.info(f"✅ Parsed response: {parsed_response.get_call_summary()}")
        return parsed_response
    
    def _parse_tool_call(self, xml_content: str) -> ToolCall:
        """Parse tool call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)
            
            # Get tool name
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is None:
                logger.warning("❌ Missing tool_name in tool_use XML")
                return None
            
            tool_name = (tool_name_elem.text or "").strip()
            
            # Get parameters
            parameters = {}
            params_elem = root.find('parameter')
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    
                    # Check if parameter has child elements (nested XML structure)
                    if len(param) > 0:
                        param_value = self.xml_parser.parse_nested_xml_to_dict(param)
                    else:
                        # Handle both regular text and CDATA content
                        if param.text is not None:
                            param_value = param.text
                            if param.tail:
                                param_value = ''.join(param.itertext())
                        else:
                            param_value = ''.join(param.itertext()) or ""
                        
                        import html
                        param_value = html.unescape(param_value)
                        
                        if param_value.strip().startswith(('{', '[')):
                            param_value = param_value.strip()
                        else:
                            param_value = param_value.strip()
                    
                    parameters[param_name] = param_value
            
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                xml_content=xml_content
            )
            
        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in tool call: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing tool call: {e}")
            return None
    
    def _parse_sub_agent_call(self, xml_content: str) -> SubAgentCall:
        """Parse sub-agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)
            
            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                logger.warning("❌ Missing agent_name in sub-agent XML")
                return None
            
            agent_name = (agent_name_elem.text or "").strip()
            
            # Get message
            message_elem = root.find('message')
            if message_elem is None:
                logger.warning("❌ Missing message in sub-agent XML")
                return None
            
            message = (message_elem.text or "").strip()
            
            return SubAgentCall(
                agent_name=agent_name,
                message=message,
                xml_content=xml_content
            )
            
        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in sub-agent call: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing sub-agent call: {e}")
            return None
    
    def _parse_batch_agent_call(self, xml_content: str) -> BatchAgentCall:
        """Parse batch agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)
            
            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                logger.warning("❌ Missing agent_name in batch agent XML")
                return None
            
            agent_name = (agent_name_elem.text or "").strip()
            
            # Get input data source
            input_data_elem = root.find('input_data_source')
            if input_data_elem is None:
                logger.warning("❌ Missing input_data_source in batch agent XML")
                return None
            
            file_name_elem = input_data_elem.find('file_name')
            if file_name_elem is None:
                logger.warning("❌ Missing file_name in input_data_source")
                return None
            
            file_path = (file_name_elem.text or "").strip()
            
            format_elem = input_data_elem.find('format')
            data_format = (format_elem.text or "jsonl").strip() if format_elem is not None else "jsonl"
            
            # Get message template
            message_elem = root.find('message')
            if message_elem is None:
                logger.warning("❌ Missing message in batch agent XML")
                return None
            
            message_template = (message_elem.text or "").strip()
            
            return BatchAgentCall(
                agent_name=agent_name,
                file_path=file_path,
                data_format=data_format,
                message_template=message_template,
                xml_content=xml_content
            )
            
        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in batch agent call: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing batch agent call: {e}")
            return None