"""Unified parser for LLM responses containing tool calls, sub-agent calls, and batch operations."""
import logging
import re
import xml.etree.ElementTree as ET

from ..utils.xml_utils import XMLParser
from .parse_structures import BatchAgentCall
from .parse_structures import ParsedResponse
from .parse_structures import SubAgentCall
from .parse_structures import ToolCall

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
        logger.info('🔍 Parsing LLM response for executable calls')

        tool_calls = []
        sub_agent_calls = []
        batch_agent_calls = []
        is_parallel_tools = False
        is_parallel_sub_agents = False

        # Check for batch agent calls first (they take priority)
        batch_agent_pattern = r'<use_batch_agent>(.*?)</use_batch_agent>'
        batch_agent_match = re.search(batch_agent_pattern, response, re.DOTALL)

        if batch_agent_match:
            logger.info('📊 Found batch agent call')
            batch_call = self._parse_batch_agent_call(
                batch_agent_match.group(1),
            )
            if batch_call:
                batch_agent_calls.append(batch_call)

        # Check for parallel execution formats
        parallel_tool_calls_pattern = r'<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>'
        parallel_tool_calls_match = re.search(
            parallel_tool_calls_pattern, response, re.DOTALL,
        )

        if parallel_tool_calls_match:
            logger.info('🔧⚡ Found parallel tool calls')
            is_parallel_tools = True
            parallel_content = parallel_tool_calls_match.group(1)
            tool_pattern = r'<parallel_tool>(.*?)</parallel_tool>'
            tool_matches = re.findall(
                tool_pattern, parallel_content, re.DOTALL,
            )
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                if tool_call.tool_name.startswith('agent:'):
                    sub_agent_call = self._parse_sub_agent_call(tool_xml)
                    if sub_agent_call:
                        sub_agent_calls.append(sub_agent_call)
                else:
                    if tool_call:
                        tool_calls.append(tool_call)

        else:
            # Fall back to individual tool calls and sub-agent calls
            logger.info('🔧 Looking for individual tool and sub-agent calls')

            # Find individual tool calls
            tool_pattern = r'<tool_use>(.*?)</tool_use>'
            tool_matches = re.findall(tool_pattern, response, re.DOTALL)
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                logger.info(
                    f"🔍 Parsed tool call: {tool_call} with tool_name: {tool_call.tool_name}",
                )
                if tool_call.tool_name.startswith('agent:'):
                    sub_agent_call = self._parse_sub_agent_call(tool_xml)
                    if sub_agent_call:
                        sub_agent_calls.append(sub_agent_call)
                else:
                    if tool_call:
                        tool_calls.append(tool_call)

        parsed_response = ParsedResponse(
            original_response=response,
            tool_calls=tool_calls,
            sub_agent_calls=sub_agent_calls,
            batch_agent_calls=batch_agent_calls,
            is_parallel_tools=is_parallel_tools,
            is_parallel_sub_agents=is_parallel_sub_agents,
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
                logger.warning('❌ Missing tool_name in tool_use XML')
                return None

            tool_name = (tool_name_elem.text or '').strip()

            # Get parameters
            parameters = {}
            params_elem = root.find('parameter')
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag

                    # Check if parameter has child elements (nested XML structure)
                    if len(param) > 0:
                        param_value = self.xml_parser.parse_nested_xml_to_dict(
                            param,
                        )
                    else:
                        # Handle both regular text and CDATA content
                        if param.text is not None:
                            param_value = param.text
                            if param.tail:
                                param_value = ''.join(param.itertext())
                        else:
                            param_value = ''.join(param.itertext()) or ''

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
                xml_content=xml_content,
            )

        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in tool call: {e}")
            logger.debug(f"XML content preview: {xml_content[:200]}...")
            return None
        except ValueError as e:
            logger.error(
                f"❌ XML parsing strategies exhausted for tool call: {e}",
            )
            logger.debug(f"Full XML content: {xml_content}")

            # Try one final fallback: extract tool name and treat all content as raw
            from ..utils.xml_utils import XMLUtils
            tool_name = XMLUtils.extract_tool_name_from_xml(xml_content)
            if tool_name != 'unknown':
                logger.info(
                    f"🔧 Fallback: Creating minimal tool call for {tool_name}",
                )
                return ToolCall(
                    tool_name=tool_name,
                    parameters={'raw_xml_content': xml_content},
                    xml_content=xml_content,
                )
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing tool call: {e}")
            logger.debug(f"XML content: {xml_content}")
            return None

    def _parse_sub_agent_call(self, xml_content: str) -> SubAgentCall:
        """Parse sub-agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)

            # Get agent name
            agent_name_elem = root.find('tool_name')
            if agent_name_elem is None:
                logger.warning('❌ Missing agent_name in sub-agent XML')
                return None

            agent_name = (agent_name_elem.text or '').strip()
            agent_name = agent_name.replace('agent:', '')

            # Get parameters
            parameters = {}
            params_elem = root.find('parameter')
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    param_value = param.text
                    parameters[param_name] = param_value

            message = parameters.get('message', '')

            return SubAgentCall(
                agent_name=agent_name,
                message=message,
                xml_content=xml_content,
            )

        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in sub-agent call: {e}")
            logger.debug(f"XML content preview: {xml_content[:200]}...")
            return None
        except ValueError as e:
            logger.error(
                f"❌ XML parsing strategies exhausted for sub-agent call: {e}",
            )
            logger.debug(f"Full XML content: {xml_content}")

            # Try one final fallback: extract agent name and treat message as raw
            from ..utils.xml_utils import XMLUtils
            agent_name = XMLUtils.extract_agent_name_from_xml(xml_content)
            if agent_name != 'unknown':
                logger.info(
                    f"🤖 Fallback: Creating minimal sub-agent call for {agent_name}",
                )
                # Try to extract some message content
                message_match = re.search(
                    r'<message[^>]*>(.*?)</message>', xml_content, re.DOTALL | re.IGNORECASE,
                )
                message = message_match.group(1).strip(
                ) if message_match else 'Unable to parse message content'
                return SubAgentCall(
                    agent_name=agent_name,
                    message=message,
                    xml_content=xml_content,
                )
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing sub-agent call: {e}")
            logger.debug(f"XML content: {xml_content}")
            return None

    def _parse_batch_agent_call(self, xml_content: str) -> BatchAgentCall:
        """Parse batch agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)

            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                logger.warning('❌ Missing agent_name in batch agent XML')
                return None

            agent_name = (agent_name_elem.text or '').strip()

            # Get input data source
            input_data_elem = root.find('input_data_source')
            if input_data_elem is None:
                logger.warning(
                    '❌ Missing input_data_source in batch agent XML',
                )
                return None

            file_name_elem = input_data_elem.find('file_name')
            if file_name_elem is None:
                logger.warning('❌ Missing file_name in input_data_source')
                return None

            file_path = (file_name_elem.text or '').strip()

            format_elem = input_data_elem.find('format')
            data_format = (format_elem.text or 'jsonl').strip(
            ) if format_elem is not None else 'jsonl'

            # Get message template
            message_elem = root.find('message')
            if message_elem is None:
                logger.warning('❌ Missing message in batch agent XML')
                return None

            message_template = (message_elem.text or '').strip()

            return BatchAgentCall(
                agent_name=agent_name,
                file_path=file_path,
                data_format=data_format,
                message_template=message_template,
                xml_content=xml_content,
            )

        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in batch agent call: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing batch agent call: {e}")
            return None
