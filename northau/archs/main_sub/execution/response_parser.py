"""Unified parser for LLM responses containing tool calls, sub-agent calls, and batch operations."""

import logging
import re
import xml.etree.ElementTree as ET

from ..utils.xml_utils import XMLParser
from .parse_structures import BatchAgentCall, ParsedResponse, SubAgentCall, ToolCall

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
        batch_agent_pattern = r"<use_batch_agent>(.*?)</use_batch_agent>"
        batch_agent_match = re.search(batch_agent_pattern, response, re.DOTALL)

        if batch_agent_match:
            logger.info("📊 Found batch agent call")
            batch_call = self._parse_batch_agent_call(
                batch_agent_match.group(1),
            )
            if batch_call:
                batch_agent_calls.append(batch_call)

        # Check for parallel execution formats
        parallel_tool_calls_pattern = r"<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>"
        parallel_tool_calls_match = re.search(
            parallel_tool_calls_pattern,
            response,
            re.DOTALL,
        )

        if parallel_tool_calls_match:
            logger.info("🔧⚡ Found parallel tool calls")
            is_parallel_tools = True
            parallel_content = parallel_tool_calls_match.group(1)
            tool_pattern = r"<parallel_tool>(.*?)</parallel_tool>"
            tool_matches = re.findall(
                tool_pattern,
                parallel_content,
                re.DOTALL,
            )
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                if tool_call.tool_name.startswith("agent:"):
                    sub_agent_call = self._parse_sub_agent_call(tool_xml)
                    if sub_agent_call:
                        sub_agent_calls.append(sub_agent_call)
                else:
                    if tool_call:
                        tool_calls.append(tool_call)

        else:
            # Fall back to individual tool calls and sub-agent calls
            logger.info("🔧 Looking for individual tool and sub-agent calls")

            # Find individual tool calls
            tool_pattern = r"<tool_use>(.*?)</tool_use>"
            tool_matches = re.findall(tool_pattern, response, re.DOTALL)
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                logger.info(
                    f"🔍 Parsed tool call: {tool_call} with tool_name: {tool_call.tool_name}",
                )
                if tool_call.tool_name.startswith("agent:"):
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
            tool_name_elem = root.find("tool_name")
            if tool_name_elem is None:
                logger.warning("❌ Missing tool_name in tool_use XML")
                return None

            tool_name = (tool_name_elem.text or "").strip()

            # Get parameters
            parameters = {}
            params_elem = root.find("parameter")
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    param_value = param.text or ""

                    # Handle nested content (like HTML) by getting all text content
                    if not param_value.strip() and len(param) > 0:
                        # If the parameter has child elements, get the full XML content
                        param_value = ET.tostring(param, encoding="unicode", method="html")
                        # Remove the outer tag to get just the content
                        if param_value.startswith(f"<{param_name}"):
                            start_tag_end = param_value.find(">") + 1
                            end_tag_start = param_value.rfind(f"</{param_name}>")
                            if start_tag_end > 0 and end_tag_start > start_tag_end:
                                param_value = param_value[start_tag_end:end_tag_start]

                    parameters[param_name] = param_value.strip() if param_value else ""

            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                xml_content=xml_content,
            )

        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in tool call: {e}")
            logger.debug(f"XML content preview: {xml_content[:200]}...")

            # Try regex-based parsing as fallback for malformed XML
            tool_call = self._parse_tool_call_with_regex(xml_content)
            if tool_call:
                return tool_call
            return None
        except ValueError as e:
            logger.error(
                f"❌ XML parsing strategies exhausted for tool call: {e}",
            )
            logger.debug(f"Full XML content: {xml_content}")

            # Try regex-based parsing as fallback before raw content fallback
            tool_call = self._parse_tool_call_with_regex(xml_content)
            if tool_call:
                return tool_call

            # Try one final fallback: extract tool name and treat all content as raw
            from ..utils.xml_utils import XMLUtils

            tool_name = XMLUtils.extract_tool_name_from_xml(xml_content)
            if tool_name != "unknown":
                logger.info(
                    f"🔧 Fallback: Creating minimal tool call for {tool_name}",
                )
                return ToolCall(
                    tool_name=tool_name,
                    parameters={"raw_xml_content": xml_content},
                    xml_content=xml_content,
                )
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing tool call: {e}")
            logger.debug(f"XML content: {xml_content}")
            return None

    def _parse_tool_call_with_regex(self, xml_content: str) -> ToolCall:
        """Parse tool call using regex as fallback for malformed XML."""
        try:
            # Extract tool name
            tool_name_match = re.search(r"<tool_name>\s*([^<]+)\s*</tool_name>", xml_content, re.DOTALL)
            if not tool_name_match:
                logger.warning("❌ Missing tool_name in regex parsing")
                return None

            tool_name = tool_name_match.group(1).strip()

            # Extract parameters using shared helper
            parameters = self._extract_parameters_with_regex(
                xml_content=xml_content,
                prefer_parameter_block=True,
                allow_global_fallback=False,
            )

            logger.info(f"🔧 Regex fallback parsed {len(parameters)} parameters for {tool_name}")
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                xml_content=xml_content,
            )

        except Exception as e:
            logger.error(f"❌ Regex parsing also failed: {e}")
            return None

    def _extract_parameters_with_regex(
        self,
        xml_content: str,
        prefer_parameter_block: bool = True,
        allow_global_fallback: bool = False,
        exclude_tags: set[str] | None = None,
    ) -> dict[str, str]:
        """Extract parameters from XML-ish content using regex with basic nesting support.

        - If prefer_parameter_block is True, will first try within the first <parameter>...</parameter> block.
        - If allow_global_fallback is True and no parameter block is found, will scan the entire content.
        - exclude_tags can be provided to skip structural tags when doing global scanning.
        """
        parameters: dict[str, str] = {}

        # Try to find a <parameter>...</parameter> block
        param_block_match = None
        if prefer_parameter_block:
            param_block_match = re.search(
                r"<parameter[^>]*>(.*?)</parameter>",
                xml_content,
                re.DOTALL | re.IGNORECASE,
            )

        if param_block_match:
            param_content = param_block_match.group(1)

            # Walk through param_content and capture child tags with basic nesting support
            current_pos = 0
            content_len = len(param_content)
            while current_pos < content_len:
                tag_start = param_content.find("<", current_pos)
                if tag_start == -1:
                    break

                tag_end = param_content.find(">", tag_start)
                if tag_end == -1:
                    break

                raw_tag_header = param_content[tag_start + 1 : tag_end].strip()
                if not raw_tag_header or raw_tag_header.startswith("/"):
                    current_pos = tag_end + 1
                    continue

                # Support attributes by splitting on whitespace
                tag_name = raw_tag_header.split()[0]

                # Basic validation: tag names are alnum or underscore
                if not tag_name.replace("_", "").isalnum():
                    current_pos = tag_end + 1
                    continue

                # Prepare search tokens
                closing_tag = f"</{tag_name}>"
                # For nested same-name tags, consider openings with optional attributes
                open_token = f"<{tag_name}"

                search_start = tag_end + 1
                open_count = 1
                pos = search_start
                while pos < content_len and open_count > 0:
                    next_open = param_content.find(open_token, pos)
                    next_close = param_content.find(closing_tag, pos)

                    if next_close == -1:
                        break

                    if next_open != -1 and next_open < next_close:
                        open_count += 1
                        pos = next_open + len(open_token)
                    else:
                        open_count -= 1
                        if open_count == 0:
                            inner_value = param_content[search_start:next_close]
                            parameters[tag_name] = inner_value.strip()
                            current_pos = next_close + len(closing_tag)
                            break
                        else:
                            pos = next_close + len(closing_tag)

                if open_count > 0:
                    # Could not find a matching closing tag; skip this tag
                    current_pos = tag_end + 1
            return parameters

        # If no parameter block found, optionally perform a global scan
        if allow_global_fallback:
            exclude = set(exclude_tags or {"parameter", "tool_name", "root"})
            # Capture simple pairs with optional attributes; not robust to cross nesting
            for match in re.finditer(r"<([^/>\s]+)[^>]*>(.*?)</\1>", xml_content, re.DOTALL | re.IGNORECASE):
                name = match.group(1)
                if name in exclude:
                    continue
                value = match.group(2).strip()
                if value:
                    parameters[name] = value
        return parameters

    def _parse_sub_agent_call(self, xml_content: str) -> SubAgentCall:
        """Parse sub-agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)

            # Get agent name
            agent_name_elem = root.find("tool_name")
            if agent_name_elem is None:
                logger.warning("❌ Missing agent_name in sub-agent XML")
                return None

            agent_name = (agent_name_elem.text or "").strip()
            agent_name = agent_name.replace("agent:", "")

            # Get parameters
            parameters = {}
            params_elem = root.find("parameter")
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    param_value = param.text
                    parameters[param_name] = param_value

            # Build formatted message with all parameters
            message_parts = []
            for param_name, param_value in parameters.items():
                if param_value:  # Only include non-empty parameters
                    message_parts.append(f"{param_name}: {param_value}")

            message = "\n".join(message_parts)

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

            # Try one final fallback: extract agent name and collect all parameters
            from ..utils.xml_utils import XMLUtils

            agent_name = XMLUtils.extract_agent_name_from_xml(xml_content)
            if agent_name != "unknown":
                logger.info(
                    f"🤖 Fallback: Creating minimal sub-agent call for {agent_name}",
                )
                # Use shared regex parameter extraction helper
                extracted_params = self._extract_parameters_with_regex(
                    xml_content=xml_content,
                    prefer_parameter_block=True,
                    allow_global_fallback=True,
                    exclude_tags={"parameter", "tool_name", "root"},
                )

                message_parts = [
                    f"{param_name}: {param_value.strip()}"
                    for param_name, param_value in extracted_params.items()
                    if param_value and param_value.strip()
                ]

                message = "\n".join(message_parts) if message_parts else "Unable to parse message content"
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
            agent_name_elem = root.find("agent_name")
            if agent_name_elem is None:
                logger.warning("❌ Missing agent_name in batch agent XML")
                return None

            agent_name = (agent_name_elem.text or "").strip()

            # Get input data source
            input_data_elem = root.find("input_data_source")
            if input_data_elem is None:
                logger.warning(
                    "❌ Missing input_data_source in batch agent XML",
                )
                return None

            file_name_elem = input_data_elem.find("file_name")
            if file_name_elem is None:
                logger.warning("❌ Missing file_name in input_data_source")
                return None

            file_path = (file_name_elem.text or "").strip()

            format_elem = input_data_elem.find("format")
            data_format = (format_elem.text or "jsonl").strip() if format_elem is not None else "jsonl"

            # Get message template
            message_elem = root.find("message")
            if message_elem is None:
                logger.warning("❌ Missing message in batch agent XML")
                return None

            message_template = (message_elem.text or "").strip()

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
