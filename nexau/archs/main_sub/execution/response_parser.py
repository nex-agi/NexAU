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

"""Unified parser for LLM responses containing tool calls, sub-agent calls, and batch operations."""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from ..sub_agent_naming import extract_sub_agent_name, is_sub_agent_tool_name
from ..utils.xml_utils import XMLParser
from .model_response import ModelResponse
from .parse_structures import BatchAgentCall, ParsedResponse, SubAgentCall, ToolCall

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses LLM responses to extract all executable calls."""

    def __init__(self) -> None:
        self.xml_parser = XMLParser()

    def parse_response(self, response: str | ModelResponse) -> ParsedResponse:
        """Parse LLM response to extract all tool calls, sub-agent calls, and batch operations."""
        logger.info("🔍 Parsing LLM response for executable calls")

        tool_calls: list[ToolCall] = []
        sub_agent_calls: list[SubAgentCall] = []
        batch_agent_calls: list[BatchAgentCall] = []
        is_parallel_tools = False
        is_parallel_sub_agents = False

        model_response = response if isinstance(response, ModelResponse) else None
        response_text = (model_response.content or "") if model_response else (response or "")

        # Parse OpenAI tool calls first if present
        if model_response and model_response.tool_calls:
            logger.info("🔧 Found structured OpenAI tool calls, normalizing")
            for call in model_response.tool_calls:
                parameters = call.arguments if isinstance(call.arguments, dict) else {"raw_arguments": call.arguments}

                normalized_tool_call = ToolCall(
                    tool_name=call.name,
                    parameters=parameters,
                    raw_content=call.raw_arguments,
                    tool_call_id=call.call_id,
                    source="openai",
                )

                if is_sub_agent_tool_name(call.name):
                    agent_name = extract_sub_agent_name(call.name)
                    if not agent_name:
                        continue
                    message = self._format_parameters_for_message(parameters)
                    sub_agent_calls.append(
                        SubAgentCall(
                            agent_name=agent_name,
                            message=message,
                            raw_content=call.raw_arguments,
                            tool_call_id=call.call_id,
                        ),
                    )
                else:
                    tool_calls.append(normalized_tool_call)

        # Parse XML-based constructs (batch calls, parallel sections, fallback tool parsing)
        xml_report = self._parse_xml_constructs(
            response_text,
            tool_calls,
            sub_agent_calls,
            batch_agent_calls,
        )
        if xml_report.get("is_parallel_tools"):
            is_parallel_tools = True

        parsed_response = ParsedResponse(
            original_response=response_text,
            tool_calls=tool_calls,
            sub_agent_calls=sub_agent_calls,
            batch_agent_calls=batch_agent_calls,
            is_parallel_tools=is_parallel_tools,
            is_parallel_sub_agents=is_parallel_sub_agents,
            model_response=model_response,
        )

        logger.info(f"✅ Parsed response: {parsed_response.get_call_summary()}")
        return parsed_response

    # ---------------------------------------------------------------------
    # XML parsing helpers
    # ---------------------------------------------------------------------

    def _parse_xml_constructs(
        self,
        response_text: str,
        tool_calls: list[ToolCall],
        sub_agent_calls: list[SubAgentCall],
        batch_agent_calls: list[BatchAgentCall],
    ) -> dict[str, bool]:
        """Parse XML-based constructs from the response text."""
        report = {"is_parallel_tools": False}
        # Batch agent calls
        batch_agent_pattern = r"<use_batch_agent>(.*?)</use_batch_agent>"
        batch_agent_match = re.search(batch_agent_pattern, response_text, re.DOTALL)
        if batch_agent_match:
            logger.info("📊 Found batch agent call")
            batch_call = self._parse_batch_agent_call(batch_agent_match.group(1))
            if batch_call:
                batch_agent_calls.append(batch_call)

        # Parallel tool calls (XML-only feature)
        parallel_tool_calls_pattern = r"<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>"
        parallel_tool_calls_match = re.search(
            parallel_tool_calls_pattern,
            response_text,
            re.DOTALL,
        )

        if parallel_tool_calls_match:
            logger.info("🔧⚡ Found parallel tool calls")
            report["is_parallel_tools"] = True
            parallel_content = parallel_tool_calls_match.group(1)
            tool_pattern = r"<parallel_tool>(.*?)</parallel_tool>"
            tool_matches = re.findall(tool_pattern, parallel_content, re.DOTALL)
            for tool_xml in tool_matches:
                tool_call = self._parse_tool_call(tool_xml)
                if tool_call is None:
                    continue
                if is_sub_agent_tool_name(tool_call.tool_name):
                    sub_agent_call = self._parse_sub_agent_call(tool_xml)
                    if sub_agent_call:
                        sub_agent_calls.append(sub_agent_call)
                else:
                    tool_calls.append(tool_call)
            return report  # Parallel block already parsed individual calls

        # Individual XML tool calls (only if not already parsed in parallel block)
        tool_pattern = r"<tool_use>(.*?)</tool_use>"
        tool_matches = re.findall(tool_pattern, response_text, re.DOTALL)
        for tool_xml in tool_matches:
            tool_call = self._parse_tool_call(tool_xml)
            if tool_call is None:
                continue
            logger.info(
                "🔍 Parsed tool call: %s with tool_name: %s",
                tool_call,
                tool_call.tool_name,
            )
            if is_sub_agent_tool_name(tool_call.tool_name):
                sub_agent_call = self._parse_sub_agent_call(tool_xml)
                if sub_agent_call:
                    sub_agent_calls.append(sub_agent_call)
            else:
                tool_calls.append(tool_call)

        return report

    def _parse_tool_call(self, xml_content: str) -> ToolCall | None:
        """Parse tool call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)

            tool_name_elem = root.find("tool_name")
            if tool_name_elem is None:
                logger.warning("❌ Missing tool_name in tool_use XML")
                return None

            tool_name = (tool_name_elem.text or "").strip()

            parameters = {}
            params_elem = root.find("parameter")
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    param_value = param.text or ""

                    if not param_value.strip() and len(param) > 0:
                        param_value = ET.tostring(param, encoding="unicode", method="html")
                        if param_value.startswith(f"<{param_name}"):
                            start_tag_end = param_value.find(">") + 1
                            end_tag_start = param_value.rfind(f"</{param_name}>")
                            if start_tag_end > 0 and end_tag_start > start_tag_end:
                                param_value = param_value[start_tag_end:end_tag_start]

                    parameters[param_name] = param_value.strip() if param_value else ""

            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                raw_content=xml_content,
                source="xml",
            )

        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in tool call: {e}")
            logger.debug(f"XML content preview: {xml_content[:200]}...")
            tool_call = self._parse_tool_call_with_regex(xml_content)
            if tool_call:
                return tool_call
            return None
        except ValueError as e:
            logger.error(
                f"❌ XML parsing strategies exhausted for tool call: {e}",
            )
            logger.debug(f"Full XML content: {xml_content}")

            tool_call = self._parse_tool_call_with_regex(xml_content)
            if tool_call:
                return tool_call

            from ..utils.xml_utils import XMLUtils

            tool_name = XMLUtils.extract_tool_name_from_xml(xml_content)
            if tool_name != "unknown":
                logger.info(
                    f"🔧 Fallback: Creating minimal tool call for {tool_name}",
                )
                return ToolCall(
                    tool_name=tool_name,
                    parameters={"raw_xml_content": xml_content},
                    raw_content=xml_content,
                    source="xml",
                )
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing tool call: {e}")
            logger.debug(f"XML content: {xml_content}")
            return None

    def _parse_tool_call_with_regex(self, xml_content: str) -> ToolCall | None:
        """Parse tool call using regex as fallback for malformed XML."""
        try:
            tool_name_match = re.search(r"<tool_name>\s*([^<]+)\s*</tool_name>", xml_content, re.DOTALL)
            if not tool_name_match:
                logger.warning("❌ Missing tool_name in regex parsing")
                return None

            tool_name = tool_name_match.group(1).strip()

            parameters = self._extract_parameters_with_regex(
                xml_content=xml_content,
                prefer_parameter_block=True,
                allow_global_fallback=False,
            )

            logger.info(f"🔧 Regex fallback parsed {len(parameters)} parameters for {tool_name}")
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                raw_content=xml_content,
                source="xml",
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
        """Extract parameters from XML-ish content using regex with basic nesting support."""
        parameters: dict[str, str] = {}

        param_block_match = None
        if prefer_parameter_block:
            param_block_match = re.search(
                r"<parameter[^>]*>(.*?)</parameter>",
                xml_content,
                re.DOTALL | re.IGNORECASE,
            )

        if param_block_match:
            param_content = param_block_match.group(1)

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

                tag_name = raw_tag_header.split()[0]

                if not tag_name.replace("_", "").isalnum():
                    current_pos = tag_end + 1
                    continue

                closing_tag = f"</{tag_name}>"
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
                    current_pos = tag_end + 1
            return parameters

        if allow_global_fallback:
            exclude = set(exclude_tags or {"parameter", "tool_name", "root"})
            for match in re.finditer(r"<([^/>\s]+)[^>]*>(.*?)</\1>", xml_content, re.DOTALL | re.IGNORECASE):
                name = match.group(1)
                if name in exclude:
                    continue
                value = match.group(2).strip()
                if value:
                    parameters[name] = value
        return parameters

    def _parse_sub_agent_call(self, xml_content: str) -> SubAgentCall | None:
        """Parse sub-agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)

            agent_name_elem = root.find("tool_name")
            if agent_name_elem is None:
                logger.warning("❌ Missing agent_name in sub-agent XML")
                return None

            agent_name = extract_sub_agent_name((agent_name_elem.text or "").strip())
            if not agent_name:
                logger.warning("❌ Unable to extract sub-agent name from tool identifier")
                return None

            parameters: dict[str, Any] = {}
            params_elem = root.find("parameter")
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    param_value = param.text
                    parameters[param_name] = param_value

            message = self._format_parameters_for_message(parameters)

            return SubAgentCall(
                agent_name=agent_name,
                message=message,
                raw_content=xml_content,
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

            from ..utils.xml_utils import XMLUtils

            agent_name = XMLUtils.extract_agent_name_from_xml(xml_content)
            if agent_name != "unknown":
                logger.info(
                    f"🤖 Fallback: Creating minimal sub-agent call for {agent_name}",
                )
                extracted_params = self._extract_parameters_with_regex(
                    xml_content=xml_content,
                    prefer_parameter_block=True,
                    allow_global_fallback=True,
                    exclude_tags={"parameter", "tool_name", "root"},
                )
                message = self._format_parameters_for_message(extracted_params)
                return SubAgentCall(
                    agent_name=agent_name,
                    message=message if message else "Unable to parse message content",
                    raw_content=xml_content,
                )
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing sub-agent call: {e}")
            logger.debug(f"XML content: {xml_content}")
            return None

    def _parse_batch_agent_call(self, xml_content: str) -> BatchAgentCall | None:
        """Parse batch agent call XML content."""
        try:
            root = self.xml_parser.parse_xml_content(xml_content)

            agent_name_elem = root.find("agent_name")
            if agent_name_elem is None:
                logger.warning("❌ Missing agent_name in batch agent XML")
                return None

            agent_name = (agent_name_elem.text or "").strip()

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
                raw_content=xml_content,
            )

        except ET.ParseError as e:
            logger.error(f"❌ Invalid XML format in batch agent call: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing batch agent call: {e}")
            return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _format_parameters_for_message(self, parameters: dict[str, Any]) -> str:
        """Format parameters dictionary into human-readable message string."""
        message_parts: list[str] = []
        for param_name, param_value in parameters.items():
            if param_value is None:
                continue
            if isinstance(param_value, (dict, list)):
                value_str = json.dumps(param_value, ensure_ascii=False)
            else:
                value_str = str(param_value)
            value_str = value_str.strip()
            if value_str:
                message_parts.append(f"{param_name}: {value_str}")
        return "\n".join(message_parts)
