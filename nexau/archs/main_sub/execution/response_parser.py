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

"""Unified parser for LLM responses containing tool calls.

RFC-0015: Removed batch-agent and sub-agent special parsing; all calls are ToolCall.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, cast

from ..utils.xml_utils import XMLParser
from .model_response import ModelResponse
from .parse_structures import ParsedResponse, ToolCall

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses LLM responses to extract all executable calls."""

    def __init__(self) -> None:
        self.xml_parser = XMLParser()

    def parse_response(self, response: str | ModelResponse) -> ParsedResponse:
        """Parse LLM response to extract all tool calls and batch operations."""
        logger.info("🔍 Parsing LLM response for executable calls")

        tool_calls: list[ToolCall] = []
        is_parallel_tools = False

        model_response = response if isinstance(response, ModelResponse) else None
        if model_response:
            response_text = model_response.content or ""
        else:
            response_text = cast(str, response) or ""

        # Parse OpenAI tool calls first if present
        if model_response and model_response.tool_calls:
            logger.info("🔧 Found structured OpenAI tool calls, normalizing")
            for call in model_response.tool_calls:
                tool_name = (call.name or "").strip()
                if not tool_name:
                    logger.warning("❌ Skipping structured tool call with empty name (call_id=%s)", call.call_id)
                    continue
                parameters: dict[str, Any] = call.arguments

                tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        parameters=parameters,
                        raw_content=call.raw_arguments,
                        tool_call_id=call.call_id,
                        source="structured",
                    ),
                )

        # Parse XML-based constructs (parallel sections, fallback tool parsing)
        xml_report = self._parse_xml_constructs(
            response_text,
            tool_calls,
        )
        if xml_report.get("is_parallel_tools"):
            is_parallel_tools = True

        parsed_response = ParsedResponse(
            original_response=response_text,
            tool_calls=tool_calls,
            is_parallel_tools=is_parallel_tools,
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
    ) -> dict[str, bool]:
        """Parse XML-based constructs from the response text.

        RFC-0015: Removed batch-agent parsing; only parallel and individual tool calls remain.
        """
        report = {"is_parallel_tools": False}

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
                if tool_call is not None:
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

            parameters: dict[str, Any] = {}
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
