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
Unit tests for ResponseParser class.

Tests cover parsing of LLM responses containing:
- Tool calls
- Sub-agent calls
- Batch agent calls
- Parallel operations
- Malformed XML handling
- Edge cases and error scenarios
"""

import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest

from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.execution.parse_structures import (
    BatchAgentCall,
    SubAgentCall,
    ToolCall,
)
from nexau.archs.main_sub.execution.response_parser import ResponseParser


class TestResponseParser:
    """Test cases for ResponseParser class."""

    @pytest.fixture
    def parser(self):
        """Create a ResponseParser instance for testing."""
        return ResponseParser()

    # ========== Basic Parse Response Tests ==========

    def test_parse_empty_response(self, parser):
        """Test parsing empty response."""
        result = parser.parse_response("")

        assert result.original_response == ""
        assert len(result.tool_calls) == 0
        assert len(result.sub_agent_calls) == 0
        assert len(result.batch_agent_calls) == 0
        assert not result.has_calls()
        assert result.get_call_summary() == "no calls"

    def test_parse_response_with_no_calls(self, parser):
        """Test parsing response with no tool calls."""
        response = "This is just a regular text response without any tool calls."
        result = parser.parse_response(response)

        assert result.original_response == response
        assert len(result.tool_calls) == 0
        assert not result.has_calls()

    def test_parse_single_tool_call(self, parser):
        """Test parsing a single tool call."""
        response = """
<tool_use>
<tool_name>sample_tool</tool_name>
<parameter>
<x>42</x>
<y>test_value</y>
</parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "sample_tool"
        assert result.tool_calls[0].parameters["x"] == "42"
        assert result.tool_calls[0].parameters["y"] == "test_value"
        assert result.get_call_summary() == "1 tool calls"

    def test_parse_multiple_tool_calls(self, parser):
        """Test parsing multiple tool calls."""
        response = """
<tool_use>
<tool_name>tool_one</tool_name>
<parameter>
<param1>value1</param1>
</parameter>
</tool_use>
<tool_use>
<tool_name>tool_two</tool_name>
<parameter>
<param2>value2</param2>
</parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "tool_one"
        assert result.tool_calls[1].tool_name == "tool_two"
        assert result.get_call_summary() == "2 tool calls"

    def test_parse_openai_tool_call(self, parser):
        """Test parsing tool calls from OpenAI structured response."""
        model_response = ModelResponse(
            content=None,
            tool_calls=[
                ModelToolCall(
                    call_id="call_123",
                    name="sample_tool",
                    arguments={"param": "value"},
                    raw_arguments='{"param": "value"}',
                ),
            ],
        )

        result = parser.parse_response(model_response)

        assert result.model_response is model_response
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.tool_name == "sample_tool"
        assert tool_call.parameters["param"] == "value"
        assert tool_call.source == "openai"

    def test_parse_openai_sub_agent_call(self, parser):
        """Test parsing OpenAI tool call that targets a sub-agent."""
        model_response = ModelResponse(
            content=None,
            tool_calls=[
                ModelToolCall(
                    call_id="call_agent",
                    name="sub-agent-researcher",
                    arguments={"task": "Analyze data"},
                    raw_arguments='{"task": "Analyze data"}',
                ),
            ],
        )

        result = parser.parse_response(model_response)

        assert len(result.sub_agent_calls) == 1
        sub_agent_call = result.sub_agent_calls[0]
        assert sub_agent_call.agent_name == "researcher"
        assert "task: Analyze data" in sub_agent_call.message
        assert result.get_call_summary() == "1 sub-agent calls"

    # ========== Sub-Agent Call Tests ==========

    def test_parse_sub_agent_call(self, parser):
        """Test parsing a sub-agent call."""
        response = """
<tool_use>
<tool_name>sub-agent-researcher</tool_name>
<parameter>
<task>Research quantum computing</task>
<deadline>2025-12-31</deadline>
</parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert len(result.sub_agent_calls) == 1
        assert result.sub_agent_calls[0].agent_name == "researcher"
        assert "task: Research quantum computing" in result.sub_agent_calls[0].message
        assert "deadline: 2025-12-31" in result.sub_agent_calls[0].message
        assert result.get_call_summary() == "1 sub-agent calls"

    def test_parse_sub_agent_call_empty_parameters(self, parser):
        """Test parsing sub-agent call with empty parameters."""
        response = """
<tool_use>
<tool_name>sub-agent-worker</tool_name>
<parameter>
<task></task>
</parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert len(result.sub_agent_calls) == 1
        # Empty parameters should not appear in message
        assert result.sub_agent_calls[0].message == ""

    def test_parse_sub_agent_call_legacy_prefix(self, parser):
        """Legacy agent: prefix should still be parsed for backward compatibility."""
        response = """
<tool_use>
<tool_name>agent:legacy</tool_name>
<parameter>
<task>Legacy task</task>
</parameter>
</tool_use>
"""

        result = parser.parse_response(response)

        assert len(result.sub_agent_calls) == 1
        assert result.sub_agent_calls[0].agent_name == "legacy"

    # ========== Batch Agent Call Tests ==========

    def test_parse_batch_agent_call(self, parser):
        """Test parsing a batch agent call."""
        response = """
<use_batch_agent>
<agent_name>data_processor</agent_name>
<input_data_source>
<file_name>data.jsonl</file_name>
<format>jsonl</format>
</input_data_source>
<message>Process this data: {item}</message>
</use_batch_agent>
"""
        result = parser.parse_response(response)

        assert len(result.batch_agent_calls) == 1
        batch_call = result.batch_agent_calls[0]
        assert batch_call.agent_name == "data_processor"
        assert batch_call.file_path == "data.jsonl"
        assert batch_call.data_format == "jsonl"
        assert batch_call.message_template == "Process this data: {item}"
        assert result.get_call_summary() == "1 batch agent calls"

    def test_parse_batch_agent_call_default_format(self, parser):
        """Test parsing batch agent call with default format."""
        response = """
<use_batch_agent>
<agent_name>processor</agent_name>
<input_data_source>
<file_name>data.jsonl</file_name>
</input_data_source>
<message>Process: {data}</message>
</use_batch_agent>
"""
        result = parser.parse_response(response)

        assert len(result.batch_agent_calls) == 1
        assert result.batch_agent_calls[0].data_format == "jsonl"

    def test_parse_batch_agent_call_missing_agent_name(self, parser):
        """Test parsing batch agent call with missing agent name."""
        response = """
<use_batch_agent>
<input_data_source>
<file_name>data.jsonl</file_name>
</input_data_source>
<message>Process</message>
</use_batch_agent>
"""
        result = parser.parse_response(response)

        assert len(result.batch_agent_calls) == 0

    def test_parse_batch_agent_call_missing_file_name(self, parser):
        """Test parsing batch agent call with missing file name."""
        response = """
<use_batch_agent>
<agent_name>processor</agent_name>
<input_data_source>
<format>jsonl</format>
</input_data_source>
<message>Process</message>
</use_batch_agent>
"""
        result = parser.parse_response(response)

        assert len(result.batch_agent_calls) == 0

    def test_parse_batch_agent_call_missing_message(self, parser):
        """Test parsing batch agent call with missing message."""
        response = """
<use_batch_agent>
<agent_name>processor</agent_name>
<input_data_source>
<file_name>data.jsonl</file_name>
</input_data_source>
</use_batch_agent>
"""
        result = parser.parse_response(response)

        assert len(result.batch_agent_calls) == 0

    # ========== Parallel Tool Calls Tests ==========

    def test_parse_parallel_tool_calls(self, parser):
        """Test parsing parallel tool calls."""
        response = """
<use_parallel_tool_calls>
<parallel_tool>
<tool_name>tool_one</tool_name>
<parameter>
<param>value1</param>
</parameter>
</parallel_tool>
<parallel_tool>
<tool_name>tool_two</tool_name>
<parameter>
<param>value2</param>
</parameter>
</parallel_tool>
</use_parallel_tool_calls>
"""
        result = parser.parse_response(response)

        assert len(result.tool_calls) == 2
        assert result.is_parallel_tools is True
        assert result.tool_calls[0].tool_name == "tool_one"
        assert result.tool_calls[1].tool_name == "tool_two"

    def test_parse_parallel_with_sub_agent(self, parser):
        """Test parsing parallel calls containing sub-agent."""
        response = """
<use_parallel_tool_calls>
<parallel_tool>
<tool_name>regular_tool</tool_name>
<parameter>
<param>value</param>
</parameter>
</parallel_tool>
<parallel_tool>
<tool_name>sub-agent-worker</tool_name>
<parameter>
<task>Do work</task>
</parameter>
</parallel_tool>
</use_parallel_tool_calls>
"""
        result = parser.parse_response(response)

        assert len(result.tool_calls) == 1
        assert len(result.sub_agent_calls) == 1
        assert result.tool_calls[0].tool_name == "regular_tool"
        assert result.sub_agent_calls[0].agent_name == "worker"

    # ========== Tool Call Parsing Tests ==========

    def test_parse_tool_call_with_nested_html(self, parser):
        """Test parsing tool call with nested HTML content."""
        xml_content = """
<tool_name>write_file</tool_name>
<parameter>
<content><h1>Title</h1><p>Content</p></content>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "write_file"
        assert "h1" in tool_call.parameters["content"]
        assert "Title" in tool_call.parameters["content"]

    def test_parse_tool_call_missing_tool_name(self, parser):
        """Test parsing tool call with missing tool name."""
        xml_content = """
<parameter>
<param>value</param>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is None

    def test_parse_tool_call_empty_parameters(self, parser):
        """Test parsing tool call with no parameters."""
        xml_content = """
<tool_name>simple_tool</tool_name>
<parameter>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "simple_tool"
        assert len(tool_call.parameters) == 0

    def test_parse_tool_call_with_whitespace(self, parser):
        """Test parsing tool call with extra whitespace."""
        xml_content = """
<tool_name>  spaced_tool  </tool_name>
<parameter>
<param>  spaced_value  </param>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "spaced_tool"
        assert tool_call.parameters["param"] == "spaced_value"

    def test_parse_tool_call_invalid_xml(self, parser):
        """Test parsing tool call with invalid XML."""
        xml_content = """
<tool_name>broken_tool
<parameter>
<param>value</param>
"""
        tool_call = parser._parse_tool_call(xml_content)

        # Should fall back to regex parsing or minimal tool call
        assert tool_call is not None
        assert tool_call.tool_name == "broken_tool"

    def test_parse_tool_call_with_special_characters(self, parser):
        """Test parsing tool call with special characters in parameters."""
        xml_content = """
<tool_name>search_tool</tool_name>
<parameter>
<query>Find items with & and < and ></query>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "search_tool"
        # Parameter should be captured even with special chars
        assert "query" in tool_call.parameters

    def test_parse_tool_call_unexpected_error(self, parser):
        """Test handling of unexpected errors during tool call parsing."""
        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=RuntimeError("Unexpected error")):
            xml_content = """
<tool_name>test_tool</tool_name>
<parameter>
<param>value</param>
</parameter>
"""
            tool_call = parser._parse_tool_call(xml_content)

            # Should return None on unexpected errors
            assert tool_call is None

    # ========== Regex Fallback Parsing Tests ==========

    def test_parse_tool_call_with_regex(self, parser):
        """Test regex-based fallback parsing."""
        xml_content = """
<tool_name>regex_tool</tool_name>
<parameter>
<param1>value1</param1>
<param2>value2</param2>
</parameter>
"""
        tool_call = parser._parse_tool_call_with_regex(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "regex_tool"
        assert tool_call.parameters["param1"] == "value1"
        assert tool_call.parameters["param2"] == "value2"

    def test_parse_tool_call_with_regex_missing_tool_name(self, parser):
        """Test regex parsing with missing tool name."""
        xml_content = """
<parameter>
<param>value</param>
</parameter>
"""
        tool_call = parser._parse_tool_call_with_regex(xml_content)

        assert tool_call is None

    def test_parse_tool_call_with_regex_nested_tags(self, parser):
        """Test regex parsing with nested tags."""
        xml_content = """
<tool_name>nested_tool</tool_name>
<parameter>
<outer><inner>nested_value</inner></outer>
</parameter>
"""
        tool_call = parser._parse_tool_call_with_regex(xml_content)

        assert tool_call is not None
        assert "outer" in tool_call.parameters

    def test_parse_tool_call_with_regex_error(self, parser):
        """Test regex parsing error handling."""
        # Create malformed content that causes regex to fail
        with patch("nexau.archs.main_sub.execution.response_parser.re.search", side_effect=Exception("Regex error")):
            xml_content = "<tool_name>test</tool_name>"
            tool_call = parser._parse_tool_call_with_regex(xml_content)

            assert tool_call is None

    # ========== Extract Parameters with Regex Tests ==========

    def test_extract_parameters_with_parameter_block(self, parser):
        """Test extracting parameters from parameter block."""
        xml_content = """
<parameter>
<param1>value1</param1>
<param2>value2</param2>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        assert len(params) == 2
        assert params["param1"] == "value1"
        assert params["param2"] == "value2"

    def test_extract_parameters_nested_same_tag(self, parser):
        """Test extracting parameters with nested same-name tags."""
        xml_content = """
<parameter>
<item>
<item>nested</item>
outer
</item>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        assert "item" in params

    def test_extract_parameters_with_attributes(self, parser):
        """Test extracting parameters with tag attributes."""
        xml_content = """
<parameter>
<field type="text">value</field>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        assert params["field"] == "value"

    def test_extract_parameters_global_fallback(self, parser):
        """Test extracting parameters with global fallback."""
        xml_content = """
<tool_name>test</tool_name>
<param1>value1</param1>
<param2>value2</param2>
"""
        params = parser._extract_parameters_with_regex(
            xml_content,
            prefer_parameter_block=True,
            allow_global_fallback=True,
        )

        assert params["param1"] == "value1"
        assert params["param2"] == "value2"

    def test_extract_parameters_exclude_tags(self, parser):
        """Test extracting parameters with excluded tags."""
        xml_content = """
<tool_name>test</tool_name>
<param1>value1</param1>
<parameter>should_exclude</parameter>
"""
        params = parser._extract_parameters_with_regex(
            xml_content,
            prefer_parameter_block=False,
            allow_global_fallback=True,
            exclude_tags={"tool_name", "parameter"},
        )

        assert "param1" in params
        assert "tool_name" not in params
        assert "parameter" not in params

    def test_extract_parameters_unclosed_tag(self, parser):
        """Test extracting parameters with unclosed tag."""
        xml_content = """
<parameter>
<param1>value1
<param2>value2</param2>
</parameter>
"""
        # Should handle unclosed tags gracefully
        params = parser._extract_parameters_with_regex(xml_content)

        # At least the properly closed tag should be extracted
        assert "param2" in params

    def test_extract_parameters_no_parameter_block_no_fallback(self, parser):
        """Test extraction with no parameter block and fallback disabled."""
        xml_content = """
<param1>value1</param1>
<param2>value2</param2>
"""
        params = parser._extract_parameters_with_regex(
            xml_content,
            prefer_parameter_block=True,
            allow_global_fallback=False,
        )

        # Should return empty dict when no parameter block and no fallback
        assert len(params) == 0

    def test_extract_parameters_empty_values(self, parser):
        """Test extracting parameters with empty values."""
        xml_content = """
<parameter>
<param1></param1>
<param2>  </param2>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # Empty values should still be included
        assert "param1" in params
        assert "param2" in params

    # ========== Sub-Agent Call Parsing Tests ==========

    def test_parse_sub_agent_call_direct(self, parser):
        """Test parsing sub-agent call directly."""
        xml_content = """
<tool_name>sub-agent-researcher</tool_name>
<parameter>
<task>Research topic</task>
<priority>high</priority>
</parameter>
"""
        sub_agent_call = parser._parse_sub_agent_call(xml_content)

        assert sub_agent_call is not None
        assert sub_agent_call.agent_name == "researcher"
        assert "task: Research topic" in sub_agent_call.message
        assert "priority: high" in sub_agent_call.message

    def test_parse_sub_agent_call_missing_agent_name(self, parser):
        """Test parsing sub-agent call with missing agent name."""
        xml_content = """
<parameter>
<task>Do something</task>
</parameter>
"""
        sub_agent_call = parser._parse_sub_agent_call(xml_content)

        assert sub_agent_call is None

    def test_parse_sub_agent_call_invalid_xml(self, parser):
        """Test parsing sub-agent call with invalid XML."""
        xml_content = """
<tool_name>sub-agent.broken
<parameter>
<task>Do something</task>
"""
        sub_agent_call = parser._parse_sub_agent_call(xml_content)

        # Invalid XML without successful fallback returns None
        assert sub_agent_call is None

    def test_parse_sub_agent_call_value_error_with_fallback(self, parser):
        """Test parsing sub-agent call with value error that has successful fallback."""
        # Create well-formed XML that will trigger fallback parsing
        xml_content = """
<tool_name>sub-agent-worker</tool_name>
<parameter>
<task>Do work</task>
</parameter>
"""
        # Mock the XML parser to raise ValueError, triggering fallback
        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ValueError("Parse error")):
            # Mock the XMLUtils module (imported inside the function)
            with patch("nexau.archs.main_sub.utils.xml_utils.XMLUtils") as mock_utils:
                mock_utils.extract_agent_name_from_xml.return_value = "worker"

                sub_agent_call = parser._parse_sub_agent_call(xml_content)

                # Should fall back to minimal creation
                assert sub_agent_call is not None
                assert sub_agent_call.agent_name == "worker"

    def test_parse_sub_agent_call_no_parameters(self, parser):
        """Test parsing sub-agent call with no parameters."""
        xml_content = """
<tool_name>sub-agent-simple</tool_name>
"""
        sub_agent_call = parser._parse_sub_agent_call(xml_content)

        assert sub_agent_call is not None
        assert sub_agent_call.agent_name == "simple"
        assert sub_agent_call.message == ""

    def test_parse_sub_agent_call_unexpected_error(self, parser):
        """Test handling unexpected error in sub-agent parsing."""
        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=RuntimeError("Unexpected")):
            xml_content = """
<tool_name>agent:test</tool_name>
<parameter>
<task>Test</task>
</parameter>
"""
            sub_agent_call = parser._parse_sub_agent_call(xml_content)

            assert sub_agent_call is None

    # ========== Batch Agent Call Parsing Tests ==========

    def test_parse_batch_agent_call_direct(self, parser):
        """Test parsing batch agent call directly."""
        xml_content = """
<agent_name>batch_processor</agent_name>
<input_data_source>
<file_name>input.jsonl</file_name>
<format>jsonl</format>
</input_data_source>
<message>Process item: {data}</message>
"""
        batch_call = parser._parse_batch_agent_call(xml_content)

        assert batch_call is not None
        assert batch_call.agent_name == "batch_processor"
        assert batch_call.file_path == "input.jsonl"
        assert batch_call.data_format == "jsonl"
        assert batch_call.message_template == "Process item: {data}"

    def test_parse_batch_agent_call_invalid_xml(self, parser):
        """Test parsing batch agent call with invalid XML."""
        xml_content = """
<agent_name>broken
<input_data_source>
<file_name>data.jsonl</file_name>
"""
        batch_call = parser._parse_batch_agent_call(xml_content)

        assert batch_call is None

    def test_parse_batch_agent_call_general_error(self, parser):
        """Test handling general error in batch agent parsing."""
        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=RuntimeError("Error")):
            xml_content = """
<agent_name>test</agent_name>
<input_data_source>
<file_name>test.jsonl</file_name>
</input_data_source>
<message>Test</message>
"""
            batch_call = parser._parse_batch_agent_call(xml_content)

            assert batch_call is None

    def test_parse_batch_agent_call_missing_input_data_source(self, parser):
        """Test parsing batch agent call without input data source."""
        xml_content = """
<agent_name>processor</agent_name>
<message>Process</message>
"""
        batch_call = parser._parse_batch_agent_call(xml_content)

        assert batch_call is None

    # ========== Integration Tests ==========

    def test_parse_mixed_calls(self, parser):
        """Test parsing response with mixed call types."""
        response = """
Here's what I'll do:
<tool_use>
<tool_name>search</tool_name>
<parameter>
<query>test</query>
</parameter>
</tool_use>
<tool_use>
<tool_name>agent:helper</tool_name>
<parameter>
<task>help me</task>
</parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert len(result.tool_calls) == 1
        assert len(result.sub_agent_calls) == 1
        assert result.tool_calls[0].tool_name == "search"
        assert result.sub_agent_calls[0].agent_name == "helper"
        assert result.get_call_summary() == "1 tool calls, 1 sub-agent calls"

    def test_parse_response_batch_takes_priority(self, parser):
        """Test that batch agent calls take priority."""
        response = """
<use_batch_agent>
<agent_name>batch_proc</agent_name>
<input_data_source>
<file_name>data.jsonl</file_name>
</input_data_source>
<message>Process</message>
</use_batch_agent>
<tool_use>
<tool_name>regular_tool</tool_name>
<parameter>
<param>value</param>
</parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        # Batch agent call should be found
        assert len(result.batch_agent_calls) == 1
        # Regular tool call should also be found
        assert len(result.tool_calls) == 1

    def test_parse_response_with_text_around_calls(self, parser):
        """Test parsing response with text before and after calls."""
        response = """
I will now search for information.

<tool_use>
<tool_name>search</tool_name>
<parameter>
<query>test query</query>
</parameter>
</tool_use>

And that's what I'll do.
"""
        result = parser.parse_response(response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

    # ========== Edge Cases ==========

    def test_parse_tool_call_with_empty_nested_elements(self, parser):
        """Test parsing tool call with empty nested elements."""
        xml_content = """
<tool_name>test_tool</tool_name>
<parameter>
<nested><inner></inner></nested>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "test_tool"

    def test_extract_parameters_malformed_tag_names(self, parser):
        """Test extracting parameters with malformed tag names."""
        xml_content = """
<parameter>
<123invalid>value</123invalid>
<valid_tag>value2</valid_tag>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # The regex-based parser is permissive and may capture malformed tags
        # Valid tag should definitely be captured
        assert "valid_tag" in params
        assert params["valid_tag"] == "value2"

    def test_parse_response_multiple_parallel_blocks(self, parser):
        """Test parsing response with multiple parallel tool call blocks."""
        response = """
<use_parallel_tool_calls>
<parallel_tool>
<tool_name>tool1</tool_name>
<parameter>
<param>value1</param>
</parameter>
</parallel_tool>
</use_parallel_tool_calls>
<use_parallel_tool_calls>
<parallel_tool>
<tool_name>tool2</tool_name>
<parameter>
<param>value2</param>
</parameter>
</parallel_tool>
</use_parallel_tool_calls>
"""
        result = parser.parse_response(response)

        # Should only process the first parallel block
        assert result.is_parallel_tools is True
        # Both tools should be found (from first block matching)
        assert len(result.tool_calls) >= 1

    def test_parsed_response_get_all_calls(self, parser):
        """Test ParsedResponse.get_all_calls method."""
        response = """
<tool_use>
<tool_name>tool1</tool_name>
<parameter><p>v</p></parameter>
</tool_use>
<tool_use>
<tool_name>agent:sub</tool_name>
<parameter><task>work</task></parameter>
</tool_use>
<use_batch_agent>
<agent_name>batch</agent_name>
<input_data_source>
<file_name>data.jsonl</file_name>
</input_data_source>
<message>msg</message>
</use_batch_agent>
"""
        result = parser.parse_response(response)
        all_calls = result.get_all_calls()

        assert len(all_calls) == 3
        assert isinstance(all_calls[0], ToolCall)
        assert isinstance(all_calls[1], SubAgentCall)
        assert isinstance(all_calls[2], BatchAgentCall)

    def test_tool_call_id_generation(self, parser):
        """Test that tool calls get unique IDs."""
        response = """
<tool_use>
<tool_name>tool1</tool_name>
<parameter><p>v</p></parameter>
</tool_use>
<tool_use>
<tool_name>tool2</tool_name>
<parameter><p>v</p></parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert result.tool_calls[0].tool_call_id is not None
        assert result.tool_calls[1].tool_call_id is not None
        assert result.tool_calls[0].tool_call_id != result.tool_calls[1].tool_call_id

    def test_sub_agent_call_id_generation(self, parser):
        """Test that sub-agent calls get unique IDs."""
        response = """
<tool_use>
<tool_name>agent:sub1</tool_name>
<parameter><task>t1</task></parameter>
</tool_use>
<tool_use>
<tool_name>agent:sub2</tool_name>
<parameter><task>t2</task></parameter>
</tool_use>
"""
        result = parser.parse_response(response)

        assert result.sub_agent_calls[0].sub_agent_call_id is not None
        assert result.sub_agent_calls[1].sub_agent_call_id is not None
        assert result.sub_agent_calls[0].sub_agent_call_id != result.sub_agent_calls[1].sub_agent_call_id

    def test_xml_content_preserved_in_calls(self, parser):
        """Test that original XML content is preserved in parsed calls."""
        xml = """<tool_name>test</tool_name>
<parameter><p>v</p></parameter>"""

        tool_call = parser._parse_tool_call(xml)

        assert tool_call is not None
        assert tool_call.xml_content == xml

    def test_parse_tool_call_with_deeply_nested_content(self, parser):
        """Test parsing tool call with deeply nested content requiring ET.tostring."""
        xml_content = """
<tool_name>complex_tool</tool_name>
<parameter>
<html_content><div><span>Nested</span></div></html_content>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "complex_tool"
        assert "html_content" in tool_call.parameters

    def test_parse_tool_call_parseerror_with_fallback(self, parser):
        """Test tool call parsing when ParseError occurs and fallback succeeds."""
        # Create intentionally malformed XML
        xml_content = "<tool_name>test</tool_name><parameter><x>1</x>"

        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ET.ParseError("Parse error")):
            tool_call = parser._parse_tool_call(xml_content)

            # Should use regex fallback
            assert tool_call is not None
            assert tool_call.tool_name == "test"

    def test_extract_parameters_closing_tag_not_found(self, parser):
        """Test parameter extraction when closing tag is not found."""
        xml_content = """
<parameter>
<unclosed>value
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # Unclosed tag should be skipped
        assert len(params) >= 0

    def test_extract_parameters_empty_tag_header(self, parser):
        """Test parameter extraction with empty tag header."""
        xml_content = """
<parameter>
<>invalid</>
<valid>value</valid>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # Valid tag should be captured
        assert "valid" in params

    def test_extract_parameters_closing_tag_only(self, parser):
        """Test parameter extraction with closing tag only."""
        xml_content = """
<parameter>
</closing>
<valid>value</valid>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # Closing tag should be skipped
        assert "valid" in params

    def test_extract_parameters_invalid_alnum_tag(self, parser):
        """Test parameter extraction with non-alphanumeric tag names."""
        xml_content = """
<parameter>
<tag-with-dash>value1</tag-with-dash>
<valid_tag>value2</valid_tag>
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # Valid tag should be captured
        assert "valid_tag" in params

    def test_parse_sub_agent_call_parseerror(self, parser):
        """Test sub-agent call parsing with ParseError."""
        xml_content = "<tool_name>agent:test</tool_name><parameter>"

        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ET.ParseError("Parse error")):
            sub_agent_call = parser._parse_sub_agent_call(xml_content)

            # Should return None when fallback fails
            assert sub_agent_call is None

    def test_parse_batch_agent_call_parseerror(self, parser):
        """Test batch agent call parsing with ParseError."""
        xml_content = "<agent_name>test</agent_name><input_data_source>"

        batch_call = parser._parse_batch_agent_call(xml_content)

        # Should return None with invalid XML
        assert batch_call is None

    def test_parse_tool_call_value_error_with_failed_regex(self, parser):
        """Test tool call parsing when ValueError occurs and regex fallback fails."""
        xml_content = "<invalid>no tool name</invalid>"

        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ValueError("Parse error")):
            with patch.object(parser, "_parse_tool_call_with_regex", return_value=None):
                with patch("nexau.archs.main_sub.utils.xml_utils.XMLUtils") as mock_utils:
                    mock_utils.extract_tool_name_from_xml.return_value = "unknown"

                    tool_call = parser._parse_tool_call(xml_content)

                    # Should return None when all fallbacks fail
                    assert tool_call is None

    def test_parse_tool_call_value_error_with_successful_fallback(self, parser):
        """Test tool call parsing when ValueError occurs but final fallback succeeds."""
        xml_content = "<tool_name>recovered_tool</tool_name>"

        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ValueError("Parse error")):
            with patch.object(parser, "_parse_tool_call_with_regex", return_value=None):
                with patch("nexau.archs.main_sub.utils.xml_utils.XMLUtils") as mock_utils:
                    mock_utils.extract_tool_name_from_xml.return_value = "recovered_tool"

                    tool_call = parser._parse_tool_call(xml_content)

                    # Should create minimal tool call
                    assert tool_call is not None
                    assert tool_call.tool_name == "recovered_tool"
                    assert "raw_xml_content" in tool_call.parameters

    def test_parse_tool_call_parseerror_with_failed_regex(self, parser):
        """Test tool call ParseError when regex fallback also fails."""
        xml_content = "<invalid>content</invalid>"

        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ET.ParseError("Parse error")):
            with patch.object(parser, "_parse_tool_call_with_regex", return_value=None):
                tool_call = parser._parse_tool_call(xml_content)

                # Should return None when regex fallback fails
                assert tool_call is None

    def test_extract_parameters_no_closing_bracket_after_tag_start(self, parser):
        """Test parameter extraction when tag has no closing bracket."""
        xml_content = """
<parameter>
<incomplete
</parameter>
"""
        params = parser._extract_parameters_with_regex(xml_content)

        # Should handle gracefully - breaks when no closing bracket found
        assert len(params) >= 0  # Just checking it doesn't crash

    def test_parse_tool_call_empty_parameter_with_nested_structure(self, parser):
        """Test parsing tool call where parameter text is empty but has nested elements."""
        xml_content = """
<tool_name>nested_tool</tool_name>
<parameter>
<content><div><p>Nested HTML</p></div></content>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "nested_tool"
        # Should handle nested content
        assert "content" in tool_call.parameters

    def test_parse_tool_call_value_error_with_regex_success(self, parser):
        """Test tool call parsing when ValueError occurs but regex succeeds."""
        xml_content = """
<tool_name>regex_recovered</tool_name>
<parameter>
<param>value</param>
</parameter>
"""
        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ValueError("Parse error")):
            # Let the actual regex parsing work
            tool_call = parser._parse_tool_call(xml_content)

            # Should use regex fallback successfully
            assert tool_call is not None
            assert tool_call.tool_name == "regex_recovered"

    def test_parse_batch_agent_call_with_parseerror_logging(self, parser, caplog):
        """Test batch agent call ParseError logging."""
        import logging

        caplog.set_level(logging.ERROR)

        xml_content = "<agent_name>test<input_data_source>"

        with patch.object(parser.xml_parser, "parse_xml_content", side_effect=ET.ParseError("Parse error")):
            batch_call = parser._parse_batch_agent_call(xml_content)

            assert batch_call is None
            # Check logging occurred
            assert any("Invalid XML format in batch agent call" in record.message for record in caplog.records)

    def test_parse_tool_call_with_parameter_containing_only_nested_elements(self, parser):
        """Test parsing when parameter has no text but contains nested XML elements.

        This tests the ET.tostring path (lines 137-143) where param.text is empty
        but the parameter contains child elements.
        """
        # Create XML where parameter has child elements but no direct text
        from xml.etree import ElementTree as ET

        # Manually construct the XML structure
        root_str = """<root>
<tool_name>html_writer</tool_name>
<parameter>
<html><body><h1>Title</h1><p>Content</p></body></html>
</parameter>
</root>"""

        # Parse and extract the inner content
        root = ET.fromstring(root_str)
        params_elem = root.find("parameter")

        # Get the html parameter element
        html_param = params_elem.find("html")

        # This element has child elements but no text attribute
        assert html_param.text is None or html_param.text.strip() == ""
        assert len(html_param) > 0  # Has child elements

        # Now test with the parser using the full XML
        xml_content = """
<tool_name>html_writer</tool_name>
<parameter>
<html><body><h1>Title</h1><p>Content</p></body></html>
</parameter>
"""
        tool_call = parser._parse_tool_call(xml_content)

        assert tool_call is not None
        assert tool_call.tool_name == "html_writer"
        assert "html" in tool_call.parameters
        # The parameter should contain the nested HTML content
        assert "body" in tool_call.parameters["html"] or "Title" in tool_call.parameters["html"]


class TestResponseParserLogging:
    """Test logging behavior of ResponseParser."""

    @pytest.fixture
    def parser(self):
        """Create a ResponseParser instance for testing."""
        return ResponseParser()

    def test_logging_on_parse_response(self, parser, caplog):
        """Test that parsing logs appropriate messages."""
        import logging

        caplog.set_level(logging.INFO)

        response = """
<tool_use>
<tool_name>test_tool</tool_name>
<parameter><p>v</p></parameter>
</tool_use>
"""
        parser.parse_response(response)

        # Should log parsing activity
        assert any("Parsing LLM response" in record.message for record in caplog.records)

    def test_logging_on_invalid_xml(self, parser, caplog):
        """Test logging when encountering invalid XML."""
        import logging

        caplog.set_level(logging.ERROR)

        xml_content = "<tool_name>broken<parameter>"
        parser._parse_tool_call(xml_content)

        # Should log error about invalid XML
        assert any("Invalid XML format" in record.message or "XML parsing" in record.message for record in caplog.records)
