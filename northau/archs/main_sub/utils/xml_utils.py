"""XML parsing utilities for agent tool and sub-agent calls."""
import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)


class XMLUtils:
    """Utility functions for XML processing."""

    @staticmethod
    def restore_closing_tags(response: str) -> str:
        """Restore XML closing tags that may have been removed by stop sequences."""
        restored_response = response

        # List of tag pairs to check (opening_tag, closing_tag)
        tag_pairs = [
            ('<tool_use>', '</tool_use>'),
            ('<parallel_tool>', '</parallel_tool>'),
            ('<use_parallel_tool_calls>', '</use_parallel_tool_calls>'),
            ('<use_batch_agent>', '</use_batch_agent>'),
        ]

        for open_tag, close_tag in tag_pairs:
            if open_tag in restored_response and not restored_response.rstrip().endswith(close_tag):
                # Count open and close tags
                open_count = restored_response.count(open_tag)
                close_count = restored_response.count(close_tag)
                if open_count > close_count:
                    restored_response += close_tag

        return restored_response

    @staticmethod
    def extract_tool_name_from_xml(xml_content: str) -> str:
        """Extract tool name from potentially malformed XML using multiple strategies."""
        # Strategy 1: Try simple regex extraction
        tool_name_match = re.search(
            r'<tool_name>\s*([^<]+)\s*</tool_name>', xml_content, re.IGNORECASE | re.DOTALL,
        )
        if tool_name_match:
            return tool_name_match.group(1).strip()

        # Strategy 2: Try parsing as valid XML
        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is not None and tool_name_elem.text:
                return tool_name_elem.text.strip()
        except ET.ParseError:
            pass

        # Strategy 3: Try cleaning up common XML issues
        try:
            # Remove potential unclosed tags and extra content
            cleaned_xml = xml_content.strip()

            # Try to fix unclosed tags by finding the pattern and closing them
            lines = cleaned_xml.split('\n')
            for i, line in enumerate(lines):
                # Check if this line has an opening tag but no closing tag
                tag_match = re.search(r'<(\w+)>[^<]*$', line.strip())
                if tag_match:
                    tag_name = tag_match.group(1)
                    lines[i] = line.rstrip() + f'</{tag_name}>'

            cleaned_xml = '\n'.join(lines)
            root = ET.fromstring(f"<root>{cleaned_xml}</root>")
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is not None and tool_name_elem.text:
                return tool_name_elem.text.strip()
        except (ET.ParseError, AttributeError):
            pass

        return 'unknown'

    @staticmethod
    def extract_agent_name_from_xml(xml_content: str) -> str:
        """Extract agent name from potentially malformed XML using multiple strategies."""
        # Strategy 1: Try simple regex extraction
        agent_name_match = re.search(
            r'<agent_name>\s*([^<]+)\s*</agent_name>', xml_content, re.IGNORECASE | re.DOTALL,
        )
        if agent_name_match:
            return agent_name_match.group(1).strip()

        # Strategy 2: Try parsing as valid XML
        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is not None and agent_name_elem.text:
                return agent_name_elem.text.strip()
        except ET.ParseError:
            pass

        # Strategy 3: Try cleaning up common XML issues
        try:
            # Remove potential unclosed tags and extra content
            cleaned_xml = xml_content.strip()

            # Try to fix unclosed tags by finding the pattern and closing them
            lines = cleaned_xml.split('\n')
            for i, line in enumerate(lines):
                # Check if this line has an opening tag but no closing tag
                tag_match = re.search(r'<(\w+)>[^<]*$', line.strip())
                if tag_match:
                    tag_name = tag_match.group(1)
                    lines[i] = line.rstrip() + f'</{tag_name}>'

            cleaned_xml = '\n'.join(lines)
            root = ET.fromstring(f"<root>{cleaned_xml}</root>")
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is not None and agent_name_elem.text:
                return agent_name_elem.text.strip()
        except (ET.ParseError, AttributeError):
            pass

        return 'unknown'


class XMLParser:
    """Robust XML parser with multiple fallback strategies."""

    def parse_xml_content(self, xml_content: str) -> ET.Element:
        """Parse XML content using multiple strategies to handle malformed XML."""
        # Strategy 0: Always preprocess parameter content with CDATA
        # This ensures parameter values are treated as plain text
        try:
            preprocessed_xml = self._wrap_parameter_content_in_cdata(xml_content)
            return ET.fromstring(f"<root>{preprocessed_xml}</root>")
        except ET.ParseError as e:
            logger.warning(
                f"CDATA preprocessing failed: {e}. Trying fallback strategies...",
            )

        # Strategy 1: Try as-is (fallback)
        try:
            return ET.fromstring(f"<root>{xml_content}</root>")
        except ET.ParseError as e:
            logger.warning(
                f"Initial XML parsing failed: {e}. Attempting recovery strategies...",
            )

        # Strategy 1.5: Handle JSON content and URLs within XML parameters (legacy fallback)
        try:
            def handle_json_param_content(match):
                param_name = match.group(1)
                param_content = match.group(2).strip()

                # Check if content looks like JSON
                if param_content.startswith(('{', '[')):
                    # Wrap JSON content in CDATA to preserve it
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                # Check if content looks like a URL or contains characters that need escaping
                elif (
                    'http' in param_content or 'www.' in param_content
                    or '%' in param_content or '&' in param_content
                    or '<' in param_content or '>' in param_content
                ):
                    # Wrap URL or complex content in CDATA to preserve it
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                elif '<' in param_content and '>' in param_content:
                    # Escape HTML/XML content if it contains < >
                    escaped_content = html.escape(param_content)
                    return f"<{param_name}>{escaped_content}</{param_name}>"
                return match.group(0)

            # Find and handle parameter content within parameters block
            escaped_xml = xml_content
            params_match = re.search(
                r'<parameter>(.*?)</parameter>', xml_content, re.DOTALL,
            )
            if params_match:
                params_content = params_match.group(1)
                # Pattern to match individual parameter tags with their content
                param_pattern = r'<(\w+)>(.*?)</\1>'
                escaped_params = re.sub(
                    param_pattern, handle_json_param_content, params_content, flags=re.DOTALL,
                )
                escaped_xml = xml_content.replace(
                    params_match.group(1), escaped_params,
                )

            return ET.fromstring(f"<root>{escaped_xml}</root>")

        except ET.ParseError:
            pass


        # Final fallback: raise with detailed error
        raise ValueError(
            f"Unable to parse XML content after multiple strategies. Content preview: {xml_content[:200]}...",
        )

    def _wrap_parameter_content_in_cdata(self, xml_content: str) -> str:
        """Wrap all parameter element content in CDATA to treat as plain text."""
        # Find all <parameter>...</parameter> blocks
        def wrap_parameter_block(match):
            parameter_content = match.group(1)
            
            # Within each parameter block, find individual parameter elements
            def wrap_individual_param(param_match):
                param_name = param_match.group(1)
                param_content = param_match.group(2)
                
                # Skip if already wrapped in CDATA
                if '[CDATA[' in param_content:
                    return param_match.group(0)
                
                # Wrap the content in CDATA
                return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
            
            # Apply CDATA wrapping to all individual parameters
            wrapped_content = re.sub(
                r'<(\w+)>(.*?)</\1>',
                wrap_individual_param,
                parameter_content,
                flags=re.DOTALL
            )
            
            return f"<parameter>{wrapped_content}</parameter>"
        
        # Apply to all parameter blocks
        result = re.sub(
            r'<parameter>(.*?)</parameter>',
            wrap_parameter_block,
            xml_content,
            flags=re.DOTALL
        )
        
        return result
