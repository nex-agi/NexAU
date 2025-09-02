"""XML parsing utilities for agent tool and sub-agent calls."""

import xml.etree.ElementTree as ET
import re
import html
import json
import logging
from typing import Dict, Any, List, Optional

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
            ('<sub_agent>', '</sub_agent>'),
            ('<parallel_tool>', '</parallel_tool>'),
            ('<parallel_agent>', '</parallel_agent>'),
            ('<use_parallel_tool_calls>', '</use_parallel_tool_calls>'),
            ('<use_parallel_sub_agents>', '</use_parallel_sub_agents>'),
            ('<use_batch_agent>', '</use_batch_agent>')
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
        tool_name_match = re.search(r'<tool_name>\s*([^<]+)\s*</tool_name>', xml_content, re.IGNORECASE | re.DOTALL)
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
        
        return "unknown"
    
    @staticmethod
    def extract_agent_name_from_xml(xml_content: str) -> str:
        """Extract agent name from potentially malformed XML using multiple strategies."""
        # Strategy 1: Try simple regex extraction
        agent_name_match = re.search(r'<agent_name>\s*([^<]+)\s*</agent_name>', xml_content, re.IGNORECASE | re.DOTALL)
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
        
        return "unknown"


class XMLParser:
    """Robust XML parser with multiple fallback strategies."""
    
    def parse_xml_content(self, xml_content: str) -> ET.Element:
        """Parse XML content using multiple strategies to handle malformed XML."""
        # Strategy 1: Try as-is
        try:
            return ET.fromstring(f"<root>{xml_content}</root>")
        except ET.ParseError as e:
            logger.warning(f"Initial XML parsing failed: {e}. Attempting recovery strategies...")
        
        # Strategy 1.5: Pre-process to wrap problematic content in CDATA
        try:
            # Look for parameter content that might contain URLs or special characters
            def wrap_parameter_content(match):
                full_param = match.group(0)
                param_name = match.group(1)
                param_content = match.group(2)
                
                # Skip if content is already wrapped in CDATA
                if param_content.strip().startswith('<![CDATA['):
                    return full_param
                
                # Check if content contains problematic XML characters
                if ('&' in param_content and 'http' in param_content):
                    # Content has unescaped & characters in URLs, wrap in CDATA
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                elif ('&' in param_content or '%' in param_content or '<' in param_content):
                    # Content has problematic characters, wrap in CDATA
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                elif len(param_content) > 200:  # Very long content likely needs CDATA
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                return full_param
            
            # Apply to all tags (both simple tags and nested parameter tags)
            preprocessed_xml = re.sub(
                r'<(\w+)>([^<>]*(?:<[^>]*>[^<>]*</[^>]*>[^<>]*)*)</\1>', 
                wrap_parameter_content, 
                xml_content, 
                flags=re.DOTALL
            )
            
            # Also handle simple leaf tags  
            preprocessed_xml = re.sub(
                r'<(\w+)>([^<]*)</\1>', 
                wrap_parameter_content, 
                preprocessed_xml, 
                flags=re.DOTALL
            )
            
            return ET.fromstring(f"<root>{preprocessed_xml}</root>")
        except ET.ParseError:
            pass
        
        # Strategy 2: Clean up common issues (unclosed tags, extra whitespace)
        try:
            cleaned_xml = xml_content.strip()
            
            # Fix potential unclosed tags by ensuring proper closing
            lines = cleaned_xml.split('\n')
            corrected_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for unclosed tags (opening tag without closing)
                tag_matches = re.findall(r'<(\w+)(?:\s+[^>]*)?>(.*?)(?:</\1>|$)', line)
                if tag_matches:
                    # Line has proper tag structure
                    corrected_lines.append(line)
                else:
                    # Check if line has opening tag but no closing tag
                    opening_match = re.match(r'<(\w+)(?:\s+[^>]*)?>\s*([^<]*)\s*$', line)
                    if opening_match:
                        tag_name = opening_match.group(1)
                        content = opening_match.group(2)
                        corrected_lines.append(f"<{tag_name}>{content}</{tag_name}>")
                    else:
                        corrected_lines.append(line)
            
            cleaned_xml = '\n'.join(corrected_lines)
            return ET.fromstring(f"<root>{cleaned_xml}</root>")
            
        except ET.ParseError:
            pass
        
        # Strategy 3: Handle JSON content and URLs within XML parameters
        try:
            def handle_json_param_content(match):
                param_name = match.group(1)
                param_content = match.group(2).strip()
                
                # Check if content looks like JSON
                if param_content.startswith(('{', '[')):
                    # Wrap JSON content in CDATA to preserve it
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                # Check if content looks like a URL or contains characters that need escaping
                elif ('http' in param_content or 'www.' in param_content or 
                      '%' in param_content or '&' in param_content or 
                      '<' in param_content or '>' in param_content):
                    # Wrap URL or complex content in CDATA to preserve it
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                elif '<' in param_content and '>' in param_content:
                    # Escape HTML/XML content if it contains < >
                    escaped_content = html.escape(param_content)
                    return f"<{param_name}>{escaped_content}</{param_name}>"
                return match.group(0)
            
            # Find and handle parameter content within parameters block
            escaped_xml = xml_content
            params_match = re.search(r'<parameter>(.*?)</parameter>', xml_content, re.DOTALL)
            if params_match:
                params_content = params_match.group(1)
                # Pattern to match individual parameter tags with their content
                param_pattern = r'<(\w+)>(.*?)</\1>'
                escaped_params = re.sub(param_pattern, handle_json_param_content, params_content, flags=re.DOTALL)
                escaped_xml = xml_content.replace(params_match.group(1), escaped_params)
            
            return ET.fromstring(f"<root>{escaped_xml}</root>")
            
        except ET.ParseError:
            pass
        
        # Strategy 4: Fallback - escape all content and selectively unescape XML tags
        try:
            escaped_content = html.escape(xml_content, quote=False)
            # Unescape the XML tags we need
            escaped_content = escaped_content.replace("&lt;", "<").replace("&gt;", ">")
            return ET.fromstring(f"<root>{escaped_content}</root>")
        except ET.ParseError:
            pass
        
        # Strategy 5: Advanced parameter content extraction and reconstruction
        try:
            tool_name_match = re.search(r'<tool_name>\s*([^<]+)\s*</tool_name>', xml_content, re.IGNORECASE | re.DOTALL)
            tool_name = tool_name_match.group(1).strip() if tool_name_match else "unknown"
            
            # Build minimal XML structure
            minimal_xml = f"<tool_name>{tool_name}</tool_name>"
            
            # Try to extract parameters with more robust handling
            params_match = re.search(r'<parameter>(.*?)</parameter>', xml_content, re.DOTALL | re.IGNORECASE)
            if params_match:
                params_content = params_match.group(1).strip()
                
                # Try to fix common parameter issues
                fixed_params = self._fix_parameter_content(params_content)
                minimal_xml += f"<parameter>{fixed_params}</parameter>"
            
            return ET.fromstring(f"<root>{minimal_xml}</root>")
        except (ET.ParseError, AttributeError):
            pass
        
        # Strategy 6: Regex-based parameter extraction with CDATA wrapping
        try:
            tool_name_match = re.search(r'<tool_name>\s*([^<]+)\s*</tool_name>', xml_content, re.IGNORECASE | re.DOTALL)
            tool_name = tool_name_match.group(1).strip() if tool_name_match else "unknown"
            
            minimal_xml = f"<tool_name>{tool_name}</tool_name>"
            
            # Extract all parameter tags individually using regex
            param_pattern = r'<(\w+)>(.*?)</\1>'
            param_matches = re.findall(param_pattern, xml_content, re.DOTALL | re.IGNORECASE)
            
            if param_matches:
                params_xml = "<parameter>"
                for param_name, param_value in param_matches:
                    # Skip tool_name if it appears in parameters
                    if param_name.lower() == 'tool_name':
                        continue
                    
                    # Wrap parameter content in CDATA to preserve formatting
                    clean_param_value = param_value.strip()
                    if clean_param_value:
                        params_xml += f"<{param_name}><![CDATA[{clean_param_value}]]></{param_name}>"
                    else:
                        params_xml += f"<{param_name}></{param_name}>"
                
                params_xml += "</parameter>"
                minimal_xml += params_xml
            
            return ET.fromstring(f"<root>{minimal_xml}</root>")
        except (ET.ParseError, AttributeError):
            pass
        
        # Final fallback: raise with detailed error
        raise ValueError(f"Unable to parse XML content after multiple strategies. Content preview: {xml_content[:200]}...")
    
    def parse_nested_xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Parse nested XML element into a dictionary."""
        result = {}
        
        for child in element:
            child_name = child.tag
            
            # Check if child has its own children (nested structure)
            if len(child) > 0:
                # Recursively parse nested elements
                result[child_name] = self.parse_nested_xml_to_dict(child)
            else:
                # Get text content
                if child.text is not None:
                    child_value = child.text.strip()
                else:
                    child_value = ''.join(child.itertext()).strip()
                
                # Try to convert to appropriate Python type
                result[child_name] = self._convert_xml_value_to_python_type(child_value)
        
        return result
    
    def _convert_xml_value_to_python_type(self, value: str):
        """Convert XML string value to appropriate Python type."""
        if not value:
            return ""
        
        # Try boolean conversion first
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON parsing (for arrays or objects)
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string if no other type matches
        return value
    
    def _fix_parameter_content(self, params_content: str) -> str:
        """Fix common issues in parameter content that cause XML parsing to fail."""
        try:
            # Strategy 1: Wrap individual parameters in CDATA if they contain problematic content
            def wrap_problematic_content(match):
                param_name = match.group(1)
                param_content = match.group(2)
                
                # Check if content has problematic characters or is very long
                if ('&' in param_content or '<' in param_content or '>' in param_content or 
                    len(param_content) > 100 or '\n' in param_content):
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                return match.group(0)
            
            # Apply CDATA wrapping to parameters that need it
            fixed_content = re.sub(
                r'<(\w+)>(.*?)</\1>', 
                wrap_problematic_content, 
                params_content, 
                flags=re.DOTALL
            )
            
            return fixed_content
            
        except Exception:
            # If fixing fails, wrap the entire content in a single CDATA section
            return f"<raw_content><![CDATA[{params_content}]]></raw_content>"