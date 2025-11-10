"""
Unit tests for BatchProcessor class.
"""

import json
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import pytest

from nexau.archs.main_sub.execution.batch_processor import BatchProcessor
from nexau.archs.main_sub.utils.xml_utils import XMLParser


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""

    @pytest.fixture
    def mock_subagent_manager(self):
        """Create a mock SubAgentManager."""
        manager = Mock()
        manager.call_sub_agent = Mock(return_value="agent response")
        return manager

    @pytest.fixture
    def batch_processor(self, mock_subagent_manager):
        """Create a BatchProcessor instance."""
        return BatchProcessor(subagent_manager=mock_subagent_manager, max_workers=5)

    @pytest.fixture
    def temp_jsonl_file(self, tmp_path):
        """Create a temporary JSONL file with test data."""
        test_file = tmp_path / "test_data.jsonl"
        test_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "San Francisco"},
            {"name": "Charlie", "age": 35, "city": "Seattle"},
        ]
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        return str(test_file)

    @pytest.fixture
    def valid_xml_content(self, temp_jsonl_file):
        """Create valid XML content for batch processing."""
        return f"""
        <agent_name>test_agent</agent_name>
        <input_data_source>
            <file_name>{temp_jsonl_file}</file_name>
            <format>jsonl</format>
        </input_data_source>
        <message>Process person: {{name}}, age {{age}}, from {{city}}</message>
        """

    # Initialization Tests
    def test_initialization(self, mock_subagent_manager):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(subagent_manager=mock_subagent_manager, max_workers=3)

        assert processor.subagent_manager == mock_subagent_manager
        assert processor.max_workers == 3
        assert isinstance(processor.xml_parser, XMLParser)

    def test_initialization_default_max_workers(self, mock_subagent_manager):
        """Test BatchProcessor initialization with default max_workers."""
        processor = BatchProcessor(subagent_manager=mock_subagent_manager)

        assert processor.max_workers == 5

    # XML Parsing Tests
    def test_execute_batch_agent_missing_agent_name(self, batch_processor, temp_jsonl_file):
        """Test execution with missing agent_name."""
        xml_content = f"""
        <input_data_source>
            <file_name>{temp_jsonl_file}</file_name>
        </input_data_source>
        <message>Test message</message>
        """

        with pytest.raises(ValueError, match="Missing agent_name"):
            batch_processor.execute_batch_agent_from_xml(xml_content)

    def test_execute_batch_agent_missing_input_data_source(self, batch_processor):
        """Test execution with missing input_data_source."""
        xml_content = """
        <agent_name>test_agent</agent_name>
        <message>Test message</message>
        """

        with pytest.raises(ValueError, match="Missing input_data_source"):
            batch_processor.execute_batch_agent_from_xml(xml_content)

    def test_execute_batch_agent_missing_file_name(self, batch_processor):
        """Test execution with missing file_name."""
        xml_content = """
        <agent_name>test_agent</agent_name>
        <input_data_source>
            <format>jsonl</format>
        </input_data_source>
        <message>Test message</message>
        """

        with pytest.raises(ValueError, match="Missing file_name"):
            batch_processor.execute_batch_agent_from_xml(xml_content)

    def test_execute_batch_agent_missing_message(self, batch_processor, temp_jsonl_file):
        """Test execution with missing message."""
        xml_content = f"""
        <agent_name>test_agent</agent_name>
        <input_data_source>
            <file_name>{temp_jsonl_file}</file_name>
        </input_data_source>
        """

        with pytest.raises(ValueError, match="Missing message"):
            batch_processor.execute_batch_agent_from_xml(xml_content)

    def test_execute_batch_agent_invalid_xml(self, batch_processor):
        """Test execution with invalid XML format."""
        xml_content = """
        <agent_name>test_agent
        <message>Broken XML
        """

        with pytest.raises(ValueError, match="Unable to parse XML content"):
            batch_processor.execute_batch_agent_from_xml(xml_content)

    def test_execute_batch_agent_xml_parse_error(self, batch_processor):
        """Test execution when XML parser directly raises ParseError."""
        xml_content = "<test>content</test>"

        # Mock xml_parser to raise ET.ParseError directly
        with patch.object(batch_processor.xml_parser, "parse_xml_content", side_effect=ET.ParseError("Parse error")):
            with pytest.raises(ValueError, match="Invalid XML format"):
                batch_processor.execute_batch_agent_from_xml(xml_content)

    def test_execute_batch_agent_default_format(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test execution with default format (jsonl)."""
        xml_content = f"""
        <agent_name>test_agent</agent_name>
        <input_data_source>
            <file_name>{temp_jsonl_file}</file_name>
        </input_data_source>
        <message>Process: {{name}}</message>
        """

        result = batch_processor.execute_batch_agent_from_xml(xml_content)

        assert "successful" in result
        assert mock_subagent_manager.call_sub_agent.called

    def test_execute_batch_agent_file_not_exists(self, batch_processor, tmp_path):
        """Test execution when file doesn't exist - should create empty file."""
        non_existent_file = tmp_path / "subdir" / "nonexistent.jsonl"
        xml_content = f"""
        <agent_name>test_agent</agent_name>
        <input_data_source>
            <file_name>{non_existent_file}</file_name>
        </input_data_source>
        <message>Test message</message>
        """

        result = batch_processor.execute_batch_agent_from_xml(xml_content)

        assert "0 items processed" in result
        assert non_existent_file.exists()

    # Data Processing Tests
    def test_process_batch_data_unsupported_format(self, batch_processor, temp_jsonl_file):
        """Test processing with unsupported data format."""
        with pytest.raises(ValueError, match="Unsupported data format"):
            batch_processor._process_batch_data("test_agent", temp_jsonl_file, "csv", "test message")

    def test_process_batch_data_empty_file(self, batch_processor, tmp_path):
        """Test processing an empty JSONL file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        result = batch_processor._process_batch_data("test_agent", str(empty_file), "jsonl", "test message")

        assert "0 items processed" in result
        assert "no valid JSON objects found" in result

    def test_process_batch_data_invalid_json_lines(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test processing file with invalid JSON lines."""
        test_file = tmp_path / "invalid.jsonl"
        test_file.write_text('{"valid": "json"}\ninvalid json line\n{"another": "valid"}\n')

        result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Process: {valid}")

        result_data = json.loads(result)
        # Should process only valid lines (both {"valid": "json"} and {"another": "valid"} have the "valid" key)
        assert result_data["total_items"] == 2

    def test_process_batch_data_non_dict_json(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test processing file with non-dict JSON lines."""
        test_file = tmp_path / "non_dict.jsonl"
        test_file.write_text('["array", "not", "dict"]\n{"valid": "dict"}\n"just a string"\n')

        result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Process: {valid}")

        result_data = json.loads(result)
        # Should process only dict lines
        assert result_data["total_items"] == 1

    def test_process_batch_data_whitespace_lines(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test processing file with whitespace-only lines."""
        test_file = tmp_path / "whitespace.jsonl"
        test_file.write_text('{"item": 1}\n   \n\t\n{"item": 2}\n\n')

        result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Process: {item}")

        result_data = json.loads(result)
        assert result_data["total_items"] == 2

    def test_process_batch_data_invalid_template_keys(self, batch_processor, temp_jsonl_file):
        """Test processing with invalid template keys."""
        with pytest.raises(ValueError, match="invalid keys"):
            batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process: {invalid_key}")

    def test_process_batch_data_success(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test successful batch data processing."""
        result = batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process person: {name}, age {age}")

        result_data = json.loads(result)

        assert result_data["total_items"] == 3
        assert result_data["successful_items"] == 3
        assert result_data["failed_items"] == 0
        assert len(result_data["displayed_results"]) == 3
        assert result_data["remaining_items"] == 0

        # Verify agent was called for each item
        assert mock_subagent_manager.call_sub_agent.call_count == 3

    def test_process_batch_data_parallel_execution(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test parallel execution with multiple items."""
        # Create file with more items
        test_file = tmp_path / "parallel.jsonl"
        test_data = [{"id": i} for i in range(10)]
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Process ID: {id}")

        result_data = json.loads(result)

        assert result_data["total_items"] == 10
        assert result_data["successful_items"] == 10
        assert mock_subagent_manager.call_sub_agent.call_count == 10

    def test_process_batch_data_limited_display_results(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test that results display is limited to first 3 items."""
        # Create file with more than 3 items
        test_file = tmp_path / "many_items.jsonl"
        test_data = [{"id": i} for i in range(5)]
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Process ID: {id}")

        result_data = json.loads(result)

        assert result_data["total_items"] == 5
        assert len(result_data["displayed_results"]) == 3
        assert result_data["remaining_items"] == 2
        assert "note" in result_data
        assert "2 additional results" in result_data["note"]

    def test_process_batch_data_with_failures(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test processing with some failures."""
        # Make agent fail on second call
        mock_subagent_manager.call_sub_agent.side_effect = [
            "success 1",
            Exception("Agent error"),
            "success 3",
        ]

        result = batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process: {name}")

        result_data = json.loads(result)

        assert result_data["total_items"] == 3
        assert result_data["successful_items"] == 2
        assert result_data["failed_items"] == 1

        # Check that failed item has error status
        failed_items = [r for r in result_data["displayed_results"] if r["status"] == "error"]
        assert len(failed_items) == 1
        assert "error" in failed_items[0]

    def test_process_batch_data_template_rendering_error(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test handling of template rendering errors."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"name": "Alice"}\n')

        # Use template with key that doesn't exist in data
        # First item should fail during template rendering
        with patch.object(batch_processor, "_render_message_template", side_effect=ValueError("Template error")):
            result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Bad template")

            result_data = json.loads(result)
            assert result_data["failed_items"] == 1
            assert "Template rendering failed" in result_data["displayed_results"][0]["error"]

    def test_process_batch_data_results_sorted_by_line(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test that results are sorted by line number."""
        # Mock to simulate out-of-order completion
        mock_subagent_manager.call_sub_agent.side_effect = ["result 1", "result 2", "result 3"]

        result = batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process: {name}")

        result_data = json.loads(result)
        displayed = result_data["displayed_results"]

        # Verify results are sorted by line number
        line_numbers = [r["line"] for r in displayed]
        assert line_numbers == sorted(line_numbers)

    def test_process_batch_data_file_read_error(self, batch_processor):
        """Test handling of file read errors."""
        with pytest.raises(ValueError, match="Error reading file"):
            batch_processor._process_batch_data("test_agent", "/nonexistent/path/file.jsonl", "jsonl", "test message")

    def test_process_batch_data_encoding_utf8(self, batch_processor, tmp_path, mock_subagent_manager):
        """Test processing file with UTF-8 encoding."""
        test_file = tmp_path / "utf8.jsonl"
        test_file.write_text('{"text": "Hello 世界"}\n{"text": "Привет мир"}\n', encoding="utf-8")

        result = batch_processor._process_batch_data("test_agent", str(test_file), "jsonl", "Process: {text}")

        result_data = json.loads(result)
        assert result_data["total_items"] == 2
        assert result_data["successful_items"] == 2

    # Template Key Extraction Tests
    def test_extract_template_keys_single_key(self, batch_processor):
        """Test extracting a single template key."""
        template = "Process {name}"
        keys = batch_processor._extract_template_keys(template)

        assert keys == ["name"]

    def test_extract_template_keys_multiple_keys(self, batch_processor):
        """Test extracting multiple template keys."""
        template = "Process {name}, age {age}, from {city}"
        keys = batch_processor._extract_template_keys(template)

        assert set(keys) == {"name", "age", "city"}

    def test_extract_template_keys_duplicate_keys(self, batch_processor):
        """Test extracting duplicate template keys."""
        template = "Name: {name}, repeat: {name}"
        keys = batch_processor._extract_template_keys(template)

        assert keys.count("name") == 2

    def test_extract_template_keys_no_keys(self, batch_processor):
        """Test extracting keys from template without placeholders."""
        template = "Static message"
        keys = batch_processor._extract_template_keys(template)

        assert keys == []

    def test_extract_template_keys_complex_names(self, batch_processor):
        """Test extracting keys with underscores and numbers."""
        template = "Process {user_name}, ID {id_123}, {data_value_2}"
        keys = batch_processor._extract_template_keys(template)

        assert set(keys) == {"user_name", "id_123", "data_value_2"}

    # Message Template Rendering Tests
    def test_render_message_template_success(self, batch_processor):
        """Test successful message template rendering."""
        template = "Hello {name}, age {age}"
        data = {"name": "Alice", "age": 30}

        result = batch_processor._render_message_template(template, data)

        assert result == "Hello Alice, age 30"

    def test_render_message_template_missing_key(self, batch_processor):
        """Test rendering with missing key."""
        template = "Hello {name}, age {age}"
        data = {"name": "Alice"}

        with pytest.raises(ValueError, match="Template key 'age' not found"):
            batch_processor._render_message_template(template, data)

    def test_render_message_template_no_placeholders(self, batch_processor):
        """Test rendering template without placeholders."""
        template = "Static message"
        data = {"key": "value"}

        result = batch_processor._render_message_template(template, data)

        assert result == "Static message"

    def test_render_message_template_extra_keys(self, batch_processor):
        """Test rendering with extra keys in data (should work fine)."""
        template = "Hello {name}"
        data = {"name": "Alice", "age": 30, "city": "NYC"}

        result = batch_processor._render_message_template(template, data)

        assert result == "Hello Alice"

    def test_render_message_template_special_chars(self, batch_processor):
        """Test rendering template with special characters."""
        template = "Message: {text}"
        data = {"text": 'Hello <world> & "friends"'}

        result = batch_processor._render_message_template(template, data)

        assert result == 'Message: Hello <world> & "friends"'

    def test_render_message_template_numeric_values(self, batch_processor):
        """Test rendering template with numeric values."""
        template = "Value: {num}, Float: {flt}"
        data = {"num": 42, "flt": 3.14}

        result = batch_processor._render_message_template(template, data)

        assert result == "Value: 42, Float: 3.14"

    # Execute Batch Item Safe Tests
    def test_execute_batch_item_safe_success(self, batch_processor, mock_subagent_manager):
        """Test successful execution of a batch item."""
        result = batch_processor._execute_batch_item_safe("test_agent", "test message", 1)

        assert result == "agent response"
        mock_subagent_manager.call_sub_agent.assert_called_once_with("test_agent", "test message")

    def test_execute_batch_item_safe_agent_error(self, batch_processor, mock_subagent_manager):
        """Test execution when agent raises an error."""
        mock_subagent_manager.call_sub_agent.side_effect = Exception("Agent failed")

        with pytest.raises(Exception, match="Agent failed"):
            batch_processor._execute_batch_item_safe("test_agent", "test message", 1)

    def test_execute_batch_item_safe_different_messages(self, batch_processor, mock_subagent_manager):
        """Test execution with different messages."""
        mock_subagent_manager.call_sub_agent.return_value = "response"

        result1 = batch_processor._execute_batch_item_safe("test_agent", "message 1", 1)
        result2 = batch_processor._execute_batch_item_safe("test_agent", "message 2", 2)

        assert result1 == "response"
        assert result2 == "response"
        assert mock_subagent_manager.call_sub_agent.call_count == 2

    # Integration Tests
    def test_full_batch_processing_workflow(self, batch_processor, valid_xml_content, mock_subagent_manager):
        """Test complete batch processing workflow."""
        result = batch_processor.execute_batch_agent_from_xml(valid_xml_content)

        result_data = json.loads(result)

        # Verify complete workflow
        assert "summary" in result_data
        assert result_data["total_items"] == 3
        assert result_data["successful_items"] == 3
        assert mock_subagent_manager.call_sub_agent.call_count == 3

        # Verify each call had proper message rendering
        calls = mock_subagent_manager.call_sub_agent.call_args_list
        assert any("Alice" in str(call) for call in calls)
        assert any("Bob" in str(call) for call in calls)
        assert any("Charlie" in str(call) for call in calls)

    def test_batch_processing_with_custom_max_workers(self, mock_subagent_manager, temp_jsonl_file):
        """Test batch processing with custom max_workers."""
        processor = BatchProcessor(subagent_manager=mock_subagent_manager, max_workers=2)

        xml_content = f"""
        <agent_name>test_agent</agent_name>
        <input_data_source>
            <file_name>{temp_jsonl_file}</file_name>
            <format>jsonl</format>
        </input_data_source>
        <message>Process: {{name}}</message>
        """

        result = processor.execute_batch_agent_from_xml(xml_content)
        result_data = json.loads(result)

        assert result_data["successful_items"] == 3

    def test_batch_processing_context_propagation(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test that context is properly propagated in parallel execution."""
        call_count = 0

        def mock_call_with_context(agent_name, message):
            nonlocal call_count
            call_count += 1
            return f"result {call_count}"

        mock_subagent_manager.call_sub_agent.side_effect = mock_call_with_context

        result = batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process: {name}")

        result_data = json.loads(result)
        assert result_data["total_items"] == 3
        assert call_count == 3

    def test_execute_batch_agent_whitespace_handling(self, batch_processor, temp_jsonl_file):
        """Test that whitespace in XML elements is properly handled."""
        xml_content = f"""
        <agent_name>  test_agent  </agent_name>
        <input_data_source>
            <file_name>  {temp_jsonl_file}  </file_name>
            <format>  jsonl  </format>
        </input_data_source>
        <message>  Process: {{name}}  </message>
        """

        result = batch_processor.execute_batch_agent_from_xml(xml_content)
        result_data = json.loads(result)

        assert result_data["successful_items"] == 3

    def test_process_batch_data_summary_format(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test the format and content of the summary."""
        result = batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process: {name}")

        result_data = json.loads(result)

        # Verify all expected keys are present
        assert "summary" in result_data
        assert "total_items" in result_data
        assert "successful_items" in result_data
        assert "failed_items" in result_data
        assert "displayed_results" in result_data
        assert "remaining_items" in result_data

        # Verify summary string format
        summary = result_data["summary"]
        assert "Batch processing completed" in summary
        assert "3/3" in summary
        assert "successful" in summary

    def test_process_batch_data_result_structure(self, batch_processor, temp_jsonl_file, mock_subagent_manager):
        """Test the structure of individual results."""
        result = batch_processor._process_batch_data("test_agent", temp_jsonl_file, "jsonl", "Process: {name}")

        result_data = json.loads(result)
        first_result = result_data["displayed_results"][0]

        # Verify result structure
        assert "line" in first_result
        assert "status" in first_result
        assert first_result["status"] == "success"
        assert "result" in first_result
        assert "data" in first_result
        assert isinstance(first_result["data"], dict)
