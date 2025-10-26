"""
Comprehensive unit tests for the TraceDumper class.

This module tests trace file output utilities including
file creation, JSON formatting, and error handling.
"""

import json
import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from northau.archs.main_sub.tracing.trace_dumper import TraceDumper


class TestTraceDumperBasic:
    """Tests for basic TraceDumper functionality."""

    def test_dump_trace_to_file_creates_file(self, temp_dir):
        """Test that dump_trace_to_file creates a trace file."""
        trace_data = [{"type": "test", "message": "Test entry"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        assert os.path.exists(dump_path)

    def test_dump_trace_to_file_writes_valid_json(self, temp_dir):
        """Test that dumped trace is valid JSON."""
        trace_data = [{"type": "test", "message": "Test entry"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        # Read and parse the JSON file
        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert isinstance(loaded_data, dict)
        assert "metadata" in loaded_data
        assert "trace" in loaded_data

    def test_dump_trace_to_file_includes_metadata(self, temp_dir):
        """Test that dumped trace includes metadata."""
        trace_data = [{"type": "test", "message": "Test entry"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        metadata = loaded_data["metadata"]
        assert metadata["agent_name"] == "test_agent"
        assert "dump_timestamp" in metadata
        assert metadata["total_entries"] == 1
        assert "entry_types" in metadata

    def test_dump_trace_to_file_preserves_trace_data(self, temp_dir):
        """Test that trace data is preserved correctly."""
        trace_data = [
            {"type": "llm_request", "iteration": 1, "model": "gpt-4"},
            {"type": "llm_response", "iteration": 1, "content": "Response"},
            {"type": "tool_request", "tool_name": "search"},
        ]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"] == trace_data

    def test_dump_trace_to_file_counts_entries(self, temp_dir):
        """Test that total_entries is counted correctly."""
        trace_data = [{"type": "test"} for _ in range(10)]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["metadata"]["total_entries"] == 10

    def test_dump_trace_to_file_identifies_entry_types(self, temp_dir):
        """Test that entry_types are identified correctly."""
        trace_data = [
            {"type": "llm_request"},
            {"type": "llm_response"},
            {"type": "tool_request"},
            {"type": "tool_response"},
            {"type": "error"},
            {"type": "llm_request"},  # Duplicate type
        ]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        entry_types = loaded_data["metadata"]["entry_types"]
        assert set(entry_types) == {"llm_request", "llm_response", "tool_request", "tool_response", "error"}


class TestTraceDumperDirectoryHandling:
    """Tests for directory creation and path handling."""

    def test_dump_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created if they don't exist."""
        trace_data = [{"type": "test"}]
        nested_path = os.path.join(temp_dir, "level1", "level2", "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, nested_path, "test_agent")

        assert os.path.exists(nested_path)

    def test_dump_with_existing_directory(self, temp_dir):
        """Test dumping to an existing directory."""
        trace_data = [{"type": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        # Directory already exists (temp_dir)
        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        assert os.path.exists(dump_path)

    def test_dump_with_no_directory_part(self):
        """Test dumping with just a filename (no directory part)."""
        trace_data = [{"type": "test"}]

        # Use current directory
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                TraceDumper.dump_trace_to_file(trace_data, "trace.json", "test_agent")
                assert os.path.exists("trace.json")
            finally:
                os.chdir(original_cwd)

    def test_dump_with_complex_path_structure(self, temp_dir):
        """Test dumping with complex directory structure."""
        trace_data = [{"type": "test"}]
        complex_path = os.path.join(temp_dir, "project", "logs", "2024", "agent_traces", "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, complex_path, "test_agent")

        assert os.path.exists(complex_path)


class TestTraceDumperEncoding:
    """Tests for character encoding and special characters."""

    def test_dump_with_unicode_characters(self, temp_dir):
        """Test dumping trace with unicode characters."""
        trace_data = [
            {"type": "test", "message": "测试 unicode 字符"},
            {"type": "test", "emoji": "🚀 🎉 💯"},
            {"type": "test", "accents": "café résumé naïve"},
        ]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"][0]["message"] == "测试 unicode 字符"
        assert loaded_data["trace"][1]["emoji"] == "🚀 🎉 💯"
        assert loaded_data["trace"][2]["accents"] == "café résumé naïve"

    def test_dump_preserves_unicode_without_ascii_escape(self, temp_dir):
        """Test that unicode is preserved as-is, not ASCII-escaped."""
        trace_data = [{"type": "test", "text": "中文"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        # Read file as text to check raw content
        with open(dump_path, encoding="utf-8") as f:
            content = f.read()

        # Should contain actual unicode, not escaped version like \u4e2d\u6587
        assert "中文" in content

    def test_dump_with_special_json_characters(self, temp_dir):
        """Test dumping with special JSON characters."""
        trace_data = [
            {"type": "test", "text": "Line 1\nLine 2"},
            {"type": "test", "text": "Tab\tseparated"},
            {"type": "test", "text": 'Quote: "hello"'},
            {"type": "test", "text": "Backslash: \\path\\to\\file"},
        ]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"][0]["text"] == "Line 1\nLine 2"
        assert loaded_data["trace"][1]["text"] == "Tab\tseparated"
        assert loaded_data["trace"][2]["text"] == 'Quote: "hello"'
        assert loaded_data["trace"][3]["text"] == "Backslash: \\path\\to\\file"


class TestTraceDumperFormatting:
    """Tests for JSON formatting and indentation."""

    def test_dump_uses_indentation(self, temp_dir):
        """Test that dumped JSON is properly indented."""
        trace_data = [{"type": "test", "nested": {"key": "value"}}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            content = f.read()

        # Check for indentation (2 spaces)
        assert "  " in content
        # Should have proper formatting, not single-line
        assert "\n" in content

    def test_dump_json_structure(self, temp_dir):
        """Test the complete JSON structure."""
        trace_data = [{"type": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Top-level structure
        assert set(loaded_data.keys()) == {"metadata", "trace"}

        # Metadata structure
        assert "agent_name" in loaded_data["metadata"]
        assert "dump_timestamp" in loaded_data["metadata"]
        assert "total_entries" in loaded_data["metadata"]
        assert "entry_types" in loaded_data["metadata"]


class TestTraceDumperEmptyAndEdgeCases:
    """Tests for empty traces and edge cases."""

    def test_dump_empty_trace(self, temp_dir):
        """Test dumping an empty trace."""
        trace_data = []
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"] == []
        assert loaded_data["metadata"]["total_entries"] == 0
        assert loaded_data["metadata"]["entry_types"] == []

    def test_dump_with_entries_missing_type(self, temp_dir):
        """Test dumping entries that don't have a 'type' field."""
        trace_data = [{"message": "Entry without type"}, {"type": "valid", "data": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Entry without type should be labeled as 'unknown'
        assert "unknown" in loaded_data["metadata"]["entry_types"]
        assert "valid" in loaded_data["metadata"]["entry_types"]

    def test_dump_with_none_values(self, temp_dir):
        """Test dumping trace with None values."""
        trace_data = [{"type": "test", "value": None}, {"type": "test", "nested": {"key": None}}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"][0]["value"] is None
        assert loaded_data["trace"][1]["nested"]["key"] is None

    def test_dump_with_complex_nested_structures(self, temp_dir):
        """Test dumping deeply nested data structures."""
        trace_data = [
            {"type": "complex", "level1": {"level2": {"level3": {"level4": ["array", "of", "items"], "another_key": {"deep": "value"}}}}}
        ]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        # Verify the complex structure is preserved
        assert loaded_data["trace"][0]["level1"]["level2"]["level3"]["level4"] == ["array", "of", "items"]

    def test_dump_with_various_data_types(self, temp_dir):
        """Test dumping trace with various JSON data types."""
        trace_data = [
            {
                "type": "test",
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean_true": True,
                "boolean_false": False,
                "null": None,
                "array": [1, 2, 3],
                "object": {"key": "value"},
            }
        ]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        entry = loaded_data["trace"][0]
        assert entry["string"] == "text"
        assert entry["integer"] == 42
        assert entry["float"] == 3.14
        assert entry["boolean_true"] is True
        assert entry["boolean_false"] is False
        assert entry["null"] is None
        assert entry["array"] == [1, 2, 3]
        assert entry["object"] == {"key": "value"}


class TestTraceDumperErrorHandling:
    """Tests for error handling in TraceDumper."""

    def test_dump_logs_success_message(self, temp_dir, caplog):
        """Test that successful dump logs appropriate message."""
        trace_data = [{"type": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        with caplog.at_level(logging.INFO):
            TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        assert any("Trace dumped" in record.message for record in caplog.records)
        assert any(dump_path in record.message for record in caplog.records)
        assert any("1 entries" in record.message for record in caplog.records)

    def test_dump_handles_write_error(self, caplog):
        """Test that write errors are caught and logged."""
        trace_data = [{"type": "test"}]
        invalid_path = "/invalid/path/that/does/not/exist/trace.json"

        with caplog.at_level(logging.ERROR):
            TraceDumper.dump_trace_to_file(trace_data, invalid_path, "test_agent")

        # Should log error but not raise exception
        assert any("Failed to dump trace" in record.message for record in caplog.records)

    def test_dump_handles_permission_error(self, temp_dir, caplog):
        """Test handling of permission errors."""
        trace_data = [{"type": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        # Create a read-only directory (simulate permission error)
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with caplog.at_level(logging.ERROR):
                TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

            assert any("Failed to dump trace" in record.message for record in caplog.records)

    def test_dump_handles_makedirs_error(self, caplog):
        """Test handling of directory creation errors."""
        trace_data = [{"type": "test"}]
        dump_path = "/some/path/trace.json"

        with patch("os.makedirs", side_effect=OSError("Cannot create directory")):
            with caplog.at_level(logging.ERROR):
                TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

            assert any("Failed to dump trace" in record.message for record in caplog.records)

    def test_dump_does_not_raise_exception_on_error(self):
        """Test that errors don't raise exceptions (fail gracefully)."""
        trace_data = [{"type": "test"}]
        invalid_path = "/completely/invalid/path/trace.json"

        # Should not raise any exception
        try:
            TraceDumper.dump_trace_to_file(trace_data, invalid_path, "test_agent")
        except Exception as e:
            pytest.fail(f"dump_trace_to_file should not raise exceptions, but raised: {e}")


class TestTraceDumperMultipleAgents:
    """Tests for dumping traces from multiple agents."""

    def test_dump_different_agent_names(self, temp_dir):
        """Test dumping traces with different agent names."""
        agents = ["agent1", "agent2", "agent3"]

        for agent_name in agents:
            trace_data = [{"type": "test", "agent": agent_name}]
            dump_path = os.path.join(temp_dir, f"{agent_name}_trace.json")

            TraceDumper.dump_trace_to_file(trace_data, dump_path, agent_name)

            with open(dump_path, encoding="utf-8") as f:
                loaded_data = json.load(f)

            assert loaded_data["metadata"]["agent_name"] == agent_name

    def test_dump_sub_agent_traces(self, temp_dir):
        """Test dumping traces in sub-agent directory structure."""
        sub_agents = ["researcher", "analyzer", "writer"]

        # Create sub_agents directory
        sub_agents_dir = os.path.join(temp_dir, "main_trace_sub_agents")
        os.makedirs(sub_agents_dir, exist_ok=True)

        for i, sub_agent in enumerate(sub_agents, 1):
            trace_data = [{"type": "subagent_task", "name": sub_agent}]
            dump_path = os.path.join(sub_agents_dir, f"{sub_agent}_{i}.json")

            TraceDumper.dump_trace_to_file(trace_data, dump_path, sub_agent)

            assert os.path.exists(dump_path)


class TestTraceDumperLargeTraces:
    """Tests for handling large trace files."""

    def test_dump_large_trace(self, temp_dir):
        """Test dumping a large trace with many entries."""
        # Create a large trace
        trace_data = [{"type": "test", "index": i, "data": "x" * 100} for i in range(1000)]
        dump_path = os.path.join(temp_dir, "large_trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["metadata"]["total_entries"] == 1000
        assert len(loaded_data["trace"]) == 1000

    def test_dump_trace_with_large_content(self, temp_dir):
        """Test dumping trace with very large individual entries."""
        # Create an entry with large content
        large_content = "x" * 100000  # 100KB of data
        trace_data = [{"type": "test", "large_field": large_content}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"][0]["large_field"] == large_content


class TestTraceDumperTimestamp:
    """Tests for timestamp handling."""

    def test_dump_includes_timestamp(self, temp_dir):
        """Test that dump includes a timestamp."""
        trace_data = [{"type": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert "dump_timestamp" in loaded_data["metadata"]

        # Verify it's a valid ISO format timestamp
        from datetime import datetime

        timestamp = loaded_data["metadata"]["dump_timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_dump_timestamp_is_recent(self, temp_dir):
        """Test that dump timestamp is recent (within last minute)."""
        from datetime import datetime

        trace_data = [{"type": "test"}]
        dump_path = os.path.join(temp_dir, "trace.json")

        before = datetime.now()
        TraceDumper.dump_trace_to_file(trace_data, dump_path, "test_agent")
        after = datetime.now()

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        timestamp = datetime.fromisoformat(loaded_data["metadata"]["dump_timestamp"])

        # Timestamp should be between before and after
        assert before <= timestamp <= after


class TestTraceDumperIntegration:
    """Integration tests for realistic usage scenarios."""

    def test_complete_agent_execution_trace(self, temp_dir):
        """Test dumping a complete agent execution trace."""
        trace_data = [
            {"type": "llm_request", "iteration": 1, "model": "gpt-4"},
            {"type": "llm_response", "iteration": 1, "content": "Response"},
            {"type": "tool_request", "tool_name": "search", "parameters": {"query": "test"}},
            {"type": "tool_response", "tool_name": "search", "result": {"data": []}},
            {"type": "llm_request", "iteration": 2, "model": "gpt-4"},
            {"type": "llm_response", "iteration": 2, "content": "Final response"},
            {"type": "shutdown", "reason": "Task completed"},
        ]
        dump_path = os.path.join(temp_dir, "execution_trace.json")

        TraceDumper.dump_trace_to_file(trace_data, dump_path, "execution_agent")

        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["metadata"]["total_entries"] == 7
        expected_types = {"llm_request", "llm_response", "tool_request", "tool_response", "shutdown"}
        assert set(loaded_data["metadata"]["entry_types"]) == expected_types

    def test_dump_multiple_traces_same_directory(self, temp_dir):
        """Test dumping multiple traces to the same directory."""
        traces = {"trace1": [{"type": "test1"}], "trace2": [{"type": "test2"}], "trace3": [{"type": "test3"}]}

        for trace_name, trace_data in traces.items():
            dump_path = os.path.join(temp_dir, f"{trace_name}.json")
            TraceDumper.dump_trace_to_file(trace_data, dump_path, trace_name)

        # All traces should exist
        for trace_name in traces.keys():
            assert os.path.exists(os.path.join(temp_dir, f"{trace_name}.json"))

    def test_overwrite_existing_trace(self, temp_dir):
        """Test that dumping overwrites existing trace file."""
        dump_path = os.path.join(temp_dir, "trace.json")

        # First dump
        trace_data1 = [{"type": "first", "data": "original"}]
        TraceDumper.dump_trace_to_file(trace_data1, dump_path, "agent1")

        # Second dump (overwrite)
        trace_data2 = [{"type": "second", "data": "updated"}]
        TraceDumper.dump_trace_to_file(trace_data2, dump_path, "agent2")

        # Read and verify it has the second data
        with open(dump_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["trace"][0]["type"] == "second"
        assert loaded_data["metadata"]["agent_name"] == "agent2"
