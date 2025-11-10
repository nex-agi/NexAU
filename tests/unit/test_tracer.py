"""
Comprehensive unit tests for the Tracer class.

This module tests trace collection, management, and thread safety
for agent execution tracing.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from nexau.archs.main_sub.tracing.tracer import Tracer


class TestTracerInitialization:
    """Tests for Tracer initialization."""

    def test_init_basic(self):
        """Test basic tracer initialization."""
        tracer = Tracer(agent_name="test_agent")

        assert tracer.agent_name == "test_agent"
        assert tracer._trace_data is None
        assert tracer._dump_trace_path is None
        assert tracer._sub_agent_counter == 0
        # Check locks exist and have the acquire/release methods
        assert hasattr(tracer._trace_lock, "acquire")
        assert hasattr(tracer._trace_lock, "release")
        assert hasattr(tracer._sub_agent_counter_lock, "acquire")
        assert hasattr(tracer._sub_agent_counter_lock, "release")

    def test_init_different_agent_names(self):
        """Test initialization with various agent names."""
        names = ["simple", "agent-123", "my_agent_v2", "AgentWithCaps"]

        for name in names:
            tracer = Tracer(agent_name=name)
            assert tracer.agent_name == name
            assert not tracer.is_tracing()


class TestTracingLifecycle:
    """Tests for tracer lifecycle (start/stop/is_tracing)."""

    def test_start_tracing(self):
        """Test starting trace collection."""
        tracer = Tracer(agent_name="test_agent")
        dump_path = "/tmp/trace.json"

        tracer.start_tracing(dump_path)

        assert tracer.is_tracing()
        assert tracer._trace_data == []
        assert tracer._dump_trace_path == dump_path

    def test_start_tracing_logs_message(self, caplog):
        """Test that starting tracing logs appropriate message."""
        tracer = Tracer(agent_name="test_agent")
        dump_path = "/tmp/trace.json"

        with caplog.at_level(logging.INFO):
            tracer.start_tracing(dump_path)

        assert any("Trace logging enabled" in record.message for record in caplog.records)
        assert any("test_agent" in record.message for record in caplog.records)
        assert any(dump_path in record.message for record in caplog.records)

    def test_stop_tracing(self):
        """Test stopping trace collection."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")
        tracer.add_entry({"type": "test"})

        tracer.stop_tracing()

        assert not tracer.is_tracing()
        assert tracer._trace_data is None
        assert tracer._dump_trace_path is None

    def test_stop_tracing_clears_data(self):
        """Test that stopping tracing clears all trace data."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        # Add multiple entries
        for i in range(5):
            tracer.add_entry({"type": f"test_{i}"})

        tracer.stop_tracing()

        assert tracer.get_trace_data() is None
        assert tracer.get_dump_path() is None

    def test_is_tracing_initially_false(self):
        """Test that is_tracing returns False initially."""
        tracer = Tracer(agent_name="test_agent")
        assert not tracer.is_tracing()

    def test_restart_tracing(self):
        """Test restarting tracing after stopping."""
        tracer = Tracer(agent_name="test_agent")

        # First trace session
        tracer.start_tracing("/tmp/trace1.json")
        tracer.add_entry({"type": "first"})
        tracer.stop_tracing()

        # Second trace session
        tracer.start_tracing("/tmp/trace2.json")

        assert tracer.is_tracing()
        assert tracer._trace_data == []  # Should be empty
        assert tracer.get_dump_path() == "/tmp/trace2.json"


class TestAddEntry:
    """Tests for adding generic trace entries."""

    def test_add_entry_basic(self):
        """Test adding a basic entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        entry = {"type": "test", "data": "value"}
        tracer.add_entry(entry)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "test"
        assert trace_data[0]["data"] == "value"

    def test_add_entry_without_tracing(self):
        """Test that entries are not added when tracing is disabled."""
        tracer = Tracer(agent_name="test_agent")

        entry = {"type": "test"}
        tracer.add_entry(entry)

        assert tracer.get_trace_data() is None

    def test_add_entry_adds_timestamp(self):
        """Test that timestamp is automatically added if missing."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        entry = {"type": "test"}
        tracer.add_entry(entry)

        trace_data = tracer.get_trace_data()
        assert "timestamp" in trace_data[0]
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(trace_data[0]["timestamp"])

    def test_add_entry_preserves_timestamp(self):
        """Test that existing timestamp is preserved."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        custom_timestamp = "2023-01-01T12:00:00"
        entry = {"type": "test", "timestamp": custom_timestamp}
        tracer.add_entry(entry)

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["timestamp"] == custom_timestamp

    def test_add_entry_adds_agent_name(self):
        """Test that agent_name is automatically added if missing."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        entry = {"type": "test"}
        tracer.add_entry(entry)

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["agent_name"] == "test_agent"

    def test_add_entry_preserves_agent_name(self):
        """Test that existing agent_name is preserved."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        entry = {"type": "test", "agent_name": "custom_agent"}
        tracer.add_entry(entry)

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["agent_name"] == "custom_agent"

    def test_add_multiple_entries(self):
        """Test adding multiple entries."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        for i in range(10):
            tracer.add_entry({"type": "test", "index": i})

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 10
        for i, entry in enumerate(trace_data):
            assert entry["index"] == i


class TestSpecializedTraceEntries:
    """Tests for specialized trace entry methods."""

    def test_add_llm_request(self):
        """Test adding LLM request trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        api_params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}],
            "temperature": 0.7,
        }

        tracer.add_llm_request(iteration=1, api_params=api_params)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "llm_request"
        assert trace_data[0]["iteration"] == 1
        assert trace_data[0]["api_params"]["model"] == "gpt-4"
        assert trace_data[0]["api_params"]["temperature"] == 0.7
        assert len(trace_data[0]["api_params"]["messages"]) == 2

    def test_add_llm_request_truncates_long_messages(self):
        """Test that long message content is truncated."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        long_content = "a" * 2000
        api_params = {"model": "gpt-4", "messages": [{"role": "user", "content": long_content}]}

        tracer.add_llm_request(iteration=1, api_params=api_params)

        trace_data = tracer.get_trace_data()
        message_content = trace_data[0]["api_params"]["messages"][0]["content"]
        assert len(message_content) == 1003  # 1000 chars + '...'
        assert message_content.endswith("...")

    def test_add_llm_request_short_messages_not_truncated(self):
        """Test that short messages are not truncated."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        short_content = "Short message"
        api_params = {"model": "gpt-4", "messages": [{"role": "user", "content": short_content}]}

        tracer.add_llm_request(iteration=1, api_params=api_params)

        trace_data = tracer.get_trace_data()
        message_content = trace_data[0]["api_params"]["messages"][0]["content"]
        assert message_content == short_content

    def test_add_llm_response(self):
        """Test adding LLM response trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        response_content = "This is the LLM response"
        tracer.add_llm_response(iteration=1, response_content=response_content)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "llm_response"
        assert trace_data[0]["iteration"] == 1
        assert trace_data[0]["response"]["content"] == response_content

    def test_add_tool_request(self):
        """Test adding tool request trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        parameters = {"param1": "value1", "param2": 42}
        tracer.add_tool_request(tool_name="test_tool", parameters=parameters)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "tool_request"
        assert trace_data[0]["tool_name"] == "test_tool"
        assert trace_data[0]["parameters"] == parameters

    def test_add_tool_response(self):
        """Test adding tool response trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        result = {"success": True, "data": [1, 2, 3]}
        tracer.add_tool_response(tool_name="test_tool", result=result)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "tool_response"
        assert trace_data[0]["tool_name"] == "test_tool"
        assert trace_data[0]["result"] == result

    def test_add_subagent_request(self):
        """Test adding sub-agent request trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        message = "Execute this task"
        tracer.add_subagent_request(subagent_name="sub_agent_1", message=message)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "subagent_request"
        assert trace_data[0]["subagent_name"] == "sub_agent_1"
        assert trace_data[0]["message"] == message

    def test_add_subagent_response(self):
        """Test adding sub-agent response trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        result = "Task completed successfully"
        tracer.add_subagent_response(subagent_name="sub_agent_1", result=result)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "subagent_response"
        assert trace_data[0]["subagent_name"] == "sub_agent_1"
        assert trace_data[0]["result"] == result

    def test_add_error(self):
        """Test adding error trace entry."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        error = ValueError("Test error message")
        tracer.add_error(error)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "error"
        assert trace_data[0]["error"] == "Test error message"
        assert trace_data[0]["error_type"] == "ValueError"

    def test_add_error_different_types(self):
        """Test adding different types of errors."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        errors = [ValueError("value error"), TypeError("type error"), RuntimeError("runtime error"), Exception("generic exception")]

        for error in errors:
            tracer.add_error(error)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 4
        assert trace_data[0]["error_type"] == "ValueError"
        assert trace_data[1]["error_type"] == "TypeError"
        assert trace_data[2]["error_type"] == "RuntimeError"
        assert trace_data[3]["error_type"] == "Exception"

    def test_add_shutdown_default_reason(self):
        """Test adding shutdown trace entry with default reason."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        tracer.add_shutdown()

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "shutdown"
        assert trace_data[0]["reason"] == "Signal interrupt (Ctrl+C)"

    def test_add_shutdown_custom_reason(self):
        """Test adding shutdown trace entry with custom reason."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        custom_reason = "Max iterations reached"
        tracer.add_shutdown(reason=custom_reason)

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1
        assert trace_data[0]["type"] == "shutdown"
        assert trace_data[0]["reason"] == custom_reason


class TestComplexTraceScenarios:
    """Tests for complex tracing scenarios."""

    def test_complete_execution_trace(self):
        """Test a complete execution trace with multiple entry types."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        # Simulate a complete execution
        tracer.add_llm_request(iteration=1, api_params={"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]})
        tracer.add_llm_response(iteration=1, response_content="Use the tool")
        tracer.add_tool_request(tool_name="search", parameters={"query": "test"})
        tracer.add_tool_response(tool_name="search", result={"results": []})
        tracer.add_subagent_request(subagent_name="researcher", message="Research this")
        tracer.add_subagent_response(subagent_name="researcher", result="Done")
        tracer.add_shutdown(reason="Task completed")

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 7

        # Verify order and types
        assert trace_data[0]["type"] == "llm_request"
        assert trace_data[1]["type"] == "llm_response"
        assert trace_data[2]["type"] == "tool_request"
        assert trace_data[3]["type"] == "tool_response"
        assert trace_data[4]["type"] == "subagent_request"
        assert trace_data[5]["type"] == "subagent_response"
        assert trace_data[6]["type"] == "shutdown"

    def test_multiple_iterations_trace(self):
        """Test trace with multiple LLM iterations."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        for i in range(1, 4):
            tracer.add_llm_request(iteration=i, api_params={"model": "gpt-4", "messages": [{"role": "user", "content": f"iter {i}"}]})
            tracer.add_llm_response(iteration=i, response_content=f"response {i}")

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 6

        # Verify iterations
        assert trace_data[0]["iteration"] == 1
        assert trace_data[1]["iteration"] == 1
        assert trace_data[2]["iteration"] == 2
        assert trace_data[3]["iteration"] == 2
        assert trace_data[4]["iteration"] == 3
        assert trace_data[5]["iteration"] == 3


class TestGetTraceData:
    """Tests for retrieving trace data."""

    def test_get_trace_data_returns_copy(self):
        """Test that get_trace_data returns a copy, not reference."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        tracer.add_entry({"type": "test"})

        trace_data_1 = tracer.get_trace_data()
        trace_data_2 = tracer.get_trace_data()

        # They should be equal but not the same object
        assert trace_data_1 == trace_data_2
        assert trace_data_1 is not trace_data_2

    def test_get_trace_data_when_not_tracing(self):
        """Test get_trace_data returns None when not tracing."""
        tracer = Tracer(agent_name="test_agent")
        assert tracer.get_trace_data() is None

    def test_get_trace_data_immutability(self):
        """Test that modifying returned trace data doesn't affect internal data."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        tracer.add_entry({"type": "test"})

        trace_data = tracer.get_trace_data()
        trace_data.append({"type": "should_not_appear"})

        # Original data should remain unchanged
        original_data = tracer.get_trace_data()
        assert len(original_data) == 1
        assert original_data[0]["type"] == "test"


class TestGetDumpPath:
    """Tests for retrieving dump path."""

    def test_get_dump_path(self):
        """Test getting dump path."""
        tracer = Tracer(agent_name="test_agent")
        dump_path = "/tmp/test_trace.json"

        tracer.start_tracing(dump_path)

        assert tracer.get_dump_path() == dump_path

    def test_get_dump_path_when_not_tracing(self):
        """Test get_dump_path returns None when not tracing."""
        tracer = Tracer(agent_name="test_agent")
        assert tracer.get_dump_path() is None

    def test_get_dump_path_after_stop(self):
        """Test get_dump_path returns None after stopping."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")
        tracer.stop_tracing()

        assert tracer.get_dump_path() is None


class TestSubAgentTracePathGeneration:
    """Tests for sub-agent trace path generation."""

    def test_generate_sub_agent_trace_path_basic(self):
        """Test basic sub-agent trace path generation."""
        tracer = Tracer(agent_name="main_agent")

        main_path = "/tmp/main_trace.json"
        sub_path = tracer.generate_sub_agent_trace_path("researcher", main_path)

        assert sub_path is not None
        assert "main_trace_sub_agents" in sub_path
        assert "researcher_1.json" in sub_path

    def test_generate_sub_agent_trace_path_increments_counter(self):
        """Test that sub-agent counter increments."""
        tracer = Tracer(agent_name="main_agent")
        main_path = "/tmp/main_trace.json"

        path1 = tracer.generate_sub_agent_trace_path("researcher", main_path)
        path2 = tracer.generate_sub_agent_trace_path("researcher", main_path)
        path3 = tracer.generate_sub_agent_trace_path("writer", main_path)

        assert "researcher_1.json" in path1
        assert "researcher_2.json" in path2
        assert "writer_3.json" in path3

    def test_generate_sub_agent_trace_path_structure(self):
        """Test the structure of generated sub-agent trace path."""
        tracer = Tracer(agent_name="main_agent")
        main_path = "/home/user/traces/my_trace.json"

        sub_path = tracer.generate_sub_agent_trace_path("researcher", main_path)

        assert sub_path is not None
        path_obj = Path(sub_path)

        # Check that parent directory is my_trace_sub_agents
        assert path_obj.parent.name == "my_trace_sub_agents"

        # Check that grandparent is traces
        assert path_obj.parent.parent.name == "traces"

        # Check filename format
        assert path_obj.name == "researcher_1.json"

    def test_generate_sub_agent_trace_path_different_extensions(self):
        """Test sub-agent path generation with different file extensions."""
        tracer = Tracer(agent_name="main_agent")

        paths = ["/tmp/trace.json", "/tmp/trace.txt", "/tmp/trace", "/tmp/trace.log.json"]

        for main_path in paths:
            sub_path = tracer.generate_sub_agent_trace_path("sub", main_path)
            assert sub_path is not None
            assert sub_path.endswith(".json")

    def test_generate_sub_agent_trace_path_with_complex_directory(self):
        """Test sub-agent path generation with complex directory structure."""
        tracer = Tracer(agent_name="main_agent")
        main_path = "/home/user/project/logs/2024/trace_v2.json"

        sub_path = tracer.generate_sub_agent_trace_path("analyzer", main_path)

        assert sub_path is not None
        assert "/home/user/project/logs/2024/trace_v2_sub_agents/analyzer_1.json" == sub_path

    def test_generate_sub_agent_trace_path_error_handling(self):
        """Test error handling in sub-agent path generation."""
        tracer = Tracer(agent_name="main_agent")

        # This might cause issues in some scenarios, but should not crash
        result = tracer.generate_sub_agent_trace_path("sub", "")

        # Should either return a valid path or None, but not crash
        assert result is None or isinstance(result, str)

    def test_generate_sub_agent_trace_path_logs_error_on_exception(self, caplog):
        """Test that errors in path generation are logged."""
        tracer = Tracer(agent_name="main_agent")

        with patch("nexau.archs.main_sub.tracing.tracer.Path", side_effect=Exception("Mock error")):
            with caplog.at_level(logging.ERROR):
                result = tracer.generate_sub_agent_trace_path("sub", "/tmp/trace.json")

            assert result is None
            assert any("Failed to generate sub-agent trace path" in record.message for record in caplog.records)


class TestThreadSafety:
    """Tests for thread safety of Tracer."""

    def test_concurrent_add_entries(self):
        """Test adding entries from multiple threads."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        num_threads = 10
        entries_per_thread = 100
        threads = []

        def add_entries(thread_id):
            for i in range(entries_per_thread):
                tracer.add_entry({"type": "test", "thread": thread_id, "index": i})

        # Start all threads
        for i in range(num_threads):
            thread = threading.Thread(target=add_entries, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all entries were added
        trace_data = tracer.get_trace_data()
        assert len(trace_data) == num_threads * entries_per_thread

    def test_concurrent_start_stop_tracing(self):
        """Test starting and stopping tracing from multiple threads."""
        tracer = Tracer(agent_name="test_agent")

        def toggle_tracing(iterations):
            for i in range(iterations):
                tracer.start_tracing(f"/tmp/trace_{i}.json")
                time.sleep(0.001)
                tracer.stop_tracing()

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=toggle_tracing, args=(10,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should end in a consistent state
        # Final state depends on last operation, but should not crash
        assert isinstance(tracer.is_tracing(), bool)

    def test_concurrent_sub_agent_counter(self):
        """Test sub-agent counter increments correctly with concurrent access."""
        tracer = Tracer(agent_name="test_agent")

        num_threads = 10
        paths = []
        paths_lock = threading.Lock()

        def generate_paths(count):
            thread_paths = []
            for _ in range(count):
                path = tracer.generate_sub_agent_trace_path("sub", "/tmp/trace.json")
                thread_paths.append(path)

            with paths_lock:
                paths.extend(thread_paths)

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=generate_paths, args=(10,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All paths should be unique due to counter
        assert len(paths) == num_threads * 10
        assert len(set(paths)) == num_threads * 10

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes to trace data."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        stop_flag = threading.Event()

        def writer():
            counter = 0
            while not stop_flag.is_set():
                tracer.add_entry({"type": "test", "counter": counter})
                counter += 1
                time.sleep(0.001)

        def reader():
            while not stop_flag.is_set():
                data = tracer.get_trace_data()
                assert data is not None
                time.sleep(0.001)

        # Start writer and readers
        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        writer_thread.start()
        for thread in reader_threads:
            thread.start()

        # Let them run for a bit
        time.sleep(0.1)
        stop_flag.set()

        # Wait for completion
        writer_thread.join()
        for thread in reader_threads:
            thread.join()

        # Should complete without errors
        assert tracer.is_tracing()
        trace_data = tracer.get_trace_data()
        assert len(trace_data) > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_trace_data(self):
        """Test getting trace data when no entries have been added."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        # When no entries added, _trace_data is [] which is falsy, so get_trace_data returns None
        # This is the current behavior in tracer.py line 163
        trace_data = tracer.get_trace_data()
        # Note: Empty list is falsy, so get_trace_data returns None for empty traces
        assert trace_data is None or trace_data == []

    def test_add_entry_with_none_values(self):
        """Test adding entry with None values."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        entry = {"type": "test", "value": None, "nested": {"key": None}}
        tracer.add_entry(entry)

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["value"] is None
        assert trace_data[0]["nested"]["key"] is None

    def test_add_llm_request_with_empty_messages(self):
        """Test adding LLM request with empty messages list."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        api_params = {"model": "gpt-4", "messages": []}
        tracer.add_llm_request(iteration=1, api_params=api_params)

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["api_params"]["messages"] == []

    def test_add_tool_request_with_empty_parameters(self):
        """Test adding tool request with empty parameters."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        tracer.add_tool_request(tool_name="test_tool", parameters={})

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["parameters"] == {}

    def test_special_characters_in_agent_name(self):
        """Test tracer with special characters in agent name."""
        special_names = ["agent-123", "agent_v2", "agent.test", "agent@host"]

        for name in special_names:
            tracer = Tracer(agent_name=name)
            tracer.start_tracing("/tmp/trace.json")
            tracer.add_entry({"type": "test"})

            trace_data = tracer.get_trace_data()
            assert trace_data[0]["agent_name"] == name

    def test_unicode_in_trace_entries(self):
        """Test tracer with unicode characters in entries."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        unicode_text = "测试 🚀 émojis and ñoñ-ASCII"
        tracer.add_entry({"type": "test", "content": unicode_text})

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["content"] == unicode_text

    def test_very_large_trace_data(self):
        """Test tracer with very large trace data."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        # Add many entries
        for i in range(1000):
            tracer.add_entry({"type": "test", "index": i, "data": "x" * 100})

        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 1000

    def test_nested_data_structures(self):
        """Test tracer with deeply nested data structures."""
        tracer = Tracer(agent_name="test_agent")
        tracer.start_tracing("/tmp/trace.json")

        nested_data = {"type": "test", "level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}

        tracer.add_entry(nested_data)

        trace_data = tracer.get_trace_data()
        assert trace_data[0]["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


class TestIntegrationScenarios:
    """Integration tests simulating real-world usage patterns."""

    def test_typical_agent_execution_flow(self):
        """Test a typical agent execution flow with tracing."""
        tracer = Tracer(agent_name="research_agent")

        # Start tracing
        tracer.start_tracing("/tmp/research_trace.json")

        # Iteration 1
        tracer.add_llm_request(iteration=1, api_params={"model": "gpt-4", "messages": [{"role": "user", "content": "Research topic X"}]})
        tracer.add_llm_response(iteration=1, response_content="I will use the search tool")

        # Tool execution
        tracer.add_tool_request(tool_name="web_search", parameters={"query": "topic X"})
        tracer.add_tool_response(tool_name="web_search", result={"results": ["result1", "result2"]})

        # Iteration 2
        tracer.add_llm_request(
            iteration=2,
            api_params={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Research topic X"}, {"role": "assistant", "content": "Search results: ..."}],
            },
        )
        tracer.add_llm_response(iteration=2, response_content="Based on the research, here is the answer...")

        # Complete
        tracer.add_shutdown(reason="Task completed successfully")

        # Verify trace
        trace_data = tracer.get_trace_data()
        assert len(trace_data) == 7

        # Verify timestamps are in order
        timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in trace_data]
        assert timestamps == sorted(timestamps)

    def test_error_recovery_scenario(self):
        """Test tracing during error and recovery."""
        tracer = Tracer(agent_name="resilient_agent")
        tracer.start_tracing("/tmp/error_trace.json")

        # First attempt
        tracer.add_tool_request(tool_name="api_call", parameters={"url": "https://api.example.com"})
        tracer.add_error(ConnectionError("API timeout"))

        # Retry
        tracer.add_tool_request(tool_name="api_call", parameters={"url": "https://api.example.com", "retry": 1})
        tracer.add_tool_response(tool_name="api_call", result={"status": "success"})

        trace_data = tracer.get_trace_data()

        # Should have both attempts logged
        tool_requests = [e for e in trace_data if e["type"] == "tool_request"]
        errors = [e for e in trace_data if e["type"] == "error"]

        assert len(tool_requests) == 2
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ConnectionError"

    def test_multi_subagent_scenario(self):
        """Test tracing with multiple sub-agents."""
        tracer = Tracer(agent_name="orchestrator")
        tracer.start_tracing("/tmp/orchestrator_trace.json")

        # Spawn multiple sub-agents
        subagents = ["researcher", "analyzer", "writer"]

        for subagent in subagents:
            # Request
            tracer.add_subagent_request(subagent_name=subagent, message=f"Execute {subagent} task")

            # Generate sub-agent trace path
            sub_path = tracer.generate_sub_agent_trace_path(subagent, "/tmp/orchestrator_trace.json")
            assert sub_path is not None

            # Response
            tracer.add_subagent_response(subagent_name=subagent, result=f"{subagent} completed")

        trace_data = tracer.get_trace_data()

        # Verify all sub-agent interactions are logged
        subagent_requests = [e for e in trace_data if e["type"] == "subagent_request"]
        subagent_responses = [e for e in trace_data if e["type"] == "subagent_response"]

        assert len(subagent_requests) == 3
        assert len(subagent_responses) == 3

        # Verify sub-agent names
        request_names = [e["subagent_name"] for e in subagent_requests]
        assert set(request_names) == set(subagents)

    def test_trace_persistence_across_sessions(self):
        """Test that tracer state can be restarted between sessions."""
        tracer = Tracer(agent_name="persistent_agent")

        # Session 1
        tracer.start_tracing("/tmp/session1.json")
        tracer.add_entry({"type": "session", "number": 1})
        session1_data = tracer.get_trace_data()
        tracer.stop_tracing()

        # Session 2
        tracer.start_tracing("/tmp/session2.json")
        tracer.add_entry({"type": "session", "number": 2})
        session2_data = tracer.get_trace_data()
        tracer.stop_tracing()

        # Sessions should be independent
        assert len(session1_data) == 1
        assert len(session2_data) == 1
        assert session1_data[0]["number"] == 1
        assert session2_data[0]["number"] == 2
