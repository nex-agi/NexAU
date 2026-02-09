import os
import shutil
import tempfile
from pathlib import Path

import pytest

from nexau.archs.sandbox.base_sandbox import (
    CodeLanguage,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
)
from nexau.archs.sandbox.e2b_sandbox import E2BSandbox, E2BSandboxManager
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine

pytestmark = pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set, skipping E2B tests")


@pytest.fixture
def e2b_sandbox():
    """Create a real E2B sandbox for testing with proper cleanup."""
    # Note: In real usage, E2BSandboxManager.start() would be called with session_manager, user_id, session_id, sandbox_config
    # For testing, we create a sandbox instance directly
    sandbox_manager = E2BSandboxManager()
    sandbox = sandbox_manager.start(
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        user_id="test_user",
        session_id="test_session",
        sandbox_config={},
    )
    yield sandbox
    sandbox_manager.stop()
    # Cleanup would be handled by E2BSandboxManager.stop() in real usage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


class TestE2BBashExecution:
    def test_execute_successful_command(self, e2b_sandbox):
        """Test executing a successful bash command."""
        result = e2b_sandbox.execute_bash("echo 'Hello World'")
        assert result.status == SandboxStatus.SUCCESS
        assert "Hello World" in result.stdout
        assert result.exit_code == 0

    def test_execute_command_with_error(self, e2b_sandbox):
        """Test executing a command that fails."""
        result = e2b_sandbox.execute_bash("exit 1")
        assert result.status == SandboxStatus.ERROR
        assert result.exit_code == 1
        assert result.error is not None

    def test_execute_command_with_output(self, e2b_sandbox):
        """Test command with stdout output."""
        result = e2b_sandbox.execute_bash("ls /")
        assert result.status == SandboxStatus.SUCCESS
        assert len(result.stdout) > 0

    def test_execute_multiline_command(self, e2b_sandbox):
        """Test executing multiline bash commands."""
        cmd = """
        echo "Line 1"
        echo "Line 2"
        echo "Line 3"
        """
        result = e2b_sandbox.execute_bash(cmd)
        assert result.status == SandboxStatus.SUCCESS
        assert "Line 1" in result.stdout
        assert "Line 2" in result.stdout
        assert "Line 3" in result.stdout

    def test_execute_without_sandbox_instance(self):
        """Test that executing without internal sandbox raises error."""
        sandbox = E2BSandbox(sandbox_id="test")
        with pytest.raises((SandboxError, AttributeError)):
            sandbox.execute_bash("echo test")

    def test_execute_command_with_user_parameter(self, e2b_sandbox):
        """Test executing a command with specific user parameter."""
        result = e2b_sandbox.execute_bash("whoami", user="root")
        assert result.status == SandboxStatus.SUCCESS
        assert "root" in result.stdout.strip()

        result_user = e2b_sandbox.execute_bash("whoami", user="user")
        assert result_user.status == SandboxStatus.SUCCESS
        assert "user" in result_user.stdout.strip()

    def test_execute_command_with_envs_parameter(self, e2b_sandbox):
        """Test executing a command with environment variables."""
        envs = {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        result = e2b_sandbox.execute_bash("echo $TEST_VAR $ANOTHER_VAR", envs=envs)
        assert result.status == SandboxStatus.SUCCESS
        assert "test_value" in result.stdout
        assert "another_value" in result.stdout


class TestE2BCodeExecution:
    def test_execute_simple_python_code(self, e2b_sandbox):
        """Test executing simple Python code."""
        code = "print('Hello from Python')"
        result = e2b_sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.SUCCESS
        assert result.language == CodeLanguage.PYTHON
        assert result.outputs is not None
        assert any("Hello from Python" in str(output) for output in result.outputs)

    def test_execute_python_with_calculation(self, e2b_sandbox):
        """Test Python code with calculations."""
        code = """
result = 2 + 2
print(f"Result: {result}")
"""
        result = e2b_sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.SUCCESS
        assert any("Result: 4" in str(output) for output in result.outputs or [])

    def test_execute_code_with_error(self, e2b_sandbox):
        """Test Python code that raises an error."""
        code = "print(undefined_variable)"
        result = e2b_sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.ERROR
        assert result.error_type is not None

    def test_execute_code_with_imports(self, e2b_sandbox):
        """Test Python code with imports."""
        code = """
import json
data = {"key": "value"}
print(json.dumps(data))
"""
        result = e2b_sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.SUCCESS
        assert any("key" in str(output) for output in result.outputs or [])

    def test_execute_unsupported_language(self, e2b_sandbox):
        """Test that unsupported language returns error."""
        result = e2b_sandbox.execute_code("console.log('test')", "javascript")
        assert result.status == SandboxStatus.ERROR
        assert "Unsupported language" in (result.error_value or "")


class TestE2BFileOperations:
    def test_write_and_read_text_file(self, e2b_sandbox):
        """Test writing and reading a text file."""
        content = "Test content for E2B"
        write_result = e2b_sandbox.write_file("test.txt", content)
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("test.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == content

    def test_write_and_read_binary_file(self, e2b_sandbox):
        """Test writing and reading a binary file."""
        binary_content = b"\x00\x01\x02\x03\x04"
        write_result = e2b_sandbox.write_file("binary.dat", binary_content, binary=True)
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("binary.dat", binary=True)
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == binary_content

    def test_write_file_with_subdirectories(self, e2b_sandbox):
        """Test writing file in nested subdirectories."""
        content = "Nested file content"
        write_result = e2b_sandbox.write_file("subdir/nested/file.txt", content)
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("subdir/nested/file.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == content

    def test_delete_file(self, e2b_sandbox):
        """Test deleting a file."""
        e2b_sandbox.write_file("to_delete.txt", "content")
        assert e2b_sandbox.file_exists("to_delete.txt")

        delete_result = e2b_sandbox.delete_file("to_delete.txt")
        assert delete_result.status == SandboxStatus.SUCCESS
        assert not e2b_sandbox.file_exists("to_delete.txt")

    def test_delete_nonexistent_file(self, e2b_sandbox):
        """Test deleting a non-existent file returns error."""
        result = e2b_sandbox.delete_file("nonexistent.txt")
        assert result.status == SandboxStatus.ERROR

    def test_file_exists(self, e2b_sandbox):
        """Test file existence check."""
        assert not e2b_sandbox.file_exists("missing.txt")

        e2b_sandbox.write_file("exists.txt", "content")
        assert e2b_sandbox.file_exists("exists.txt")

    def test_get_file_info(self, e2b_sandbox):
        """Test getting file information."""
        content = "File info test"
        e2b_sandbox.write_file("info_test.txt", content)

        info = e2b_sandbox.get_file_info("info_test.txt")
        assert info.exists
        assert info.is_file
        assert not info.is_directory
        assert info.size > 0

    def test_get_file_info_nonexistent(self, e2b_sandbox):
        """Test getting info for non-existent file."""
        info = e2b_sandbox.get_file_info("nonexistent.txt")
        assert not info.exists


class TestE2BDirectoryOperations:
    def test_create_directory(self, e2b_sandbox):
        """Test creating a directory."""
        assert e2b_sandbox.create_directory("test_dir")
        assert e2b_sandbox.file_exists("test_dir")

    def test_create_nested_directory(self, e2b_sandbox):
        """Test creating nested directories."""
        assert e2b_sandbox.create_directory("level1/level2/level3", parents=True)
        assert e2b_sandbox.file_exists("level1/level2/level3")

    def test_list_files_in_directory(self, e2b_sandbox):
        """Test listing files in a directory."""
        e2b_sandbox.write_file("file1.txt", "content1")
        e2b_sandbox.write_file("file2.txt", "content2")
        e2b_sandbox.create_directory("subdir")

        files = e2b_sandbox.list_files(".")
        assert len(files) >= 3
        file_names = [Path(f.path).name for f in files]
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names

    def test_list_files_recursive(self, e2b_sandbox):
        """Test listing files recursively."""
        e2b_sandbox.write_file("root.txt", "root")
        e2b_sandbox.write_file("subdir/nested.txt", "nested")
        e2b_sandbox.write_file("subdir/deep/file.txt", "deep")

        files = e2b_sandbox.list_files(".", recursive=True)
        file_paths = [f.path for f in files]
        assert any("root.txt" in p for p in file_paths)
        assert any("nested.txt" in p for p in file_paths)

    def test_list_nonexistent_directory(self, e2b_sandbox):
        """Test listing non-existent directory raises error."""
        with pytest.raises(SandboxFileError):
            e2b_sandbox.list_files("/nonexistent_dir_12345")


class TestE2BFileEditing:
    def test_create_file_with_edit(self, e2b_sandbox):
        """Test creating a new file using edit."""
        result = e2b_sandbox.edit_file("new_file.txt", "", "New content")
        assert result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("new_file.txt")
        assert read_result.content == "New content"

    def test_update_file_content(self, e2b_sandbox):
        """Test updating file content."""
        e2b_sandbox.write_file("update_test.txt", "Original content")

        result = e2b_sandbox.edit_file("update_test.txt", "Original", "Updated")
        assert result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("update_test.txt")
        assert read_result.content == "Updated content"

    def test_remove_content_from_file(self, e2b_sandbox):
        """Test removing content from file."""
        e2b_sandbox.write_file("remove_test.txt", "Keep this. Remove this.")

        result = e2b_sandbox.edit_file("remove_test.txt", " Remove this.", "")
        assert result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("remove_test.txt")
        assert read_result.content == "Keep this."

    def test_edit_string_not_found(self, e2b_sandbox):
        """Test editing with non-existent string returns error."""
        e2b_sandbox.write_file("test.txt", "Content")

        result = e2b_sandbox.edit_file("test.txt", "NonExistent", "New")
        assert result.status == SandboxStatus.ERROR
        assert "not found" in result.error.lower()

    def test_edit_multiple_matches(self, e2b_sandbox):
        """Test editing with multiple matches returns error."""
        e2b_sandbox.write_file("test.txt", "test test test")

        result = e2b_sandbox.edit_file("test.txt", "test", "replaced")
        assert result.status == SandboxStatus.ERROR
        assert "matches" in result.error.lower()


class TestE2BGlobPattern:
    def test_glob_simple_pattern(self, e2b_sandbox):
        """Test simple glob pattern matching."""
        e2b_sandbox.write_file("test1.py", "")
        e2b_sandbox.write_file("test2.py", "")
        e2b_sandbox.write_file("test.txt", "")

        matches = e2b_sandbox.glob("*.py")
        assert len(matches) >= 2
        assert all(".py" in m for m in matches)

    def test_glob_recursive_pattern(self, e2b_sandbox):
        """Test recursive glob pattern."""
        e2b_sandbox.write_file("root.txt", "")
        e2b_sandbox.write_file("subdir/nested.txt", "")
        e2b_sandbox.write_file("subdir/deep/file.txt", "")

        matches = e2b_sandbox.glob("**/*.txt", recursive=True)
        assert len(matches) >= 3


class TestE2BFileTransfer:
    def test_upload_file(self, e2b_sandbox, temp_dir):
        """Test uploading a file from local to sandbox."""
        source_file = Path(temp_dir) / "source.txt"
        source_file.write_text("Upload test content")

        result = e2b_sandbox.upload_file(str(source_file), "uploaded.txt")
        assert result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("uploaded.txt")
        assert read_result.content == "Upload test content"

    def test_upload_nonexistent_file(self, e2b_sandbox):
        """Test uploading non-existent file returns error."""
        result = e2b_sandbox.upload_file("/nonexistent/file.txt", "dest.txt")
        assert result.status == SandboxStatus.ERROR
        assert "does not exist" in result.error.lower()

    def test_download_file(self, e2b_sandbox, temp_dir):
        """Test downloading a file from sandbox to local."""
        e2b_sandbox.write_file("download_test.txt", "Download content")

        dest_file = Path(temp_dir) / "downloaded.txt"
        result = e2b_sandbox.download_file("download_test.txt", str(dest_file))
        assert result.status == SandboxStatus.SUCCESS
        assert dest_file.exists()
        assert dest_file.read_text() == "Download content"

    def test_download_nonexistent_file(self, e2b_sandbox, temp_dir):
        """Test downloading non-existent file returns error."""
        dest_file = Path(temp_dir) / "dest.txt"
        result = e2b_sandbox.download_file("nonexistent.txt", str(dest_file))
        assert result.status == SandboxStatus.ERROR

    def test_upload_directory(self, e2b_sandbox, temp_dir):
        """Test uploading a directory."""
        source_dir = Path(temp_dir) / "upload_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "file2.txt").write_text("content2")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file3.txt").write_text("content3")

        assert e2b_sandbox.upload_directory(str(source_dir), "uploaded_dir")
        assert e2b_sandbox.file_exists("uploaded_dir/file1.txt")
        assert e2b_sandbox.file_exists("uploaded_dir/file2.txt")
        assert e2b_sandbox.file_exists("uploaded_dir/subdir/file3.txt")

    def test_download_directory(self, e2b_sandbox, temp_dir):
        """Test downloading a directory."""
        e2b_sandbox.write_file("download_dir/file1.txt", "content1")
        e2b_sandbox.write_file("download_dir/file2.txt", "content2")
        e2b_sandbox.write_file("download_dir/subdir/file3.txt", "content3")

        dest_dir = Path(temp_dir) / "downloaded_dir"
        assert e2b_sandbox.download_directory("download_dir", str(dest_dir))
        assert (dest_dir / "file1.txt").exists()
        assert (dest_dir / "file2.txt").exists()
        assert (dest_dir / "subdir" / "file3.txt").exists()


class TestE2BBackgroundExecution:
    def test_execute_bash_background(self, e2b_sandbox):
        """Test starting a background task returns immediately with a pid."""
        result = e2b_sandbox.execute_bash("sleep 10", background=True)
        assert result.status == SandboxStatus.SUCCESS
        assert result.background_pid is not None
        assert result.background_pid > 0
        assert "pid" in result.stdout.lower()
        # Cleanup
        e2b_sandbox.kill_background_task(result.background_pid)

    def test_background_task_status_running(self, e2b_sandbox):
        """Test checking status of a still-running background task."""
        result = e2b_sandbox.execute_bash("sleep 10", background=True)
        pid = result.background_pid
        assert pid is not None

        status = e2b_sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.RUNNING
        assert status.background_pid == pid
        assert status.exit_code == -1
        # Cleanup
        e2b_sandbox.kill_background_task(pid)

    def test_background_task_status_finished(self, e2b_sandbox):
        """Test checking status of a finished background task."""
        result = e2b_sandbox.execute_bash("echo done", background=True)
        pid = result.background_pid
        assert pid is not None

        import time

        time.sleep(2)  # Wait for the short command to finish

        status = e2b_sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.SUCCESS
        assert status.exit_code == 0
        assert "done" in status.stdout
        assert status.background_pid == pid

    def test_background_task_status_not_found(self, e2b_sandbox):
        """Test checking status of a non-existent background task."""
        status = e2b_sandbox.get_background_task_status(99999)
        assert status.status == SandboxStatus.ERROR
        assert "not found" in status.error.lower()

    def test_kill_background_task(self, e2b_sandbox):
        """Test killing a running background task."""
        result = e2b_sandbox.execute_bash("sleep 60", background=True)
        pid = result.background_pid
        assert pid is not None

        kill_result = e2b_sandbox.kill_background_task(pid)
        assert kill_result.status == SandboxStatus.SUCCESS
        assert kill_result.background_pid == pid

        # Task should no longer be found
        status = e2b_sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.ERROR

    def test_kill_background_task_not_found(self, e2b_sandbox):
        """Test killing a non-existent background task."""
        result = e2b_sandbox.kill_background_task(99999)
        assert result.status == SandboxStatus.ERROR
        assert "not found" in result.error.lower()

    def test_multiple_background_tasks(self, e2b_sandbox):
        """Test running multiple background tasks simultaneously."""
        result1 = e2b_sandbox.execute_bash("sleep 10", background=True)
        result2 = e2b_sandbox.execute_bash("sleep 10", background=True)
        pid1 = result1.background_pid
        pid2 = result2.background_pid
        assert pid1 is not None
        assert pid2 is not None
        assert pid1 != pid2

        status1 = e2b_sandbox.get_background_task_status(pid1)
        status2 = e2b_sandbox.get_background_task_status(pid2)
        assert status1.status == SandboxStatus.RUNNING
        assert status2.status == SandboxStatus.RUNNING

        # Cleanup
        e2b_sandbox.kill_background_task(pid1)
        e2b_sandbox.kill_background_task(pid2)


class TestE2BEdgeCases:
    def test_empty_file_operations(self, e2b_sandbox):
        """Test operations with empty files."""
        write_result = e2b_sandbox.write_file("empty.txt", "")
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = e2b_sandbox.read_file("empty.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == ""

    def test_unicode_content(self, e2b_sandbox):
        """Test handling unicode content."""
        unicode_content = "Hello 世界 🌍"
        e2b_sandbox.write_file("unicode.txt", unicode_content)

        read_result = e2b_sandbox.read_file("unicode.txt")
        assert read_result.content == unicode_content

    def test_dict_representation(self, e2b_sandbox):
        """Test sandbox dict representation."""

        result = e2b_sandbox.dict()
        assert "sandbox_id" in result
        assert "_work_dir" in result
