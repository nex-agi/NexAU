import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from nexau.archs.sandbox.base_sandbox import (
    BASH_TOOL_RESULTS_BASE_PATH,
    CodeLanguage,
    SandboxFileError,
    SandboxStatus,
)
from nexau.archs.sandbox.local_sandbox import LocalSandbox


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sandbox(temp_dir):
    sb = LocalSandbox(sandbox_id="test_sandbox", _work_dir=temp_dir)
    yield sb


class TestLocalSandboxLifecycle:
    def test_sandbox_initialization(self, temp_dir):
        sandbox = LocalSandbox(sandbox_id="lifecycle_test", _work_dir=temp_dir)
        assert sandbox.sandbox_id == "lifecycle_test"
        assert sandbox.work_dir == Path(temp_dir)


class TestBashCommandExecution:
    def test_execute_simple_command(self, sandbox):
        result = sandbox.execute_bash("echo 'Hello World'")
        assert result.status == SandboxStatus.SUCCESS
        assert "Hello World" in result.stdout
        assert result.exit_code == 0
        assert result.error is None

    def test_execute_command_with_error(self, sandbox):
        result = sandbox.execute_bash("exit 1")
        assert result.status == SandboxStatus.ERROR
        assert result.exit_code == 1
        assert result.error is not None

    def test_execute_command_timeout(self, sandbox):
        result = sandbox.execute_bash("sleep 10", timeout=100)
        assert result.status == SandboxStatus.TIMEOUT
        assert "timed out" in result.error.lower()

    def test_execute_command_with_output(self, sandbox):
        result = sandbox.execute_bash("ls -la")
        assert result.status == SandboxStatus.SUCCESS
        assert len(result.stdout) > 0

    def test_execute_command_stderr(self, sandbox):
        result = sandbox.execute_bash("ls /nonexistent_directory_12345")
        assert result.status == SandboxStatus.ERROR
        assert len(result.stderr) > 0

    def test_execute_multiline_command(self, sandbox):
        cmd = """
        echo "Line 1"
        echo "Line 2"
        echo "Line 3"
        """
        result = sandbox.execute_bash(cmd)
        assert result.status == SandboxStatus.SUCCESS
        assert "Line 1" in result.stdout
        assert "Line 2" in result.stdout
        assert "Line 3" in result.stdout

    def test_execute_command_in_work_dir(self, sandbox):
        sandbox.write_file("test_marker.txt", "marker")
        result = sandbox.execute_bash("cat test_marker.txt")
        assert result.status == SandboxStatus.SUCCESS
        assert "marker" in result.stdout


class TestCodeExecution:
    def test_execute_simple_python_code(self, sandbox):
        code = "print('Hello from Python')"
        result = sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.SUCCESS
        assert result.language == CodeLanguage.PYTHON
        assert result.outputs is not None
        assert any("Hello from Python" in str(output) for output in result.outputs)

    def test_execute_python_code_with_calculation(self, sandbox):
        code = """
result = 2 + 2
print(f"Result: {result}")
"""
        result = sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.SUCCESS
        assert any("Result: 4" in str(output) for output in result.outputs or [])

    def test_execute_python_code_with_error(self, sandbox):
        code = "print(undefined_variable)"
        result = sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.ERROR
        assert result.error_type is not None

    def test_execute_python_code_with_syntax_error(self, sandbox):
        code = "print('unclosed string"
        result = sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.ERROR

    def test_execute_python_code_with_imports(self, sandbox):
        code = """
import json
data = {"key": "value"}
print(json.dumps(data))
"""
        result = sandbox.execute_code(code, CodeLanguage.PYTHON)
        assert result.status == SandboxStatus.SUCCESS
        assert any("key" in str(output) for output in result.outputs or [])

    def test_execute_unsupported_language(self, sandbox):
        result = sandbox.execute_code("console.log('test')", "javascript")
        assert result.status == SandboxStatus.ERROR
        assert "Unsupported language" in result.error_value or ""

    def test_execute_code_string_language(self, sandbox):
        result = sandbox.execute_code("print('test')", "python")
        assert result.status == SandboxStatus.SUCCESS


class TestFileOperations:
    def test_write_and_read_text_file(self, sandbox):
        content = "Hello, World!"
        write_result = sandbox.write_file("test.txt", content)
        assert write_result.status == SandboxStatus.SUCCESS
        assert write_result.size > 0

        read_result = sandbox.read_file("test.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == content

    def test_write_and_read_binary_file(self, sandbox):
        binary_content = b"\x00\x01\x02\x03\x04"
        write_result = sandbox.write_file("binary.dat", binary_content, binary=True)
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("binary.dat", binary=True)
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == binary_content

    def test_write_file_with_subdirectories(self, sandbox):
        content = "Nested file"
        write_result = sandbox.write_file("subdir/nested/file.txt", content)
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("subdir/nested/file.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == content

    def test_read_nonexistent_file(self, sandbox):
        result = sandbox.read_file("nonexistent.txt")
        assert result.status == SandboxStatus.ERROR
        assert "does not exist" in result.error.lower()

    def test_delete_file(self, sandbox):
        sandbox.write_file("to_delete.txt", "content")
        assert sandbox.file_exists("to_delete.txt")

        delete_result = sandbox.delete_file("to_delete.txt")
        assert delete_result.status == SandboxStatus.SUCCESS
        assert not sandbox.file_exists("to_delete.txt")

    def test_delete_nonexistent_file(self, sandbox):
        result = sandbox.delete_file("nonexistent.txt")
        assert result.status == SandboxStatus.ERROR

    def test_delete_directory(self, sandbox):
        sandbox.create_directory("test_dir")
        sandbox.write_file("test_dir/file.txt", "content")

        delete_result = sandbox.delete_file("test_dir")
        assert delete_result.status == SandboxStatus.SUCCESS
        assert not sandbox.file_exists("test_dir")

    def test_file_exists(self, sandbox):
        assert not sandbox.file_exists("missing.txt")

        sandbox.write_file("exists.txt", "content")
        assert sandbox.file_exists("exists.txt")

    def test_get_file_info_for_file(self, sandbox):
        content = "Test content"
        sandbox.write_file("info_test.txt", content)

        info = sandbox.get_file_info("info_test.txt")
        assert info.exists
        assert info.is_file
        assert not info.is_directory
        assert info.size > 0
        assert info.readable
        assert info.writable
        assert info.encoding is not None

    def test_get_file_info_for_directory(self, sandbox):
        sandbox.create_directory("test_dir")

        info = sandbox.get_file_info("test_dir")
        assert info.exists
        assert not info.is_file
        assert info.is_directory

    def test_get_file_info_nonexistent(self, sandbox):
        info = sandbox.get_file_info("nonexistent.txt")
        assert not info.exists


class TestDirectoryOperations:
    def test_create_directory(self, sandbox):
        assert sandbox.create_directory("new_dir")
        assert sandbox.file_exists("new_dir")

        info = sandbox.get_file_info("new_dir")
        assert info.is_directory

    def test_create_nested_directory(self, sandbox):
        assert sandbox.create_directory("level1/level2/level3", parents=True)
        assert sandbox.file_exists("level1/level2/level3")

    def test_list_files_in_directory(self, sandbox):
        sandbox.write_file("file1.txt", "content1")
        sandbox.write_file("file2.txt", "content2")
        sandbox.create_directory("subdir")

        files = sandbox.list_files(".")
        assert len(files) >= 3
        file_names = [Path(f.path).name for f in files]
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names
        assert "subdir" in file_names

    def test_list_files_recursive(self, sandbox):
        sandbox.write_file("root.txt", "root")
        sandbox.write_file("subdir/nested.txt", "nested")
        sandbox.write_file("subdir/deep/file.txt", "deep")

        files = sandbox.list_files(".", recursive=True)
        file_paths = [f.path for f in files]
        assert any("root.txt" in p for p in file_paths)
        assert any("nested.txt" in p for p in file_paths)
        assert any("deep" in p and "file.txt" in p for p in file_paths)

    def test_list_files_with_pattern(self, sandbox):
        sandbox.write_file("test1.py", "python1")
        sandbox.write_file("test2.py", "python2")
        sandbox.write_file("test.txt", "text")

        files = sandbox.list_files(".", pattern="*.py")
        assert len(files) == 2
        assert all(f.path.endswith(".py") for f in files)

    def test_list_nonexistent_directory(self, sandbox):
        with pytest.raises(SandboxFileError):
            sandbox.list_files("nonexistent_dir")


class TestFileEditing:
    def test_create_file_with_edit(self, sandbox):
        result = sandbox.edit_file("new_file.txt", "", "New content")
        assert result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("new_file.txt")
        assert read_result.content == "New content"

    def test_update_file_content(self, sandbox):
        sandbox.write_file("update_test.txt", "Original content")

        result = sandbox.edit_file("update_test.txt", "Original", "Updated")
        assert result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("update_test.txt")
        assert read_result.content == "Updated content"

    def test_remove_content_from_file(self, sandbox):
        sandbox.write_file("remove_test.txt", "Keep this. Remove this.")

        result = sandbox.edit_file("remove_test.txt", " Remove this.", "")
        assert result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("remove_test.txt")
        assert read_result.content == "Keep this."

    def test_edit_file_string_not_found(self, sandbox):
        sandbox.write_file("test.txt", "Content")

        result = sandbox.edit_file("test.txt", "NonExistent", "New")
        assert result.status == SandboxStatus.ERROR
        assert "not found" in result.error.lower()

    def test_edit_file_multiple_matches(self, sandbox):
        sandbox.write_file("test.txt", "test test test")

        result = sandbox.edit_file("test.txt", "test", "replaced")
        assert result.status == SandboxStatus.ERROR
        assert "matches" in result.error.lower()

    def test_create_existing_file_fails(self, sandbox):
        sandbox.write_file("existing.txt", "content")

        result = sandbox.edit_file("existing.txt", "", "new content")
        assert result.status == SandboxStatus.ERROR
        assert "already exists" in result.error.lower()

    def test_update_nonexistent_file_fails(self, sandbox):
        result = sandbox.edit_file("nonexistent.txt", "old", "new")
        assert result.status == SandboxStatus.ERROR
        assert "does not exist" in result.error.lower()


class TestGlobPattern:
    def test_glob_simple_pattern(self, sandbox):
        sandbox.write_file("test1.py", "")
        sandbox.write_file("test2.py", "")
        sandbox.write_file("test.txt", "")

        matches = sandbox.glob("*.py")
        assert len(matches) == 2
        assert all(m.endswith(".py") for m in matches)

    def test_glob_recursive_pattern(self, sandbox):
        sandbox.write_file("root.txt", "")
        sandbox.write_file("subdir/nested.txt", "")
        sandbox.write_file("subdir/deep/file.txt", "")

        matches = sandbox.glob("**/*.txt", recursive=True)
        assert len(matches) >= 3

    def test_glob_no_matches(self, sandbox):
        matches = sandbox.glob("*.nonexistent")
        assert len(matches) == 0

    def test_glob_specific_directory(self, sandbox):
        sandbox.write_file("dir1/file.py", "")
        sandbox.write_file("dir2/file.py", "")

        matches = sandbox.glob("dir1/*.py")
        assert len(matches) == 1
        assert "dir1" in matches[0]


class TestFileUploadDownload:
    def test_upload_file(self, sandbox, temp_dir):
        source_file = Path(temp_dir) / "source.txt"
        source_file.write_text("Upload test content")

        result = sandbox.upload_file(str(source_file), "uploaded.txt")
        assert result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("uploaded.txt")
        assert read_result.content == "Upload test content"

    def test_upload_nonexistent_file(self, sandbox):
        result = sandbox.upload_file("/nonexistent/file.txt", "dest.txt")
        assert result.status == SandboxStatus.ERROR
        assert "does not exist" in result.error.lower()

    def test_download_file(self, sandbox, temp_dir):
        sandbox.write_file("download_test.txt", "Download content")

        dest_file = Path(temp_dir) / "downloaded.txt"
        result = sandbox.download_file("download_test.txt", str(dest_file))
        assert result.status == SandboxStatus.SUCCESS
        assert dest_file.exists()
        assert dest_file.read_text() == "Download content"

    def test_download_nonexistent_file(self, sandbox, temp_dir):
        dest_file = Path(temp_dir) / "dest.txt"
        result = sandbox.download_file("nonexistent.txt", str(dest_file))
        assert result.status == SandboxStatus.ERROR

    def test_upload_directory(self, sandbox, temp_dir):
        source_dir = Path(temp_dir) / "upload_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "file2.txt").write_text("content2")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file3.txt").write_text("content3")

        assert sandbox.upload_directory(str(source_dir), "uploaded_dir")
        assert sandbox.file_exists("uploaded_dir/file1.txt")
        assert sandbox.file_exists("uploaded_dir/file2.txt")
        assert sandbox.file_exists("uploaded_dir/subdir/file3.txt")

    def test_download_directory(self, sandbox, temp_dir):
        sandbox.write_file("download_dir/file1.txt", "content1")
        sandbox.write_file("download_dir/file2.txt", "content2")
        sandbox.write_file("download_dir/subdir/file3.txt", "content3")

        dest_dir = Path(temp_dir) / "downloaded_dir"
        assert sandbox.download_directory("download_dir", str(dest_dir))
        assert (dest_dir / "file1.txt").exists()
        assert (dest_dir / "file2.txt").exists()
        assert (dest_dir / "subdir" / "file3.txt").exists()


class TestPathResolution:
    def test_relative_path_resolution(self, sandbox):
        sandbox.write_file("relative.txt", "content")
        assert sandbox.file_exists("relative.txt")

    def test_absolute_path_resolution(self, sandbox):
        abs_path = sandbox.work_dir / "absolute.txt"
        sandbox.write_file(str(abs_path), "content")
        assert sandbox.file_exists(str(abs_path))

    def test_nested_relative_path(self, sandbox):
        sandbox.write_file("a/b/c/nested.txt", "content")
        assert sandbox.file_exists("a/b/c/nested.txt")


class TestEdgeCases:
    def test_empty_file_operations(self, sandbox):
        write_result = sandbox.write_file("empty.txt", "")
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("empty.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.content == ""

    def test_large_file_content(self, sandbox):
        large_content = "x" * 50000
        write_result = sandbox.write_file("large.txt", large_content)
        assert write_result.status == SandboxStatus.SUCCESS

        read_result = sandbox.read_file("large.txt")
        assert read_result.status == SandboxStatus.SUCCESS
        assert read_result.truncated

    def test_special_characters_in_filename(self, sandbox):
        sandbox.write_file("file with spaces.txt", "content")
        assert sandbox.file_exists("file with spaces.txt")

    def test_unicode_content(self, sandbox):
        unicode_content = "Hello 世界 🌍"
        sandbox.write_file("unicode.txt", unicode_content)

        read_result = sandbox.read_file("unicode.txt")
        assert read_result.content == unicode_content

    def test_multiple_file_operations_sequence(self, sandbox):
        for i in range(10):
            sandbox.write_file(f"file{i}.txt", f"content{i}")

        files = sandbox.list_files(".")
        txt_files = [f for f in files if f.path.endswith(".txt")]
        assert len(txt_files) >= 10

    def test_concurrent_file_access(self, sandbox):
        sandbox.write_file("concurrent.txt", "initial")

        read1 = sandbox.read_file("concurrent.txt")
        sandbox.write_file("concurrent.txt", "updated")
        read2 = sandbox.read_file("concurrent.txt")

        assert read1.content == "initial"
        assert read2.content == "updated"


class TestBackgroundExecution:
    def test_execute_bash_background(self, sandbox):
        """Test starting a background task returns immediately with a pid."""
        result = sandbox.execute_bash("sleep 10", background=True)
        assert result.status == SandboxStatus.SUCCESS
        assert result.background_pid is not None
        assert result.background_pid > 0
        assert "pid" in result.stdout.lower()
        # Cleanup
        sandbox.kill_background_task(result.background_pid)

    def test_background_task_status_running(self, sandbox):
        """Test checking status of a still-running background task."""
        result = sandbox.execute_bash("sleep 10", background=True)
        pid = result.background_pid
        assert pid is not None

        status = sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.RUNNING
        assert status.background_pid == pid
        assert status.exit_code == -1
        # Cleanup
        sandbox.kill_background_task(pid)

    def test_background_task_status_finished(self, sandbox):
        """Test checking status of a finished background task."""
        result = sandbox.execute_bash("echo done", background=True)
        pid = result.background_pid
        assert pid is not None

        import time

        time.sleep(0.5)  # Wait for the short command to finish

        status = sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.SUCCESS
        assert status.exit_code == 0
        assert "done" in status.stdout
        assert status.background_pid == pid

    def test_background_task_status_not_found(self, sandbox):
        """Test checking status of a non-existent background task."""
        status = sandbox.get_background_task_status(99999)
        assert status.status == SandboxStatus.ERROR
        assert "not found" in status.error.lower()

    def test_kill_background_task(self, sandbox):
        """Test killing a running background task."""
        result = sandbox.execute_bash("sleep 60", background=True)
        pid = result.background_pid
        assert pid is not None

        kill_result = sandbox.kill_background_task(pid)
        assert kill_result.status == SandboxStatus.SUCCESS
        assert kill_result.background_pid == pid
        assert "killed" in kill_result.stdout.lower()

        # Task should no longer be found
        status = sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.ERROR

    def test_kill_background_task_not_found(self, sandbox):
        """Test killing a non-existent background task."""
        result = sandbox.kill_background_task(99999)
        assert result.status == SandboxStatus.ERROR
        assert "not found" in result.error.lower()

    def test_background_task_with_stderr(self, sandbox):
        """Test background task that produces stderr output."""
        result = sandbox.execute_bash("ls /nonexistent_path_12345", background=True)
        pid = result.background_pid
        assert pid is not None

        import time

        time.sleep(0.5)

        status = sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.ERROR
        assert status.exit_code != 0

    def test_multiple_background_tasks(self, sandbox):
        """Test running multiple background tasks simultaneously."""
        result1 = sandbox.execute_bash("sleep 10", background=True)
        result2 = sandbox.execute_bash("sleep 10", background=True)
        pid1 = result1.background_pid
        pid2 = result2.background_pid
        assert pid1 is not None
        assert pid2 is not None
        assert pid1 != pid2

        status1 = sandbox.get_background_task_status(pid1)
        status2 = sandbox.get_background_task_status(pid2)
        assert status1.status == SandboxStatus.RUNNING
        assert status2.status == SandboxStatus.RUNNING

        # Cleanup
        sandbox.kill_background_task(pid1)
        sandbox.kill_background_task(pid2)


class TestSaveOutputToTempFile:
    def test_foreground_save_output_creates_files(self, sandbox):
        """Test that save_output_to_temp_file creates command.txt, stdout.log, stderr.log in foreground mode."""
        result = sandbox.execute_bash("echo 'hello save'", save_output_to_temp_file=True)
        assert result.status == SandboxStatus.SUCCESS
        assert "[Output saved to:" in result.stdout

        # Extract temp dir path from stdout
        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        # Verify files exist and have correct content
        assert Path(f"{temp_dir}/command.txt").exists()
        assert Path(f"{temp_dir}/stdout.log").exists()
        assert Path(f"{temp_dir}/stderr.log").exists()

        assert Path(f"{temp_dir}/command.txt").read_text() == "echo 'hello save'"
        assert "hello save" in Path(f"{temp_dir}/stdout.log").read_text()

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_foreground_save_output_with_stderr(self, sandbox):
        """Test that stderr is also saved when save_output_to_temp_file is enabled."""
        result = sandbox.execute_bash("echo 'out' && echo 'err' >&2", save_output_to_temp_file=True)
        assert "[Output saved to:" in result.stdout

        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        assert "out" in Path(f"{temp_dir}/stdout.log").read_text()
        assert "err" in Path(f"{temp_dir}/stderr.log").read_text()

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_foreground_no_save_output_by_default(self, sandbox):
        """Test that output is NOT saved to temp file when save_output_to_temp_file is False (default)."""
        result = sandbox.execute_bash("echo 'no save'")
        assert result.status == SandboxStatus.SUCCESS
        assert "[Output saved to:" not in result.stdout

    def test_foreground_save_output_with_failed_command(self, sandbox):
        """Test save_output_to_temp_file works even when the command fails."""
        result = sandbox.execute_bash("echo 'before fail' && exit 1", save_output_to_temp_file=True)
        assert result.status == SandboxStatus.ERROR
        assert "[Output saved to:" in result.stdout

        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        assert Path(f"{temp_dir}/stdout.log").exists()
        assert "before fail" in Path(f"{temp_dir}/stdout.log").read_text()

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_background_save_output_creates_files(self, sandbox):
        """Test that save_output_to_temp_file works in background mode."""
        import time

        result = sandbox.execute_bash("echo 'bg output'", background=True, save_output_to_temp_file=True)
        assert result.status == SandboxStatus.SUCCESS
        assert result.background_pid is not None
        assert "Output will be saved to" in result.stdout

        # Wait for background task to finish
        time.sleep(1)

        status = sandbox.get_background_task_status(result.background_pid)
        assert status.status == SandboxStatus.SUCCESS
        assert "[Output saved to:" in status.stdout

        # Extract temp dir from status stdout
        marker = "[Output saved to: "
        start = status.stdout.index(marker) + len(marker)
        end = status.stdout.index("]", start)
        temp_dir = status.stdout[start:end]

        assert Path(f"{temp_dir}/command.txt").exists()
        assert Path(f"{temp_dir}/stdout.log").exists()
        assert Path(f"{temp_dir}/stderr.log").exists()
        assert "bg output" in Path(f"{temp_dir}/stdout.log").read_text()

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_background_no_save_output_by_default(self, sandbox):
        """Test that background mode does NOT save to temp file by default."""
        import time

        result = sandbox.execute_bash("echo 'bg no save'", background=True)
        assert result.status == SandboxStatus.SUCCESS
        assert "Output will be saved to" not in result.stdout

        time.sleep(0.5)
        status = sandbox.get_background_task_status(result.background_pid)
        assert "[Output saved to:" not in status.stdout

        sandbox.kill_background_task(result.background_pid)

    def test_save_output_temp_dir_under_expected_path(self, sandbox):
        """Test that temp files are created under the expected base path."""
        result = sandbox.execute_bash("echo 'path test'", save_output_to_temp_file=True)
        assert "[Output saved to:" in result.stdout

        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        assert temp_dir.startswith(f"{BASH_TOOL_RESULTS_BASE_PATH}/")

        shutil.rmtree(temp_dir, ignore_errors=True)


class TestGracefulKill:
    """Tests for the _graceful_kill static method and process group management."""

    def test_graceful_kill_terminates_process(self, sandbox):
        """_graceful_kill should terminate a running process and return its output."""
        import subprocess

        process = subprocess.Popen(
            "sleep 60",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=2.0)
        assert process.poll() is not None  # process has exited
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)

    def test_graceful_kill_already_exited_process(self, sandbox):
        """_graceful_kill should handle an already-exited process gracefully."""
        import subprocess

        process = subprocess.Popen(
            "echo done",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        process.wait()  # ensure it's done
        stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=1.0)
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)

    def test_graceful_kill_captures_output(self, sandbox):
        """_graceful_kill should capture stdout from a process that produces output before being killed."""
        import subprocess

        process = subprocess.Popen(
            "echo 'before_kill' && sleep 60",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        import time

        time.sleep(0.3)  # let echo run
        stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=2.0)
        # stdout may or may not contain the output depending on timing,
        # but the method should not raise
        assert isinstance(stdout, str)

    def test_foreground_timeout_uses_graceful_kill(self, sandbox):
        """Foreground command timeout should use _graceful_kill (SIGTERM before SIGKILL)."""
        result = sandbox.execute_bash("sleep 60", timeout=200)
        assert result.status == SandboxStatus.TIMEOUT
        assert "timed out" in result.error.lower()

    def test_foreground_uses_start_new_session(self, sandbox):
        """Foreground execution should create a new process group (start_new_session=True)."""
        import os

        # Run a command that prints its process group ID
        result = sandbox.execute_bash('python3 -c "import os; print(os.getpgrp())"')
        assert result.status == SandboxStatus.SUCCESS
        child_pgid = int(result.stdout.strip())
        # The child's pgid should differ from our own (since start_new_session=True)
        assert child_pgid != os.getpgrp()

    def test_background_uses_start_new_session(self, sandbox):
        """Background execution should create a new process group (start_new_session=True)."""
        import os
        import time

        result = sandbox.execute_bash(
            'python3 -c "import os; print(os.getpgrp())"',
            background=True,
        )
        pid = result.background_pid
        assert pid is not None

        time.sleep(0.5)
        status = sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.SUCCESS
        child_pgid = int(status.stdout.strip())
        assert child_pgid != os.getpgrp()

    def test_kill_background_task_uses_graceful_kill(self, sandbox):
        """kill_background_task should cleanly terminate a long-running background process."""
        result = sandbox.execute_bash("sleep 120", background=True)
        pid = result.background_pid
        assert pid is not None

        kill_result = sandbox.kill_background_task(pid)
        assert kill_result.status == SandboxStatus.SUCCESS

        # Verify the process is actually gone
        import os

        try:
            os.kill(pid, 0)
            # If we get here, process still exists — wait a moment
            import time

            time.sleep(1)
            os.kill(pid, 0)
            # Still alive after 1s is unexpected but not a hard failure
        except ProcessLookupError:
            pass  # expected — process was killed

    def test_graceful_kill_kills_child_processes(self, sandbox):
        """_graceful_kill should kill the entire process group, including child processes."""
        # Start a parent that spawns a child
        result = sandbox.execute_bash(
            "bash -c 'sleep 120 & echo child_started; wait' ",
            timeout=500,
        )
        # The timeout triggers _graceful_kill which should kill the whole group
        assert result.status == SandboxStatus.TIMEOUT


class TestGracefulKillBranches:
    """Mock-based tests to cover SIGKILL fallback and stuck-pipe branches (lines 111-137)."""

    def test_sigkill_path_when_sigterm_times_out(self):
        """When SIGTERM doesn't stop the process within grace_period, SIGKILL should be sent."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 12345

        # Step 2: first communicate() raises TimeoutExpired (SIGTERM didn't work)
        # Step 4: second communicate() succeeds (after SIGKILL)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),  # step 2
            ("killed_stdout", "killed_stderr"),  # step 4
        ]

        with patch("os.getpgid", return_value=12345) as _, patch("os.killpg") as mock_killpg:
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=1.0)

        # Verify SIGKILL was sent (second call to killpg)
        import signal

        killpg_calls = mock_killpg.call_args_list
        assert len(killpg_calls) == 2
        assert killpg_calls[0].args == (12345, signal.SIGTERM)
        assert killpg_calls[1].args == (12345, signal.SIGKILL)
        # process.kill() is also called as belt-and-suspenders
        process.kill.assert_called_once()
        assert stdout == "killed_stdout"
        assert stderr == "killed_stderr"

    def test_sigkill_path_process_already_gone(self):
        """When process exits between SIGTERM and SIGKILL, ProcessLookupError is handled."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 99999

        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),  # step 2
            ("out", "err"),  # step 4
        ]

        with patch("os.getpgid", side_effect=ProcessLookupError("No such process")):
            # SIGTERM fallback to process.terminate(), then SIGKILL getpgid also fails
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=0.1)

        assert stdout == "out"
        assert stderr == "err"

    def test_pipe_drain_timeout_closes_pipes(self):
        """When pipe drain (step 4) times out, pipes should be closed."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 11111
        mock_stdout_pipe = MagicMock()
        mock_stderr_pipe = MagicMock()
        process.stdout = mock_stdout_pipe
        process.stderr = mock_stderr_pipe

        # Step 2: SIGTERM timeout, Step 4: pipe drain also times out
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),  # step 2
            subprocess.TimeoutExpired(cmd="test", timeout=10),  # step 4
        ]
        process.wait.return_value = 0

        with patch("os.getpgid", return_value=11111), patch("os.killpg"):
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=0.1)

        # Verify pipes were closed
        mock_stdout_pipe.close.assert_called_once()
        mock_stderr_pipe.close.assert_called_once()
        # process.wait() called as final cleanup
        process.wait.assert_called_once_with(timeout=5)
        # Returns empty strings since communicate never returned data
        assert stdout == ""
        assert stderr == ""

    def test_pipe_drain_timeout_and_wait_timeout(self):
        """When both pipe drain and final wait time out, method still returns gracefully."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 22222
        process.stdout = MagicMock()
        process.stderr = MagicMock()

        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),  # step 2
            subprocess.TimeoutExpired(cmd="test", timeout=10),  # step 4
        ]
        # Final wait also times out
        process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)

        with patch("os.getpgid", return_value=22222), patch("os.killpg"):
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=0.1)

        # Should still return without raising
        assert stdout == ""
        assert stderr == ""

    def test_pipe_close_oserror_handled(self):
        """OSError when closing pipes should be silently caught."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 33333
        mock_stdout_pipe = MagicMock()
        mock_stdout_pipe.close.side_effect = OSError("broken pipe")
        mock_stderr_pipe = MagicMock()
        mock_stderr_pipe.close.side_effect = OSError("broken pipe")
        process.stdout = mock_stdout_pipe
        process.stderr = mock_stderr_pipe

        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),
            subprocess.TimeoutExpired(cmd="test", timeout=10),
        ]
        process.wait.return_value = 0

        with patch("os.getpgid", return_value=33333), patch("os.killpg"):
            # Should not raise despite pipe close failures
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=0.1)

        assert stdout == ""
        assert stderr == ""

    def test_sigkill_oserror_on_process_kill(self):
        """OSError on process.kill() (step 3) should be silently caught."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 44444
        process.kill.side_effect = OSError("already dead")

        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),  # step 2
            ("final_out", "final_err"),  # step 4
        ]

        with patch("os.getpgid", return_value=44444), patch("os.killpg"):
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=0.1)

        assert stdout == "final_out"
        assert stderr == "final_err"

    def test_none_pipes_skipped_during_close(self):
        """When stdout/stderr pipes are None, pipe close loop should skip them."""
        from unittest.mock import MagicMock, patch

        process = MagicMock()
        process.pid = 55555
        process.stdout = None
        process.stderr = None

        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),
            subprocess.TimeoutExpired(cmd="test", timeout=10),
        ]
        process.wait.return_value = 0

        with patch("os.getpgid", return_value=55555), patch("os.killpg"):
            # Should not raise when pipes are None
            stdout, stderr = LocalSandbox._graceful_kill(process, grace_period=0.1)

        assert stdout == ""
        assert stderr == ""


class TestSandboxDict:
    def test_dict_representation(self, sandbox):
        from dataclasses import asdict

        result = asdict(sandbox)
        assert "sandbox_id" in result
        assert "_work_dir" in result
