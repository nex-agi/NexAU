import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from nexau.archs.sandbox.base_sandbox import (
    BASH_TOOL_RESULTS_BASE_PATH,
    CodeLanguage,
    E2BSandboxConfig,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
)
from nexau.archs.sandbox.e2b_sandbox import E2BSandbox, E2BSandboxManager
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine

pytestmark = pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set, skipping E2B tests")

_FORCE_HTTP_MANAGER_CASES = [
    pytest.param(False, id="force_http_off"),
    pytest.param(
        True,
        id="force_http_on",
        marks=pytest.mark.skipif(
            not (os.getenv("E2B_API_URL") and os.getenv("E2B_DOMAIN")),
            reason="force_http e2e path requires self-host E2B_API_URL and E2B_DOMAIN",
        ),
    ),
]


def _create_sandbox() -> tuple[E2BSandboxManager, E2BSandbox]:
    """Create a fresh E2B sandbox with retry on transient failures."""
    max_attempts = 3
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        manager = E2BSandboxManager()
        try:
            sandbox = manager.start(
                session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
                user_id="test_user",
                session_id=f"test_session_{attempt}",
                sandbox_config=E2BSandboxConfig(),
            )
            # Smoke-test: retry the healthcheck a few times within the same sandbox
            # because self-host envd may need a moment to stabilize after creation.
            for _hc in range(5):
                result = sandbox.execute_bash("echo __healthcheck__", timeout=15000)
                if result.status == SandboxStatus.SUCCESS and "__healthcheck__" in result.stdout:
                    return manager, sandbox
                time.sleep(2)
            last_exc = RuntimeError(f"Healthcheck failed after retries: {result.error or result.stderr}")
        except Exception as exc:
            last_exc = exc
        try:
            manager.stop()
        except Exception:
            pass
        if attempt < max_attempts:
            wait = attempt * 3
            print(f"[e2b_sandbox fixture] attempt {attempt}/{max_attempts} failed ({last_exc}), retrying in {wait}s …")
            time.sleep(wait)
    raise last_exc or RuntimeError("Failed to create sandbox")


@pytest.fixture(scope="class")
def e2b_sandbox():
    """Create a real E2B sandbox shared across tests in a class.

    Using class scope so each test class gets its own sandbox. This avoids
    sandbox timeout on self-hosted E2B (default 300s) while still reducing
    creation overhead vs per-test scope.  The work directory is cleaned
    between tests via the autouse ``_clean_workdir`` fixture below.

    Includes retry logic: if the sandbox fails the initial healthcheck
    (common on self-hosted E2B due to transient connection resets), it
    tears down and retries up to 3 times.
    """
    manager, sandbox = _create_sandbox()
    yield sandbox
    manager.stop()


@pytest.fixture(autouse=True)
def _clean_workdir(request):
    """Reset the sandbox work directory before each test to avoid cross-test pollution.

    Also adds a small delay between tests to reduce request density on
    self-hosted E2B instances and avoid connection-reset errors.

    Only runs for tests that actually use the shared e2b_sandbox fixture (skipped for
    Manager tests that manage their own sandbox lifecycle).
    """
    if "e2b_sandbox" not in request.fixturenames:
        yield
        return
    sandbox = request.getfixturevalue("e2b_sandbox")
    # Retry cleanup — transient disconnects should not poison the whole class
    for attempt in range(3):
        try:
            sandbox.execute_bash("rm -rf /home/user/* /home/user/.* 2>/dev/null || true", timeout=15000)
            break
        except Exception:
            if attempt < 2:
                time.sleep(1)
    yield
    # Delay between tests to let self-host envd recover
    time.sleep(0.5)


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

    def test_glob_absolute_path_with_doublestar(self, e2b_sandbox):
        """Absolute path + ** should search the correct directory."""
        e2b_sandbox.write_file("absglob/foo.py", "")
        e2b_sandbox.write_file("absglob/sub/bar.py", "")
        work_dir = str(e2b_sandbox.work_dir)
        matches = e2b_sandbox.glob(f"{work_dir}/absglob/**/*.py")
        assert len(matches) >= 2
        assert any("foo.py" in m for m in matches)
        assert any("bar.py" in m for m in matches)

    def test_glob_absolute_path_no_doublestar(self, e2b_sandbox):
        """Absolute path without ** should find files in that directory."""
        e2b_sandbox.write_file("absdir/hello.md", "")
        work_dir = str(e2b_sandbox.work_dir)
        matches = e2b_sandbox.glob(f"{work_dir}/absdir/*.md")
        assert any("hello.md" in m for m in matches)

    def test_glob_bare_doublestar(self, e2b_sandbox):
        """** alone should find all files recursively."""
        e2b_sandbox.write_file("bareglob_marker.txt", "")
        matches = e2b_sandbox.glob("**")
        assert len(matches) >= 1
        assert any("bareglob_marker.txt" in m for m in matches)

    def test_glob_doublestar_trailing_no_slash(self, e2b_sandbox):
        """/path/** (no trailing slash) should find all files under path."""
        e2b_sandbox.write_file("tdir/a.txt", "")
        e2b_sandbox.write_file("tdir/sub/b.txt", "")
        work_dir = str(e2b_sandbox.work_dir)
        matches = e2b_sandbox.glob(f"{work_dir}/tdir/**")
        assert len(matches) >= 2

    def test_glob_relative_doublestar(self, e2b_sandbox):
        """src/**/*.ts style relative pattern."""
        e2b_sandbox.write_file("srcg/index.ts", "")
        e2b_sandbox.write_file("srcg/lib/util.ts", "")
        matches = e2b_sandbox.glob("srcg/**/*.ts")
        assert len(matches) >= 2
        assert any("index.ts" in m for m in matches)
        assert any("util.ts" in m for m in matches)

    def test_glob_no_matches_returns_empty(self, e2b_sandbox):
        """Pattern that matches nothing should return empty list, not raise."""
        matches = e2b_sandbox.glob("**/*.nonexistent_extension_xyz")
        assert matches == []

    def test_glob_results_are_sorted(self, e2b_sandbox):
        """Results should be returned in sorted order."""
        e2b_sandbox.write_file("sortglob/c.py", "")
        e2b_sandbox.write_file("sortglob/a.py", "")
        e2b_sandbox.write_file("sortglob/b.py", "")
        matches = e2b_sandbox.glob("sortglob/*.py")
        filenames = [m.split("/")[-1] for m in matches]
        assert filenames == sorted(filenames)

    def test_glob_wildcard_in_dir_finds_directories(self, e2b_sandbox):
        """Test that a wildcard in the directory portion of a pattern finds matching directories."""
        e2b_sandbox.write_file("testlibs/python3.12/site-packages/.keep", "")
        e2b_sandbox.write_file("testlibs/python3.11/site-packages/.keep", "")

        # Trailing slash → directory search via `find -type d -path`
        matches = e2b_sandbox.glob("testlibs/python*/site-packages/", recursive=True)
        assert len(matches) >= 2
        assert any("python3.12" in m for m in matches)
        assert any("python3.11" in m for m in matches)
        assert all("site-packages" in m for m in matches)

    def test_glob_wildcard_in_dir_finds_files(self, e2b_sandbox):
        """Test that a wildcard in the directory portion of a pattern finds matching files."""
        e2b_sandbox.write_file("testlibs/python3.12/site-packages/requests.py", "")
        e2b_sandbox.write_file("testlibs/python3.11/site-packages/flask.py", "")

        # No trailing slash, file pattern at end → `find -path`
        matches = e2b_sandbox.glob("testlibs/python*/site-packages/*.py", recursive=True)
        assert len(matches) >= 2
        assert any("requests.py" in m for m in matches)
        assert any("flask.py" in m for m in matches)

    def test_glob_absolute_path_wildcard_in_dir(self, e2b_sandbox):
        """Test absolute pattern with wildcard in directory component.

        Previously this raised 'directory does not exist' because bash tried to expand the glob.
        Now find handles the wildcard itself.
        """
        work_dir = str(e2b_sandbox.work_dir)
        e2b_sandbox.write_file("abslib/python3.12/site-packages/.keep", "")
        e2b_sandbox.write_file("abslib/python3.11/site-packages/.keep", "")

        matches = e2b_sandbox.glob(f"{work_dir}/abslib/python*/site-packages/", recursive=True)
        assert isinstance(matches, list)
        assert len(matches) >= 2
        assert all("site-packages" in m for m in matches)

    def test_glob_double_slash_normalized(self, e2b_sandbox):
        """Test that double slashes within an absolute pattern are normalized before processing.

        /home/user//testpkg/requests/models.py should be treated as
        /home/user/testpkg/requests/models.py and find the file.
        """
        e2b_sandbox.write_file("testpkg/requests/models.py", "")

        work_dir = str(e2b_sandbox.work_dir)  # e.g. /home/user

        # Absolute path with // injected in the middle
        double_slash_pattern = f"{work_dir}//testpkg/requests/models.py"
        matches = e2b_sandbox.glob(double_slash_pattern, recursive=True)

        assert len(matches) >= 1
        assert any("models.py" in m for m in matches)

    def test_glob_double_slash_no_permission_error(self, e2b_sandbox):
        """Test that //pattern does not raise due to permission-denied errors.

        Before the fix, //astropy*/units/core.py would compute root_dir='/'
        and run `find /`, hitting /proc /sys etc. and failing with exit code 1.
        """
        # This should complete without raising SandboxFileError, returning an empty list
        # (no /astropy* directory exists in a stock sandbox).
        from nexau.archs.sandbox.base_sandbox import SandboxFileError

        try:
            matches = e2b_sandbox.glob("//astropy*/units/core.py", recursive=True)
            assert isinstance(matches, list)
        except SandboxFileError:
            raise AssertionError("//pattern raised SandboxFileError; double-slash normalization or 2>/dev/null fix is missing")


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


class TestE2BSaveOutputToTempFile:
    def test_foreground_save_output_creates_files(self, e2b_sandbox):
        """Test that save_output_to_temp_file creates command.txt, stdout.log, stderr.log in the sandbox."""
        result = e2b_sandbox.execute_bash("echo 'hello save'", save_output_to_temp_file=True)
        assert result.status == SandboxStatus.SUCCESS
        assert "[Output saved to:" in result.stdout

        # Extract temp dir path from stdout
        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        # Verify files exist in the sandbox
        assert e2b_sandbox.file_exists(f"{temp_dir}/command.txt")
        assert e2b_sandbox.file_exists(f"{temp_dir}/stdout.log")
        assert e2b_sandbox.file_exists(f"{temp_dir}/stderr.log")

        # Verify content
        cmd_content = e2b_sandbox.read_file(f"{temp_dir}/command.txt")
        assert cmd_content.status == SandboxStatus.SUCCESS
        assert cmd_content.content == "echo 'hello save'"

        stdout_content = e2b_sandbox.read_file(f"{temp_dir}/stdout.log")
        assert stdout_content.status == SandboxStatus.SUCCESS
        assert "hello save" in stdout_content.content

    def test_foreground_save_output_with_stderr(self, e2b_sandbox):
        """Test that stderr is also saved when save_output_to_temp_file is enabled."""
        result = e2b_sandbox.execute_bash("echo 'out' && echo 'err' >&2", save_output_to_temp_file=True)
        assert "[Output saved to:" in result.stdout

        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        stdout_content = e2b_sandbox.read_file(f"{temp_dir}/stdout.log")
        assert "out" in stdout_content.content

        stderr_content = e2b_sandbox.read_file(f"{temp_dir}/stderr.log")
        assert "err" in stderr_content.content

    def test_foreground_no_save_output_by_default(self, e2b_sandbox):
        """Test that output is NOT saved to temp file when save_output_to_temp_file is False (default)."""
        result = e2b_sandbox.execute_bash("echo 'no save'")
        assert result.status == SandboxStatus.SUCCESS
        assert "[Output saved to:" not in result.stdout

    def test_foreground_save_output_with_failed_command(self, e2b_sandbox):
        """Test save_output_to_temp_file works even when the command fails."""
        result = e2b_sandbox.execute_bash("echo 'before fail' && exit 1", save_output_to_temp_file=True)
        assert result.status == SandboxStatus.ERROR
        assert "[Output saved to:" in result.stdout

        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        stdout_content = e2b_sandbox.read_file(f"{temp_dir}/stdout.log")
        assert stdout_content.status == SandboxStatus.SUCCESS
        assert "before fail" in stdout_content.content

    def test_background_save_output_creates_files(self, e2b_sandbox):
        """Test that save_output_to_temp_file works in background mode."""
        result = e2b_sandbox.execute_bash("echo 'bg output'", background=True, save_output_to_temp_file=True)
        assert result.status == SandboxStatus.SUCCESS
        assert result.background_pid is not None
        assert "Output will be saved to" in result.stdout

        # Poll until background task finishes (self-host can be slow)
        for _ in range(15):
            time.sleep(2)
            status = e2b_sandbox.get_background_task_status(result.background_pid)
            if status.status != SandboxStatus.RUNNING:
                break

        assert status.status == SandboxStatus.SUCCESS
        assert "[Output saved to:" in status.stdout

        # Extract temp dir from status stdout
        marker = "[Output saved to: "
        start = status.stdout.index(marker) + len(marker)
        end = status.stdout.index("]", start)
        temp_dir = status.stdout[start:end]

        # Verify files exist in the sandbox
        assert e2b_sandbox.file_exists(f"{temp_dir}/command.txt")
        assert e2b_sandbox.file_exists(f"{temp_dir}/stdout.log")
        assert e2b_sandbox.file_exists(f"{temp_dir}/stderr.log")

        stdout_content = e2b_sandbox.read_file(f"{temp_dir}/stdout.log")
        assert "bg output" in stdout_content.content

    def test_background_no_save_output_by_default(self, e2b_sandbox):
        """Test that background mode does NOT save to temp file by default."""
        result = e2b_sandbox.execute_bash("echo 'bg no save'", background=True)
        assert result.status == SandboxStatus.SUCCESS
        assert "Output will be saved to" not in result.stdout

        time.sleep(2)
        status = e2b_sandbox.get_background_task_status(result.background_pid)
        assert "[Output saved to:" not in status.stdout

        e2b_sandbox.kill_background_task(result.background_pid)

    def test_save_output_temp_dir_under_expected_path(self, e2b_sandbox):
        """Test that temp files are created under the expected base path in the sandbox."""
        result = e2b_sandbox.execute_bash("echo 'path test'", save_output_to_temp_file=True)
        assert "[Output saved to:" in result.stdout

        marker = "[Output saved to: "
        start = result.stdout.index(marker) + len(marker)
        end = result.stdout.index("]", start)
        temp_dir = result.stdout[start:end]

        assert temp_dir.startswith(f"{BASH_TOOL_RESULTS_BASE_PATH}/")


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

    def test_read_binary_file_as_text_returns_error(self, e2b_sandbox):
        """Reading a binary file without binary=True should return error (UnicodeDecodeError)."""
        e2b_sandbox.write_file("binary.dat", b"\x80\x81\x82\xff\xfe", binary=True)
        result = e2b_sandbox.read_file("binary.dat")
        assert result.status == SandboxStatus.ERROR

    def test_read_file_truncates_large_content(self, e2b_sandbox):
        """Large file content should be truncated at 30000 chars."""
        big_content = "x" * 31000
        e2b_sandbox.write_file("big.txt", big_content)
        result = e2b_sandbox.read_file("big.txt")
        assert result.status == SandboxStatus.SUCCESS
        assert result.truncated is True
        assert len(result.content) == 30000

    def test_list_files_with_pattern(self, e2b_sandbox):
        """list_files with pattern should filter results."""
        e2b_sandbox.write_file("a.py", "")
        e2b_sandbox.write_file("b.txt", "")
        e2b_sandbox.write_file("c.py", "")
        files = e2b_sandbox.list_files(".", pattern="*.py")
        names = [Path(f.path).name for f in files]
        assert "a.py" in names
        assert "c.py" in names
        assert "b.txt" not in names

    def test_get_file_info_directory(self, e2b_sandbox):
        """get_file_info on a directory should have encoding=None."""
        e2b_sandbox.create_directory("info_dir")
        info = e2b_sandbox.get_file_info("info_dir")
        assert info.exists
        assert info.is_directory
        assert not info.is_file
        assert info.encoding is None

    def test_execute_bash_invalid_user(self, e2b_sandbox):
        """Invalid user parameter should raise ValueError."""
        with pytest.raises(ValueError, match="root.*user"):
            e2b_sandbox.execute_bash("echo hi", user="nobody")

    def test_execute_code_unsupported_language_enum(self, e2b_sandbox):
        """Passing a CodeLanguage enum that is not PYTHON should return error."""
        # CodeLanguage only has PYTHON, so use string that parses to unknown
        result = e2b_sandbox.execute_code("code", "ruby")
        assert result.status == SandboxStatus.ERROR
        assert "Unsupported" in (result.error_value or "")

    def test_edit_file_create_on_existing_returns_error(self, e2b_sandbox):
        """CREATE (old_string='') on existing file should fail."""
        e2b_sandbox.write_file("exists.txt", "content")
        result = e2b_sandbox.edit_file("exists.txt", "", "new content")
        assert result.status == SandboxStatus.ERROR
        assert "already exists" in result.error.lower()

    def test_edit_file_update_on_nonexistent_returns_error(self, e2b_sandbox):
        """UPDATE on non-existent file should fail."""
        result = e2b_sandbox.edit_file("missing.txt", "old", "new")
        assert result.status == SandboxStatus.ERROR
        assert "does not exist" in result.error.lower()

    def test_edit_file_normalizes_escaped_newlines(self, e2b_sandbox):
        """edit_file should normalize \\n escape sequences from LLM output."""
        e2b_sandbox.write_file("escape.txt", "line1\nline2")
        result = e2b_sandbox.edit_file("escape.txt", "line1\\nline2", "replaced")
        assert result.status == SandboxStatus.SUCCESS
        read = e2b_sandbox.read_file("escape.txt")
        assert read.content == "replaced"

    def test_glob_with_directory_pattern(self, e2b_sandbox):
        """glob with dir/pattern should work."""
        e2b_sandbox.write_file("gdir/a.py", "")
        e2b_sandbox.write_file("gdir/b.txt", "")
        matches = e2b_sandbox.glob("gdir/*.py")
        assert any("a.py" in m for m in matches)
        assert not any("b.txt" in m for m in matches)

    def test_glob_non_recursive(self, e2b_sandbox):
        """glob with recursive=False should use ls."""
        e2b_sandbox.write_file("nr_test.txt", "")
        # Non-recursive glob uses ls -1
        matches = e2b_sandbox.glob("*.txt", recursive=False)
        # May or may not find files depending on ls behavior, but should not raise
        assert isinstance(matches, list)

    def test_upload_empty_directory(self, e2b_sandbox, temp_dir):
        """Uploading an empty directory should succeed."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        assert e2b_sandbox.upload_directory(str(empty_dir), "empty_dest")

    def test_write_file_resolves_relative_path(self, e2b_sandbox):
        """write_file with relative path should resolve against work_dir."""
        e2b_sandbox.write_file("rel_test.txt", "content")
        # Verify via absolute path
        result = e2b_sandbox.read_file("/home/user/rel_test.txt")
        assert result.status == SandboxStatus.SUCCESS
        assert result.content == "content"

    def test_delete_file_resolves_relative_path(self, e2b_sandbox):
        """delete_file with relative path should resolve against work_dir."""
        e2b_sandbox.write_file("del_rel.txt", "content")
        result = e2b_sandbox.delete_file("del_rel.txt")
        assert result.status == SandboxStatus.SUCCESS
        assert not e2b_sandbox.file_exists("/home/user/del_rel.txt")


class TestE2BManagerStartPriorities:
    """E2e tests for E2BSandboxManager.start() Priority 1 (connect from config)
    and Priority 2 (restore from session state).

    The default e2b_sandbox fixture only exercises Priority 3 (create new).
    """

    @pytest.fixture(autouse=True)
    def _throttle(self):
        """Small delay between Manager tests to avoid overwhelming self-host."""
        yield
        time.sleep(1)

    @pytest.mark.parametrize("force_http", _FORCE_HTTP_MANAGER_CASES)
    def test_priority1_connect_from_config(self, force_http: bool):
        """Priority 1: pass sandbox_id in config to connect to an existing sandbox."""
        case = "http" if force_http else "sdk"
        session_mgr = SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance())
        connector: E2BSandboxManager | None = None

        creator = E2BSandboxManager()
        created = creator.start(
            session_manager=session_mgr,
            user_id=f"p1_user_{case}",
            session_id=f"p1_create_{case}",
            sandbox_config=E2BSandboxConfig(status_after_run="none", force_http=force_http),
        )
        sandbox_id = created.sandbox_id
        assert sandbox_id is not None

        try:
            connector = E2BSandboxManager()
            connected = connector.start(
                session_manager=session_mgr,
                user_id=f"p1_user_{case}",
                session_id=f"p1_connect_{case}",
                sandbox_config=E2BSandboxConfig(sandbox_id=sandbox_id, force_http=force_http),
            )

            assert connected.sandbox_id == sandbox_id
            assert connected.sandbox is not None
        finally:
            if connector is not None:
                connector.on_run_complete()
            creator.stop()

    @pytest.mark.parametrize("force_http", _FORCE_HTTP_MANAGER_CASES)
    def test_priority2_restore_from_session_state(self, force_http: bool):
        """Priority 2: restore sandbox from persisted session state."""
        case = "http" if force_http else "sdk"
        engine = InMemoryDatabaseEngine.get_shared_instance()
        session_mgr = SessionManager(engine=engine)
        user_id = f"p2_user_{case}"
        session_id = f"p2_session_{case}"
        restorer: E2BSandboxManager | None = None

        creator = E2BSandboxManager()
        created = creator.start(
            session_manager=session_mgr,
            user_id=user_id,
            session_id=session_id,
            sandbox_config=E2BSandboxConfig(status_after_run="none", force_http=force_http),
        )
        sandbox_id = created.sandbox_id
        assert sandbox_id is not None

        try:
            restorer = E2BSandboxManager()
            restored = restorer.start(
                session_manager=session_mgr,
                user_id=user_id,
                session_id=session_id,
                sandbox_config=E2BSandboxConfig(force_http=force_http),
            )

            assert restored.sandbox_id == sandbox_id
            assert restored.sandbox is not None
        finally:
            if restorer is not None:
                restorer.on_run_complete()
            creator.stop()


class TestE2BManagerLifecycle:
    """E2e tests for E2BSandboxManager stop/pause/is_running/on_run_complete."""

    @pytest.fixture(autouse=True)
    def _throttle(self):
        """Small delay between Manager tests to avoid overwhelming self-host."""
        yield
        time.sleep(1)

    def test_is_running(self):
        """is_running should return True for a live sandbox."""
        manager, _sandbox = _create_sandbox()
        try:
            assert manager.is_running() is True
        finally:
            manager.stop()

    def test_is_running_no_instance(self):
        """is_running should return False when no sandbox has been started."""
        manager = E2BSandboxManager()
        assert manager.is_running() is False

    def test_stop_returns_true(self):
        """stop() should return True and kill the sandbox."""
        manager, _sandbox = _create_sandbox()
        assert manager.stop() is True

    def test_stop_no_instance(self):
        """stop() with no sandbox should return False."""
        manager = E2BSandboxManager()
        assert manager.stop() is False

    def test_on_run_complete(self):
        """on_run_complete should stop keepalive without error."""
        manager, _sandbox = _create_sandbox()
        try:
            # Should not raise
            manager.on_run_complete()
            # Sandbox should still be running (on_run_complete only stops keepalive)
            assert manager.is_running() is True
        finally:
            manager.stop()

    @pytest.mark.parametrize("force_http", _FORCE_HTTP_MANAGER_CASES)
    def test_pause(self, force_http: bool):
        """pause() should return True for both default and force_http paths."""
        manager = E2BSandboxManager()
        manager.start(
            session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
            user_id="pause_user",
            session_id="pause_session",
            sandbox_config=E2BSandboxConfig(force_http=force_http),
        )
        assert manager.pause() is True

    def test_pause_no_instance(self):
        """pause() with no sandbox should return False."""
        manager = E2BSandboxManager()
        assert manager.pause() is False

    def test_start_with_config_overrides(self):
        """start() should apply api_key/api_url/template from config."""
        manager = E2BSandboxManager()
        config = E2BSandboxConfig(
            api_key=os.getenv("E2B_API_KEY"),
            api_url=os.getenv("E2B_API_URL"),
        )
        sandbox = manager.start(
            session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
            user_id="cfg_user",
            session_id="cfg_session",
            sandbox_config=config,
        )
        assert sandbox.sandbox is not None
        manager.stop()

    def test_start_wrong_config_type(self):
        """start() with non-E2BSandboxConfig should raise ValueError."""
        from nexau.archs.sandbox.base_sandbox import LocalSandboxConfig

        manager = E2BSandboxManager()
        with pytest.raises(ValueError, match="E2BSandboxConfig"):
            manager.start(
                session_manager=None,
                user_id="x",
                session_id="x",
                sandbox_config=LocalSandboxConfig(),
            )
