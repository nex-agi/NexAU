import shutil
import tempfile
from pathlib import Path

import pytest

from nexau.archs.sandbox.base_sandbox import (
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


class TestSandboxDict:
    def test_dict_representation(self, sandbox):
        from dataclasses import asdict

        result = asdict(sandbox)
        assert "sandbox_id" in result
        assert "_work_dir" in result
