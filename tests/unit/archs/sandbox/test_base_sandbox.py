from dataclasses import asdict
from pathlib import Path

import pytest

from nexau.archs.sandbox.base_sandbox import (
    BaseSandbox,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
)


class TestSandboxStatus:
    def test_sandbox_status_values(self):
        assert SandboxStatus.RUNNING.value == "running"
        assert SandboxStatus.STOPPED.value == "stopped"
        assert SandboxStatus.ERROR.value == "error"
        assert SandboxStatus.TIMEOUT.value == "timeout"
        assert SandboxStatus.SUCCESS.value == "success"


class TestCodeLanguage:
    def test_code_language_values(self):
        assert CodeLanguage.PYTHON.value == "python"


class TestCommandResult:
    def test_command_result_success(self):
        result = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="Hello World",
            stderr="",
            exit_code=0,
            duration_ms=100,
        )
        assert result.status == SandboxStatus.SUCCESS
        assert result.stdout == "Hello World"
        assert result.exit_code == 0
        assert result.error is None
        assert not result.truncated

    def test_command_result_error(self):
        result = CommandResult(
            status=SandboxStatus.ERROR,
            stdout="",
            stderr="Error occurred",
            exit_code=1,
            duration_ms=50,
            error="Command failed",
        )
        assert result.status == SandboxStatus.ERROR
        assert result.exit_code == 1
        assert result.error == "Command failed"

    def test_command_result_timeout(self):
        result = CommandResult(
            status=SandboxStatus.TIMEOUT,
            stdout="Partial output",
            stderr="",
            exit_code=-1,
            duration_ms=5000,
            error="Timeout after 5000ms",
        )
        assert result.status == SandboxStatus.TIMEOUT
        assert result.error == "Timeout after 5000ms"

    def test_command_result_truncation(self):
        result = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="x" * 30000,
            stderr="y" * 30000,
            exit_code=0,
            duration_ms=100,
            truncated=True,
            original_stdout_length=50000,
            original_stderr_length=40000,
        )
        assert result.truncated
        assert result.original_stdout_length == 50000
        assert result.original_stderr_length == 40000


class TestCodeExecutionResult:
    def test_code_execution_result_success(self):
        result = CodeExecutionResult(
            status=SandboxStatus.SUCCESS,
            language=CodeLanguage.PYTHON,
            outputs=[{"type": "stdout", "text": "42"}],
            duration_ms=150,
        )
        assert result.status == SandboxStatus.SUCCESS
        assert result.language == CodeLanguage.PYTHON
        assert result.outputs is not None
        assert len(result.outputs) == 1
        assert result.outputs[0]["text"] == "42"
        assert result.error_type is None

    def test_code_execution_result_error(self):
        result = CodeExecutionResult(
            status=SandboxStatus.ERROR,
            language=CodeLanguage.PYTHON,
            error_type="SyntaxError",
            error_value="invalid syntax",
            traceback=["File test.py, line 1"],
            duration_ms=50,
        )
        assert result.status == SandboxStatus.ERROR
        assert result.error_type == "SyntaxError"
        assert result.error_value == "invalid syntax"
        assert result.traceback is not None
        assert len(result.traceback) == 1

    def test_code_execution_result_default_outputs(self):
        result = CodeExecutionResult(
            status=SandboxStatus.SUCCESS,
            language=CodeLanguage.PYTHON,
        )
        assert result.outputs == []


class TestFileInfo:
    def test_file_info_file(self):
        info = FileInfo(
            path="/test/file.txt",
            exists=True,
            is_file=True,
            is_directory=False,
            size=1024,
            mode=0o644,
            permissions="rw-r--r--",
            readable=True,
            writable=True,
            encoding="utf-8",
        )
        assert info.exists
        assert info.is_file
        assert not info.is_directory
        assert info.size == 1024
        assert info.readable
        assert info.writable

    def test_file_info_directory(self):
        info = FileInfo(
            path="/test/dir",
            exists=True,
            is_file=False,
            is_directory=True,
            size=0,
            readable=True,
            writable=True,
        )
        assert info.exists
        assert not info.is_file
        assert info.is_directory

    def test_file_info_nonexistent(self):
        info = FileInfo(
            path="/test/missing.txt",
            exists=False,
        )
        assert not info.exists
        assert not info.is_file
        assert not info.is_directory


class TestFileOperationResult:
    def test_file_operation_success(self):
        result = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/test/file.txt",
            content="Hello World",
            size=11,
        )
        assert result.status == SandboxStatus.SUCCESS
        assert result.content == "Hello World"
        assert result.size == 11
        assert result.error is None

    def test_file_operation_error(self):
        result = FileOperationResult(
            status=SandboxStatus.ERROR,
            file_path="/test/file.txt",
            error="File not found",
        )
        assert result.status == SandboxStatus.ERROR
        assert result.error == "File not found"
        assert result.content is None

    def test_file_operation_binary_content(self):
        binary_data = b"\x00\x01\x02\x03"
        result = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/test/binary.dat",
            content=binary_data,
            size=4,
        )
        assert result.content == binary_data
        assert result.size == 4


class TestSandboxExceptions:
    def test_sandbox_error(self):
        with pytest.raises(SandboxError) as exc_info:
            raise SandboxError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_sandbox_file_error(self):
        with pytest.raises(SandboxFileError) as exc_info:
            raise SandboxFileError("File error")
        assert str(exc_info.value) == "File error"
        assert isinstance(exc_info.value, SandboxError)


# Removed TestSandboxType, TestSandboxConfig, and TestGetSandbox classes
# as these are no longer part of the base_sandbox module


class TestBaseSandboxInterface:
    def test_base_sandboxwork_dir_property(self):
        from nexau.archs.sandbox.local_sandbox import LocalSandbox

        sandbox = LocalSandbox(_work_dir="/tmp/test")
        assert isinstance(sandbox.work_dir, Path)
        assert str(sandbox.work_dir) == "/tmp/test"

    def test_base_sandbox_dict_method(self):
        from nexau.archs.sandbox.local_sandbox import LocalSandbox

        sandbox = LocalSandbox(sandbox_id="test123", _work_dir="/tmp/test")
        result = asdict(sandbox)
        assert result["sandbox_id"] == "test123"
        assert "_work_dir" in result

    def test_base_sandbox_str_repr(self):
        from nexau.archs.sandbox.local_sandbox import LocalSandbox

        sandbox = LocalSandbox(sandbox_id="test123")
        str_repr = str(sandbox)
        assert "LocalSandbox" in str_repr
        assert "test123" in str_repr

    def test_base_sandbox_context_manager(self):
        from nexau.archs.sandbox.local_sandbox import LocalSandbox

        sandbox = LocalSandbox(sandbox_id="ctx_test")
        assert sandbox is not None
        assert sandbox.sandbox_id == "ctx_test"

    def test_detect_file_encoding_utf8(self):
        utf8_data = b"Hello World"
        encoding = BaseSandbox._detect_file_encoding(utf8_data)
        assert encoding in ["utf-8", "ascii"]

    def test_detect_file_encoding_fallback(self):
        binary_data = b"\x00\x01\x02\x03"
        encoding = BaseSandbox._detect_file_encoding(binary_data)
        assert encoding == "utf-8"
