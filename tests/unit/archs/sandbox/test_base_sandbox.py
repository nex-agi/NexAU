from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from nexau.archs.sandbox.base_sandbox import (
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    LocalSandboxConfig,
    SandboxConfig,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
    extract_dataclass_init_kwargs,
)
from nexau.archs.session.session_manager import SessionManager


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

    def test_command_result_background_pid_default(self):
        result = CommandResult(status=SandboxStatus.SUCCESS)
        assert result.background_pid is None

    def test_command_result_background_pid(self):
        result = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="Background task started (pid: 12345)",
            exit_code=0,
            duration_ms=10,
            background_pid=12345,
        )
        assert result.background_pid == 12345


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


class TestExtractDataclassInitKwargs:
    def test_extract_kwargs_ignores_extra(self):
        from dataclasses import dataclass

        @dataclass
        class Example:
            name: str
            count: int

        result = extract_dataclass_init_kwargs(Example, {"name": "a", "count": 2, "extra": "ignore"})
        assert result == {"name": "a", "count": 2}

    def test_extract_kwargs_rejects_extra(self):
        from dataclasses import dataclass

        @dataclass
        class Example:
            name: str

        with pytest.raises(TypeError):
            extract_dataclass_init_kwargs(Example, {"name": "a", "extra": "bad"}, ignore_extra=False)

    def test_extract_kwargs_non_dataclass(self):
        with pytest.raises(TypeError):
            extract_dataclass_init_kwargs(dict, {"name": "a"})


# Removed TestSandboxType, TestSandboxConfig, and TestGetSandbox classes
# as these are no longer part of the base_sandbox module


class DummySandbox(BaseSandbox):
    uploaded: list[tuple[str, str]]

    def __init__(self, sandbox_id: str | None = None, _work_dir: str = "/tmp") -> None:
        super().__init__(sandbox_id=sandbox_id, _work_dir=_work_dir)
        self.uploaded = []

    def detect_encoding(self, data: bytes) -> str:
        return self._detect_file_encoding(data)

    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
        save_output_to_temp_file: bool = False,
    ) -> CommandResult:
        return CommandResult(status=SandboxStatus.SUCCESS)

    def get_background_task_status(self, pid: int) -> CommandResult:
        return CommandResult(status=SandboxStatus.ERROR, error=f"Not found: pid={pid}")

    def kill_background_task(self, pid: int) -> CommandResult:
        return CommandResult(status=SandboxStatus.ERROR, error=f"Not found: pid={pid}")

    def execute_code(
        self,
        code: str,
        language: CodeLanguage | str,
        timeout: int | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionResult:
        return CodeExecutionResult(status=SandboxStatus.SUCCESS, language=CodeLanguage.PYTHON)

    def read_file(self, file_path: str, encoding: str = "utf-8", binary: bool = False) -> FileOperationResult:
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=file_path, content="")

    def write_file(
        self,
        file_path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        binary: bool = False,
        create_directories: bool = True,
        user: str | None = None,
    ) -> FileOperationResult:
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=file_path)

    def delete_file(self, file_path: str) -> FileOperationResult:
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=file_path)

    def list_files(self, directory_path: str, recursive: bool = False, pattern: str | None = None) -> list[FileInfo]:
        return []

    def file_exists(self, file_path: str) -> bool:
        return False

    def get_file_info(self, file_path: str) -> FileInfo:
        return FileInfo(path=file_path, exists=False)

    def create_directory(self, directory_path: str, parents: bool = True, user: str | None = None) -> bool:
        return True

    def edit_file(self, file_path: str, old_string: str, new_string: str) -> FileOperationResult:
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=file_path)

    def glob(self, pattern: str, recursive: bool = True, user: str | None = None) -> list[str]:
        return []

    def upload_file(self, local_path: str, sandbox_path: str, create_directories: bool = True) -> FileOperationResult:
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=sandbox_path)

    def download_file(self, sandbox_path: str, local_path: str, create_directories: bool = True) -> FileOperationResult:
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=local_path)

    def upload_directory(self, local_path: str, sandbox_path: str) -> bool:
        self.uploaded.append((local_path, sandbox_path))
        return True

    def download_directory(self, sandbox_path: str, local_path: str) -> bool:
        return True


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
        encoding = DummySandbox().detect_encoding(utf8_data)
        assert encoding in ["utf-8", "ascii"]

    def test_detect_file_encoding_fallback(self):
        """Binary data should fall back to utf-8 or ascii (chardet may vary)."""
        binary_data = b"\x00\x01\x02\x03"
        encoding = DummySandbox().detect_encoding(binary_data)
        assert encoding in ["utf-8", "ascii"]

    def test_upload_skill_calls_create_directory_and_upload(self) -> None:
        """upload_skill should create .skills subdir and upload skill folder."""
        sandbox = DummySandbox(_work_dir="/tmp/sandbox")
        skill = type("Skill", (), {"folder": "/local/skill_name"})()
        result = sandbox.upload_skill(skill)
        assert result == "/tmp/sandbox/.skills/skill_name"
        assert sandbox.uploaded == [("/local/skill_name", "/tmp/sandbox/.skills/skill_name")]

    def test_dict_returns_init_fields_only(self) -> None:
        """dict() should return only dataclass init fields."""
        sandbox = DummySandbox(sandbox_id="sid", _work_dir="/tmp")
        d = sandbox.dict()
        assert "sandbox_id" in d
        assert "_work_dir" in d
        assert d["sandbox_id"] == "sid"
        assert d["_work_dir"] == "/tmp"

    def test_detect_file_encoding_chardet_high_confidence(self) -> None:
        """When chardet returns high confidence, use detected encoding."""
        data = "你好世界".encode()
        encoding = DummySandbox().detect_encoding(data)
        assert encoding in ["utf-8", "ascii", "UTF-8"]

    def test_detect_file_encoding_exception_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When chardet raises, fall back to utf-8."""

        def raise_err(_: bytes) -> dict:
            raise ValueError("detect failed")

        monkeypatch.setattr("chardet.detect", raise_err)
        encoding = DummySandbox().detect_encoding(b"hello")
        assert encoding == "utf-8"


class TestBaseSandboxManager:
    def test_instance_raises_without_context(self) -> None:
        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(
                self,
                session_manager: object | None,
                user_id: str,
                session_id: str,
                sandbox_config: SandboxConfig,
            ) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return False

        manager = DummyManager()
        with pytest.raises(SandboxError):
            _ = manager.instance

    def test_instance_starts_with_context(self) -> None:
        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(
                self,
                session_manager: object | None,
                user_id: str,
                session_id: str,
                sandbox_config: object,
            ) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return False

        manager = DummyManager()
        manager.prepare_session_context(
            session_manager=None,
            user_id="user",
            session_id="session",
            sandbox_config=LocalSandboxConfig(),
        )
        instance = manager.instance
        assert instance is not None

    def test_start_sync_without_context_returns_none(self) -> None:
        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(
                self,
                session_manager: object | None,
                user_id: str,
                session_id: str,
                sandbox_config: SandboxConfig,
            ) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        manager = DummyManager()
        assert manager.start_sync() is None

    def test_start_sync_with_context(self) -> None:
        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(
                self,
                session_manager: object | None,
                user_id: str,
                session_id: str,
                sandbox_config: object,
            ) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        manager = DummyManager()
        manager.prepare_session_context(
            session_manager=None,
            user_id="user",
            session_id="session",
            sandbox_config=LocalSandboxConfig(),
        )
        instance = manager.start_sync()
        assert instance is not None

    def test_persist_sandbox_state_event_loop_conflict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummySessionManager:
            async def update_session_sandbox(self, user_id: str, session_id: str, sandbox_state: dict[str, object]) -> None:
                return None

        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(
                self,
                session_manager: object | None,
                user_id: str,
                session_id: str,
                sandbox_config: SandboxConfig,
            ) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        def raise_conflict(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Event loop is closed")

        manager = DummyManager()
        monkeypatch.setattr("nexau.archs.sandbox.base_sandbox.run_async_function_sync", raise_conflict)
        result = manager.persist_sandbox_state(
            session_manager=cast(SessionManager, DummySessionManager()),
            user_id="user",
            session_id="session",
            sandbox=DummySandbox(),
        )
        assert result is None

    def test_load_sandbox_state_event_loop_conflict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummySessionManager:
            async def get_session(self, user_id: str, session_id: str) -> object | None:
                return None

        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(
                self,
                session_manager: object | None,
                user_id: str,
                session_id: str,
                sandbox_config: SandboxConfig,
            ) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        def raise_conflict(*args: object, **kwargs: object) -> None:
            raise RuntimeError("bound to a different event loop")

        manager = DummyManager()
        monkeypatch.setattr("nexau.archs.sandbox.base_sandbox.run_async_function_sync", raise_conflict)
        result = manager.load_sandbox_state(
            session_manager=cast(SessionManager, DummySessionManager()),
            user_id="user",
            session_id="session",
        )
        assert result is None

    def test_persist_sandbox_state_success(self) -> None:
        """persist_sandbox_state should call update_session_sandbox on success."""
        updated = []

        class SessionManagerWithUpdate:
            async def update_session_sandbox(self, user_id: str, session_id: str, sandbox_state: dict) -> None:
                updated.append((user_id, session_id, sandbox_state))

        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(self, *args: object, **kwargs: object) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        sm = cast(SessionManager, SessionManagerWithUpdate())
        manager = DummyManager()
        manager._instance = DummySandbox(sandbox_id="x", _work_dir="/tmp")
        manager.persist_sandbox_state(sm, "u", "s", manager._instance)
        assert len(updated) == 1
        assert updated[0][0] == "u"
        assert updated[0][1] == "s"
        assert updated[0][2]["sandbox_id"] == "x"
        assert updated[0][2]["sandbox_type"] == "DummySandbox"

    def test_load_sandbox_state_success(self) -> None:
        """load_sandbox_state should return sandbox_state when session has it."""
        state = {"sandbox_id": "loaded", "sandbox_type": "LocalSandbox"}

        class SessionManagerWithGet:
            async def get_session(self, user_id: str, session_id: str) -> object:
                return type("Session", (), {"sandbox_state": state})()

        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(self, *args: object, **kwargs: object) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        sm = cast(SessionManager, SessionManagerWithGet())
        manager = DummyManager()
        result = manager.load_sandbox_state(sm, "u", "s")
        assert result == state

    def test_start_sync_returns_cached_instance(self) -> None:
        """start_sync should return _instance when already set."""

        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(self, *args: object, **kwargs: object) -> DummySandbox:
                return DummySandbox(sandbox_id="started")

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        manager = DummyManager()
        existing = DummySandbox(sandbox_id="cached")
        manager._instance = existing
        result = manager.start_sync()
        assert result is existing
        assert result.sandbox_id == "cached"

    def test_pause_no_wait_starts_thread(self) -> None:
        """pause_no_wait should start a thread and return it."""

        class DummyManager(BaseSandboxManager["DummySandbox"]):
            def start(self, *args: object, **kwargs: object) -> DummySandbox:
                return DummySandbox()

            def stop(self) -> bool:
                return True

            def pause(self) -> bool:
                return True

            def is_running(self) -> bool:
                return True

        manager = DummyManager()
        thread = manager.pause_no_wait()
        assert thread is not None
        assert hasattr(thread, "start")
        thread.join(timeout=2)
