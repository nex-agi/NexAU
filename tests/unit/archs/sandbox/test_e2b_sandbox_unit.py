"""Unit tests for E2BSandbox — pure logic only, no SDK interaction.

These tests cover deterministic, client-side logic that does not depend on
the E2B SDK or a running sandbox instance.  All SDK-dependent behavior is
tested via e2e tests in test_e2b_sandbox.py.
"""

from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from packaging.version import Version

from nexau.archs.sandbox import e2b_sandbox as e2b_module
from nexau.archs.sandbox.base_sandbox import (
    E2BSandboxConfig,
    FileInfo,
    FileOperationResult,
    SandboxFileError,
    SandboxStatus,
)
from nexau.archs.sandbox.e2b_sandbox import E2BSandbox

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeCommandResult:
    stdout: str | None
    stderr: str | None
    exit_code: int


class _FakeCommands:
    def __init__(self, behavior: Sequence[Exception | _FakeCommandResult] | None = None):
        self.behavior: list[Exception | _FakeCommandResult] = list(behavior or [])
        self.calls: list[tuple[str, dict]] = []

    def run(self, cmd: str, **kwargs: object) -> _FakeCommandResult:
        self.calls.append((cmd, kwargs))
        if self.behavior:
            action = self.behavior.pop(0)
            if isinstance(action, Exception):
                raise action
            return action
        return _FakeCommandResult(stdout="", stderr="", exit_code=0)


@dataclass(frozen=True)
class _FakeEntry:
    name: str
    path: str
    type: object
    size: int | None
    mode: int | None
    permissions: str | None
    modified_time: datetime
    symlink_target: str | None


class _FakeFilesystem:
    def __init__(self, read_content: bytes = b"", exists_error: Exception | None = None):
        self.read_content = read_content
        self.exists_error = exists_error
        self.written: list[tuple[str, bytes | str]] = []
        self.removed: list[str] = []
        self.get_info_entry: _FakeEntry | None = None

    def read(self, path: str, format: str = "bytes") -> bytes:
        return self.read_content

    def write(self, path: str, content: bytes | str, **kw: object) -> None:
        self.written.append((path, content))

    def remove(self, path: str) -> None:
        self.removed.append(path)

    def list(self, path: str) -> list[_FakeEntry]:
        return []

    def exists(self, path: str) -> bool:
        if self.exists_error:
            raise self.exists_error
        return True

    def get_info(self, path: str) -> _FakeEntry:
        if self.get_info_entry is None:
            raise RuntimeError("Missing entry info")
        return self.get_info_entry

    def write_files(self, entries: builtins.list, **kw: object) -> None:  # type: ignore[type-arg]
        pass


class _FakeConnectionConfig:
    def __init__(self, sandbox_url: str = "", extra_sandbox_headers: dict | None = None) -> None:
        self.sandbox_url = sandbox_url
        self.sandbox_headers = extra_sandbox_headers or {}


class _FakeSandbox:
    def __init__(
        self,
        commands: _FakeCommands | None = None,
        filesystem: _FakeFilesystem | None = None,
        sandbox_id: str = "sbx",
    ):
        self.commands = commands or _FakeCommands()
        self._filesystem = filesystem or _FakeFilesystem()
        self.sandbox_id = sandbox_id
        self.connection_config = _FakeConnectionConfig()
        self._transport = object()
        self.sandbox_domain: str | None = "example.com"
        self._envd_access_token: str | None = "token"
        self._envd_version: Version = Version("0.1.4")
        self._killed = False

    def kill(self, **kw: object) -> None:
        self._killed = True

    def beta_pause(self, **kw: object) -> None:
        pass

    def is_running(self) -> bool:
        return not self._killed


class _FakeSandboxClass:
    def __init__(self, sandbox: _FakeSandbox):
        self._sandbox = sandbox
        self.connect_called = False
        self.init_kwargs: dict | None = None

    def __call__(self, **kwargs: object) -> _FakeSandbox:
        self.init_kwargs = kwargs
        return self._sandbox

    def connect(self, sandbox_id: str, **kwargs: object) -> _FakeSandbox:
        self.connect_called = True
        return self._sandbox

    def beta_create(self, **kwargs: object) -> _FakeSandbox:
        return self._sandbox

    def set_timeout(self, sandbox_id: str, timeout: int, **kw: object) -> None:
        pass


@dataclass(frozen=True)
class _FakeFileType:
    FILE: str = "file"
    DIR: str = "dir"


class _FakeCommandExitError(Exception):
    def __init__(self, stdout: str, stderr: str, exit_code: int):
        super().__init__("command failed")
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _FakeTimeoutError(Exception):
    pass


def _enable_fake_e2b(monkeypatch: pytest.MonkeyPatch, sandbox: _FakeSandbox | None = None) -> _FakeSandboxClass:
    sandbox = sandbox or _FakeSandbox()
    fake_class = _FakeSandboxClass(sandbox)
    monkeypatch.setattr(e2b_module, "E2B_AVAILABLE", True)
    monkeypatch.setattr(e2b_module, "FileType", _FakeFileType())
    monkeypatch.setattr(e2b_module, "Sandbox", fake_class)
    # Mock e2b exception classes used in execute_bash
    if "e2b" not in sys.modules:
        fake_e2b = types.ModuleType("e2b")
        sys.modules["e2b"] = fake_e2b
    fake_e2b = sys.modules["e2b"]
    monkeypatch.setattr(fake_e2b, "CommandExitException", _FakeCommandExitError, raising=False)
    if "e2b.exceptions" not in sys.modules:
        fake_exc = types.ModuleType("e2b.exceptions")
        sys.modules["e2b.exceptions"] = fake_exc
        fake_e2b.exceptions = fake_exc  # type: ignore
    monkeypatch.setattr(sys.modules["e2b.exceptions"], "TimeoutException", _FakeTimeoutError, raising=False)
    return fake_class


def _attach_backend(sandbox: E2BSandbox, backend: _FakeSandbox) -> None:
    sandbox.__dict__["_sandbox"] = backend
    sandbox.sandbox_id = backend.sandbox_id


# =============================================================================
# _resolve_path tests
# =============================================================================


class TestResolvePath:
    """Tests for E2BSandbox._resolve_path — the client-side path resolution
    that ensures consistent behavior across SaaS and self-host envd."""

    def _make_sandbox(self, monkeypatch: pytest.MonkeyPatch, work_dir: str = "/home/user") -> E2BSandbox:
        _enable_fake_e2b(monkeypatch)
        return E2BSandbox(sandbox_id="sbx", _work_dir=work_dir)

    def test_absolute_path_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch)
        assert sandbox._resolve_path("/etc/hosts") == "/etc/hosts"

    def test_relative_path_resolved_against_work_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch)
        assert sandbox._resolve_path("project/main.py") == "/home/user/project/main.py"

    def test_relative_path_with_custom_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch)
        assert sandbox._resolve_path("file.txt", cwd="/tmp") == "/tmp/file.txt"

    def test_absolute_path_ignores_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch)
        assert sandbox._resolve_path("/var/log/app.log", cwd="/tmp") == "/var/log/app.log"

    def test_dot_relative_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch)
        assert sandbox._resolve_path("./file.txt") == "/home/user/./file.txt"

    def test_custom_work_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch, work_dir="/workspace")
        assert sandbox._resolve_path("src/app.py") == "/workspace/src/app.py"

    def test_bare_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = self._make_sandbox(monkeypatch)
        assert sandbox._resolve_path("readme.md") == "/home/user/readme.md"


# =============================================================================
# E2B_DEFAULT_WORK_DIR constant tests
# =============================================================================


class TestDefaultWorkDir:
    """Verify E2B_DEFAULT_WORK_DIR is the single source of truth."""

    def test_constant_value(self) -> None:
        from nexau.archs.sandbox.base_sandbox import E2B_DEFAULT_WORK_DIR

        assert E2B_DEFAULT_WORK_DIR == "/home/user"

    def test_e2b_sandbox_default_work_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nexau.archs.sandbox.base_sandbox import E2B_DEFAULT_WORK_DIR

        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        assert str(sandbox.work_dir) == E2B_DEFAULT_WORK_DIR

    def test_e2b_sandbox_config_default_work_dir(self) -> None:
        from nexau.archs.sandbox.base_sandbox import E2B_DEFAULT_WORK_DIR

        config = E2BSandboxConfig()
        assert config.work_dir == E2B_DEFAULT_WORK_DIR

    def test_manager_default_work_dir(self) -> None:
        from nexau.archs.sandbox.base_sandbox import E2B_DEFAULT_WORK_DIR

        manager = e2b_module.E2BSandboxManager(api_key="key")
        assert str(manager.work_dir) == E2B_DEFAULT_WORK_DIR


# =============================================================================
# Defensive code paths — exercised via fakes to reach 100% coverage
# =============================================================================


class TestExecuteBashDefensive:
    """Cover execute_bash error/edge paths that can't be triggered via e2e."""

    def test_reconnects_on_event_loop_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(
            behavior=[
                RuntimeError("Event loop is closed"),
                _FakeCommandResult(stdout="ok", stderr="", exit_code=0),
            ]
        )
        backend = _FakeSandbox(commands=commands)
        cls = _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        result = sandbox.execute_bash("echo hi")
        assert result.status == SandboxStatus.SUCCESS
        assert cls.connect_called

    def test_command_exit_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(behavior=[_FakeCommandExitError("out", "err", 2)])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        result = sandbox.execute_bash("false")
        assert result.status == SandboxStatus.ERROR
        assert result.exit_code == 2

    def test_timeout_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(behavior=[_FakeTimeoutError()])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        result = sandbox.execute_bash("sleep 999", timeout=1000)
        assert result.status == SandboxStatus.TIMEOUT

    def test_generic_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(behavior=[ValueError("unexpected")])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        result = sandbox.execute_bash("echo hi")
        assert result.status == SandboxStatus.ERROR
        assert "unexpected" in (result.error or "")

    def test_invalid_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(ValueError, match="root.*user"):
            sandbox.execute_bash("echo hi", user="nobody")

    def test_no_sandbox_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.execute_bash("echo hi")


class TestExecuteCodeDefensive:
    def test_no_sandbox_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.execute_code("print(1)", "python")

    def test_invalid_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(ValueError, match="root.*user"):
            sandbox.execute_code("print(1)", "python", user="nobody")

    def test_unsupported_language_enum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CodeLanguage enum that is not PYTHON."""

        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        # CodeLanguage only has PYTHON; test with a string that doesn't parse
        result = sandbox.execute_code("code", "ruby")
        assert result.status == SandboxStatus.ERROR


class TestFileOpsDefensive:
    """Cover file operation defensive paths."""

    def test_read_file_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.read_file("f.txt")

    def test_write_file_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.write_file("f.txt", "c")

    def test_write_file_invalid_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(ValueError, match="root.*user"):
            sandbox.write_file("f.txt", "c", user="nobody")

    def test_write_file_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fs = _FakeFilesystem()
        fs.write = lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))  # type: ignore
        backend = _FakeSandbox(filesystem=fs)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        result = sandbox.write_file("/tmp/f.txt", "c")
        assert result.status == SandboxStatus.ERROR

    def test_delete_file_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.delete_file("f.txt")

    def test_delete_file_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fs = _FakeFilesystem()
        fs.remove = lambda p: (_ for _ in ()).throw(OSError("perm"))  # type: ignore
        backend = _FakeSandbox(filesystem=fs)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        result = sandbox.delete_file("/tmp/f.txt")
        assert result.status == SandboxStatus.ERROR

    def test_list_files_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.list_files(".")

    def test_file_exists_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.file_exists("f.txt")

    def test_file_exists_exception_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fs = _FakeFilesystem(exists_error=RuntimeError("boom"))
        backend = _FakeSandbox(filesystem=fs)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        assert sandbox.file_exists("/missing") is False

    def test_get_file_info_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.get_file_info("f.txt")

    def test_get_file_info_get_info_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_info raises → returns FileInfo(exists=False)."""
        fs = _FakeFilesystem()
        fs.get_info_entry = None  # will raise RuntimeError
        backend = _FakeSandbox(filesystem=fs)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        info = sandbox.get_file_info("/some/file")
        assert not info.exists

    def test_create_directory_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.create_directory("d")

    def test_create_directory_invalid_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(ValueError, match="root.*user"):
            sandbox.create_directory("d", user="nobody")

    def test_create_directory_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(behavior=[_FakeCommandResult("", "perm denied", 1)])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(SandboxFileError):
            sandbox.create_directory("d")


class TestEditFileDefensive:
    def test_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.edit_file("f.txt", "a", "b")

    def test_read_failure_during_edit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        monkeypatch.setattr(sandbox, "file_exists", lambda p: True)
        monkeypatch.setattr(
            sandbox, "read_file", lambda p, **kw: FileOperationResult(status=SandboxStatus.ERROR, file_path=p, error="read failed")
        )
        result = sandbox.edit_file("f.txt", "old", "new")
        assert result.status == SandboxStatus.ERROR


class TestGlobDefensive:
    def test_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.glob("*.py")

    def test_invalid_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(ValueError, match="root.*user"):
            sandbox.glob("*.py", user="nobody")

    def test_glob_command_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(behavior=[_FakeCommandResult("", "bad", 2)])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(SandboxFileError):
            sandbox.glob("*.txt")

    def test_glob_with_slash_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """glob('dir/*.py') should use find with dir and pattern."""
        commands = _FakeCommands(behavior=[_FakeCommandResult("dir/a.py\n", "", 0)])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        matches = sandbox.glob("dir/*.py")
        assert "dir/a.py" in matches

    def test_glob_non_recursive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands = _FakeCommands(behavior=[_FakeCommandResult("a.txt\n", "", 0)])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        matches = sandbox.glob("*.txt", recursive=False)
        assert "a.txt" in matches

    def test_glob_leading_slash_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """glob('**//file') — file_pattern starts with /."""
        commands = _FakeCommands(behavior=[_FakeCommandResult("", "", 0)])
        backend = _FakeSandbox(commands=commands)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        sandbox.glob("**//*.py")
        cmd = commands.calls[0][0]
        assert "find" in cmd


class TestFileTransferDefensive:
    def test_upload_file_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.upload_file("/tmp/f", "dest")

    def test_download_file_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.download_file("src", "/tmp/dest")

    def test_download_file_empty_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        monkeypatch.setattr(
            sandbox, "read_file", lambda p, **kw: FileOperationResult(status=SandboxStatus.SUCCESS, file_path=p, content=None)
        )
        result = sandbox.download_file("remote.txt", "/tmp/local.txt")
        assert result.status == SandboxStatus.ERROR
        assert "Empty content" in (result.error or "")

    def test_download_file_string_content(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """download_file with string content should encode to utf-8."""
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        monkeypatch.setattr(
            sandbox, "read_file", lambda p, **kw: FileOperationResult(status=SandboxStatus.SUCCESS, file_path=p, content="hello")
        )
        dest = tmp_path / "out.txt"
        result = sandbox.download_file("remote.txt", str(dest))
        assert result.status == SandboxStatus.SUCCESS
        assert dest.read_text() == "hello"

    def test_upload_directory_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.upload_directory("/tmp", "dest")

    def test_upload_directory_missing_source(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(SandboxFileError, match="does not exist"):
            sandbox.upload_directory(str(tmp_path / "missing"), "dest")

    def test_upload_directory_not_a_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(SandboxFileError, match="not a directory"):
            sandbox.upload_directory(str(f), "dest")

    def test_upload_directory_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        assert sandbox.upload_directory(str(d), "dest") is True

    def test_download_directory_no_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_fake_e2b(monkeypatch)
        sandbox = E2BSandbox(sandbox_id="sbx")
        with pytest.raises(Exception):
            sandbox.download_directory("src", "/tmp/dest")

    def test_download_directory_missing_source(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fs = _FakeFilesystem()
        fs.exists = lambda p: False  # type: ignore
        backend = _FakeSandbox(filesystem=fs)
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        with pytest.raises(SandboxFileError, match="does not exist"):
            sandbox.download_directory("missing", str(tmp_path))

    def test_download_directory_skips_unrelated_paths(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        monkeypatch.setattr(sandbox, "file_exists", lambda p: True)
        monkeypatch.setattr(sandbox, "list_files", lambda p, **kw: [FileInfo(path="/other/path.txt", exists=True, is_file=True)])
        downloaded: list[str] = []
        monkeypatch.setattr(
            sandbox,
            "download_file",
            lambda s, d, **kw: (
                downloaded.append(s) or FileOperationResult(status=SandboxStatus.SUCCESS, file_path=d)  # type: ignore
            ),
        )
        assert sandbox.download_directory("sandbox_dir", str(tmp_path))
        assert downloaded == []

    def test_download_directory_failed_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """download_directory should warn on failed individual file downloads."""
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        monkeypatch.setattr(sandbox, "file_exists", lambda p: True)
        monkeypatch.setattr(
            sandbox, "list_files", lambda p, **kw: [FileInfo(path="/home/user/sandbox_dir/fail.txt", exists=True, is_file=True)]
        )
        monkeypatch.setattr(
            sandbox,
            "download_file",
            lambda s, d, **kw: FileOperationResult(status=SandboxStatus.ERROR, file_path=d, error="download failed"),
        )
        # Should not raise, just warn
        assert sandbox.download_directory("sandbox_dir", str(tmp_path))


class TestManagerDefensive:
    """Cover Manager start/stop/pause/keepalive defensive paths."""

    def test_start_p1_connect_and_rebuild(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1: connect from config with force_http=True triggers rebuild."""
        backend = _FakeSandbox(sandbox_id="sbx-p1")
        cls = _enable_fake_e2b(monkeypatch, backend)
        # Mock ConnectionConfig
        if "e2b.connection_config" not in sys.modules:
            mod = types.ModuleType("e2b.connection_config")
            mod.ConnectionConfig = _FakeConnectionConfig  # type: ignore
            sys.modules["e2b.connection_config"] = mod
        else:
            monkeypatch.setattr(sys.modules["e2b.connection_config"], "ConnectionConfig", _FakeConnectionConfig)

        # Mock TransportWithLogger
        class _FakeTWL:
            singleton = object()

        if "e2b.api" not in sys.modules:
            sys.modules["e2b.api"] = types.ModuleType("e2b.api")
        if "e2b.api.client_sync" not in sys.modules:
            mod2 = types.ModuleType("e2b.api.client_sync")
            mod2.TransportWithLogger = _FakeTWL  # type: ignore
            sys.modules["e2b.api.client_sync"] = mod2
        else:
            monkeypatch.setattr(sys.modules["e2b.api.client_sync"], "TransportWithLogger", _FakeTWL)

        manager = e2b_module.E2BSandboxManager(api_key="key", api_url="http://api")
        monkeypatch.setattr(manager, "load_sandbox_state", lambda *a: None)
        sandbox = manager.start(object(), "u", "s", E2BSandboxConfig(sandbox_id="sbx-p1", force_http=True))
        assert sandbox.sandbox_id == "sbx-p1"
        assert cls.connect_called
        assert cls.init_kwargs is not None  # rebuilt

    def test_start_p1_connect_failure_falls_to_p3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1 connect failure should fall through to P3."""
        backend = _FakeSandbox(sandbox_id="sbx-new")
        cls = _enable_fake_e2b(monkeypatch, backend)

        # Make connect raise
        def failing_connect(sandbox_id: str, **kw: object) -> _FakeSandbox:
            raise RuntimeError("connect failed")

        cls.connect = failing_connect  # type: ignore

        manager = e2b_module.E2BSandboxManager(api_key="key")
        monkeypatch.setattr(manager, "load_sandbox_state", lambda *a: None)
        monkeypatch.setattr(manager, "persist_sandbox_state", lambda *a: None)
        sandbox = manager.start(object(), "u", "s", E2BSandboxConfig(sandbox_id="sbx-fail", force_http=False))
        assert sandbox.sandbox is not None

    def test_start_p2_restore_and_rebuild(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2: restore from state with force_http=True triggers rebuild."""
        backend = _FakeSandbox(sandbox_id="sbx-p2")
        cls = _enable_fake_e2b(monkeypatch, backend)
        if "e2b.connection_config" not in sys.modules:
            mod = types.ModuleType("e2b.connection_config")
            mod.ConnectionConfig = _FakeConnectionConfig  # type: ignore
            sys.modules["e2b.connection_config"] = mod
        else:
            monkeypatch.setattr(sys.modules["e2b.connection_config"], "ConnectionConfig", _FakeConnectionConfig)

        class _FakeTWL:
            singleton = object()

        if "e2b.api.client_sync" not in sys.modules:
            if "e2b.api" not in sys.modules:
                sys.modules["e2b.api"] = types.ModuleType("e2b.api")
            mod2 = types.ModuleType("e2b.api.client_sync")
            mod2.TransportWithLogger = _FakeTWL  # type: ignore
            sys.modules["e2b.api.client_sync"] = mod2
        else:
            monkeypatch.setattr(sys.modules["e2b.api.client_sync"], "TransportWithLogger", _FakeTWL)

        manager = e2b_module.E2BSandboxManager(api_key="key")
        monkeypatch.setattr(manager, "load_sandbox_state", lambda *a: {"sandbox_id": "sbx-p2"})
        sandbox = manager.start(object(), "u", "s", E2BSandboxConfig(force_http=True))
        assert sandbox.sandbox_id == "sbx-p2"
        assert cls.connect_called

    def test_start_p2_restore_failure_falls_to_p3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2 restore failure should fall through to P3."""
        backend = _FakeSandbox(sandbox_id="sbx-new")
        cls = _enable_fake_e2b(monkeypatch, backend)
        call_count = [0]

        def counting_connect(sandbox_id: str, **kw: object) -> _FakeSandbox:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("connect failed")
            return backend

        cls.connect = counting_connect  # type: ignore

        manager = e2b_module.E2BSandboxManager(api_key="key")
        monkeypatch.setattr(manager, "load_sandbox_state", lambda *a: {"sandbox_id": "sbx-old"})
        monkeypatch.setattr(manager, "persist_sandbox_state", lambda *a: None)
        sandbox = manager.start(object(), "u", "s", E2BSandboxConfig(force_http=False))
        assert sandbox.sandbox is not None

    def test_start_p2_missing_sandbox_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2 with empty sandbox_id in state should fall to P3."""
        backend = _FakeSandbox(sandbox_id="sbx-new")
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        monkeypatch.setattr(manager, "load_sandbox_state", lambda *a: {"sandbox_id": ""})
        monkeypatch.setattr(manager, "persist_sandbox_state", lambda *a: None)
        sandbox = manager.start(object(), "u", "s", E2BSandboxConfig())
        assert sandbox.sandbox is not None

    def test_start_config_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """start() should apply api_key/api_url/template from config."""
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="old")
        monkeypatch.setattr(manager, "load_sandbox_state", lambda *a: None)
        monkeypatch.setattr(manager, "persist_sandbox_state", lambda *a: None)
        config = E2BSandboxConfig(api_key="new-key", api_url="http://new", template="custom")
        manager.start(object(), "u", "s", config)
        assert manager.api_key == "new-key"
        assert manager.api_url == "http://new"
        assert manager.template == "custom"

    def test_maybe_rebuild_warning_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_maybe_rebuild_for_http with missing domain/token should warn and return original."""
        backend = _FakeSandbox()
        backend.sandbox_domain = None
        backend._envd_access_token = None
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        result = manager._maybe_rebuild_for_http(backend, E2BSandboxConfig(force_http=True))  # type: ignore
        assert result is backend  # returned unchanged

    def test_stop_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stop() should return False on kill exception."""
        backend = _FakeSandbox()
        backend.kill = lambda **kw: (_ for _ in ()).throw(RuntimeError("kill failed"))  # type: ignore
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        manager._instance = sandbox
        assert manager.stop() is False

    def test_pause_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pause() should return False on beta_pause exception."""
        backend = _FakeSandbox()
        backend.beta_pause = lambda **kw: (_ for _ in ()).throw(RuntimeError("pause failed"))  # type: ignore
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        manager._instance = sandbox
        assert manager.pause() is False

    def test_is_running_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_running() should return False on exception."""
        backend = _FakeSandbox()
        backend.is_running = lambda: (_ for _ in ()).throw(RuntimeError("check failed"))  # type: ignore
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        sandbox = E2BSandbox(sandbox_id="sbx")
        _attach_backend(sandbox, backend)
        manager._instance = sandbox
        assert manager.is_running() is False

    def test_keepalive_skipped_when_interval_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        manager._start_keepalive("sbx", 0)
        assert manager._keepalive_thread is None

    def test_keepalive_starts_and_stops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        backend = _FakeSandbox()
        cls = _enable_fake_e2b(monkeypatch, backend)
        set_timeout_calls = []
        cls.set_timeout = lambda sid, t, **kw: set_timeout_calls.append(sid)  # type: ignore
        manager = e2b_module.E2BSandboxManager(api_key="key")
        manager._start_keepalive("sbx-ka", 1)
        assert manager._keepalive_thread is not None
        time.sleep(1.5)
        manager._stop_keepalive()
        assert manager._keepalive_thread is None
        assert len(set_timeout_calls) >= 1

    def test_keepalive_handles_409_auto_resume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        backend = _FakeSandbox()
        cls = _enable_fake_e2b(monkeypatch, backend)
        call_count = [0]

        def failing_set_timeout(sid: str, t: int, **kw: object) -> None:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("409 sandbox not running")

        cls.set_timeout = failing_set_timeout  # type: ignore
        connect_calls: list[str] = []

        def tracking_connect(sandbox_id: str, **kw: object) -> _FakeSandbox:
            connect_calls.append(sandbox_id)
            return backend

        cls.connect = tracking_connect  # type: ignore
        manager = e2b_module.E2BSandboxManager(api_key="key")
        manager._start_keepalive("sbx-409", 1)
        time.sleep(1.5)
        manager._stop_keepalive()
        assert len(connect_calls) >= 1

    def test_keepalive_handles_generic_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        backend = _FakeSandbox()
        cls = _enable_fake_e2b(monkeypatch, backend)
        cls.set_timeout = lambda sid, t, **kw: (_ for _ in ()).throw(RuntimeError("network error"))  # type: ignore
        manager = e2b_module.E2BSandboxManager(api_key="key")
        manager._start_keepalive("sbx-err", 1)
        time.sleep(1.5)
        manager._stop_keepalive()
        # Should not crash

    def test_keepalive_auto_resume_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        backend = _FakeSandbox()
        cls = _enable_fake_e2b(monkeypatch, backend)
        cls.set_timeout = lambda sid, t, **kw: (_ for _ in ()).throw(RuntimeError("409 not running"))  # type: ignore

        def failing_connect(sandbox_id: str, **kw: object) -> _FakeSandbox:
            raise RuntimeError("resume failed")

        cls.connect = failing_connect  # type: ignore
        manager = e2b_module.E2BSandboxManager(api_key="key")
        manager._start_keepalive("sbx-rf", 1)
        time.sleep(1.5)
        manager._stop_keepalive()
        # Should not crash

    def test_on_run_complete_stops_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        cls = _enable_fake_e2b(monkeypatch, backend)
        cls.set_timeout = lambda sid, t, **kw: None  # type: ignore
        manager = e2b_module.E2BSandboxManager(api_key="key")
        manager._start_keepalive("sbx-orc", 60)
        assert manager._keepalive_thread is not None
        manager.on_run_complete()
        assert manager._keepalive_thread is None


class TestBuildSandboxResetsSingleton:
    def test_transport_singleton_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeSandbox()
        _enable_fake_e2b(monkeypatch, backend)
        if "e2b.connection_config" not in sys.modules:
            mod = types.ModuleType("e2b.connection_config")
            mod.ConnectionConfig = _FakeConnectionConfig  # type: ignore
            sys.modules["e2b.connection_config"] = mod
        else:
            monkeypatch.setattr(sys.modules["e2b.connection_config"], "ConnectionConfig", _FakeConnectionConfig)

        class _FakeTWL:
            singleton = object()

        if "e2b.api" not in sys.modules:
            sys.modules["e2b.api"] = types.ModuleType("e2b.api")
        if "e2b.api.client_sync" not in sys.modules:
            mod2 = types.ModuleType("e2b.api.client_sync")
            mod2.TransportWithLogger = _FakeTWL  # type: ignore
            sys.modules["e2b.api.client_sync"] = mod2
        else:
            monkeypatch.setattr(sys.modules["e2b.api.client_sync"], "TransportWithLogger", _FakeTWL)
        manager = e2b_module.E2BSandboxManager(api_key="key")
        assert _FakeTWL.singleton is not None
        manager._build_sandbox_with_connection_config("sbx", "example.com", "token", Version("0.1.4"))
        assert _FakeTWL.singleton is None
