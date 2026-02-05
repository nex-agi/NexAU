from __future__ import annotations

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
    FileInfo,
    FileOperationResult,
    SandboxFileError,
    SandboxStatus,
)
from nexau.archs.sandbox.e2b_sandbox import E2BSandbox


@dataclass(frozen=True)
class _FakeCommandResult:
    stdout: str | None
    stderr: str | None
    exit_code: int


class _FakeCommands:
    def __init__(self, behavior: Sequence[Exception | _FakeCommandResult] | None = None):
        self.behavior: list[Exception | _FakeCommandResult] = list(behavior or [])
        self.calls: list[tuple[str, int | None, str | None]] = []

    def run(self, cmd: str, timeout: int | None = None, cwd: str | None = None) -> _FakeCommandResult:
        self.calls.append((cmd, timeout, cwd))
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
        self.read_calls: list[tuple[str, str]] = []
        self.get_info_entry: _FakeEntry | None = None

    def read(self, path: str, format: str = "bytes") -> bytes:
        self.read_calls.append((path, format))
        return self.read_content

    def write(self, path: str, content: bytes | str) -> None:
        return None

    def remove(self, path: str) -> None:
        return None

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


class _FakeConnectionConfig:
    def __init__(self, sandbox_url: str, extra_sandbox_headers: dict[str, str]) -> None:
        self.sandbox_url = sandbox_url
        self.sandbox_headers = extra_sandbox_headers


class _FakeSandbox:
    def __init__(self, commands: _FakeCommands, filesystem: _FakeFilesystem, sandbox_id: str = "sbx"):
        self.commands = commands
        self._filesystem = filesystem
        self.sandbox_id = sandbox_id
        self.connection_config = _FakeConnectionConfig(sandbox_url="", extra_sandbox_headers={})
        self._transport = object()

    def kill(self) -> None:
        return None

    def beta_pause(self) -> None:
        return None

    def is_running(self) -> bool:
        return True


class _FakeSandboxClass:
    def __init__(self, sandbox: _FakeSandbox):
        self._sandbox = sandbox
        self.connect_called = False

    def __call__(self, **kwargs: object) -> _FakeSandbox:
        return self._sandbox

    def connect(self, sandbox_id: str, **kwargs: object) -> _FakeSandbox:
        self.connect_called = True
        return self._sandbox

    def beta_create(self, **kwargs: object) -> _FakeSandbox:
        return self._sandbox


@dataclass
class _FakeManagerSandbox:
    sandbox_id: str
    connection_config: _FakeConnectionConfig
    _transport: object
    _SandboxBase__envd_api_url: str
    _envd_api: object | None = None

    @property
    def envd_api_url(self) -> str:
        return self._SandboxBase__envd_api_url

    @property
    def envd_api(self) -> object | None:
        return self._envd_api


class _RecordingSandboxFactory:
    def __init__(self, sandbox: _FakeManagerSandbox):
        self._sandbox = sandbox
        self.init_kwargs: dict[str, object] | None = None
        self.connected_kwargs: dict[str, object] | None = None
        self.created_kwargs: dict[str, object] | None = None
        self.connect_called = False
        self.beta_create_called = False

    def __call__(self, **kwargs: object) -> _FakeManagerSandbox:
        self.init_kwargs = kwargs
        return self._sandbox

    def connect(self, sandbox_id: str, **kwargs: object) -> _FakeManagerSandbox:
        self.connect_called = True
        self.connected_kwargs = {"sandbox_id": sandbox_id, **kwargs}
        return self._sandbox

    def beta_create(self, **kwargs: object) -> _FakeManagerSandbox:
        self.beta_create_called = True
        self.created_kwargs = kwargs
        return self._sandbox


class _FakeHttpxClient:
    def __init__(self, base_url: str, transport: object, headers: dict[str, str]) -> None:
        self.base_url = base_url
        self.transport = transport
        self.headers = headers


class _FakeCommandExitError(Exception):
    def __init__(self, stdout: str, stderr: str, exit_code: int):
        super().__init__("command failed")
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


@dataclass(frozen=True)
class _FakeFileType:
    FILE: str = "file"
    DIR: str = "dir"


def _enable_fake_e2b(monkeypatch: pytest.MonkeyPatch, sandbox: _FakeSandbox) -> _FakeSandboxClass:
    fake_class = _FakeSandboxClass(sandbox)
    monkeypatch.setattr(e2b_module, "E2B_AVAILABLE", True)
    monkeypatch.setattr(e2b_module, "FileType", _FakeFileType())
    monkeypatch.setattr(e2b_module, "Sandbox", fake_class)
    monkeypatch.setattr(e2b_module, "CommandExitException", _FakeCommandExitError)
    return fake_class


def _enable_fake_e2b_with_factory(monkeypatch: pytest.MonkeyPatch, factory: _RecordingSandboxFactory) -> None:
    monkeypatch.setattr(e2b_module, "E2B_AVAILABLE", True)
    monkeypatch.setattr(e2b_module, "Sandbox", factory)


def _attach_backend(sandbox: E2BSandbox, backend: _FakeSandbox) -> None:
    sandbox.__dict__["_sandbox"] = backend
    sandbox.sandbox_id = backend.sandbox_id


def test_execute_bash_reconnects_on_event_loop_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _FakeCommands(
        behavior=[
            RuntimeError("Event loop is closed"),
            _FakeCommandResult(stdout="ok", stderr="", exit_code=0),
        ]
    )
    filesystem = _FakeFilesystem()
    sandbox_backend = _FakeSandbox(commands, filesystem, sandbox_id="sbx-123")
    fake_sandbox_class = _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx-123")
    _attach_backend(sandbox, sandbox_backend)
    result = sandbox.execute_bash("echo hi")

    assert result.status == SandboxStatus.SUCCESS
    assert "ok" in result.stdout
    assert fake_sandbox_class.connect_called


def test_execute_bash_handles_command_exit_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _FakeCommands(behavior=[_FakeCommandExitError(stdout="out", stderr="err", exit_code=2)])
    filesystem = _FakeFilesystem()
    sandbox_backend = _FakeSandbox(commands, filesystem)
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)
    result = sandbox.execute_bash("false")

    assert result.status == SandboxStatus.ERROR
    assert result.exit_code == 2
    assert result.truncated is False
    assert "exit code 2" in (result.error or "")


def test_read_file_truncates_large_content(monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"a" * 30010
    filesystem = _FakeFilesystem(read_content=content)
    sandbox_backend = _FakeSandbox(_FakeCommands(), filesystem)
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    def fake_get_file_info(path: str) -> FileInfo:
        return FileInfo(path=path, exists=True, size=len(content))

    monkeypatch.setattr(sandbox, "get_file_info", fake_get_file_info)

    result = sandbox.read_file("big.txt")
    assert result.status == SandboxStatus.SUCCESS
    assert isinstance(result.content, str)
    assert len(result.content) == 30000
    assert result.truncated


def test_file_exists_returns_false_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    filesystem = _FakeFilesystem(exists_error=RuntimeError("boom"))
    sandbox_backend = _FakeSandbox(_FakeCommands(), filesystem)
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)
    assert sandbox.file_exists("/missing") is False


def test_get_file_info_detects_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    filesystem = _FakeFilesystem(read_content=b"hello")
    entry = _FakeEntry(
        name="file.txt",
        path="/file.txt",
        type="file",
        size=5,
        mode=0o600,
        permissions="rw-------",
        modified_time=datetime(2024, 1, 1, 12, 0, 0),
        symlink_target=None,
    )
    filesystem.get_info_entry = entry
    sandbox_backend = _FakeSandbox(_FakeCommands(), filesystem)
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    def fake_detect(self: E2BSandbox, raw: bytes) -> str:
        return "utf-16"

    monkeypatch.setattr(E2BSandbox, "_detect_file_encoding", fake_detect)

    info = sandbox.get_file_info("/file.txt")
    assert info.exists
    assert info.is_file
    assert info.readable is True
    assert info.writable is True
    assert info.encoding == "utf-16"


def test_edit_file_normalizes_escaped_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_backend = _FakeSandbox(_FakeCommands(), _FakeFilesystem())
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    def fake_exists(path: str) -> bool:
        return True

    def fake_read(path: str, encoding: str = "utf-8", binary: bool = False) -> FileOperationResult:
        return FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path=path,
            content="line1\nline2",
        )

    def fake_write(
        path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        binary: bool = False,
        create_directories: bool = True,
    ) -> FileOperationResult:
        return FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path=path,
            content=content,
        )

    monkeypatch.setattr(sandbox, "file_exists", fake_exists)
    monkeypatch.setattr(sandbox, "read_file", fake_read)
    monkeypatch.setattr(sandbox, "write_file", fake_write)

    result = sandbox.edit_file("file.txt", "line1\\nline2", "replaced")
    assert result.status == SandboxStatus.SUCCESS


def test_edit_file_multiple_matches_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_backend = _FakeSandbox(_FakeCommands(), _FakeFilesystem())
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    def fake_exists(path: str) -> bool:
        return True

    def fake_read(path: str, encoding: str = "utf-8", binary: bool = False) -> FileOperationResult:
        return FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path=path,
            content="test test",
        )

    monkeypatch.setattr(sandbox, "file_exists", fake_exists)
    monkeypatch.setattr(sandbox, "read_file", fake_read)

    result = sandbox.edit_file("file.txt", "test", "new")
    assert result.status == SandboxStatus.ERROR
    assert "matches" in (result.error or "")


def test_glob_raises_on_command_error(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _FakeCommands(behavior=[_FakeCommandResult(stdout="", stderr="bad", exit_code=2)])
    sandbox_backend = _FakeSandbox(commands, _FakeFilesystem())
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    with pytest.raises(SandboxFileError):
        sandbox.glob("*.txt", recursive=True)


def test_download_file_empty_content_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_backend = _FakeSandbox(_FakeCommands(), _FakeFilesystem())
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    def fake_read(path: str, encoding: str = "utf-8", binary: bool = False) -> FileOperationResult:
        return FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path=path,
            content=None,
        )

    monkeypatch.setattr(sandbox, "read_file", fake_read)

    result = sandbox.download_file("remote.txt", "local.txt")
    assert result.status == SandboxStatus.ERROR
    assert "Empty content" in (result.error or "")


def test_upload_directory_invalid_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sandbox_backend = _FakeSandbox(_FakeCommands(), _FakeFilesystem())
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    with pytest.raises(SandboxFileError):
        sandbox.upload_directory(str(tmp_path / "missing"), "dest")

    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    with pytest.raises(SandboxFileError):
        sandbox.upload_directory(str(file_path), "dest")


def test_download_directory_skips_unrelated_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sandbox_backend = _FakeSandbox(_FakeCommands(), _FakeFilesystem())
    _enable_fake_e2b(monkeypatch, sandbox_backend)

    sandbox = E2BSandbox(sandbox_id="sbx")
    _attach_backend(sandbox, sandbox_backend)

    def fake_exists(path: str) -> bool:
        return True

    def fake_list(path: str, recursive: bool = True, pattern: str | None = None) -> list[FileInfo]:
        return [FileInfo(path="/other/path.txt", exists=True, is_file=True)]

    monkeypatch.setattr(sandbox, "file_exists", fake_exists)
    monkeypatch.setattr(sandbox, "list_files", fake_list)
    downloaded: list[str] = []

    def fake_download(src: str, dst: str, create_directories: bool = True) -> FileOperationResult:
        downloaded.append(src)
        return FileOperationResult(status=SandboxStatus.SUCCESS, file_path=dst)

    monkeypatch.setattr(sandbox, "download_file", fake_download)

    assert sandbox.download_directory("sandbox_dir", str(tmp_path))
    assert downloaded == []


def test_manager_start_connects_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeManagerSandbox(
        sandbox_id="sbx-123",
        connection_config=_FakeConnectionConfig(sandbox_url="", extra_sandbox_headers={}),
        _transport=object(),
        _SandboxBase__envd_api_url="https://envd-service",
    )
    factory = _RecordingSandboxFactory(backend)
    _enable_fake_e2b_with_factory(monkeypatch, factory)
    monkeypatch.setattr(e2b_module, "_get_connection_config_class", lambda: _FakeConnectionConfig)

    manager = e2b_module.E2BSandboxManager(api_key="api-key", api_url="https://api", force_http=False)
    sandbox_config = {
        "sandbox_id": "sbx-123",
        "sandbox_domain": "example.com",
        "envd_access_token": "token",
        "envd_version": "1.2.3",
    }

    sandbox = manager.start(object(), "user", "session", sandbox_config)

    assert sandbox.sandbox_id == "sbx-123"
    assert sandbox.sandbox is not None
    assert sandbox.sandbox_id == backend.sandbox_id
    assert factory.init_kwargs is not None
    connection_config = factory.init_kwargs["connection_config"]
    assert isinstance(connection_config, _FakeConnectionConfig)
    assert connection_config.sandbox_url == "http://49983-sbx-123.example.com"
    assert connection_config.sandbox_headers == {"X-Access-Token": "token"}
    assert isinstance(factory.init_kwargs["envd_version"], Version)


def test_manager_start_restores_state_and_patches_envd_url(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeManagerSandbox(
        sandbox_id="sbx-state",
        connection_config=_FakeConnectionConfig(
            sandbox_url="",
            extra_sandbox_headers={"X-Access-Token": "token"},
        ),
        _transport=object(),
        _SandboxBase__envd_api_url="https://envd-service",
    )
    factory = _RecordingSandboxFactory(backend)
    _enable_fake_e2b_with_factory(monkeypatch, factory)

    fake_httpx = types.ModuleType("httpx")
    setattr(fake_httpx, "Client", _FakeHttpxClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    manager = e2b_module.E2BSandboxManager(api_key="api-key", api_url="https://api", force_http=True)

    def fake_load_sandbox_state(session_manager: object, user_id: str, session_id: str) -> dict[str, str]:
        return {"sandbox_id": "sbx-state"}

    monkeypatch.setattr(manager, "load_sandbox_state", fake_load_sandbox_state)

    sandbox = manager.start(object(), "user", "session", {})

    assert factory.connect_called
    assert sandbox.sandbox is not None
    assert sandbox.sandbox_id == backend.sandbox_id
    assert factory.connected_kwargs == {"sandbox_id": "sbx-state", "api_key": "api-key", "api_url": "https://api"}
    assert backend.envd_api_url == "http://envd-service"
    assert isinstance(backend.envd_api, _FakeHttpxClient)
    assert backend.envd_api.base_url == "http://envd-service"


def test_manager_start_creates_sandbox_and_patches_envd_url(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeManagerSandbox(
        sandbox_id="sbx-new",
        connection_config=_FakeConnectionConfig(
            sandbox_url="",
            extra_sandbox_headers={"X-Access-Token": "token"},
        ),
        _transport=object(),
        _SandboxBase__envd_api_url="https://envd-service",
    )
    factory = _RecordingSandboxFactory(backend)
    _enable_fake_e2b_with_factory(monkeypatch, factory)

    fake_httpx = types.ModuleType("httpx")
    setattr(fake_httpx, "Client", _FakeHttpxClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    manager = e2b_module.E2BSandboxManager(api_key="api-key", api_url="https://api", force_http=True)

    def fake_load_sandbox_state(session_manager: object, user_id: str, session_id: str) -> None:
        return None

    monkeypatch.setattr(manager, "load_sandbox_state", fake_load_sandbox_state)
    persisted: list[E2BSandbox] = []

    def fake_persist_sandbox_state(session_manager: object, user_id: str, session_id: str, sandbox: E2BSandbox) -> None:
        persisted.append(sandbox)

    monkeypatch.setattr(manager, "persist_sandbox_state", fake_persist_sandbox_state)

    sandbox = manager.start(object(), "user", "session", {})

    assert factory.beta_create_called
    assert factory.created_kwargs is not None
    assert factory.created_kwargs["auto_pause"] is False
    assert sandbox.sandbox_id == "sbx-new"
    assert backend.envd_api_url == "http://envd-service"
    assert isinstance(backend.envd_api, _FakeHttpxClient)
    assert persisted == [sandbox]
