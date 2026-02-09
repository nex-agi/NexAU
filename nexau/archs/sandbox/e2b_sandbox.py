# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
E2B sandbox implementation for secure cloud-based code execution.

This implementation uses E2B (https://e2b.dev) to provide isolated cloud sandboxes
for secure code execution and file operations. E2B provides proper security isolation
and is suitable for production use.
"""

# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    import httpx

from packaging.version import Version

from .base_sandbox import (
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
    extract_dataclass_init_kwargs,
)

logger = logging.getLogger(__name__)


class _CommandResultProtocol(Protocol):
    stdout: str | None
    stderr: str | None
    exit_code: int


class _CommandHandleProtocol(Protocol):
    @property
    def pid(self) -> int: ...
    def wait(self) -> _CommandResultProtocol: ...
    def kill(self) -> bool: ...


class _SandboxCommandsProtocol(Protocol):
    def run(
        self,
        cmd: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool | None = None,
    ) -> _CommandResultProtocol: ...


class _FilesystemEntryProtocol(Protocol):
    name: str
    path: str
    type: object
    size: int | None


class _SandboxFilesystemProtocol(Protocol):
    def read(self, path: str, format: str = "bytes") -> bytes: ...

    def write(self, path: str, content: bytes | str) -> None: ...

    def remove(self, path: str) -> None: ...

    def list(self, path: str) -> list[_FilesystemEntryProtocol]: ...

    def exists(self, path: str) -> bool: ...

    def get_info(self, path: str) -> _FilesystemEntryProtocol: ...


class _SandboxProtocol(Protocol):
    sandbox_id: str
    commands: _SandboxCommandsProtocol
    _filesystem: _SandboxFilesystemProtocol
    connection_config: _ConnectionConfigProtocol
    _transport: httpx.BaseTransport | None
    _envd_api: object

    def kill(self) -> None: ...

    def beta_pause(self) -> None: ...

    def is_running(self) -> bool: ...


class _SandboxClassProtocol(Protocol):
    def __call__(self, **kwargs: object) -> _SandboxProtocol: ...

    def connect(self, sandbox_id: str, **kwargs: object) -> _SandboxProtocol: ...

    def beta_create(self, **kwargs: object) -> _SandboxProtocol: ...


class _FileTypeProtocol(Protocol):
    FILE: object
    DIR: object


class _E2BTimeoutError(Exception):
    pass


class _E2BCommandExitError(Exception):
    pass


Sandbox: _SandboxClassProtocol | None = None
TimeoutException: type[BaseException] = _E2BTimeoutError
CommandExitException: type[BaseException] = _E2BCommandExitError
FileType: _FileTypeProtocol | None = None

try:
    e2b_module = import_module("e2b")
    e2b_exceptions = import_module("e2b.exceptions")
    e2b_command_handle = import_module("e2b.sandbox.commands.command_handle")
    e2b_filesystem = import_module("e2b.sandbox.filesystem.filesystem")

    Sandbox = cast(_SandboxClassProtocol, e2b_module.Sandbox)
    TimeoutException = cast(type[BaseException], e2b_exceptions.TimeoutException)
    CommandExitException = cast(type[BaseException], e2b_command_handle.CommandExitException)
    FileType = cast(_FileTypeProtocol, e2b_filesystem.FileType)
    _e2b_available = True
except (ImportError, ModuleNotFoundError, AttributeError):
    logger.warning("E2B SDK not installed. Install it with: pip install e2b")
    _e2b_available = False

E2B_AVAILABLE = _e2b_available


class _ConnectionConfigProtocol(Protocol):
    def __init__(self, sandbox_url: str, extra_sandbox_headers: dict[str, str]) -> None: ...

    sandbox_headers: dict[str, str]


def _get_connection_config_class() -> type[_ConnectionConfigProtocol]:
    connection_config_module = import_module("e2b.connection_config")
    return cast(type[_ConnectionConfigProtocol], connection_config_module.ConnectionConfig)


@dataclass(kw_only=True)
class E2BSandbox(BaseSandbox):
    """
    E2B cloud sandbox implementation for secure code execution.

    This implementation uses E2B's cloud infrastructure to provide isolated,
    secure sandboxes for code execution and file operations. It's suitable
    for production use and provides proper security isolation.

    Note: Configuration (template, timeout, api_key, etc.) is managed by E2BSandboxManager.
    This class only contains runtime state.
    """

    default_user: str = field(default="user")
    _work_dir: str = field(default="/home/user")
    envd_version: str | None = field(default=None)
    _api_key: str | None = field(default=None, repr=False)
    _api_url: str | None = field(default=None, repr=False)

    # Unserialized fields
    _sandbox: _SandboxProtocol | None = field(default=None, repr=False, init=False)

    @property
    def sandbox(self) -> _SandboxProtocol | None:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: _SandboxProtocol):
        self._sandbox = sandbox
        self.sandbox_id = sandbox.sandbox_id

    def set_api_credentials(self, api_key: str | None, api_url: str | None) -> None:
        self._api_key = api_key
        self._api_url = api_url

    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
    ) -> CommandResult:
        """
        Execute a bash command in the E2B sandbox.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in milliseconds (overrides default)
            cwd: Optional working directory
            user: Optional user to run the command as
            envs: Optional environment variables
            background: Optional flag to run the command in the background

        Returns:
            CommandResult containing execution results
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if self._sandbox is None:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        start_time = time.time()

        if timeout is None:
            timeout = 120000  # Default: 2 minutes

        timeout_seconds = timeout / 1000.0
        max_output_size = 30000  # Default: 30000 characters

        try:
            if background:
                # Background mode: use E2B SDK's background=True to get a CommandHandle
                handle = self._sandbox.commands.run(
                    cmd=command,
                    background=True,
                    timeout=int(timeout_seconds),
                    cwd=cwd or str(self.work_dir),
                    user=user,
                    envs=envs,
                )
                cmd_handle = cast("_CommandHandleProtocol", handle)
                bg_pid: int = cmd_handle.pid

                # E2B CommandHandle requires iterating events to populate _result.
                # Start a daemon thread to consume events in the background.
                task_info: dict[str, Any] = {
                    "handle": handle,
                    "command": command,
                    "start_time": start_time,
                    "finished": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                }

                def _consume_events(h: Any, info: dict[str, Any]) -> None:
                    try:
                        for stdout_chunk, stderr_chunk, _pty in h:
                            if stdout_chunk is not None:
                                info["stdout"] += stdout_chunk
                            if stderr_chunk is not None:
                                info["stderr"] += stderr_chunk
                    except StopIteration:
                        pass
                    except Exception as exc:
                        info["error"] = str(exc)
                    finally:
                        # After iteration ends, _result should be populated
                        if h._result is not None:
                            info["exit_code"] = h._result.exit_code
                            info["stdout"] = h._result.stdout or info["stdout"]
                            info["stderr"] = h._result.stderr or info["stderr"]
                            if h._result.exit_code != 0:
                                info["error"] = info.get("error") or f"Command failed with exit code {h._result.exit_code}"
                        info["finished"] = True

                consumer_thread = threading.Thread(
                    target=_consume_events,
                    args=(handle, task_info),
                    daemon=True,
                )
                consumer_thread.start()
                task_info["thread"] = consumer_thread

                self._background_tasks[bg_pid] = task_info
                duration_ms = int((time.time() - start_time) * 1000)
                return CommandResult(
                    status=SandboxStatus.SUCCESS,
                    stdout=f"Background task started (pid: {bg_pid})",
                    stderr="",
                    exit_code=0,
                    duration_ms=duration_ms,
                    background_pid=bg_pid,
                )

            # Foreground mode: execute command directly
            # E2B SDK will raise exception on non-zero exit code
            # 功能说明1：E2B SDK 可能有事件循环问题，添加重试逻辑
            # 功能说明2：如果遇到 "Event loop is closed"，重新连接后重试
            max_retries = 2
            last_error: RuntimeError | None = None
            result: _CommandResultProtocol | None = None
            for retry in range(max_retries + 1):
                try:
                    result = self._sandbox.commands.run(
                        cmd=command,
                        timeout=int(timeout_seconds),
                        cwd=cwd or str(self.work_dir),
                        user=user,
                        envs=envs,
                    )
                    break
                except RuntimeError as e:
                    last_error = e
                    if "Event loop is closed" in str(e) and retry < max_retries:
                        logger.warning(f"Event loop closed error, reconnecting sandbox (retry {retry + 1}/{max_retries})")
                        # 尝试重新连接
                        try:
                            assert Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"
                            connect_opts: dict[str, Any] = {}
                            if self._api_key:
                                connect_opts["api_key"] = self._api_key
                            if self._api_url:
                                connect_opts["api_url"] = self._api_url
                            if not self.sandbox_id:
                                raise SandboxError("Sandbox ID not set; cannot reconnect.")
                            self._sandbox = Sandbox.connect(sandbox_id=self.sandbox_id, **connect_opts)
                            continue
                        except Exception as reconnect_error:
                            logger.error(f"Failed to reconnect sandbox: {reconnect_error}")
                    raise
            else:
                if last_error:
                    raise last_error

            if result is None:
                raise SandboxError("E2B command returned no result after retries.")

            duration_ms = int((time.time() - start_time) * 1000)

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.exit_code

            stdout_truncated = len(stdout) > max_output_size
            stderr_truncated = len(stderr) > max_output_size

            original_stdout_len = len(stdout) if stdout_truncated else None
            original_stderr_len = len(stderr) if stderr_truncated else None

            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=stdout[:max_output_size],
                stderr=stderr[:max_output_size],
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=None if exit_code == 0 else f"Command failed with exit code {exit_code}",
                truncated=stdout_truncated or stderr_truncated,
                original_stdout_length=original_stdout_len,
                original_stderr_length=original_stderr_len,
            )

        except TimeoutException:  # type: ignore
            duration_ms = int((time.time() - start_time) * 1000)
            return CommandResult(
                status=SandboxStatus.TIMEOUT,
                stdout="",
                stderr="",
                exit_code=-1,
                duration_ms=duration_ms,
                error=f"Command timed out after {timeout}ms",
                truncated=False,
            )

        except CommandExitException as e:  # type: ignore
            # CommandExitException has stdout, stderr, exit_code attributes
            duration_ms = int((time.time() - start_time) * 1000)

            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            exit_code = getattr(e, "exit_code", -1)

            stdout_truncated = len(stdout) > max_output_size
            stderr_truncated = len(stderr) > max_output_size

            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout=stdout[:max_output_size],
                stderr=stderr[:max_output_size],
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=f"Command failed with exit code {exit_code}",
                truncated=stdout_truncated or stderr_truncated,
            )

        except Exception as e:
            # Other unexpected exceptions
            duration_ms = int((time.time() - start_time) * 1000)
            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_ms=duration_ms,
                error=f"Command execution error: {str(e)[:200]}",
                truncated=False,
            )

    def get_background_task_status(self, pid: int) -> CommandResult:
        """
        Get the status and output of a background task.

        The consumer thread (started in execute_bash) continuously reads events
        from the E2B CommandHandle and populates task_info["stdout"], ["stderr"],
        ["finished"], ["exit_code"], and ["error"].

        Args:
            pid: The process ID of the background task

        Returns:
            CommandResult with current status and accumulated output
        """
        if pid not in self._background_tasks:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=f"Background task not found: pid={pid}",
                exit_code=-1,
                error=f"Background task not found: pid={pid}",
            )

        task_info = self._background_tasks[pid]
        max_output_size = 30000
        duration_ms = int((time.time() - task_info["start_time"]) * 1000)

        stdout = task_info.get("stdout", "") or ""
        stderr = task_info.get("stderr", "") or ""

        if task_info["finished"]:
            exit_code = task_info["exit_code"]
            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=stdout[:max_output_size],
                stderr=stderr[:max_output_size],
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=task_info.get("error"),
                truncated=len(stdout) > max_output_size or len(stderr) > max_output_size,
                background_pid=pid,
            )

        # Task is still running, return accumulated output so far
        return CommandResult(
            status=SandboxStatus.RUNNING,
            stdout=stdout[:max_output_size],
            stderr=stderr[:max_output_size],
            exit_code=-1,
            duration_ms=duration_ms,
            truncated=len(stdout) > max_output_size or len(stderr) > max_output_size,
            background_pid=pid,
        )

    def kill_background_task(self, pid: int) -> CommandResult:
        """
        Kill a background task.

        Args:
            pid: The process ID of the background task

        Returns:
            CommandResult with the kill operation result
        """
        if pid not in self._background_tasks:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=f"Background task not found: pid={pid}",
                exit_code=-1,
                error=f"Background task not found: pid={pid}",
            )

        task_info = self._background_tasks[pid]
        handle = task_info["handle"]

        try:
            killed = handle.kill()  # type: ignore[union-attr]
            duration_ms = int((time.time() - task_info["start_time"]) * 1000)
            del self._background_tasks[pid]
            return CommandResult(
                status=SandboxStatus.SUCCESS if killed else SandboxStatus.ERROR,
                stdout=f"Background task (pid={pid}) killed successfully" if killed else f"Failed to kill task (pid={pid})",
                exit_code=0 if killed else -1,
                duration_ms=duration_ms,
                background_pid=pid,
            )
        except Exception as e:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=str(e),
                exit_code=-1,
                error=f"Failed to kill background task: {str(e)[:200]}",
                background_pid=pid,
            )

    def execute_code(
        self,
        code: str,
        language: CodeLanguage | str,
        timeout: int | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionResult:
        """
        Execute Python code in the E2B sandbox.

        Args:
            code: The Python code to execute
            language: Programming language (must be "python" or CodeLanguage.PYTHON)
            timeout: Optional timeout in milliseconds (overrides default)
            user: Optional user to run the code as (default: root)
            envs: Optional environment variables to set

        Returns:
            CodeExecutionResult containing execution results and outputs
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        start_time = time.time()

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                return CodeExecutionResult(
                    status=SandboxStatus.ERROR,
                    language=CodeLanguage.PYTHON,
                    error_type="ValueError",
                    error_value=f"Unsupported language: {language}. Only Python is supported.",
                    duration_ms=0,
                )

        if language != CodeLanguage.PYTHON:
            return CodeExecutionResult(
                status=SandboxStatus.ERROR,
                language=language,
                error_type="ValueError",
                error_value=f"Unsupported language: {language}. Only Python is supported.",
                duration_ms=0,
            )

        temp_file_path = None
        try:
            # Create temporary Python file in /tmp (always exists)
            temp_filename = f"tmp_{uuid.uuid4().hex[:8]}.py"
            temp_file_path = f"/tmp/{temp_filename}"

            # Write code to temp file
            self.write_file(temp_file_path, code)

            # Execute the temp file
            result = self.execute_bash(f"python3 {temp_filename}", timeout, cwd="/tmp/", user=user, envs=envs)

            outputs: list[dict[str, Any]] = []
            if result.stdout:
                outputs.append({"type": "stdout", "text": result.stdout})
            if result.stderr:
                outputs.append({"type": "stderr", "text": result.stderr})

            return CodeExecutionResult(
                status=result.status,
                language=language,
                outputs=outputs,
                error_type=None if result.status == SandboxStatus.SUCCESS else "ExecutionError",
                error_value=result.error,
                traceback=[result.stderr] if result.stderr else None,
                duration_ms=result.duration_ms,
                truncated=result.truncated,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to execute code: {e}")
            return CodeExecutionResult(
                status=SandboxStatus.ERROR,
                language=language,
                error_type=type(e).__name__,
                error_value=str(e),
                duration_ms=duration_ms,
            )
        finally:
            # Clean up temp file
            if temp_file_path:
                try:
                    self.execute_bash(f"rm -f {temp_file_path}", timeout=5000)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file from the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox
            encoding: File encoding (default: utf-8)
            binary: Whether to read file in binary mode

        Returns:
            FileOperationResult containing file content
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Resolve to absolute path if relative
            resolved_path = file_path
            if not file_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_path = f"{work_dir}/{file_path}"

            max_output_size = 30000  # Default: 30000 characters

            raw_content = self._sandbox._filesystem.read(resolved_path, format="bytes")  # type: ignore
            content: str | bytes
            if binary:
                content = raw_content
            else:
                content = raw_content.decode(encoding)

            # Get file size
            file_info = self.get_file_info(resolved_path)
            file_size = file_info.size

            truncated = len(content) > max_output_size if isinstance(content, str) else len(content) > max_output_size

            if truncated:
                content = content[:max_output_size]

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                content=content,
                size=file_size,
                truncated=truncated,
            )

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR, file_path=file_path, error=f"Failed to read file: {str(e)}", content=None
            )

    def write_file(
        self,
        file_path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        binary: bool = False,
        create_directories: bool = True,
        user: str | None = None,
    ) -> FileOperationResult:
        """
        Write content to a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox
            content: Content to write (string or bytes)
            encoding: File encoding (default: utf-8)
            binary: Whether to write file in binary mode
            create_directories: Whether to create parent directories if they don't exist
            user: Optional user to run the create_directories command as

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            # Resolve to absolute path if relative
            resolved_path = file_path
            if not file_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_path = f"{work_dir}/{file_path}"

            # Create parent directories if needed
            if create_directories:
                parent_dir = str(Path(resolved_path).parent)
                if parent_dir and parent_dir != ".":
                    self._sandbox.commands.run(
                        cmd=f"mkdir -p {parent_dir}",
                        user=user,
                    )

            # Write file using E2B filesystem API
            if isinstance(content, bytes):
                # E2B expects string or bytes
                self._sandbox._filesystem.write(resolved_path, content)  # type: ignore
            else:
                self._sandbox._filesystem.write(resolved_path, content)  # type: ignore

            # Get file size - use stat command directly to avoid recursion issues
            size = 0
            try:
                result = self._sandbox.commands.run(
                    cmd=f'stat -c "%s" "{resolved_path}"',
                    user=user,
                )
                stdout_text = result.stdout or ""
                if result.exit_code == 0 and stdout_text.strip():
                    size = int(stdout_text.strip())
            except Exception:
                pass

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=resolved_path,
                size=size,
            )

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to write file: {str(e)}",
            )

    def delete_file(self, file_path: str) -> FileOperationResult:
        """
        Delete a file from the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Resolve to absolute path if relative
            resolved_path = file_path
            if not file_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_path = f"{work_dir}/{file_path}"

            # Check if file exists
            if not self.file_exists(resolved_path):
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=resolved_path,
                    error=f"File does not exist: {resolved_path}",
                )

            # Use E2B filesystem remove
            self._sandbox._filesystem.remove(resolved_path)  # type: ignore

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
            )

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to delete file: {str(e)}",
            )

    def list_files(
        self,
        directory_path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileInfo]:
        """
        List files in a directory in the E2B sandbox.

        Args:
            directory_path: Path to the directory in the sandbox
            recursive: Whether to list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of FileInfo objects for matching files
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if directory exists
            if not self.file_exists(directory_path):
                raise SandboxFileError(f"Directory does not exist: {directory_path}")

            # Use E2B filesystem list
            entries = self._sandbox._filesystem.list(directory_path)  # type: ignore

            files: list[FileInfo] = []
            for entry in entries:
                # Detect readable/writable from permissions
                readable = True
                writable = True
                if entry.mode:  # type: ignore
                    # Check owner read permission (bit 8)
                    readable = bool(entry.mode & 0o400)  # type: ignore
                    # Check owner write permission (bit 7)
                    writable = bool(entry.mode & 0o200)  # type: ignore

                # Convert E2B EntryInfo to our FileInfo
                file_info = FileInfo(
                    path=entry.path,  # type: ignore
                    exists=True,
                    is_file=entry.type == FileType.FILE,  # type: ignore
                    is_directory=entry.type == FileType.DIR,  # type: ignore
                    size=entry.size,  # type: ignore
                    mode=entry.mode,  # type: ignore
                    permissions=entry.permissions,  # type: ignore
                    modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),  # type: ignore
                    symlink_target=entry.symlink_target,  # type: ignore
                    readable=readable,
                    writable=writable,
                    encoding=None,  # Skip encoding detection in list for performance
                )
                files.append(file_info)

                # Recursively list subdirectories if requested
                if recursive and file_info.is_directory:
                    try:
                        subdir_files = self.list_files(file_info.path, recursive=True, pattern=pattern)
                        files.extend(subdir_files)
                    except Exception as e:
                        logger.warning(f"Failed to list subdirectory {file_info.path}: {e}")

            # Apply pattern filtering if specified
            if pattern:
                import fnmatch

                files = [f for f in files if fnmatch.fnmatch(Path(f.path).name, pattern)]

            return files

        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {e}")
            raise SandboxFileError(f"Failed to list files: {str(e)}")

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            True if file exists, False otherwise
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            return self._sandbox._filesystem.exists(file_path)  # type: ignore
        except Exception:
            return False

    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileInfo object containing file metadata
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if file exists
            exists = self.file_exists(file_path)

            if not exists:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Use E2B filesystem get_info API
            try:
                entry = self._sandbox._filesystem.get_info(file_path)  # type: ignore
            except Exception:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Detect readable/writable from permissions
            readable = True
            writable = True
            if entry.mode:  # type: ignore
                # Check owner read permission (bit 8)
                readable = bool(entry.mode & 0o400)  # type: ignore
                # Check owner write permission (bit 7)
                writable = bool(entry.mode & 0o200)  # type: ignore

            if entry.type == FileType.FILE:  # type: ignore
                raw_data = self._sandbox._filesystem.read(file_path, format="bytes")  # type: ignore
                encoding = self._detect_file_encoding(bytes(raw_data))
            else:
                encoding = None

            return FileInfo(
                path=file_path,
                exists=True,
                is_file=entry.type == FileType.FILE,  # type: ignore
                is_directory=entry.type == FileType.DIR,  # type: ignore
                size=entry.size,  # type: ignore
                mode=entry.mode,  # type: ignore
                permissions=entry.permissions,  # type: ignore
                modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),  # type: ignore
                symlink_target=entry.symlink_target,  # type: ignore
                readable=readable,
                writable=writable,
                encoding=encoding,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise SandboxFileError(f"Failed to get file info: {str(e)}")

    def create_directory(self, directory_path: str, parents: bool = True, user: str | None = None) -> bool:
        """
        Create a directory in the E2B sandbox.

        Args:
            directory_path: Path to the directory to create
            parents: Whether to create parent directories if they don't exist

        Returns:
            True if directory created successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            # Use mkdir command
            # Note: Use /home/user as cwd since the target directory may not exist yet
            cmd = f"mkdir -p {directory_path}" if parents else f"mkdir {directory_path}"
            result = self._sandbox.commands.run(
                cmd=cmd,
                cwd="/home/user",
                user=user,
            )

            if result.exit_code != 0:
                raise SandboxFileError(f"Failed to create directory: {result.stderr}")

            return True

        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise SandboxFileError(f"Failed to create directory: {str(e)}")

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
    ) -> FileOperationResult:
        """
        Edit a file by replacing old_string with new_string.

        Supports three operations:
        1. CREATE: Set old_string to empty string to create a new file
        2. UPDATE: Provide both old_string and new_string to update existing content
        3. REMOVE_CONTENT: Set new_string to empty string to remove the old_string content

        Args:
            file_path: Path to the file to edit
            old_string: String to replace (empty for file creation)
            new_string: Replacement string (empty for content removal)

        Returns:
            FileOperationResult containing operation status and details
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            file_exists = self.file_exists(file_path)

            # Determine operation type
            if old_string == "" and new_string != "":
                operation = "CREATE"
            elif old_string != "" and new_string == "":
                operation = "REMOVE_CONTENT"
            else:
                operation = "UPDATE"

            # Validate operation
            if operation == "CREATE" and file_exists:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File already exists: {file_path}. Use UPDATE operation instead.",
                )

            if operation != "CREATE" and not file_exists:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}. Use CREATE operation instead.",
                )

            # Read original content
            if file_exists:
                read_result = self.read_file(file_path, binary=False)
                if read_result.status != SandboxStatus.SUCCESS:
                    return read_result
                original_content = read_result.content
            else:
                original_content = ""

            assert isinstance(original_content, str), f"Unexpected content type: {type(original_content)}"

            # Validate string matching for UPDATE/REMOVE operations
            if operation != "CREATE":
                if old_string not in original_content:
                    # Try to normalize common escape sequence issues from LLM
                    def _normalize_escape_sequences(value: str) -> str:
                        return (
                            value.replace("\\\\n", "\n")
                            .replace("\\n", "\n")
                            .replace("\\\\t", "\t")
                            .replace("\\t", "\t")
                            .replace("\\\\r", "\r")
                            .replace("\\r", "\r")
                        )

                    normalized_old_string = _normalize_escape_sequences(old_string)
                    normalized_new_string = _normalize_escape_sequences(new_string)

                    if normalized_old_string != old_string and normalized_old_string in original_content:
                        # Use normalized version
                        old_string = normalized_old_string
                        new_string = normalized_new_string
                    else:
                        return FileOperationResult(
                            status=SandboxStatus.ERROR,
                            file_path=file_path,
                            error=f"String to replace not found in file: {file_path}",
                        )

                matches = original_content.count(old_string)
                if matches > 1:
                    return FileOperationResult(
                        status=SandboxStatus.ERROR,
                        file_path=file_path,
                        error=(
                            f"Found {matches} matches of the string to replace. "
                            "For safety, this tool only supports replacing exactly one occurrence at a time. "
                            "Add more lines of context to your edit and try again."
                        ),
                    )

            # Apply the edit
            if operation == "CREATE":
                updated_content = new_string
            elif operation == "REMOVE_CONTENT":
                updated_content = original_content.replace(old_string, "", 1)
            else:
                updated_content = original_content.replace(old_string, new_string, 1)

            # Write the updated content
            write_result = self.write_file(file_path, updated_content)

            return write_result

        except Exception as e:
            logger.error(f"Failed to edit file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to edit file: {str(e)}",
            )

    def glob(self, pattern: str, recursive: bool = True, user: str | None = None) -> list[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '*.py', '**/*.txt')
            recursive: Whether to search recursively (default: True)

        Returns:
            List of file paths matching the pattern
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            # Use find command with pattern matching
            if recursive:
                # Handle ** glob pattern by converting to find command
                if "**" in pattern:
                    # Remove ** and extract the file pattern
                    file_pattern = pattern.replace("**/", "").replace("**", "")
                    if file_pattern.startswith("/"):
                        file_pattern = file_pattern[1:]
                    search_dir = "."
                    cmd = f'find "{search_dir}" -type f -name "{file_pattern}"'
                elif "/" in pattern:
                    search_dir, file_pattern = pattern.rsplit("/", 1)
                    search_dir = search_dir or "."
                    cmd = f'find "{search_dir}" -name "{file_pattern}"'
                else:
                    search_dir = "."
                    file_pattern = pattern
                    cmd = f'find "{search_dir}" -name "{file_pattern}"'
            else:
                cmd = f'ls -1 "{pattern}" 2>/dev/null || true'
            result = self._sandbox.commands.run(
                cmd=cmd,
                cwd=str(self.work_dir),
                user=user,
            )

            if result.exit_code != 0 and result.exit_code != 1:
                raise SandboxFileError(f"Glob command failed: {result.stderr}")

            # Parse output
            stdout_text = result.stdout or ""
            matches = [line.strip() for line in stdout_text.split("\n") if line.strip()]

            return sorted(matches)

        except Exception as e:
            logger.error(f"Failed to glob pattern {pattern}: {e}")
            raise SandboxFileError(f"Failed to glob: {str(e)}")

    def upload_file(
        self,
        local_path: str,
        sandbox_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Upload a file from the local filesystem to the E2B sandbox.

        Args:
            local_path: Path to the file on the local filesystem
            sandbox_path: Destination path in the sandbox
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            local_file = Path(local_path)

            if not local_file.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=sandbox_path,
                    error=f"Source file does not exist: {local_path}",
                )

            # Read local file
            with open(local_file, "rb") as f:
                content = f.read()

            # Write to sandbox
            return self.write_file(sandbox_path, content, binary=True, create_directories=create_directories)

        except Exception as e:
            logger.error(f"Failed to upload file from {local_path} to {sandbox_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=sandbox_path,
                error=f"Failed to upload file: {str(e)}",
            )

    def download_file(
        self,
        sandbox_path: str,
        local_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Download a file from the E2B sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the file in the sandbox
            local_path: Destination path on the local filesystem
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Read from sandbox
            read_result = self.read_file(sandbox_path, binary=True)

            if read_result.status != SandboxStatus.SUCCESS:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=local_path,
                    error=f"Failed to read file from sandbox: {read_result.error}",
                )

            # Write to local filesystem
            local_file = Path(local_path)

            if create_directories:
                local_file.parent.mkdir(parents=True, exist_ok=True)

            with open(local_file, "wb") as f:
                content = read_result.content
                if content is None:
                    return FileOperationResult(
                        status=SandboxStatus.ERROR,
                        file_path=local_path,
                        error="Empty content returned from sandbox read",
                    )
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
                    f.write(bytes(content))

            file_size = local_file.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=local_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to download file from {sandbox_path} to {local_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=local_path,
                error=f"Failed to download file: {str(e)}",
            )

    def upload_directory(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> bool:
        """
        Upload a directory from the local filesystem to the E2B sandbox.

        Args:
            local_path: Path to the directory on the local filesystem
            sandbox_path: Destination path in the sandbox

        Returns:
            True if directory uploaded successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            local_dir = Path(local_path)

            if not local_dir.exists():
                raise SandboxFileError(f"Source directory does not exist: {local_path}")

            if not local_dir.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {local_path}")

            # Create destination directory
            self.create_directory(sandbox_path, parents=True)

            # Upload all files recursively
            for item in local_dir.rglob("*"):
                if item.is_file():
                    # Calculate relative path
                    rel_path = item.relative_to(local_dir)
                    dest_path = f"{sandbox_path}/{rel_path}"

                    # Upload file
                    result = self.upload_file(str(item), dest_path, create_directories=True)

                    if result.status != SandboxStatus.SUCCESS:
                        logger.warning(f"Failed to upload {item}: {result.error}")

            return True

        except Exception as e:
            logger.error(f"Failed to upload directory from {local_path} to {sandbox_path}: {e}")
            raise SandboxFileError(f"Failed to upload directory: {str(e)}")

    def download_directory(
        self,
        sandbox_path: str,
        local_path: str,
    ) -> bool:
        """
        Download a directory from the E2B sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the directory in the sandbox
            local_path: Destination path on the local filesystem

        Returns:
            True if directory downloaded successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if source directory exists
            if not self.file_exists(sandbox_path):
                raise SandboxFileError(f"Source directory does not exist: {sandbox_path}")

            # Create local directory
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)

            # List all files recursively
            files = self.list_files(sandbox_path, recursive=True)

            # Resolve sandbox_path to absolute path for comparison
            resolved_sandbox_path = sandbox_path
            if not sandbox_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_sandbox_path = f"{work_dir}/{sandbox_path}"

            # Download all files
            for file_info in files:
                if file_info.is_file:
                    # Calculate relative path using absolute paths
                    file_abs_path = Path(file_info.path)
                    sandbox_abs_path = Path(resolved_sandbox_path)

                    try:
                        rel_path = file_abs_path.relative_to(sandbox_abs_path)
                    except ValueError:
                        # If relative_to fails, try to extract the relative part manually
                        file_path_str = str(file_abs_path)
                        sandbox_path_str = str(sandbox_abs_path)
                        if file_path_str.startswith(sandbox_path_str):
                            rel_path = Path(file_path_str[len(sandbox_path_str) :].lstrip("/"))
                        else:
                            logger.warning(f"Cannot determine relative path for {file_info.path}")
                            continue

                    dest_path = local_dir / rel_path

                    # Download file
                    result = self.download_file(file_info.path, str(dest_path), create_directories=True)

                    if result.status != SandboxStatus.SUCCESS:
                        logger.warning(f"Failed to download {file_info.path}: {result.error}")

            return True

        except Exception as e:
            logger.error(f"Failed to download directory from {sandbox_path} to {local_path}: {e}")
            raise SandboxFileError(f"Failed to download directory: {str(e)}")


@dataclass(kw_only=True)
class E2BSandboxManager(BaseSandboxManager[E2BSandbox]):
    """
    Manager for E2B sandbox lifecycle and configuration.

    This class handles:
    - Creating and configuring E2B sandboxes
    - Connecting to existing sandboxes
    - Persisting and loading sandbox state
    - Managing sandbox lifecycle (start/stop)
    """

    # E2B configuration fields
    _work_dir: str = field(default="/home/user")
    template: str = field(default_factory=lambda: os.getenv("E2B_TEMPLATE", "base"))
    timeout: int = field(default_factory=lambda: int(os.getenv("E2B_TIMEOUT", "300")))
    api_key: str | None = field(default_factory=lambda: os.getenv("E2B_API_KEY"))
    api_url: str | None = field(default_factory=lambda: os.getenv("E2B_API_URL"))
    force_http: bool = field(default_factory=lambda: os.getenv("E2B_FORCE_HTTP") in ["true", "True", "1"])
    metadata: dict[str, str] = field(default_factory=lambda: {})
    envs: dict[str, str] = field(default_factory=lambda: {})

    def start(self, session_manager: Any, user_id: str, session_id: str, sandbox_config: dict[str, Any]) -> E2BSandbox:
        """
        Start an E2B sandbox for a session.

        Args:
            session_manager: Session manager instance
            user_id: User ID
            session_id: Session ID
            sandbox_config: Sandbox configuration dict. If contains sandbox_id, domain,
                and envd_access_token, will connect to existing sandbox instead of creating new.

        Returns:
            Configured and started E2BSandbox instance
        """
        assert E2B_AVAILABLE and Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"

        # Priority 1: Check if sandbox_config contains an existing sandbox to connect to
        # This is used by NexAU Cloud Agent Runtime which manages sandbox lifecycle externally
        config_sandbox_id = sandbox_config.get("sandbox_id")
        config_domain = sandbox_config.get("sandbox_domain")
        config_envd_token = sandbox_config.get("envd_access_token")

        if config_sandbox_id and config_domain and config_envd_token:
            try:
                logger.info(f"Connecting to existing sandbox from config: {config_sandbox_id[:16]}...")

                # Build sandbox URL (HTTP for internal K8s communication)
                sandbox_url = f"http://49983-{config_sandbox_id}.{config_domain}"

                # Create sandbox kwargs from config
                sandbox_kwargs = extract_dataclass_init_kwargs(E2BSandbox, sandbox_config)
                if "_work_dir" not in sandbox_kwargs:
                    sandbox_kwargs["_work_dir"] = "/home/user"
                sandbox = E2BSandbox(**sandbox_kwargs)
                sandbox.set_api_credentials(self.api_key, self.api_url)

                # Connect using E2B SDK with explicit URL via ConnectionConfig
                connection_config_cls = _get_connection_config_class()

                envd_version_value = sandbox.envd_version
                if envd_version_value is None:
                    envd_version_value = sandbox_config.get("envd_version")
                parsed_envd_version: Version | None = None
                if isinstance(envd_version_value, str):
                    try:
                        parsed_envd_version = Version(envd_version_value)
                    except Exception as e:
                        logger.warning(f"Invalid envd_version '{envd_version_value}': {e}. Skipping version pin.")
                        parsed_envd_version = None
                elif isinstance(envd_version_value, Version):
                    parsed_envd_version = envd_version_value
                sandbox_init_kwargs: dict[str, Any] = {
                    "sandbox_id": config_sandbox_id,
                    "sandbox_domain": config_domain,
                    "envd_access_token": config_envd_token,
                    "traffic_access_token": None,
                    "connection_config": connection_config_cls(
                        sandbox_url=sandbox_url,
                        extra_sandbox_headers={"X-Access-Token": config_envd_token},
                    ),
                }
                if parsed_envd_version is not None:
                    sandbox_init_kwargs["envd_version"] = parsed_envd_version

                e2b_sandbox = Sandbox(**sandbox_init_kwargs)

                sandbox.sandbox = e2b_sandbox
                sandbox.sandbox_id = config_sandbox_id

                logger.info(f"Connected to existing sandbox: {config_sandbox_id[:16]}... at {sandbox_url}")

                self._instance = sandbox
                return sandbox

            except Exception as e:
                logger.warning(f"Failed to connect to sandbox from config: {e}. Will try session state or create new.")

        # Priority 2: Load existing sandbox state from session_manager if available
        sandbox_state = self.load_sandbox_state(session_manager, user_id, session_id)

        # Try to restore from saved state
        if sandbox_state and sandbox_state.get("sandbox_id"):
            try:
                logger.info(f"Attempting to restore E2B sandbox from state: {sandbox_state.get('sandbox_id')}")

                # Create sandbox instance from saved state
                sandbox_kwargs = extract_dataclass_init_kwargs(E2BSandbox, sandbox_state)
                sandbox = E2BSandbox(**sandbox_kwargs)
                sandbox.set_api_credentials(self.api_key, self.api_url)

                if not sandbox.sandbox_id:
                    raise SandboxError("Sandbox ID not found in state, failed to restore.")

                # Try to reconnect to existing E2B sandbox
                connect_opts: dict[str, Any] = {"api_key": self.api_key}
                if self.api_url:
                    connect_opts["api_url"] = self.api_url

                e2b_sandbox = Sandbox.connect(sandbox_id=sandbox.sandbox_id, **connect_opts)
                sandbox.sandbox = e2b_sandbox

                # 功能说明：reconnect 时也要 patch URL 为 HTTP
                if self.force_http:
                    try:
                        envd_url = getattr(e2b_sandbox, "_SandboxBase__envd_api_url", None)
                        if envd_url and envd_url.startswith("https://"):
                            http_url = envd_url.replace("https://", "http://")
                            e2b_sandbox._SandboxBase__envd_api_url = http_url  # type: ignore[attr-defined]
                            import httpx

                            connection_config = e2b_sandbox.connection_config
                            e2b_sandbox._envd_api = httpx.Client(
                                base_url=http_url,
                                transport=e2b_sandbox._transport,
                                headers=connection_config.sandbox_headers,
                            )
                            logger.info(f"Patched reconnected sandbox envd URL to HTTP: {http_url}")
                    except Exception as e:
                        logger.warning(f"Failed to patch sandbox URL to HTTP on reconnect: {e}")

                logger.info(f"Successfully reconnected to E2B sandbox: {sandbox.sandbox_id}")

                self._instance = sandbox

                return sandbox

            except Exception as e:
                logger.warning(f"Failed to restore sandbox from state: {e}. Creating new sandbox.")

        # Create new sandbox
        logger.info(f"Creating new E2B sandbox with template: {self.template}")

        # Build create options
        create_opts: dict[str, Any] = {
            "template": self.template,
            "timeout": self.timeout,
            "api_key": self.api_key,
        }

        if self.metadata:
            create_opts["metadata"] = self.metadata
        if self.envs:
            create_opts["envs"] = self.envs
        if self.api_url:
            create_opts["api_url"] = self.api_url
        # Note: force_http is NOT passed to SDK - E2B SDK doesn't accept it
        # HTTP patching is done after sandbox creation (see below)
        # Disable E2B auto_pause; runtime controls lifecycle explicitly.

        # Disable auto_pause if pstatus_after_run is set as none (for RL training scenarios)
        # This prevents E2B from automatically pausing the sandbox during long operations
        create_opts["auto_pause"] = sandbox_config.get("status_after_run", "pause") == "pause"

        # Create E2B sandbox via SDK
        e2b_sandbox = Sandbox.beta_create(**create_opts)  # type: ignore

        sandbox_kwargs = extract_dataclass_init_kwargs(E2BSandbox, sandbox_config)
        if "_work_dir" not in sandbox_kwargs:
            sandbox_kwargs["_work_dir"] = "/home/user"

        # Create our wrapper instance
        sandbox = E2BSandbox(**sandbox_kwargs)
        sandbox.set_api_credentials(self.api_key, self.api_url)

        sandbox.sandbox = e2b_sandbox
        sandbox.sandbox_id = e2b_sandbox.sandbox_id

        # 功能说明1：patch sandbox URL 强制使用 HTTP，避免 SSL 错误
        # 功能说明2：K8s 内部服务不使用 SSL，需要覆盖 E2B SDK 的默认 HTTPS
        # 功能说明3：获取 envd API URL 并确保使用 http:// 前缀
        if self.force_http:
            try:
                # 获取 sandbox 内部的 envd API URL
                envd_url = getattr(e2b_sandbox, "_SandboxBase__envd_api_url", None)
                if envd_url and envd_url.startswith("https://"):
                    # 替换为 HTTP
                    http_url = envd_url.replace("https://", "http://")
                    e2b_sandbox._SandboxBase__envd_api_url = http_url  # type: ignore[attr-defined]

                    # 重新创建内部的 httpx 客户端
                    import httpx

                    connection_config = e2b_sandbox.connection_config
                    e2b_sandbox._envd_api = httpx.Client(
                        base_url=http_url,
                        transport=e2b_sandbox._transport,
                        headers=connection_config.sandbox_headers,
                    )
                    logger.info(f"Patched sandbox envd URL to HTTP: {http_url}")
            except Exception as e:
                logger.warning(f"Failed to patch sandbox URL to HTTP: {e}")

        logger.info(f"E2B sandbox created with ID: {sandbox.sandbox_id}")

        # Ensure work_dir exists (it may not exist if user configured a custom path)
        try:
            sandbox.create_directory(str(sandbox.work_dir))
            logger.debug(f"Work directory ensured: {sandbox.work_dir}")
        except Exception as e:
            logger.warning(f"Failed to create work directory {sandbox.work_dir}: {e}")

        # Persist sandbox state
        self.persist_sandbox_state(session_manager, user_id, session_id, sandbox)

        self._instance = sandbox

        return sandbox

    def stop(self) -> bool:
        """
        Stop the E2B sandbox for a session.
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"

        instance: E2BSandbox | None = self._instance

        if not instance:
            logger.info("No sandbox instance found, nothing to stop")
            return False

        try:
            if instance.sandbox:
                # Kill the sandbox to destroy the container
                instance.sandbox.kill()
                logger.info(f"E2B sandbox {instance.sandbox_id} destroyed")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy E2B sandbox: {e}")
            return False

    def pause(self) -> bool:
        """
        Pause the E2B sandbox for a session.
        """
        try:
            instance = self._instance
            if not instance or not instance.sandbox:
                logger.warning("No E2B sandbox instance to pause; skipping pause")
                return False
            instance.sandbox.beta_pause()
            return True
        except Exception as e:
            logger.error(f"Failed to pause E2B sandbox: {e}")
            return False
        return True

    def is_running(self) -> bool:
        if self._instance is None or self._instance.sandbox is None:
            return False
        try:
            return self._instance.sandbox.is_running()
        except Exception as e:
            logger.error(f"Failed to check E2B sandbox status: {e}")
            return False
