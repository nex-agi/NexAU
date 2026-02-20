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
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, override

if TYPE_CHECKING:
    from e2b import CommandResult as E2BCommandResult
    from e2b import FileType as E2BFileType
    from e2b import Sandbox as E2BRawSandbox
    from e2b.sandbox.filesystem.filesystem import WriteEntry

from packaging.version import Version

from .base_sandbox import (
    BASH_TOOL_RESULTS_BASE_PATH,
    E2B_DEFAULT_WORK_DIR,
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    E2BSandboxConfig,
    FileInfo,
    FileOperationResult,
    SandboxConfig,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
    extract_dataclass_init_kwargs,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# =============================================================================
# E2B SDK dynamic import
# =============================================================================
# e2b is an optional dependency, loaded dynamically via import_module.
# Type annotations use real e2b types imported under TYPE_CHECKING.
# Fallback values only prevent NameError; actual code paths are guarded by E2B_AVAILABLE.

Sandbox: type[E2BRawSandbox] | None = None
FileType: type[E2BFileType] | None = None

try:
    _e2b = import_module("e2b")

    Sandbox = _e2b.Sandbox
    FileType = _e2b.FileType
    _e2b_available = True
except (ImportError, ModuleNotFoundError, AttributeError):
    logger.warning("E2B SDK not installed. Install it with: pip install e2b")
    _e2b_available = False

E2B_AVAILABLE = _e2b_available


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
    _work_dir: str = field(default=E2B_DEFAULT_WORK_DIR)
    envd_version: str | None = field(default=None)
    _api_key: str | None = field(default=None, repr=False)
    _api_url: str | None = field(default=None, repr=False)

    # Unserialized fields
    _sandbox: E2BRawSandbox | None = field(default=None, repr=False, init=False)

    @property
    def sandbox(self) -> E2BRawSandbox | None:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: E2BRawSandbox):
        self._sandbox = sandbox
        self.sandbox_id = sandbox.sandbox_id

    def set_api_credentials(self, api_key: str | None, api_url: str | None) -> None:
        self._api_key = api_key
        self._api_url = api_url

    def _save_output_to_temp_file(
        self,
        command: str,
        stdout: str,
        stderr: str,
        temp_dir: str | None = None,
    ) -> str:
        """
        Save command output to temporary files in the sandbox.
        If temp_dir is provided, overwrites existing files; otherwise creates new directory.

        Args:
            command: The command that was executed
            stdout: Standard output content
            stderr: Standard error content
            temp_dir: Optional existing temp directory path (for streaming updates)

        Returns:
            The directory path where files were saved
        """
        if self._sandbox is None:
            raise SandboxError("Sandbox not started. Call start() first.")

        # Generate unique ID if temp_dir not provided
        if temp_dir is None:
            temp_dir = f"{BASH_TOOL_RESULTS_BASE_PATH}/{uuid.uuid4().hex[:8]}"

        try:
            # Create the directory if it doesn't exist
            self._sandbox.commands.run(f"mkdir -p {temp_dir}")

            # Write/overwrite command to command.txt
            self._sandbox._filesystem.write(f"{temp_dir}/command.txt", command)

            # Write/overwrite stdout to stdout.log
            self._sandbox._filesystem.write(f"{temp_dir}/stdout.log", stdout)

            # Write/overwrite stderr to stderr.log
            self._sandbox._filesystem.write(f"{temp_dir}/stderr.log", stderr)

            return temp_dir
        except Exception as e:
            logger.error(f"Failed to save output to temp file: {e}")
            raise SandboxError(f"Failed to save output to temp file: {e}")

    def _resolve_path(self, path: str, cwd: str | None = None) -> str:
        """Resolve a relative path to an absolute path.

        E2B SaaS envd resolves relative paths server-side, but self-host envd
        (e.g. 0.4.2) does not. This method ensures consistent behavior across
        both environments by always resolving on the client side.

        Args:
            path: File or directory path (absolute or relative).
            cwd: Working directory for resolution. Defaults to self.work_dir.
        """
        if path.startswith("/"):
            return path
        base = cwd or str(self.work_dir)
        return f"{base}/{path}"

    # Transient error patterns that warrant a reconnect + retry
    _TRANSIENT_PATTERNS = (
        "Event loop is closed",
        "Server disconnected",
        "Connection reset",
        "Connection refused",
        "RemoteProtocolError",
    )

    def _is_transient_error(self, exc: Exception) -> bool:
        """Return True if the exception looks like a transient network error."""
        msg = str(exc)
        return any(p in msg for p in self._TRANSIENT_PATTERNS)

    def _reconnect(self) -> None:
        """Attempt to reconnect to the sandbox by sandbox_id."""
        assert Sandbox is not None, "E2B SDK not installed."
        if not self.sandbox_id:
            raise SandboxError("Sandbox ID not set; cannot reconnect.")
        self._sandbox = Sandbox.connect(
            sandbox_id=self.sandbox_id,
            api_key=self._api_key,
            api_url=self._api_url,
        )

    def _retry_on_transient(self, fn: Callable[[], _T], max_retries: int = 2) -> _T:
        """Execute *fn* with automatic reconnect + retry on transient errors.

        Args:
            fn: Zero-arg callable that performs the SDK operation.
            max_retries: How many times to retry after the first failure.

        Returns:
            Whatever *fn* returns on success.

        Raises:
            The original exception if it is not transient or retries are exhausted.
        """
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if not self._is_transient_error(e) or attempt >= max_retries:
                    raise
                logger.warning(
                    "Transient error (attempt %d/%d), reconnecting: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                try:
                    time.sleep(1 * (attempt + 1))
                    self._reconnect()
                except Exception as reconnect_err:
                    logger.error(f"Reconnect failed: {reconnect_err}")
                    raise e from reconnect_err
        # Should never reach here, but satisfy type checker
        raise SandboxError("Retries exhausted")

    @override
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
        """
        Execute a bash command in the E2B sandbox.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in milliseconds (overrides default)
            cwd: Optional working directory
            user: Optional user to run the command as
            envs: Optional environment variables
            background: Optional flag to run the command in the background
            save_output_to_temp_file: Optional flag to save the output (stdout and stderr) to a temporary file

        Returns:
            CommandResult containing execution results
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        from e2b import CommandExitException
        from e2b.exceptions import TimeoutException

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
                # Generate temp directory for streaming if requested
                temp_dir = None
                if save_output_to_temp_file:
                    temp_dir = f"{BASH_TOOL_RESULTS_BASE_PATH}/{uuid.uuid4().hex[:8]}"

                # Background mode: retry on transient network errors
                handle = self._retry_on_transient(
                    lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                        cmd=command,
                        background=True,
                        timeout=int(timeout_seconds),
                        cwd=cwd or str(self.work_dir),
                        user=user,
                        envs=self._merge_envs(envs),
                    )
                )

                bg_pid: int = handle.pid  # background=True → CommandHandle

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
                    "temp_dir": temp_dir,
                }

                def _consume_events(h: Any, info: dict[str, Any]) -> None:
                    try:
                        for stdout_chunk, stderr_chunk, _pty in h:
                            if stdout_chunk is not None:
                                info["stdout"] += stdout_chunk
                            if stderr_chunk is not None:
                                info["stderr"] += stderr_chunk

                            # Overwrite log files with complete output if streaming enabled
                            if save_output_to_temp_file:
                                try:
                                    self._save_output_to_temp_file(
                                        info["command"],
                                        info["stdout"],
                                        info["stderr"],
                                        temp_dir=temp_dir,
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to update streaming log files: {e}")
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

                stdout_msg = f"Background task started (pid: {bg_pid})"
                if save_output_to_temp_file:
                    stdout_msg += f"\nOutput will be saved to {BASH_TOOL_RESULTS_BASE_PATH}/{{unique_id}}/ when task completes"

                return CommandResult(
                    status=SandboxStatus.SUCCESS,
                    stdout=stdout_msg,
                    stderr="",
                    exit_code=0,
                    duration_ms=duration_ms,
                    background_pid=bg_pid,
                )

            # Foreground mode: execute command directly
            # Retry on transient network errors (disconnects, resets, etc.)
            result: E2BCommandResult = self._retry_on_transient(
                lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                    cmd=command,
                    timeout=int(timeout_seconds),
                    cwd=cwd or str(self.work_dir),
                    user=user,
                    envs=self._merge_envs(envs),
                )
            )

            duration_ms = int((time.time() - start_time) * 1000)

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.exit_code

            # Save output to temp file if requested
            temp_file_path = None
            if save_output_to_temp_file:
                try:
                    temp_file_path = self._save_output_to_temp_file(command, stdout, stderr)
                except Exception as e:
                    logger.error(f"Failed to save output to temp file: {e}")

            stdout_truncated = len(stdout) > max_output_size
            stderr_truncated = len(stderr) > max_output_size

            original_stdout_len = len(stdout) if stdout_truncated else None
            original_stderr_len = len(stderr) if stderr_truncated else None

            # Append temp file path info to stdout if saved
            stdout_output = stdout[:max_output_size]
            if temp_file_path:
                stdout_output += f"\n\n[Output saved to: {temp_file_path}]"

            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=stdout_output,
                stderr=stderr[:max_output_size],
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=None if exit_code == 0 else f"Command failed with exit code {exit_code}",
                truncated=stdout_truncated or stderr_truncated,
                original_stdout_length=original_stdout_len,
                original_stderr_length=original_stderr_len,
            )

        except TimeoutException:
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

        except CommandExitException as e:
            # CommandExitException has stdout, stderr, exit_code attributes
            duration_ms = int((time.time() - start_time) * 1000)

            stdout = getattr(e, "stdout", "") or ""
            stderr = getattr(e, "stderr", "") or ""
            exit_code = getattr(e, "exit_code", -1)

            # Save output to temp file if requested
            temp_file_path = None
            if save_output_to_temp_file:
                try:
                    temp_file_path = self._save_output_to_temp_file(command, stdout, stderr)
                except Exception as save_err:
                    logger.error(f"Failed to save output to temp file: {save_err}")

            stdout_truncated = len(stdout) > max_output_size
            stderr_truncated = len(stderr) > max_output_size

            original_stdout_len = len(stdout) if stdout_truncated else None
            original_stderr_len = len(stderr) if stderr_truncated else None

            # Append temp file path info to stdout if saved
            stdout_output = stdout[:max_output_size]
            if temp_file_path:
                stdout_output += f"\n\n[Output saved to: {temp_file_path}]"

            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout=stdout_output,
                stderr=stderr[:max_output_size],
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=f"Command failed with exit code {exit_code}",
                truncated=stdout_truncated or stderr_truncated,
                original_stdout_length=original_stdout_len,
                original_stderr_length=original_stderr_len,
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

    @override
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
        temp_file_path = task_info.get("temp_dir")

        if task_info["finished"]:
            exit_code = task_info["exit_code"]

            # Append temp file path info to stdout if saved
            stdout_output = stdout[:max_output_size]
            if temp_file_path:
                stdout_output += f"\n\n[Output saved to: {temp_file_path}]"

            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=stdout_output,
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

    @override
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
            killed = handle.kill()
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

    @override
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

    @override
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
            resolved_path = self._resolve_path(file_path)

            max_output_size = 30000  # Default: 30000 characters

            raw_content = self._retry_on_transient(
                lambda: self._sandbox._filesystem.read(resolved_path, format="bytes")  # type: ignore[union-attr]
            )
            content: str | bytearray
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
            # Use debug for UnicodeDecodeError - expected when reading binary as text, avoid log spam
            if isinstance(e, UnicodeDecodeError):
                logger.debug("Skipped binary file (cannot decode as text): %s", file_path)
            else:
                logger.error(f"Failed to read file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR, file_path=file_path, error=f"Failed to read file: {str(e)}", content=None
            )

    @override
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
            resolved_path = self._resolve_path(file_path)

            # Create parent directories if needed
            if create_directories:
                parent_dir = str(Path(resolved_path).parent)
                if parent_dir and parent_dir != ".":
                    self._retry_on_transient(
                        lambda: self._sandbox.commands.run(  # type: ignore[union-attr]
                            cmd=f"mkdir -p {parent_dir}",
                            user=user,
                        )
                    )

            # Write file using E2B filesystem API (with retry)
            self._retry_on_transient(
                lambda: self._sandbox._filesystem.write(resolved_path, content, request_timeout=300.0)  # type: ignore[union-attr]
            )

            # Calculate size from content directly (avoid extra round trip)
            if isinstance(content, (bytes, bytearray)):
                size = len(content)
            else:
                size = len(content.encode(encoding))

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

    @override
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
            resolved_path = self._resolve_path(file_path)

            # Check if file exists
            if not self.file_exists(resolved_path):
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=resolved_path,
                    error=f"File does not exist: {resolved_path}",
                )

            # Use E2B filesystem remove
            self._sandbox._filesystem.remove(resolved_path)

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

    @override
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
        assert FileType is not None
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            resolved_path = self._resolve_path(directory_path)

            # Check if directory exists
            if not self.file_exists(resolved_path):
                raise SandboxFileError(f"Directory does not exist: {directory_path}")

            # Use E2B filesystem list
            entries = self._sandbox._filesystem.list(resolved_path)

            files: list[FileInfo] = []
            for entry in entries:
                # Detect readable/writable from permissions
                readable = True
                writable = True
                if entry.mode:
                    # Check owner read permission (bit 8)
                    readable = bool(entry.mode & 0o400)
                    # Check owner write permission (bit 7)
                    writable = bool(entry.mode & 0o200)

                # Convert E2B EntryInfo to our FileInfo
                file_info = FileInfo(
                    path=entry.path,
                    exists=True,
                    is_file=entry.type == FileType.FILE,
                    is_directory=entry.type == FileType.DIR,
                    size=entry.size,
                    mode=entry.mode,
                    permissions=entry.permissions,
                    modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                    symlink_target=entry.symlink_target,
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

    @override
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
            resolved_path = self._resolve_path(file_path)
            return self._sandbox._filesystem.exists(resolved_path)
        except Exception:
            return False

    @override
    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileInfo object containing file metadata
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        assert FileType is not None
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            resolved_path = self._resolve_path(file_path)

            # Check if file exists
            exists = self.file_exists(resolved_path)

            if not exists:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Use E2B filesystem get_info API
            try:
                entry = self._sandbox._filesystem.get_info(resolved_path)
            except Exception:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Detect readable/writable from permissions
            readable = True
            writable = True
            if entry.mode:
                # Check owner read permission (bit 8)
                readable = bool(entry.mode & 0o400)
                # Check owner write permission (bit 7)
                writable = bool(entry.mode & 0o200)

            if entry.type == FileType.FILE:
                raw_data = self._sandbox._filesystem.read(resolved_path, format="bytes")
                encoding = self._detect_file_encoding(bytes(raw_data))
            else:
                encoding = None

            return FileInfo(
                path=file_path,
                exists=True,
                is_file=entry.type == FileType.FILE,
                is_directory=entry.type == FileType.DIR,
                size=entry.size,
                mode=entry.mode,
                permissions=entry.permissions,
                modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                symlink_target=entry.symlink_target,
                readable=readable,
                writable=writable,
                encoding=encoding,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise SandboxFileError(f"Failed to get file info: {str(e)}")

    @override
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
            cmd = f"mkdir -p {directory_path}" if parents else f"mkdir {directory_path}"
            result = self._sandbox.commands.run(
                cmd=cmd,
                cwd=str(self.work_dir),
                user=user,
            )

            if result.exit_code != 0:
                raise SandboxFileError(f"Failed to create directory: {result.stderr}")

            return True

        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise SandboxFileError(f"Failed to create directory: {str(e)}")

    @override
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

    @override
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
        from e2b import CommandExitException

        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        user = user or self.default_user or "user"

        if user not in ("root", "user"):
            raise ValueError(f"User must be 'root' or 'user' for E2B sandbox. But got {user}")

        try:
            # Normalize repeated slashes (e.g. //foo -> /foo) to avoid
            # accidentally computing root '/' as the search base.
            pattern = re.sub(r"/{2,}", "/", pattern)

            # Use find command with pattern matching
            if recursive:
                if "**" in pattern:
                    # Split on the first occurrence of ** to get search_dir and remainder.
                    # e.g. "/home/user/project/**/*.py" → search_dir="/home/user/project", file_pattern="*.py"
                    # e.g. "src/**/*.ts"               → search_dir="src",                file_pattern="*.ts"
                    # e.g. "**/*.py"                   → search_dir=".",                  file_pattern="*.py"
                    # e.g. "/home/user/project/**"     → search_dir="/home/user/project", file_pattern="*"
                    # e.g. "**"                        → search_dir=".",                  file_pattern="*"
                    idx = pattern.index("**")
                    search_dir = pattern[:idx].rstrip("/") or "."
                    remainder = pattern[idx + 2 :].lstrip("/")  # skip "**" and any trailing /
                    # If remainder contains more **, take only the final filename pattern
                    if "**" in remainder:
                        remainder = remainder.rsplit("**/", 1)[-1].lstrip("/")
                    file_pattern = remainder if remainder else "*"
                    cmd = f'find "{search_dir}" -type f -name "{file_pattern}" 2>/dev/null'
                elif "/" in pattern:
                    search_dir, file_pattern = pattern.rsplit("/", 1)
                    search_dir = search_dir or "."
                    # If the directory portion contains wildcards, let find handle
                    # them via -path instead of quoting the dir (which would cause
                    # bash to fail when the glob path doesn't literally exist).
                    if "*" in search_dir or "?" in search_dir:
                        # Find the deepest non-glob ancestor as the search root
                        parts = search_dir.lstrip("/").split("/")
                        root_parts = []
                        for part in parts:
                            if "*" in part or "?" in part:
                                break
                            root_parts.append(part)
                        root_dir = ("/" if search_dir.startswith("/") else "") + "/".join(root_parts) or "."
                        # Reconstruct the full path pattern (strip trailing slash)
                        full_pattern = pattern.rstrip("/")
                        if file_pattern:
                            cmd = f'find "{root_dir}" -path "{full_pattern}" 2>/dev/null'
                        else:
                            cmd = f'find "{root_dir}" -type d -path "{full_pattern}" -print 2>/dev/null'
                    else:
                        cmd = f'find "{search_dir}" -name "{file_pattern}" 2>/dev/null || true'
                else:
                    search_dir = "."
                    file_pattern = pattern
                    cmd = f'find "{search_dir}" -name "{file_pattern}" 2>/dev/null || true'
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

        except CommandExitException as e:
            # find returns exit code 1 when it encounters permission errors
            # (e.g. /proc, /sys) but may still have found valid matches in stdout.
            stdout = getattr(e, "stdout", "") or ""
            matches = [line.strip() for line in stdout.split("\n") if line.strip()]
            return sorted(matches)

        except Exception as e:
            logger.error(f"Failed to glob pattern {pattern}: {e}")
            raise SandboxFileError(f"Failed to glob: {str(e)}")

    @override
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

    @override
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

    @override
    def upload_directory(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> bool:
        """
        Upload a directory from the local filesystem to the E2B sandbox.

        Uses write_files() for batch upload to minimize round trips.

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

            resolved_sandbox_path = self._resolve_path(sandbox_path)

            # 1. Collect all files and directories to create
            files_to_write: list[WriteEntry] = []
            parent_dirs: set[str] = {resolved_sandbox_path}

            for item in local_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(local_dir)
                    dest_path = f"{resolved_sandbox_path}/{rel_path}"
                    parent_dirs.add(str(Path(dest_path).parent))

                    with open(item, "rb") as f:
                        files_to_write.append({"path": dest_path, "data": f.read()})

            if not files_to_write:
                return True

            # 2. Create all parent directories in one shot
            dirs_cmd = " ".join(f'"{d}"' for d in sorted(parent_dirs))
            self._sandbox.commands.run(cmd=f"mkdir -p {dirs_cmd}", user="user")

            # 3. Batch-write all files
            self._sandbox._filesystem.write_files(files_to_write, request_timeout=300.0)

            return True

        except Exception as e:
            logger.error(f"Failed to upload directory from {local_path} to {sandbox_path}: {e}")
            raise SandboxFileError(f"Failed to upload directory: {str(e)}")

    @override
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
            resolved_sandbox_path = self._resolve_path(sandbox_path)

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
    _work_dir: str = field(default=E2B_DEFAULT_WORK_DIR)
    template: str = field(default_factory=lambda: os.getenv("E2B_TEMPLATE", "base"))
    timeout: int = field(default_factory=lambda: int(os.getenv("E2B_TIMEOUT", "300")))
    api_key: str | None = field(default_factory=lambda: os.getenv("E2B_API_KEY"))
    api_url: str | None = field(default_factory=lambda: os.getenv("E2B_API_URL"))
    metadata: dict[str, str] = field(default_factory=lambda: {})
    envs: dict[str, str] = field(default_factory=lambda: {})

    # Keepalive state (not init params)
    _keepalive_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _keepalive_stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def _maybe_rebuild_for_http(
        self,
        e2b_sandbox_raw: E2BRawSandbox,
        sandbox_config: E2BSandboxConfig,
    ) -> E2BRawSandbox:
        """Conditionally rebuild a Sandbox instance for HTTP (self-host).

        If force_http is enabled and the raw sandbox has the required domain
        and access token, rebuilds the instance with an HTTP ConnectionConfig.
        Otherwise returns the original SDK instance unchanged (SaaS path).

        Args:
            e2b_sandbox_raw: Sandbox instance from connect() or beta_create()
            sandbox_config: Sandbox configuration with force_http flag

        Returns:
            The original or rebuilt Sandbox instance
        """
        if not sandbox_config.force_http:
            return e2b_sandbox_raw

        domain = e2b_sandbox_raw.sandbox_domain
        envd_token = getattr(e2b_sandbox_raw, "_envd_access_token", None)

        if domain and envd_token:
            return self._build_sandbox_with_connection_config(
                sandbox_id=e2b_sandbox_raw.sandbox_id,
                domain=domain,
                envd_access_token=envd_token,
                envd_version=e2b_sandbox_raw._envd_version,
            )

        logger.warning(
            f"force_http=True but cannot rebuild sandbox {e2b_sandbox_raw.sandbox_id[:16]}...: "
            f"domain={domain!r}, envd_token={'<set>' if envd_token else None}. "
            "Keeping original instance."
        )
        return e2b_sandbox_raw

    def _build_sandbox_with_connection_config(
        self,
        sandbox_id: str,
        domain: str,
        envd_access_token: str,
        envd_version: Version,
    ) -> E2BRawSandbox:
        """Build an HTTP Sandbox instance using ConnectionConfig (self-host only).

        Ensures all internal SDK components (_envd_api, _filesystem._envd_api, etc.)
        use the HTTP URL and X-Access-Token header from initialization.

        Important: Must reset TransportWithLogger.singleton before creating a new
        Sandbox instance. Otherwise the new instance reuses the httpcore.ConnectionPool
        from beta_create/connect, whose cached HTTPS connections to the API server
        interfere with HTTP streaming connections to envd, causing commands.run to hang.

        Args:
            sandbox_id: E2B sandbox ID
            domain: Sandbox domain for URL construction
            envd_access_token: Access token for envd API authentication
            envd_version: envd version from Sandbox Manager response

        Returns:
            Configured Sandbox instance with HTTP ConnectionConfig
        """
        assert Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"

        from e2b.api.client_sync import TransportWithLogger
        from e2b.connection_config import ConnectionConfig

        sandbox_url = f"http://49983-{sandbox_id}.{domain}"
        connection_config = ConnectionConfig(
            sandbox_url=sandbox_url,
            extra_sandbox_headers={"X-Access-Token": envd_access_token},
        )

        # Reset singleton transport to avoid reusing the connection pool from
        # beta_create/connect. Cached connections in that pool cause subsequent
        # HTTP streaming requests (e.g. commands.run) to hang indefinitely.
        TransportWithLogger.singleton = None

        return Sandbox(  # type: ignore[call-arg]
            sandbox_id=sandbox_id,
            sandbox_domain=domain,
            envd_access_token=envd_access_token,
            traffic_access_token=None,
            connection_config=connection_config,
            envd_version=envd_version,
        )

    @override
    def start(self, session_manager: Any, user_id: str, session_id: str, sandbox_config: SandboxConfig) -> E2BSandbox:
        """
        Start an E2B sandbox for a session.

        Args:
            session_manager: Session manager instance
            user_id: User ID
            session_id: Session ID
            sandbox_config: Typed sandbox configuration (E2BSandboxConfig expected)

        Returns:
            Configured and started E2BSandbox instance
        """
        assert E2B_AVAILABLE and Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"

        if not isinstance(sandbox_config, E2BSandboxConfig):
            raise ValueError(f"E2BSandboxManager requires E2BSandboxConfig, got {type(sandbox_config)}")

        # Override api_key/api_url from config if provided
        if sandbox_config.api_key:
            self.api_key = sandbox_config.api_key
        if sandbox_config.api_url:
            self.api_url = sandbox_config.api_url
        if sandbox_config.template and sandbox_config.template != "base":
            self.template = sandbox_config.template

        # Priority 1: Check if config contains an existing sandbox to connect to
        config_sandbox_id = sandbox_config.sandbox_id

        if config_sandbox_id:
            try:
                logger.info(f"Connecting to existing sandbox from config: {config_sandbox_id[:16]}...")

                # Use Sandbox.connect() — domain/token come from API response
                e2b_sandbox_raw = Sandbox.connect(
                    sandbox_id=config_sandbox_id,
                    timeout=sandbox_config.timeout,
                    api_key=self.api_key,
                    api_url=self.api_url,
                )

                # Extract connection info from response and rebuild with ConnectionConfig
                e2b_sandbox = self._maybe_rebuild_for_http(e2b_sandbox_raw, sandbox_config)
                envd_ver = e2b_sandbox_raw._envd_version

                sandbox = E2BSandbox(_work_dir=sandbox_config.work_dir)
                sandbox.set_api_credentials(self.api_key, self.api_url)
                sandbox.envd_version = str(envd_ver)

                sandbox.sandbox = e2b_sandbox
                sandbox.sandbox_id = config_sandbox_id

                logger.info(f"Connected to existing sandbox: {config_sandbox_id[:16]}...")

                self._instance = sandbox
                self._start_keepalive(config_sandbox_id, sandbox_config.keepalive_interval)
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
                # Same as Priority 1: only self-host needs rebuild
                e2b_sandbox_raw = Sandbox.connect(
                    sandbox_id=sandbox.sandbox_id,
                    api_key=self.api_key,
                    api_url=self.api_url,
                )

                e2b_sandbox = self._maybe_rebuild_for_http(e2b_sandbox_raw, sandbox_config)
                envd_ver = e2b_sandbox_raw._envd_version

                sandbox.envd_version = str(envd_ver)
                sandbox.sandbox = e2b_sandbox

                logger.info(f"Successfully reconnected to E2B sandbox: {sandbox.sandbox_id}")

                self._instance = sandbox
                self._start_keepalive(sandbox.sandbox_id, sandbox_config.keepalive_interval)
                return sandbox

            except Exception as e:
                logger.warning(f"Failed to restore sandbox from state: {e}. Creating new sandbox.")

        # Create new sandbox
        logger.info(f"Creating new E2B sandbox with template: {self.template}")

        # Create E2B sandbox via SDK (Step 1: create to get sandbox_id and connection info)
        # Disable auto_pause if status_after_run is not "pause" (for RL training scenarios)
        e2b_sandbox = Sandbox.beta_create(
            template=self.template,
            timeout=self.timeout,
            api_key=self.api_key,
            api_url=self.api_url,
            metadata=self.metadata or None,
            envs=self.envs or None,
            auto_pause=sandbox_config.status_after_run == "pause",
        )
        created_sandbox_id = e2b_sandbox.sandbox_id
        envd_ver = e2b_sandbox._envd_version

        # Step 2: Self-host needs rebuild with HTTP + X-Access-Token
        e2b_sandbox = self._maybe_rebuild_for_http(e2b_sandbox, sandbox_config)

        # Create our wrapper instance
        sandbox = E2BSandbox(_work_dir=sandbox_config.work_dir)
        sandbox.set_api_credentials(self.api_key, self.api_url)
        sandbox.envd_version = str(envd_ver)

        sandbox.sandbox = e2b_sandbox
        sandbox.sandbox_id = created_sandbox_id

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
        self._start_keepalive(sandbox.sandbox_id, sandbox_config.keepalive_interval)
        return sandbox

    # -------------------------------------------------------------------------
    # Keepalive
    # -------------------------------------------------------------------------

    def _start_keepalive(self, sandbox_id: str | None, interval: int) -> None:
        """Start a daemon thread that periodically calls set_timeout.

        Prevents the idle checker from pausing the sandbox during long agent runs.
        """
        if interval <= 0 or not sandbox_id or Sandbox is None:
            return
        self._keepalive_stop_event.clear()
        # Capture in local vars so the closure doesn't depend on module-level None check
        sandbox_cls = Sandbox
        mgr_timeout = self.timeout
        mgr_api_url = self.api_url
        mgr_api_key = self.api_key

        def _loop() -> None:
            while not self._keepalive_stop_event.wait(interval):
                try:
                    sandbox_cls.set_timeout(
                        sandbox_id,
                        mgr_timeout,
                        api_url=mgr_api_url,
                        api_key=mgr_api_key,
                    )
                    logger.debug(f"Keepalive sent: {sandbox_id[:8]}...")
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if "409" in err_msg or "not running" in err_msg:
                        logger.warning(f"Sandbox paused, auto-resuming: {sandbox_id[:8]}...")
                        try:
                            sandbox_cls.connect(
                                sandbox_id=sandbox_id,
                                timeout=mgr_timeout,
                                api_key=mgr_api_key,
                                api_url=mgr_api_url,
                            )
                            logger.info(f"Sandbox auto-resumed: {sandbox_id[:8]}...")
                        except Exception as resume_err:
                            logger.error(f"Auto-resume failed: {resume_err}")
                    else:
                        logger.warning(f"Keepalive failed: {sandbox_id[:8]}..., error={exc}")

        self._keepalive_thread = threading.Thread(target=_loop, daemon=True)
        self._keepalive_thread.start()
        logger.debug(f"Keepalive thread started: {sandbox_id[:8]}..., interval={interval}s")

    def _stop_keepalive(self) -> None:
        """Stop the keepalive thread if running."""
        if self._keepalive_thread is not None:
            self._keepalive_stop_event.set()
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None
            logger.debug("Keepalive thread stopped")

    @override
    def on_run_complete(self) -> None:
        """Stop keepalive thread when agent execution completes."""
        self._stop_keepalive()

    @override
    def stop(self) -> bool:
        """
        Stop the E2B sandbox for a session.
        """
        self._stop_keepalive()
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"

        instance: E2BSandbox | None = self._instance

        if not instance:
            logger.info("No sandbox instance found, nothing to stop")
            return False

        try:
            if instance.sandbox:
                instance.sandbox.kill(api_key=self.api_key, api_url=self.api_url)
                logger.info(f"E2B sandbox {instance.sandbox_id} destroyed")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy E2B sandbox: {e}")
            return False

    @override
    def pause(self) -> bool:
        """
        Pause the E2B sandbox for a session.
        """
        self._stop_keepalive()
        try:
            instance = self._instance
            if not instance or not instance.sandbox:
                logger.warning("No E2B sandbox instance to pause; skipping pause")
                return False
            # Pass api_key/api_url explicitly — E2B SDK's beta_pause() does not
            # inherit the key used at creation time, so without this the call
            # fails when E2B_API_KEY env var is not set (e.g. self-hosted).
            instance.sandbox.beta_pause(api_key=self.api_key, api_url=self.api_url)
            return True
        except Exception as e:
            logger.error(f"Failed to pause E2B sandbox: {e}")
            return False

    @override
    def is_running(self) -> bool:
        if self._instance is None or self._instance.sandbox is None:
            return False
        try:
            return self._instance.sandbox.is_running()
        except Exception as e:
            logger.error(f"Failed to check E2B sandbox status: {e}")
            return False
