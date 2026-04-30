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
Local sandbox implementation using native Python commands.

This is a naive implementation that executes commands directly on the local system
without any isolation. It is intended for development and testing purposes only.
For production use, consider using proper sandboxing solutions like E2B or Docker.
"""

import glob as glob_module
import logging
import os
import shlex
import shutil
import stat as stat_module
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, override

from nexau.archs.platform.path_helpers import get_local_bash_tool_results_dir, get_local_temp_root
from nexau.archs.platform.process_compat import graceful_kill
from nexau.archs.platform.shell_backend import ShellBackend, WindowsCmdBackend, WindowsPowerShellBackend, create_shell_backend

from .base_sandbox import (
    BaseSandbox,
    BaseSandboxManager,
    BashHeredocBlock,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    SandboxConfig,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
    contains_heredoc,
    extract_dataclass_init_kwargs,
    parse_single_bash_heredoc,
    smart_truncate_output,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LocalSandbox(BaseSandbox):
    """
    Local sandbox implementation that executes commands directly on the host system.

    This implementation uses subprocess for command execution and standard file operations.

    Warning: This implementation does NOT provide any security isolation and should
    only be used for development and testing purposes.
    """

    _shell_backend: ShellBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._shell_backend = create_shell_backend()

    def get_temp_dir(self) -> str:
        """Return the local host temp directory managed by NexAU."""
        return str(get_local_temp_root())

    def to_shell_path(self, path: str | Path) -> str:
        """Convert a local native path into the active shell backend format."""
        return self._shell_backend.format_path_for_shell(path)

    def get_python_command(self) -> str:
        """Return the shell-safe Python interpreter command for LocalSandbox."""
        return self._shell_backend.format_executable_for_shell(sys.executable)

    def _ensure_working_directory(self) -> Path:
        """
        Ensure the working directory exists.

        Returns:
            Path to the working directory
        """
        if self.work_dir is None:
            raise SandboxError("work_dir is not set")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.work_dir)

    @staticmethod
    def _graceful_kill(process: subprocess.Popen[bytes], grace_period: float = 5.0) -> None:
        """Gracefully terminate a process using the platform compatibility layer."""
        graceful_kill(process, grace_period)

    def _prepare_output_dir(self, command: str) -> str:
        """Create an output directory and write command.txt.

        stdout.txt and stderr.txt are created by the OS via file-descriptor
        redirection (passed to ``subprocess.Popen``), so this method only sets
        up the directory and the command metadata file.

        Returns:
            The absolute output directory path.
        """
        output_dir = str(get_local_bash_tool_results_dir() / uuid.uuid4().hex[:8])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/command.txt").write_text(command, encoding="utf-8")
        return output_dir

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path to an absolute path.

        If the path is relative, it will be resolved against the sandbox working directory.
        If the path is absolute, it will be used as-is.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Absolute Path object
        """
        p = Path(path)
        if p.is_absolute():
            return p
        else:
            if self.work_dir is None:
                raise SandboxError("work_dir is not set")
            return self.work_dir / p

    def _build_local_envs(self, per_call_envs: dict[str, str] | None = None) -> dict[str, str] | None:
        """Build environment variables for local subprocess execution.

        Merges os.environ + self.envs + per_call_envs.
        Returns None if no custom envs are set (subprocess inherits parent env).

        Args:
            per_call_envs: Optional per-call environment variables

        Returns:
            Merged envs dict including os.environ, or None to inherit parent env
        """
        merged = self._merge_envs(per_call_envs)
        if merged is None:
            return None
        # For local subprocess, we must include os.environ as base
        return {**os.environ, **merged}

    @override
    def prepare_shell_command(self, command: str, script_dir: str | None = None) -> str:
        """Prepare shell command for the active local backend.

        RFC-0019: PowerShell backend-aware heredoc handling

        PowerShell does not understand Bash heredoc syntax. Common ``cat``
        heredocs are rewritten through a UTF-8 payload file; complex heredocs
        fail before execution with guidance to use PowerShell here-strings or
        opt into Git Bash.
        """
        if not contains_heredoc(command):
            return command

        if isinstance(self._shell_backend, WindowsPowerShellBackend):
            return self._prepare_powershell_heredoc(command, script_dir)

        if isinstance(self._shell_backend, WindowsCmdBackend):
            raise SandboxError(self._unsupported_windows_heredoc_message("cmd.exe"))

        return super().prepare_shell_command(command, script_dir)

    @override
    def scriptify_heredoc(self, command: str, script_dir: str | None = None) -> str:
        """Compatibility wrapper for backend-aware command preparation."""
        return self.prepare_shell_command(command, script_dir)

    def _prepare_powershell_heredoc(self, command: str, script_dir: str | None) -> str:
        """Rewrite supported Bash ``cat`` heredocs into PowerShell commands."""
        heredoc = parse_single_bash_heredoc(command)
        if heredoc is None:
            raise SandboxError(self._unsupported_windows_heredoc_message("PowerShell"))

        mode, target = self._parse_simple_cat_heredoc_target(heredoc)
        if mode == "unsupported":
            raise SandboxError(self._unsupported_windows_heredoc_message("PowerShell"))

        target_script_dir = script_dir if script_dir is not None else self.get_temp_dir()
        payload_path = f"{target_script_dir}/_nexau_heredoc_payload_{uuid.uuid4().hex[:8]}.txt"
        self.write_file(payload_path, heredoc.body, create_directories=True)
        logger.info(
            "[sandbox] PowerShell heredoc payload prepared (%d bytes) – wrote to %s",
            len(heredoc.body.encode("utf-8", errors="replace")),
            payload_path,
        )

        payload_literal = self._quote_powershell_literal(self._shell_backend.format_path_for_shell(payload_path))
        read_payload = f"$__nexau_payload = [System.IO.File]::ReadAllText({payload_literal}, [System.Text.Encoding]::UTF8)"
        if mode == "stdout":
            return f"{read_payload}; [Console]::Out.Write($__nexau_payload)"

        if target is None:
            raise SandboxError(self._unsupported_windows_heredoc_message("PowerShell"))

        target_literal = self._quote_powershell_literal(target)
        write_method = "AppendAllText" if mode == "append" else "WriteAllText"
        utf8_encoding = "$__nexau_utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false"
        return f"{read_payload}; {utf8_encoding}; [System.IO.File]::{write_method}({target_literal}, $__nexau_payload, $__nexau_utf8)"

    def _parse_simple_cat_heredoc_target(self, heredoc: BashHeredocBlock) -> tuple[str, str | None]:
        """Return PowerShell output mode and target for supported ``cat`` heredocs."""
        prefix_tokens = self._split_heredoc_command_part(heredoc.command_prefix)
        suffix_tokens = self._split_heredoc_command_part(heredoc.command_suffix)
        if prefix_tokens is None or suffix_tokens is None or not prefix_tokens or prefix_tokens[0] != "cat":
            return "unsupported", None

        prefix_tail = prefix_tokens[1:]
        if not prefix_tail and not suffix_tokens:
            return "stdout", None

        if prefix_tail and suffix_tokens:
            return "unsupported", None

        redirect_tokens = prefix_tail if prefix_tail else suffix_tokens
        if len(redirect_tokens) != 2 or redirect_tokens[0] not in {">", ">>"}:
            return "unsupported", None

        target = redirect_tokens[1]
        if not self._is_safe_literal_heredoc_target(target):
            return "unsupported", None

        mode = "append" if redirect_tokens[0] == ">>" else "write"
        return mode, target

    @staticmethod
    def _split_heredoc_command_part(value: str) -> list[str] | None:
        """Split a simple heredoc command fragment with shell-style quoting."""
        if not value:
            return []
        try:
            return shlex.split(value, posix=True)
        except ValueError:
            return None

    @staticmethod
    def _is_safe_literal_heredoc_target(target: str) -> bool:
        """Return True for literal path tokens safe to pass to PowerShell."""
        if not target or target in {"-", ".", ".."}:
            return False
        forbidden_chars = {"$", "`", "(", ")", "|", "&", ";", "<", ">", "\n", "\r"}
        return all(char not in forbidden_chars for char in target)

    @staticmethod
    def _quote_powershell_literal(value: str) -> str:
        """Quote a literal for PowerShell single-quoted strings."""
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _unsupported_windows_heredoc_message(shell_name: str) -> str:
        """Return clear RFC-0019 guidance for unsupported Bash heredocs."""
        return (
            f"Unsupported Bash heredoc form under {shell_name}. "
            "Rewrite the command using PowerShell syntax. For simple multi-line text, "
            "a here-string is acceptable; for exact UTF-8 file content without BOM, "
            "use .NET WriteAllText with UTF8Encoding($false). "
            "Alternatively, set NEXAU_WINDOWS_SHELL_BACKEND=git-bash."
        )

    @override
    def execute_shell(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
    ) -> CommandResult:
        """
        Execute a shell command through the active local shell backend.

        Stdout and stderr are always saved to temporary files (stdout.txt, stderr.txt).
        If the combined output exceeds the threshold, the returned stdout/stderr are
        smart-truncated with hints to the full files.

        Args:
            command: The shell command to execute
            timeout: Optional timeout in milliseconds (overrides default)
            cwd: Optional working directory
            user: Optional user to run the command as (not available in LocalSandbox)
            envs: Optional environment variables
            background: Optional flag to run the command in the background

        Returns:
            CommandResult containing execution results
        """
        if user is not None:
            logger.warning(f"User {user} is not used in local sandbox.")

        start_time = time.time()
        work_dir = self._ensure_working_directory()

        if timeout is None:
            timeout = 120000  # Default: 2 minutes

        timeout_seconds = timeout / 1000.0

        try:
            # 1. 按当前 shell backend 预处理命令（例如 heredoc 脚本化 / PowerShell 安全重写）
            command = self.prepare_shell_command(command)
            launch_config = self._shell_backend.build_launch_config(command)

            if background:
                # 1. 创建输出目录，stdout/stderr 直接重定向到文件
                output_dir = self._prepare_output_dir(command)
                stdout_path = f"{output_dir}/stdout.txt"
                stderr_path = f"{output_dir}/stderr.txt"

                fout = open(stdout_path, "wb")
                ferr = open(stderr_path, "wb")

                # Background mode: redirect stdout/stderr to files, no PIPE
                process = subprocess.Popen(
                    launch_config.argv,
                    stdin=subprocess.DEVNULL,
                    stdout=fout,
                    stderr=ferr,
                    cwd=cwd or str(work_dir),
                    env=self._build_local_envs(envs),
                    creationflags=launch_config.creationflags,
                    start_new_session=launch_config.start_new_session,
                )
                bg_pid = process.pid

                task_info: dict[str, Any] = {
                    "process": process,
                    "command": command,
                    "start_time": start_time,
                    "finished": False,
                    "exit_code": -1,
                    "error": None,
                    "std_output_dir": output_dir,
                    "stdout_file": fout,
                    "stderr_file": ferr,
                }

                def _wait_process(proc: subprocess.Popen[bytes], info: dict[str, Any]) -> None:
                    try:
                        exit_code = proc.wait()
                        info["exit_code"] = exit_code
                        if exit_code != 0:
                            info["error"] = f"Command failed with exit code {exit_code}"
                    except Exception as exc:
                        info["error"] = str(exc)
                    finally:
                        # 进程结束后关闭文件句柄，确保数据刷盘
                        for f in (info.get("stdout_file"), info.get("stderr_file")):
                            if f:
                                try:
                                    f.close()
                                except Exception:
                                    pass
                        info["finished"] = True

                wait_thread = threading.Thread(target=_wait_process, args=(process, task_info), daemon=True)
                wait_thread.start()
                task_info["wait_thread"] = wait_thread

                self._background_tasks[bg_pid] = task_info
                duration_ms = int((time.time() - start_time) * 1000)

                stdout_msg = f"Background task started (pid: {bg_pid})\nOutput will be saved to {output_dir}/"

                return CommandResult(
                    status=SandboxStatus.SUCCESS,
                    stdout=stdout_msg,
                    stderr="",
                    exit_code=0,
                    duration_ms=duration_ms,
                    background_pid=bg_pid,
                    output_dir=output_dir,
                    stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                    stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
                )

            # Foreground mode: 直接重定向 stdout/stderr 到文件
            output_dir = self._prepare_output_dir(command)
            stdout_path = f"{output_dir}/stdout.txt"
            stderr_path = f"{output_dir}/stderr.txt"

            fout = open(stdout_path, "wb")
            ferr = open(stderr_path, "wb")

            try:
                process = subprocess.Popen(
                    launch_config.argv,
                    stdin=subprocess.DEVNULL,
                    stdout=fout,
                    stderr=ferr,
                    cwd=cwd or str(work_dir),
                    env=self._build_local_envs(envs),
                    creationflags=launch_config.creationflags,
                    start_new_session=launch_config.start_new_session,
                )

                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._graceful_kill(process)
                    fout.close()
                    ferr.close()

                    duration_ms = int((time.time() - start_time) * 1000)
                    stdout_raw = Path(stdout_path).read_bytes().decode("utf-8", errors="replace")
                    stderr_raw = Path(stderr_path).read_bytes().decode("utf-8", errors="replace")

                    # 智能截断
                    t_stdout, t_stderr, was_truncated, o_out, o_err = smart_truncate_output(
                        stdout_raw,
                        stderr_raw,
                        output_dir,
                        threshold=self.output_char_threshold,
                        head_chars=self.truncate_head_chars,
                        tail_chars=self.truncate_tail_chars,
                    )

                    return CommandResult(
                        status=SandboxStatus.TIMEOUT,
                        stdout=t_stdout,
                        stderr=t_stderr,
                        exit_code=process.returncode or -1,
                        duration_ms=duration_ms,
                        error=f"Command timed out after {timeout}ms",
                        truncated=was_truncated,
                        original_stdout_length=o_out,
                        original_stderr_length=o_err,
                        output_dir=output_dir,
                        stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                        stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
                    )
            finally:
                # 确保文件句柄关闭（正常退出路径）
                if not fout.closed:
                    fout.close()
                if not ferr.closed:
                    ferr.close()

            duration_ms = int((time.time() - start_time) * 1000)

            # 从文件读取完整输出
            stdout_raw = Path(stdout_path).read_bytes().decode("utf-8", errors="replace")
            stderr_raw = Path(stderr_path).read_bytes().decode("utf-8", errors="replace")

            # 智能截断
            t_stdout, t_stderr, was_truncated, orig_stdout_len, orig_stderr_len = smart_truncate_output(
                stdout_raw,
                stderr_raw,
                output_dir,
                threshold=self.output_char_threshold,
                head_chars=self.truncate_head_chars,
                tail_chars=self.truncate_tail_chars,
            )

            return CommandResult(
                status=SandboxStatus.SUCCESS if process.returncode == 0 else SandboxStatus.ERROR,
                stdout=t_stdout,
                stderr=t_stderr,
                exit_code=process.returncode,
                duration_ms=duration_ms,
                error=None if process.returncode == 0 else f"Command failed with exit code {process.returncode}",
                truncated=was_truncated,
                original_stdout_length=orig_stdout_len,
                original_stderr_length=orig_stderr_len,
                output_dir=output_dir,
                stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to execute bash command: {e}")
            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout="",
                stderr="",
                exit_code=-1,
                duration_ms=duration_ms,
                error=f"Execution failed: {str(e)}",
                truncated=False,
            )

    @override
    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
    ) -> CommandResult:
        """Deprecated legacy alias for execute_shell."""
        return self.execute_shell(
            command,
            timeout=timeout,
            cwd=cwd,
            user=user,
            envs=envs,
            background=background,
        )

    @override
    def get_background_task_status(self, pid: int) -> CommandResult:
        """
        Get the status and output of a background task.

        读取 output_dir 下的 stdout.txt / stderr.txt 获取输出（文件由 OS 重定向写入）。

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
        duration_ms = int((time.time() - task_info["start_time"]) * 1000)
        output_dir: str | None = task_info.get("std_output_dir")

        # 从文件读取输出（进程仍在写入时也可读取已刷盘部分）
        stdout = ""
        stderr = ""
        if output_dir:
            try:
                stdout = Path(f"{output_dir}/stdout.txt").read_bytes().decode("utf-8", errors="replace")
            except Exception:
                pass
            try:
                stderr = Path(f"{output_dir}/stderr.txt").read_bytes().decode("utf-8", errors="replace")
            except Exception:
                pass

        # 智能截断
        if output_dir:
            t_stdout, t_stderr, was_truncated, o_out, o_err = smart_truncate_output(
                stdout,
                stderr,
                output_dir,
                threshold=self.output_char_threshold,
                head_chars=self.truncate_head_chars,
                tail_chars=self.truncate_tail_chars,
            )
        else:
            t_stdout, t_stderr, was_truncated, o_out, o_err = stdout, stderr, False, None, None

        if task_info["finished"]:
            exit_code = task_info["exit_code"]
            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=t_stdout,
                stderr=t_stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=task_info.get("error"),
                truncated=was_truncated,
                original_stdout_length=o_out,
                original_stderr_length=o_err,
                background_pid=pid,
                output_dir=output_dir,
                stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
                stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
            )

        # Task is still running, return accumulated output so far
        return CommandResult(
            status=SandboxStatus.RUNNING,
            stdout=t_stdout,
            stderr=t_stderr,
            exit_code=-1,
            duration_ms=duration_ms,
            truncated=was_truncated,
            original_stdout_length=o_out,
            original_stderr_length=o_err,
            background_pid=pid,
            output_dir=output_dir,
            stdout_file=f"{output_dir}/stdout.txt" if output_dir else None,
            stderr_file=f"{output_dir}/stderr.txt" if output_dir else None,
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
        process: subprocess.Popen[bytes] = task_info["process"]

        try:
            self._graceful_kill(process)
            # 关闭文件句柄（如果还没被 _wait_process 关闭）
            for f in (task_info.get("stdout_file"), task_info.get("stderr_file")):
                if f and not f.closed:
                    try:
                        f.close()
                    except Exception:
                        pass
            # 等待 wait_thread 结束
            t = task_info.get("wait_thread")
            if t is not None:
                t.join(timeout=2)
            duration_ms = int((time.time() - task_info["start_time"]) * 1000)
            del self._background_tasks[pid]
            return CommandResult(
                status=SandboxStatus.SUCCESS,
                stdout=f"Background task (pid={pid}) killed successfully",
                exit_code=0,
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
        Execute Python code in the sandbox.

        Args:
            code: The Python code to execute
            language: Programming language (must be "python" or CodeLanguage.PYTHON)
            timeout: Optional timeout in milliseconds (overrides default)
            user: Optional user to run the command as (not available in LocalSandbox)
            envs: Optional environment variables to set

        Returns:
            CodeExecutionResult containing execution results and outputs
        """
        if user is not None:
            logger.warning(f"User {user} is not used in local sandbox.")

        start_time = time.time()

        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                return CodeExecutionResult(
                    status=SandboxStatus.ERROR,
                    language=CodeLanguage.PYTHON,
                    error_type="ValueError",
                    error_value=f"Unsupported language: {language}.",
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

        temp_file = None
        try:
            # Create temporary Python file in work_dir
            import tempfile

            work_path = self.work_dir if self.work_dir else Path.cwd()

            # Create temp file in work_dir
            fd, temp_file = tempfile.mkstemp(suffix=".py", dir=str(work_path), text=True)
            try:
                # Write code to temp file
                os.write(fd, code.encode("utf-8"))
            finally:
                os.close(fd)

            # Execute the temp file
            result = self.execute_shell(
                f"{self.get_python_command()} {shlex.quote(Path(temp_file).name)}",
                timeout,
                envs=envs,
            )

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
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")

    @override
    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file from the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)
            encoding: File encoding (default: utf-8)
            binary: Whether to read file in binary mode

        Returns:
            FileOperationResult containing file content
        """
        try:
            full_path = self._resolve_path(file_path)

            if not full_path.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}",
                )

            file_size = full_path.stat().st_size

            content: str | bytes
            if binary:
                with open(full_path, "rb") as f:
                    content = f.read()
            else:
                with open(full_path, encoding=encoding) as f:
                    content = f.read()

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                content=content,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to read file: {str(e)}",
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
        Write content to a file in the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)
            content: Content to write (string or bytes)
            encoding: File encoding (default: utf-8)
            binary: Whether to write file in binary mode
            create_directories: Whether to create parent directories if they don't exist
            user: Optional user to run the create_directories command as (not available in LocalSandbox)

        Returns:
            FileOperationResult containing operation status
        """
        if user is not None:
            logger.warning(f"User {user} is not used in local sandbox.")

        try:
            full_path = self._resolve_path(file_path)

            if create_directories:
                full_path.parent.mkdir(parents=True, exist_ok=True)

            if binary:
                with open(full_path, "wb") as f:
                    f.write(content if isinstance(content, bytes) else content.encode(encoding))
            else:
                with open(full_path, "w", encoding=encoding) as f:
                    f.write(content if isinstance(content, str) else content.decode(encoding))

            file_size = full_path.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                size=file_size,
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
        Delete a file from the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)

        Returns:
            FileOperationResult containing operation status
        """
        try:
            full_path = self._resolve_path(file_path)

            if not full_path.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}",
                )

            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()

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
        List files in a directory in the sandbox.

        Args:
            directory_path: Path to the directory in the sandbox (relative to working directory or absolute)
            recursive: Whether to list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of FileInfo objects for matching files
        """
        try:
            dir_path = self._resolve_path(directory_path)

            if not dir_path.exists():
                raise SandboxFileError(f"Directory does not exist: {directory_path}")

            if not dir_path.is_dir():
                raise SandboxFileError(f"Path is not a directory: {directory_path}")

            files: list[FileInfo] = []

            if pattern:
                if recursive:
                    glob_pattern = f"**/{pattern}"
                else:
                    glob_pattern = pattern
                paths = dir_path.glob(glob_pattern)
            else:
                if recursive:
                    paths = dir_path.rglob("*")
                else:
                    paths = dir_path.iterdir()

            for path in paths:
                try:
                    stat_result = path.stat()
                    files.append(
                        FileInfo(
                            path=str(path),
                            exists=True,
                            is_file=path.is_file(),
                            is_directory=path.is_dir(),
                            size=stat_result.st_size if path.is_file() else 0,
                            modified_time=datetime.fromtimestamp(stat_result.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                            readable=os.access(path, os.R_OK),
                            writable=os.access(path, os.W_OK),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to get info for {path}: {e}")
                    continue

            return files

        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {e}")
            raise SandboxFileError(f"Failed to list files: {str(e)}")

    @override
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)

        Returns:
            True if file exists, False otherwise
        """
        try:
            return self._resolve_path(file_path).exists()
        except Exception:
            return False

    @override
    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)

        Returns:
            FileInfo object containing file metadata
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            stat_result = path.stat()

            if path.is_file():
                with open(path, "rb") as f:
                    raw_data = f.read()
                    encoding = self._detect_file_encoding(raw_data)
            else:
                encoding = None

            return FileInfo(
                path=file_path,
                exists=True,
                is_file=path.is_file(),
                is_directory=path.is_dir(),
                size=stat_result.st_size,
                mode=stat_result.st_mode,
                permissions=stat_module.filemode(stat_result.st_mode)[1:],
                modified_time=datetime.fromtimestamp(stat_result.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                readable=os.access(path, os.R_OK),
                writable=os.access(path, os.W_OK),
                encoding=encoding,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise SandboxFileError(f"Failed to get file info: {str(e)}")

    @override
    def create_directory(self, directory_path: str, parents: bool = True, user: str | None = None) -> bool:
        """
        Create a directory in the sandbox.

        Args:
            directory_path: Path to the directory to create (relative to working directory or absolute)
            parents: Whether to create parent directories if they don't exist

        Returns:
            True if directory created successfully
        """
        try:
            path = self._resolve_path(directory_path)

            path.mkdir(parents=parents, exist_ok=True)
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
            file_path: Path to the file to edit (relative to working directory or absolute)
            old_string: String to replace (empty for file creation)
            new_string: Replacement string (empty for content removal)

        Returns:
            FileOperationResult containing operation status and details
        """
        try:
            full_path = self._resolve_path(file_path)
            file_exists = full_path.exists()

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
                with open(full_path, "rb") as f:
                    raw_data = f.read()
                    encoding = self._detect_file_encoding(raw_data)
                    original_content = raw_data.decode(encoding)
            else:
                original_content = ""
                encoding = "utf-8"

            # Validate string matching for UPDATE/REMOVE operations
            if operation != "CREATE":
                if old_string not in original_content:
                    # Try to normalize common escape sequence issues from LLM
                    # LLM might double-escape sequences like \\n when it should be \n
                    normalized_old_string = old_string.replace("\\\\n", "\\n").replace("\\\\t", "\\t").replace("\\\\r", "\\r")
                    normalized_new_string = new_string.replace("\\\\n", "\\n").replace("\\\\t", "\\t").replace("\\\\r", "\\r")

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

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the updated content
            with open(full_path, "w", encoding=encoding, newline="") as f:
                f.write(updated_content)

            file_size = full_path.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to edit file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to edit file: {str(e)}",
            )

    @override
    def glob(
        self,
        pattern: str,
        recursive: bool = True,
        user: str | None = None,
    ) -> list[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '*.py', '**/*.txt')
            recursive: Whether to search recursively (default: True)

        Returns:
            List of file paths matching the pattern (relative to working directory)
        """
        try:
            # If pattern is absolute, use it directly
            # Otherwise, resolve it against working directory
            pattern_path = Path(pattern)
            if self.work_dir is None:
                raise SandboxError("work_dir is not set")
            if pattern_path.is_absolute():
                search_pattern = str(pattern_path)
            else:
                search_pattern = str(Path(self.work_dir) / pattern)

            # Use glob with recursive support
            matches = glob_module.glob(search_pattern, recursive=recursive)

            # Convert absolute paths to relative paths (relative to work_dir)
            result: list[str] = []
            for match in matches:
                match_path = Path(match)
                try:
                    # Try to make it relative to work_dir
                    rel_path = match_path.relative_to(self.work_dir)
                    result.append(str(rel_path))
                except ValueError:
                    # If not under work_dir, return absolute path
                    result.append(str(match_path))

            return sorted(result)

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
        Upload a file from the local filesystem to the sandbox.

        Args:
            local_path: Path to the file on the local filesystem
            sandbox_path: Destination path in the sandbox (relative to working directory or absolute)
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        try:
            src = Path(local_path)
            dst = self._resolve_path(sandbox_path)

            if not src.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=sandbox_path,
                    error=f"Source file does not exist: {local_path}",
                )

            if create_directories:
                dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)

            file_size = dst.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=sandbox_path,
                size=file_size,
            )

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
        Download a file from the sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the file in the sandbox (relative to working directory or absolute)
            local_path: Destination path on the local filesystem
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        try:
            src = self._resolve_path(sandbox_path)
            dst = Path(local_path)

            if not src.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=local_path,
                    error=f"Source file does not exist: {sandbox_path}",
                )

            if create_directories:
                dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)

            file_size = dst.stat().st_size

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
        Upload a directory from the local filesystem to the sandbox.

        Args:
            local_path: Path to the directory on the local filesystem
            sandbox_path: Destination path in the sandbox (relative to working directory or absolute)

        Returns:
            True if directory uploaded successfully
        """
        try:
            src = Path(local_path)
            dst = self._resolve_path(sandbox_path)

            if not src.exists():
                raise SandboxFileError(f"Source directory does not exist: {local_path}")

            if not src.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {local_path}")

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(src, dst)
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
        Download a directory from the sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the directory in the sandbox (relative to working directory or absolute)
            local_path: Destination path on the local filesystem

        Returns:
            True if directory downloaded successfully
        """
        try:
            src = self._resolve_path(sandbox_path)
            dst = Path(local_path)

            if not src.exists():
                raise SandboxFileError(f"Source directory does not exist: {sandbox_path}")

            if not src.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {sandbox_path}")

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(src, dst)
            return True

        except Exception as e:
            logger.error(f"Failed to download directory from {sandbox_path} to {local_path}: {e}")
            raise SandboxFileError(f"Failed to download directory: {str(e)}")


@dataclass(kw_only=True)
class LocalSandboxManager(BaseSandboxManager[LocalSandbox]):
    """
    Manager for local sandbox lifecycle and configuration.

    This class handles:
    - Creating and configuring local sandboxes
    - Persisting and loading sandbox state
    - Managing sandbox lifecycle (start/stop)
    """

    @override
    def start(self, session_manager: Any, user_id: str, session_id: str, sandbox_config: SandboxConfig) -> LocalSandbox:
        """
        Start a local sandbox for a session.

        Args:
            session_manager: Session manager instance
            user_id: User ID
            session_id: Session ID
            sandbox_config: Typed sandbox configuration

        Returns:
            Configured and started LocalSandbox instance
        """
        # Load existing sandbox state if available
        sandbox_state = self.load_sandbox_state(session_manager, user_id, session_id)

        # Try to restore from saved state
        if sandbox_state and sandbox_state.get("sandbox_id"):
            try:
                logger.info(f"Attempting to restore local sandbox from state: {sandbox_state.get('sandbox_id')}")

                # Create sandbox instance from saved state
                sandbox_kwargs = extract_dataclass_init_kwargs(LocalSandbox, sandbox_state)
                sandbox = LocalSandbox(**sandbox_kwargs)

                logger.info(f"Successfully restored local sandbox: {sandbox.sandbox_id}")

                return sandbox

            except Exception as e:
                logger.warning(f"Failed to restore sandbox from state: {e}. Creating new sandbox.")

        # Create new sandbox
        logger.info(f"Creating new local sandbox with ID: {session_id}")

        sandbox = LocalSandbox(
            work_dir=sandbox_config.work_dir,
            envs=sandbox_config.envs,
            output_char_threshold=sandbox_config.output_char_threshold,
            truncate_head_chars=sandbox_config.truncate_head_chars,
            truncate_tail_chars=sandbox_config.truncate_tail_chars,
        )

        logger.info(f"Local sandbox created with ID: {sandbox.sandbox_id}, work_dir: {sandbox.work_dir}")

        # Persist sandbox state
        self.persist_sandbox_state(session_manager, user_id, session_id, sandbox)

        return sandbox

    @override
    def stop(
        self,
    ) -> bool:
        """
        Stop the local sandbox for a session.

        Note: For local sandboxes, this just logs the stop event.
        The working directory is preserved for future use.
        """
        return True

    @override
    def pause(self) -> bool:
        """
        Stop the local sandbox for a session.

        Note: For local sandboxes, this just logs the pause event.
        The working directory is preserved for future use.
        """
        return True

    @override
    def is_running(self) -> bool:
        return self._instance is not None
