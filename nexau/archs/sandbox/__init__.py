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

"""Sandbox module for secure code execution and file operations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_sandbox import (
        HEREDOC_PATTERN,
        BaseSandbox,
        BaseSandboxConfig,
        BaseSandboxManager,
        CodeExecutionResult,
        CodeLanguage,
        CommandResult,
        E2BSandboxConfig,
        FileInfo,
        FileOperationResult,
        LocalSandboxConfig,
        SandboxConfig,
        SandboxError,
        SandboxExecutionError,
        SandboxFileError,
        SandboxStatus,
        SandboxTimeoutError,
        contains_heredoc,
        extract_dataclass_init_kwargs,
        parse_sandbox_config,
    )
    from .e2b_sandbox import E2BSandbox, E2BSandboxManager
    from .local_sandbox import LocalSandbox, LocalSandboxManager
    from .output_utils import clean_shell_output, collapse_repetitive, resolve_cr, strip_ansi


def _cache_export(name: str, value: object) -> object:
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Lazily import sandbox exports so base types do not pull local/E2B stacks."""

    if name == "BaseSandbox":
        from .base_sandbox import BaseSandbox

        return _cache_export(name, BaseSandbox)
    if name == "SandboxStatus":
        from .base_sandbox import SandboxStatus

        return _cache_export(name, SandboxStatus)
    if name == "BaseSandboxConfig":
        from .base_sandbox import BaseSandboxConfig

        return _cache_export(name, BaseSandboxConfig)
    if name == "SandboxConfig":
        from .base_sandbox import SandboxConfig

        return _cache_export(name, SandboxConfig)
    if name == "LocalSandboxConfig":
        from .base_sandbox import LocalSandboxConfig

        return _cache_export(name, LocalSandboxConfig)
    if name == "E2BSandboxConfig":
        from .base_sandbox import E2BSandboxConfig

        return _cache_export(name, E2BSandboxConfig)
    if name == "parse_sandbox_config":
        from .base_sandbox import parse_sandbox_config

        return _cache_export(name, parse_sandbox_config)
    if name == "CodeLanguage":
        from .base_sandbox import CodeLanguage

        return _cache_export(name, CodeLanguage)
    if name == "CommandResult":
        from .base_sandbox import CommandResult

        return _cache_export(name, CommandResult)
    if name == "CodeExecutionResult":
        from .base_sandbox import CodeExecutionResult

        return _cache_export(name, CodeExecutionResult)
    if name == "FileInfo":
        from .base_sandbox import FileInfo

        return _cache_export(name, FileInfo)
    if name == "FileOperationResult":
        from .base_sandbox import FileOperationResult

        return _cache_export(name, FileOperationResult)
    if name == "SandboxError":
        from .base_sandbox import SandboxError

        return _cache_export(name, SandboxError)
    if name == "SandboxTimeoutError":
        from .base_sandbox import SandboxTimeoutError

        return _cache_export(name, SandboxTimeoutError)
    if name == "SandboxExecutionError":
        from .base_sandbox import SandboxExecutionError

        return _cache_export(name, SandboxExecutionError)
    if name == "SandboxFileError":
        from .base_sandbox import SandboxFileError

        return _cache_export(name, SandboxFileError)
    if name == "BaseSandboxManager":
        from .base_sandbox import BaseSandboxManager

        return _cache_export(name, BaseSandboxManager)
    if name == "extract_dataclass_init_kwargs":
        from .base_sandbox import extract_dataclass_init_kwargs

        return _cache_export(name, extract_dataclass_init_kwargs)
    if name == "HEREDOC_PATTERN":
        from .base_sandbox import HEREDOC_PATTERN

        return _cache_export(name, HEREDOC_PATTERN)
    if name == "contains_heredoc":
        from .base_sandbox import contains_heredoc

        return _cache_export(name, contains_heredoc)
    if name == "LocalSandbox":
        from .local_sandbox import LocalSandbox

        return _cache_export(name, LocalSandbox)
    if name == "LocalSandboxManager":
        from .local_sandbox import LocalSandboxManager

        return _cache_export(name, LocalSandboxManager)
    if name == "E2BSandbox":
        from .e2b_sandbox import E2BSandbox

        return _cache_export(name, E2BSandbox)
    if name == "E2BSandboxManager":
        from .e2b_sandbox import E2BSandboxManager

        return _cache_export(name, E2BSandboxManager)
    if name == "clean_shell_output":
        from .output_utils import clean_shell_output

        return _cache_export(name, clean_shell_output)
    if name == "strip_ansi":
        from .output_utils import strip_ansi

        return _cache_export(name, strip_ansi)
    if name == "resolve_cr":
        from .output_utils import resolve_cr

        return _cache_export(name, resolve_cr)
    if name == "collapse_repetitive":
        from .output_utils import collapse_repetitive

        return _cache_export(name, collapse_repetitive)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseSandbox",
    "LocalSandbox",
    "E2BSandbox",
    "SandboxStatus",
    "BaseSandboxConfig",
    "SandboxConfig",
    "LocalSandboxConfig",
    "E2BSandboxConfig",
    "parse_sandbox_config",
    "CodeLanguage",
    "CommandResult",
    "CodeExecutionResult",
    "FileInfo",
    "FileOperationResult",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxExecutionError",
    "SandboxFileError",
    "LocalSandboxManager",
    "E2BSandboxManager",
    "BaseSandboxManager",
    "extract_dataclass_init_kwargs",
    "HEREDOC_PATTERN",
    "contains_heredoc",
    "clean_shell_output",
    "strip_ansi",
    "resolve_cr",
    "collapse_repetitive",
]
