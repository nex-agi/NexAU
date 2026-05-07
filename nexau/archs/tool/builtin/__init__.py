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
# limitations in the License.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .background_task_manage_tool import background_task_manage_tool
    from .file_tools.apply_patch import apply_patch
    from .file_tools.glob_tool import glob
    from .file_tools.list_directory import list_directory
    from .file_tools.read_file import read_file
    from .file_tools.read_many_files import read_many_files
    from .file_tools.replace import replace
    from .file_tools.search_file_content import search_file_content
    from .file_tools.write_file import write_file
    from .mcp_client import (
        MCPClient,
        MCPManager,
        MCPServerConfig,
        MCPTool,
        get_mcp_manager,
        initialize_mcp_tools,
        sync_initialize_mcp_tools,
    )
    from .multiedit_tool import multiedit_tool
    from .session_tools.ask_user import ask_user
    from .session_tools.complete_task import complete_task
    from .session_tools.save_memory import save_memory
    from .session_tools.write_todos import write_todos
    from .shell_tools.run_shell_command import run_shell_command
    from .web_tools.google_web_search import google_web_search
    from .web_tools.web_fetch import web_fetch

__all__ = [
    "background_task_manage_tool",
    "multiedit_tool",
    "read_file",
    "write_file",
    "replace",
    "apply_patch",
    "glob",
    "list_directory",
    "read_many_files",
    "search_file_content",
    "run_shell_command",
    "google_web_search",
    "web_fetch",
    "write_todos",
    "complete_task",
    "save_memory",
    "ask_user",
    "MCPClient",
    "MCPManager",
    "MCPTool",
    "MCPServerConfig",
    "get_mcp_manager",
    "initialize_mcp_tools",
    "sync_initialize_mcp_tools",
]


def _cache_export(name: str, value: object) -> object:
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Lazily resolve built-in tools by family to keep leaf imports cheap."""

    if name == "background_task_manage_tool":
        from .background_task_manage_tool import background_task_manage_tool

        return _cache_export(name, background_task_manage_tool)
    if name == "multiedit_tool":
        from .multiedit_tool import multiedit_tool

        return _cache_export(name, multiedit_tool)
    if name == "read_file":
        from .file_tools.read_file import read_file

        return _cache_export(name, read_file)
    if name == "write_file":
        from .file_tools.write_file import write_file

        return _cache_export(name, write_file)
    if name == "replace":
        from .file_tools.replace import replace

        return _cache_export(name, replace)
    if name == "apply_patch":
        from .file_tools.apply_patch import apply_patch

        return _cache_export(name, apply_patch)
    if name == "glob":
        from .file_tools.glob_tool import glob

        return _cache_export(name, glob)
    if name == "list_directory":
        from .file_tools.list_directory import list_directory

        return _cache_export(name, list_directory)
    if name == "read_many_files":
        from .file_tools.read_many_files import read_many_files

        return _cache_export(name, read_many_files)
    if name == "search_file_content":
        from .file_tools.search_file_content import search_file_content

        return _cache_export(name, search_file_content)
    if name == "run_shell_command":
        from .shell_tools.run_shell_command import run_shell_command

        return _cache_export(name, run_shell_command)
    if name == "google_web_search":
        from .web_tools.google_web_search import google_web_search

        return _cache_export(name, google_web_search)
    if name == "web_fetch":
        from .web_tools.web_fetch import web_fetch

        return _cache_export(name, web_fetch)
    if name == "write_todos":
        from .session_tools.write_todos import write_todos

        return _cache_export(name, write_todos)
    if name == "complete_task":
        from .session_tools.complete_task import complete_task

        return _cache_export(name, complete_task)
    if name == "save_memory":
        from .session_tools.save_memory import save_memory

        return _cache_export(name, save_memory)
    if name == "ask_user":
        from .session_tools.ask_user import ask_user

        return _cache_export(name, ask_user)
    if name == "MCPClient":
        from .mcp_client import MCPClient

        return _cache_export(name, MCPClient)
    if name == "MCPManager":
        from .mcp_client import MCPManager

        return _cache_export(name, MCPManager)
    if name == "MCPTool":
        from .mcp_client import MCPTool

        return _cache_export(name, MCPTool)
    if name == "MCPServerConfig":
        from .mcp_client import MCPServerConfig

        return _cache_export(name, MCPServerConfig)
    if name == "get_mcp_manager":
        from .mcp_client import get_mcp_manager

        return _cache_export(name, get_mcp_manager)
    if name == "initialize_mcp_tools":
        from .mcp_client import initialize_mcp_tools

        return _cache_export(name, initialize_mcp_tools)
    if name == "sync_initialize_mcp_tools":
        from .mcp_client import sync_initialize_mcp_tools

        return _cache_export(name, sync_initialize_mcp_tools)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
