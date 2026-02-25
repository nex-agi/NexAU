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

"""SSE server for NexAU Building Team.

RFC-0002: NexAU 构建团队 SSE 服务器

Starts an HTTP server that exposes team endpoints for the
NexAU agent building workflow (requirements → RFC → build → test).

Usage:
    python examples/nexau_building_team/start_server.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.transports.http import HTTPConfig, SSETransportServer

from event_store import EventStore

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("nexau.archs.main_sub.execution.llm_caller").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


def main() -> None:
    """Start SSE server with NexAU building team support.

    RFC-0002: 启动 NexAU 构建团队 SSE 服务器

    Steps:
    1. 加载所有 agent 配置
    2. 创建 SSE server
    3. 注册 team 配置（leader + 2 candidate roles）
    4. 启动服务器
    """
    # 1. 加载 agent 配置
    leader_config = AgentConfig.from_yaml(SCRIPT_DIR / "leader_agent.yaml")
    rfc_writer_config = AgentConfig.from_yaml(SCRIPT_DIR / "rfc_writer_agent.yaml")
    builder_config = AgentConfig.from_yaml(SCRIPT_DIR / "builder_agent.yaml")

    # 2. 创建事件存储和 SSE server
    event_store = EventStore()
    engine = InMemoryDatabaseEngine()
    server = SSETransportServer(
        engine=engine,
        config=HTTPConfig(host="0.0.0.0", port=8000),
        default_agent_config=leader_config,
        on_stream_event=event_store.append,
        get_history=event_store.get_history,
        count_events=event_store.count,
    )

    # 3. 注册 team 配置
    registry = server.team_registry
    if registry is not None:
        registry.register_config(
            "default",
            leader_config=leader_config,
            candidates={
                "rfc_writer": rfc_writer_config,
                "builder": builder_config,
            },
        )

    # 4. 注册文件浏览 API（直接读取 sandbox 工作目录）
    sandbox_work_dir = Path(
        os.environ.get("SANDBOX_WORK_DIR", "") or (leader_config.sandbox_config.work_dir if leader_config.sandbox_config else os.getcwd())
    ).resolve()

    files_router = APIRouter(prefix="/files", tags=["files"])

    @files_router.get("/tree")
    async def file_tree(path: str = Query(default=".")) -> list[dict[str, object]]:
        """List directory contents for the file browser.

        RFC-0002: 文件树浏览 API

        返回指定目录下的文件和子目录列表，按文件夹优先、文件名排序。
        """
        # 1. 解析并校验路径安全
        target = (sandbox_work_dir / path).resolve()
        if not str(target).startswith(str(sandbox_work_dir)):
            raise HTTPException(status_code=403, detail="Path outside sandbox")
        if not target.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")

        # 2. 列出目录内容
        entries: list[dict[str, object]] = []
        for item in target.iterdir():
            entries.append(
                {
                    "name": item.name,
                    "path": str(item.relative_to(sandbox_work_dir)),
                    "isDirectory": item.is_dir(),
                    "isFile": item.is_file(),
                    "size": item.stat().st_size if item.is_file() else 0,
                }
            )

        # 3. 排序：文件夹优先，然后按名称
        entries.sort(key=lambda x: (not x["isDirectory"], str(x["name"]).lower()))
        return entries

    @files_router.get("/content")
    async def file_content(path: str = Query(...)) -> dict[str, object]:
        """Read file content for the file viewer.

        RFC-0002: 文件内容读取 API

        返回文件内容、大小和语言类型（用于语法高亮）。
        """
        # 1. 解析并校验路径安全
        target = (sandbox_work_dir / path).resolve()
        if not str(target).startswith(str(sandbox_work_dir)):
            raise HTTPException(status_code=403, detail="Path outside sandbox")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # 2. 推断语言类型
        ext_to_lang: dict[str, str] = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".sh": "bash",
            ".bash": "bash",
            ".toml": "toml",
            ".xml": "xml",
            ".sql": "sql",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby",
            ".txt": "plaintext",
        }
        language = ext_to_lang.get(target.suffix.lower(), "plaintext")

        # 3. 读取文件内容（限制 1MB）
        max_size = 1024 * 1024
        file_size = target.stat().st_size
        truncated = file_size > max_size

        try:
            content = target.read_text(encoding="utf-8")
            if truncated:
                content = content[:max_size]
        except UnicodeDecodeError:
            return {
                "path": path,
                "content": "(Binary file — cannot display)",
                "size": file_size,
                "truncated": False,
                "language": "plaintext",
            }

        return {
            "path": path,
            "content": content,
            "size": file_size,
            "truncated": truncated,
            "language": language,
        }

    server.app.include_router(files_router)

    # 5. 注册事件历史 API（用于前端刷新后恢复状态）
    history_router = APIRouter(prefix="/team", tags=["history"])

    @history_router.get("/history")
    async def team_history(
        user_id: str = Query(...),
        session_id: str = Query(...),
        after: int = Query(default=0),
    ) -> list[dict[str, object]]:
        """Return stored team events for history replay.

        前端刷新后调用此接口恢复之前的事件流。
        """
        events = event_store.get_history(user_id, session_id, after=after)
        logger.info("GET /team/history user_id=%s session_id=%s after=%d → %d events", user_id, session_id, after, len(events))
        return events

    @history_router.get("/sessions")
    async def team_sessions(
        user_id: str = Query(...),
    ) -> list[str]:
        """List all session IDs for a user."""
        return event_store.list_sessions(user_id)

    server.app.include_router(history_router)

    # 6. 启动服务器
    logger.info("Starting NexAU Building Team SSE server on http://0.0.0.0:8000")
    logger.info("Sandbox work dir: %s", sandbox_work_dir)
    logger.info("Team endpoints: /team/stream, /team/query, /team/tasks, /team/teammates, /team/history, /team/sessions")
    logger.info("File endpoints: /files/tree, /files/content")
    server.run()


if __name__ == "__main__":
    main()
