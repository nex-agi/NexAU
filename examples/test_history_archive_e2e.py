# Copyright (c) Nex-AGI. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""RFC-0021 端到端真实测试: 多轮对话强制触发 llm_summary 压缩 + 归档 + 召回。

策略:
  - SessionManager + InMemoryDatabaseEngine 让多次 agent.run() 共享历史
  - 每次 agent.run() = 1 个 USER iteration
  - llm_summary 策略 + keep_iterations=2: 第 3 个 turn 起就会压缩前面的轮次
  - 最后一轮要求 agent 回忆早期细节, 期望它去 .nexau_history_archive/ 找

运行:
    cd /Users/yiran/Projects/nexau-compact-save-history
    LANGFUSE_SECRET_KEY=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_HOST=... \\
    LLM_MODEL=... LLM_BASE_URL=... LLM_API_KEY=... \\
    uv run python examples/test_history_archive_e2e.py
"""

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from nexau import Agent, AgentConfig
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware, Event
from nexau.archs.main_sub.execution.middleware.context_compaction import ContextCompactionMiddleware
from nexau.archs.session import SessionManager
from nexau.archs.session.orm.memory_engine import InMemoryDatabaseEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_event_counter = {"text": 0, "tool_call": 0, "tool_result": 0, "compaction": 0}


def handler(event: Event) -> None:
    et = type(event).__name__
    if "Compaction" in et:
        _event_counter["compaction"] += 1
        print(f"\n>>> {et}\n")
    elif "ToolCallStart" in et:
        _event_counter["tool_call"] += 1


def main() -> int:
    project_root = Path(__file__).parent.parent

    # 0. sandbox.work_dir 用独立 tempdir 隔离 (不再借用项目根目录).
    #    这样: (a) 归档不污染项目树; (b) ripgrep 不会被项目 .gitignore 屏蔽归档目录;
    #    (c) 跑完自动清理。绝对路径 read_file 仍能读项目源码。
    sandbox_dir = Path(tempfile.mkdtemp(prefix="rfc0021-e2e-"))
    os.environ["SANDBOX_WORK_DIR"] = str(sandbox_dir)
    archive_dir = sandbox_dir / ".nexau_history_archive"

    # 1. 加载 code_agent base config, flatten + 缩小
    code_agent_dir = project_root / "examples" / "code_agent"
    config = AgentConfig.from_yaml(config_path=code_agent_dir / "code_agent.yaml")
    config.sub_agents = None
    config.max_context_tokens = 24000  # 2x: 更接近真实场景, 不再人工卡极限
    config.max_iterations = 100  # 给召回足够空间

    # 2. 接入 ContextCompactionMiddleware: 更宽松的阈值 + 更多保留窗口
    compaction_mw = ContextCompactionMiddleware(
        max_context_tokens=24000,
        auto_compact=True,
        threshold=0.65,
        compaction_strategy="llm_summary",
        keep_iterations=3,
        # RFC-0021
        save_history=True,
    )
    if config.middlewares is None:
        config.middlewares = []
    config.middlewares.append(compaction_mw)

    session_id = f"rfc0021-e2e-{int(time.time())}"
    event_mw = AgentEventsMiddleware(session_id=session_id, on_event=handler)
    config.middlewares.append(event_mw)

    # 3. SessionManager 让多次 agent.run() 共享历史
    engine = InMemoryDatabaseEngine()
    session_manager = SessionManager(engine=engine)

    agent = Agent(
        config=config,
        session_manager=session_manager,
        user_id="rfc0021-tester",
        session_id=session_id,
    )
    sandbox = agent.sandbox_manager.instance
    if sandbox is None:
        print("✗ Sandbox failed to start", file=sys.stderr)
        return 1
    sandbox_root = Path(str(sandbox.work_dir))
    print(f"Sandbox work_dir: {sandbox_root}")
    print(f"Session id:        {session_id}\n")

    # 4. 复杂多轮任务: 真实工作量 — 多文件分析 + 写 markdown 报告 + 召回
    mw_dir = f"{project_root}/nexau/archs/main_sub/execution/middleware"
    rfc_path = f"{project_root}/docs/rfcs/0021-history-archive-on-compaction.md"
    turns = [
        (
            "T1",
            f"Read this file with read_file (limit=400 lines from offset=0): "
            f"{mw_dir}/agent_events_middleware.py"
            f"\n\nThen tell me: what specific Event class names are emitted by this middleware? "
            f"List them in your reply.",
        ),
        (
            "T2",
            f"Read both of these files (limit=400 each, offset=0):\n"
            f"  1. {mw_dir}/llm_failover.py\n"
            f"  2. {mw_dir}/long_tool_output.py\n"
            f"\n\nThen explain in 3-4 sentences how their roles differ in the middleware pipeline.",
        ),
        (
            "T3",
            f"Read these three files in the context_compaction submodule (limit=500 each, offset=0):\n"
            f"  1. {mw_dir}/context_compaction/middleware.py\n"
            f"  2. {mw_dir}/context_compaction/history_archive.py\n"
            f"  3. {mw_dir}/context_compaction/config.py\n"
            f"\n\nExplain in 4-5 sentences how the three files cooperate to deliver the compaction + archive feature.",
        ),
        (
            "T4",
            f"Read these two files (limit=400 each, offset=0):\n"
            f"  1. {rfc_path}\n"
            f"  2. {mw_dir}/context_compaction/compact_stratigies/sliding_window.py\n"
            f"\n\nThen write a markdown report to `./middleware_deep_dive.md` with one section per "
            f"middleware/file analysed across this conversation (T1, T2, T3, T4): name, purpose, "
            f"3 key methods, and how it integrates with neighbours. Make the report at least 60 lines.",
        ),
        (
            "T5-RECALL",
            "Important task: way back in turn 1 we discussed agent_events_middleware.py and you listed "
            "several event class names it emits. Several rounds of compaction have happened since then, "
            "so that detail is almost certainly outside your active context now. "
            "**Recover the exact list of event class names from `.nexau_history_archive/transcript.jsonl`** "
            "—— each line is either a serialized Message or a boundary record `{\"_boundary\": ...}` "
            "marking a compaction round. Use search_file_content (grep) for keywords like "
            "`agent_events_middleware` or specific Event class names, or read_file the transcript directly. "
            "Then call complete_task with the recovered list.",
        ),
    ]

    for tag, msg in turns:
        print("\n" + "=" * 70)
        print(f"### Turn {tag}")
        print("=" * 70)
        print(f"USER: {msg[:200]}{'…' if len(msg) > 200 else ''}\n")
        try:
            response = agent.run(
                message=msg,
                context={
                    "date": get_date(),
                    "username": os.getenv("USER"),
                    "working_directory": str(sandbox_root),
                    "env_content": {
                        "date": get_date(),
                        "username": os.getenv("USER"),
                        "working_directory": str(sandbox_root),
                    },
                },
            )
            text = response if isinstance(response, str) else str(response)
            print(f"\nASSISTANT [{tag}]: {text[:1500]}")
        except Exception as exc:
            print(f"\n✗ Turn {tag} failed: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            break

    # 5. 检查归档
    print("\n" + "=" * 70)
    print("RFC-0021 Archive Inspection")
    print("=" * 70)
    print(f"Archive dir:        {archive_dir}")
    print(f"Exists:             {archive_dir.exists()}")
    print(f"Compaction events:  {_event_counter['compaction']}")
    print(f"Tool calls:         {_event_counter['tool_call']}")

    if not archive_dir.exists():
        print("\n⚠️  归档目录未创建 — 没触发实际移除")
        return 0

    files = sorted(archive_dir.iterdir())
    print(f"\nFiles ({len(files)}):")
    for f in files:
        print(f"  {f.name:30s}  {f.stat().st_size:>8d} bytes")

    # RFC-0021 单文件: transcript.jsonl 内每行是 Message JSON 或 boundary 记录
    transcript_path = archive_dir / "transcript.jsonl"
    if not transcript_path.exists():
        print("\n⚠️  transcript.jsonl 不存在 — 没触发实际压缩或归档关闭")
        return 0

    boundaries: list[dict[str, Any]] = []
    archived_msg_lines: list[str] = []
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        if isinstance(obj, dict) and "_boundary" in obj:
            obj_dict = cast("dict[str, Any]", obj)
            b_val = obj_dict["_boundary"]
            if isinstance(b_val, dict):
                boundaries.append(cast("dict[str, Any]", b_val))
        else:
            archived_msg_lines.append(ln)

    print(f"\ntranscript.jsonl: {len(archived_msg_lines)} archived messages, {len(boundaries)} boundary records")
    for b in boundaries:
        print(
            f"  Round {b['round']:2d}: "
            f"{b['removed_message_count']:>3d} msgs removed, "
            f"tokens {b.get('tokens_before')} -> {b.get('tokens_after')}, "
            f"strategy={b['strategy']}, "
            f"trigger={b['trigger_reason']}"
        )
        preview = (b.get("preview") or "")[:120]
        if preview:
            print(f"          preview: {preview!r}")

    print("\n" + "=" * 70)
    print("Invariants check")
    print("=" * 70)

    def check(label: str, ok: bool, detail: str = "") -> None:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}{(' — ' + detail) if detail else ''}")

    check("至少触发 1 轮压缩", len(boundaries) >= 1, f"rounds={len(boundaries)}")
    if len(boundaries) >= 2:
        check("✨ 多轮压缩被覆盖 (RFC-0021 关键场景)", True, f"rounds={len(boundaries)}")

    # 每行 Message JSON 都能被 Pydantic 反序列化
    from nexau.core.messages import Message
    parseable = True
    for ln in archived_msg_lines:
        try:
            Message.model_validate_json(ln)
        except Exception as exc:
            parseable = False
            print(f"  ✗ Message 反序列化失败: {exc}")
            break
    check(f"全部 {len(archived_msg_lines)} 条归档 Message 都可反序列化", parseable)

    # round 数 = boundary 行数;  removed_count 累计 = 实际归档消息数
    expected_total = sum(b.get("removed_message_count", 0) for b in boundaries)
    check(
        "Σ removed_message_count == 实际归档消息行数",
        expected_total == len(archived_msg_lines),
        f"sum={expected_total}, actual={len(archived_msg_lines)}",
    )

    # Turn 1 内容应在 transcript 中
    found_t1 = "agent_events_middleware" in transcript_path.read_text(encoding="utf-8")
    check("Turn 1 (agent_events_middleware) 内容仍可被 grep 到", found_t1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
