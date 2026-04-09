# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""Real LLM E2E tests for micro-compact.

One unified complex query, run under 4 different middleware configurations.
Compare traces to verify compaction behavior.

Requires environment variables:
    LLM_API_KEY         - API key for the LLM provider
    LLM_BASE_URL        - Base URL for the LLM API
    LLM_MODEL           - Model name (default: claude-opus-4-6)
    LANGFUSE_PUBLIC_KEY  - (optional) Langfuse public key for tracing
    LANGFUSE_SECRET_KEY  - (optional) Langfuse secret key for tracing
    LANGFUSE_HOST        - (optional) Langfuse host URL
    MC_SLOW_TOOL_SLEEP   - (optional) Slow tool sleep seconds (default: 5)
    MC_GAP_THRESHOLD     - (optional) Gap threshold in minutes (default: 0.05 = 3s)

Configurations:
    A. No compaction middleware (baseline)
    B. compactable_tools=["read_file","list_directory"], keep_iterations=1, gap=3s
    C. compactable_tools=None (compact all), keep_iterations=1, gap=3s
    D. compactable_tools=["read_file","list_directory"], keep_iterations=2, gap=3s
    E. Same as B but gap_threshold=5min (too high to trigger)
    F. compactable_tools=None (compact all), keep_iterations=2, gap=3s

Query flow (6 steps, causal dependencies force sequential execution):
    Step 1: read_file /tmp/main.py           → get function name      (iter1)
    Step 2: list_directory /tmp              → count files             (iter2)
    Step 3: write_file /tmp/result.txt       → write findings          (iter3)
    Step 4: run_shell_command (SLOW, 5s)     → verify written file     (iter4)
    Step 5: read_file /tmp/main.py again     → re-confirm              (iter5)
    Step 6: list_directory /tmp again        → see result.txt added    (iter6)

    Compaction triggers after step 4 (slow tool). At that point 4 iterations exist.

Expected results:

    |            | iter1      | iter2      | iter3      | iter4    | iter5    | iter6    |
    |            | read_file  | list_dir   | write_file | shell    | read_file| list_dir |
    |------------|------------|------------|------------|----------|----------|----------|
    | A baseline | preserved  | preserved  | preserved  | preserved| preserved| preserved|
    | B filtered | COMPACTED  | COMPACTED  | preserved  | preserved| preserved| preserved|
    | C no-filter| COMPACTED  | COMPACTED  | COMPACTED  | preserved| preserved| preserved|
    | D keep=2   | COMPACTED  | COMPACTED  | preserved* | preserved| preserved| preserved|
    | E hi-thres | preserved  | preserved  | preserved  | preserved| preserved| preserved|
    | F no-f,k=2 | COMPACTED  | COMPACTED  | preserved  | preserved| preserved| preserved|

    * iter3 in D: preserved by keep_iterations=2 AND write_file not in filter (double protection)
    * iter3 in F: preserved by keep_iterations=2 ONLY (no filter, so only keep protects it)
    D vs F: same result for write_file, isolating that keep_iterations=2 alone is sufficient
    A vs E: both no compaction, but A has no middleware, E has middleware with threshold too high
"""

from __future__ import annotations

import os
import time
from typing import Any

from nexau.archs.llm.llm_aggregators.events import (
    CompactionFinishedEvent,
    CompactionStartedEvent,
)
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
)
from nexau.archs.main_sub.execution.middleware.context_compaction import (
    ContextCompactionMiddleware,
)
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool
from nexau.core.messages import Role, ToolResultBlock, ToolUseBlock

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

LLM_CONFIG: dict[str, Any] = dict(
    model=os.environ.get("LLM_MODEL", "claude-opus-4-6"),
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:3001/v1"),
    api_key=os.environ.get("LLM_API_KEY", "test-key"),
    api_type="openai_chat_completion",
    stream=False,
    max_tokens=1024,
)

SLOW_TOOL_SLEEP = int(os.environ.get("MC_SLOW_TOOL_SLEEP", "5"))
GAP_THRESHOLD = float(os.environ.get("MC_GAP_THRESHOLD", "0.05"))  # 3s

COMPACTED_MARKER = "Tool call result has been compacted"

# ---------------------------------------------------------------------------
# Unified query — causal chain forces sequential execution
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an autonomous code agent. You will receive a multi-step task. "
    "You MUST complete ALL steps by yourself — the user will NOT send any follow-up messages. "
    "After receiving the task, execute every step autonomously until all are done.\n\n"
    "Rules:\n"
    "1. Call exactly ONE tool per turn. NEVER call multiple tools in parallel.\n"
    "2. Each step depends on the previous step's result — you MUST wait for each tool result before proceeding.\n"
    "3. Do NOT stop or ask for confirmation between steps. Keep going until ALL steps are finished.\n"
    "4. Only after completing ALL steps, give a brief summary referencing results from each step."
)

QUERY = (
    "Complete ALL 6 steps below autonomously. Do NOT stop after any step — "
    "keep calling tools until all 6 steps are done. I will not send any more messages.\n"
    "\n"
    "Step 1: Use read_file to read /tmp/main.py. Note the function name defined in it.\n"
    "\n"
    "Step 2: Use list_directory on /tmp. Count the number of files listed.\n"
    "\n"
    "Step 3: Use write_file to write to /tmp/result.txt with content:\n"
    "  'function=<name_from_step1>, file_count=<count_from_step2>'\n"
    "  (use the actual values from steps 1 and 2)\n"
    "\n"
    "Step 4: Use run_shell_command to run:\n"
    "  'sleep 5 && cat /tmp/result.txt'\n"
    "  to verify what was written in step 3.\n"
    "\n"
    "Step 5: Use read_file to read /tmp/main.py again. Confirm the function name matches step 1.\n"
    "\n"
    "Step 6: Use list_directory on /tmp again. Confirm result.txt now appears in the listing.\n"
    "\n"
    "Remember: complete ALL 6 steps. Do not stop early."
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _slow_shell(command: str) -> str:
    time.sleep(SLOW_TOOL_SLEEP)
    return f"$ {command}\ncommand completed after {SLOW_TOOL_SLEEP} seconds"


def _make_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read a file and return its content.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
            implementation=lambda path: (
                f"Content of {path}:\nimport os\nimport sys\n\n"
                f"def main():\n    print('hello world')\n    x = 42\n    return x\n\n"
                f"if __name__ == '__main__':\n    main()\n# END OF FILE"
            ),
        ),
        Tool(
            name="list_directory",
            description="List files in a directory.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path"}},
                "required": ["path"],
            },
            implementation=lambda path: "main.py\nutils.py\nREADME.md\nresult.txt",
        ),
        Tool(
            name="run_shell_command",
            description="Run a shell command. IMPORTANT: Only call ONE tool at a time.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Command"}},
                "required": ["command"],
            },
            implementation=_slow_shell,
        ),
        Tool(
            name="write_file",
            description="Write content to a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            implementation=lambda path, content: f"Successfully wrote {len(content)} bytes to {path}",
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_langfuse_tracer(session_id: str) -> Any:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        return None
    from nexau.archs.tracer.adapters.langfuse import LangfuseTracer

    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return LangfuseTracer(public_key=pk, secret_key=sk, host=host, debug=False, session_id=session_id)


def _run_config(
    *,
    config_name: str,
    middlewares: list[Any],
) -> tuple[Agent, list[Any]]:
    """Run the unified query under a given middleware config. Returns (agent, events)."""
    events: list[Any] = []
    session_id = f"micro-compact-e2e-{config_name}"
    mw_events = AgentEventsMiddleware(session_id=session_id, on_event=events.append)

    tracer = _make_langfuse_tracer(session_id)
    tracers = [tracer] if tracer is not None else []

    config = AgentConfig(
        name=f"mc_{config_name}",
        system_prompt=SYSTEM_PROMPT,
        llm_config=LLMConfig(**LLM_CONFIG),
        tools=_make_tools(),
        middlewares=[*middlewares, mw_events],
        tracers=tracers,
        max_iterations=12,
        retry_attempts=2,
        max_context_tokens=200000,
        tool_call_mode="structured",
    )
    sm = SessionManager(engine=InMemoryDatabaseEngine())
    agent = Agent(config=config, session_manager=sm, user_id="e2e_user", session_id=session_id)

    response = agent.run(message=QUERY)
    assert isinstance(response, str), f"Config {config_name}: agent returned non-string"

    # Flush langfuse
    if tracer is not None:
        tracer.flush()

    return agent, events


def _print_results(config_name: str, agent: Agent, events: list[Any]) -> None:
    """Print detailed results for a config run."""
    started = [e for e in events if isinstance(e, CompactionStartedEvent) and e.mode == "regular"]
    finished = [e for e in events if isinstance(e, CompactionFinishedEvent) and e.mode == "regular"]

    print(f"\n{'─' * 70}")
    print(f"Config {config_name}: {len(agent.history)} messages, {len(started)} compaction triggers")
    print(f"{'─' * 70}")

    for i, msg in enumerate(agent.history):
        role = msg.role.value

        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    print(f"  [{i:2d}] {role:10s} [ToolUse] name={block.name}")
                elif isinstance(block, ToolResultBlock):
                    content = block.content if isinstance(block.content, str) else str(block.content)
                    if COMPACTED_MARKER in content:
                        print(f"  [{i:2d}] {role:10s} [ToolResult] >>> COMPACTED <<<")
                    else:
                        preview = content[:80] + "..." if len(content) > 80 else content
                        print(f"  [{i:2d}] {role:10s} [ToolResult] {preview}")
                else:
                    text = getattr(block, "text", "")
                    if text.strip():
                        preview = text.strip()[:60] + "..." if len(text.strip()) > 60 else text.strip()
                        print(f"  [{i:2d}] {role:10s} [Text] {preview}")

    if started:
        for e in started:
            print(f"  Trigger: {e.trigger_reason}")
    if finished:
        for ef in finished:
            print(f"  Result: success={ef.success}, msgs {ef.original_message_count} -> {ef.compacted_message_count}")


def _count_by_tool(agent: Agent) -> dict[str, str]:
    """Map tool_name -> 'COMPACTED' or 'preserved' for each tool result, in order."""
    result: dict[str, str] = {}
    # Build tool_use_id -> name mapping
    id_to_name: dict[str, str] = {}
    for msg in agent.history:
        if msg.role == Role.ASSISTANT:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    id_to_name[block.id] = block.name

    seen_names: dict[str, int] = {}
    for msg in agent.history:
        if msg.role == Role.TOOL:
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    name = id_to_name.get(block.tool_use_id, "unknown")
                    count = seen_names.get(name, 0) + 1
                    seen_names[name] = count
                    key = f"{name}#{count}"
                    content = block.content if isinstance(block.content, str) else str(block.content)
                    result[key] = "COMPACTED" if COMPACTED_MARKER in content else "preserved"
    return result


# ---------------------------------------------------------------------------
# Config A: No compaction (baseline)
# ---------------------------------------------------------------------------


def run_config_a() -> tuple[Agent, list[Any]]:
    print("\n[Config A] No compaction middleware (baseline)")
    return _run_config(config_name="A_baseline", middlewares=[])


# ---------------------------------------------------------------------------
# Config B: compactable_tools filter + keep_iterations=1
# ---------------------------------------------------------------------------


def run_config_b() -> tuple[Agent, list[Any]]:
    print("\n[Config B] compactable_tools=[read_file,list_directory], keep_iterations=1")
    micro_ccm = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=GAP_THRESHOLD,
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        compactable_tools=["read_file", "list_directory"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )
    return _run_config(config_name="B_filtered_keep1", middlewares=[micro_ccm])


# ---------------------------------------------------------------------------
# Config C: No filter (compact all) + keep_iterations=1
# ---------------------------------------------------------------------------


def run_config_c() -> tuple[Agent, list[Any]]:
    print("\n[Config C] compactable_tools=None (compact ALL), keep_iterations=1")
    micro_ccm = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=GAP_THRESHOLD,
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        # compactable_tools not set -> None -> compact all
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )
    return _run_config(config_name="C_nofilter_keep1", middlewares=[micro_ccm])


# ---------------------------------------------------------------------------
# Config D: compactable_tools filter + keep_iterations=2
# ---------------------------------------------------------------------------


def run_config_d() -> tuple[Agent, list[Any]]:
    print("\n[Config D] compactable_tools=[read_file,list_directory], keep_iterations=2")
    micro_ccm = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=GAP_THRESHOLD,
        compaction_strategy="tool_result_compaction",
        keep_iterations=2,
        compactable_tools=["read_file", "list_directory"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )
    return _run_config(config_name="D_filtered_keep2", middlewares=[micro_ccm])


# ---------------------------------------------------------------------------
# Config E: Same as B but gap_threshold=5min (too high to trigger)
# ---------------------------------------------------------------------------


def run_config_e() -> tuple[Agent, list[Any]]:
    print("\n[Config E] Same as B but gap_threshold=5min (should NOT trigger)")
    micro_ccm = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=5,  # 5 minutes — slow tool only blocks ~5 seconds
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        compactable_tools=["read_file", "list_directory"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )
    return _run_config(config_name="E_high_threshold", middlewares=[micro_ccm])


# ---------------------------------------------------------------------------
# Config F: No filter + keep_iterations=2 (isolate keep_iterations effect on write_file)
# ---------------------------------------------------------------------------


def run_config_f() -> tuple[Agent, list[Any]]:
    print("\n[Config F] compactable_tools=None (compact ALL), keep_iterations=2")
    micro_ccm = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=GAP_THRESHOLD,
        compaction_strategy="tool_result_compaction",
        keep_iterations=2,
        # no filter — all tools compactable; only keep_iterations protects
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )
    return _run_config(config_name="F_nofilter_keep2", middlewares=[micro_ccm])


# ---------------------------------------------------------------------------
# Main — run all configs, print comparison table
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    configs = [
        ("A", "baseline (no compaction)", run_config_a),
        ("B", "filter=[read,list], keep=1, gap=3s", run_config_b),
        ("C", "no filter, keep=1, gap=3s", run_config_c),
        ("D", "filter=[read,list], keep=2, gap=3s", run_config_d),
        ("E", "same as B but gap=5min (no trigger)", run_config_e),
        ("F", "no filter, keep=2", run_config_f),
    ]

    results: dict[str, tuple[Agent, list[Any]]] = {}

    for label, desc, fn in configs:
        print(f"\n{'=' * 70}")
        print(f"Running Config {label}: {desc}")
        print(f"{'=' * 70}")
        try:
            agent, events = fn()
            results[label] = (agent, events)
            _print_results(label, agent, events)
        except Exception as e:
            print(f"\nFAILED: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison table
    print(f"\n\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")

    # Collect tool result status per config
    all_tools: dict[str, dict[str, str]] = {}
    for label, (agent, _) in results.items():
        all_tools[label] = _count_by_tool(agent)

    # Get union of all tool keys
    all_keys: list[str] = []
    for tools in all_tools.values():
        for k in tools:
            if k not in all_keys:
                all_keys.append(k)

    # Print header
    config_labels = [label for label, _, _ in configs]
    header = f"{'tool':<25s}"
    for label in config_labels:
        if label in results:
            header += f" | {label:^12s}"
    print(header)
    print("─" * len(header))

    # Print rows
    for key in all_keys:
        row = f"{key:<25s}"
        for label in config_labels:
            if label in results:
                status = all_tools.get(label, {}).get(key, "n/a")
                marker = "COMPACTED" if status == "COMPACTED" else "preserved"
                row += f" | {marker:^12s}"
        print(row)

    # Assertions
    print(f"\n{'=' * 70}")
    print("ASSERTIONS")
    print(f"{'=' * 70}")

    failures: list[str] = []

    def check(condition: bool, msg: str) -> None:
        if condition:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures.append(msg)

    if "A" in results:
        a_tools = all_tools["A"]
        a_compacted = sum(1 for v in a_tools.values() if v == "COMPACTED")
        check(a_compacted == 0, "Config A: no compaction (baseline)")

    if "B" in results:
        b_tools = all_tools["B"]
        check(b_tools.get("read_file#1") == "COMPACTED", "Config B: iter1 read_file COMPACTED")
        check(b_tools.get("list_directory#1") == "COMPACTED", "Config B: iter2 list_directory COMPACTED")
        check(b_tools.get("write_file#1") == "preserved", "Config B: iter3 write_file preserved (not in filter)")
        check(b_tools.get("run_shell_command#1") == "preserved", "Config B: iter4 shell preserved (keep=1)")
        check(b_tools.get("read_file#2") == "preserved", "Config B: iter5 read_file preserved (after compaction)")
        check(b_tools.get("list_directory#2") == "preserved", "Config B: iter6 list_directory preserved (after compaction)")

    if "C" in results:
        c_tools = all_tools["C"]
        check(c_tools.get("read_file#1") == "COMPACTED", "Config C: iter1 read_file COMPACTED")
        check(c_tools.get("list_directory#1") == "COMPACTED", "Config C: iter2 list_directory COMPACTED")
        check(c_tools.get("write_file#1") == "COMPACTED", "Config C: iter3 write_file COMPACTED (no filter)")
        check(c_tools.get("run_shell_command#1") == "preserved", "Config C: iter4 shell preserved (keep=1)")

    if "D" in results:
        d_tools = all_tools["D"]
        check(d_tools.get("read_file#1") == "COMPACTED", "Config D: iter1 read_file COMPACTED")
        check(d_tools.get("list_directory#1") == "COMPACTED", "Config D: iter2 list_directory COMPACTED")
        check(d_tools.get("write_file#1") == "preserved", "Config D: iter3 write_file preserved (keep=2 + not in filter)")
        check(d_tools.get("run_shell_command#1") == "preserved", "Config D: iter4 shell preserved (keep=2)")

    if "E" in results:
        e_tools = all_tools["E"]
        e_compacted = sum(1 for v in e_tools.values() if v == "COMPACTED")
        check(e_compacted == 0, "Config E: no compaction (gap_threshold=5min too high to trigger)")

    if "F" in results:
        f_tools = all_tools["F"]
        check(f_tools.get("read_file#1") == "COMPACTED", "Config F: iter1 read_file COMPACTED (no filter)")
        check(f_tools.get("list_directory#1") == "COMPACTED", "Config F: iter2 list_directory COMPACTED (no filter)")
        check(f_tools.get("write_file#1") == "preserved", "Config F: iter3 write_file preserved (keep=2 only)")
        check(f_tools.get("run_shell_command#1") == "preserved", "Config F: iter4 shell preserved (keep=2)")

    print(f"\n{'=' * 70}")
    if failures:
        print(f"RESULT: {len(failures)} assertions FAILED")
        for f in failures:
            print(f"  - {f}")
        exit(1)
    else:
        print(f"RESULT: All assertions passed across {len(results)} configurations")
    print(f"{'=' * 70}")
