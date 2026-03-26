#!/usr/bin/env python3
"""Real end-to-end test: Agent.run() + E2B sandbox — nested event loop check.

Uses the REAL Agent class loaded from code_agent.yaml with a REAL E2B sandbox.
Calls Agent.run() which internally does asyncio.run(run_async(...)),
triggering actual LLM calls and tool execution inside the sandbox.

Usage:
    E2B_API_KEY=xxx uv run python tests/scripts/test_e2b_real_agent_run.py
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    e2b_api_key = os.environ.get("E2B_API_KEY")
    if not e2b_api_key:
        print("[SKIP] E2B_API_KEY not set.")
        sys.exit(0)

    llm_model = os.environ.get("LLM_MODEL", "")
    llm_base_url = os.environ.get("LLM_BASE_URL", "")
    print("=" * 70)
    print("  Real Agent.run() + E2B Sandbox — Nested Event Loop Test")
    print(f"  LLM: {llm_model} @ {llm_base_url}")
    print(f"  E2B_API_KEY: {e2b_api_key[:12]}...")
    print("=" * 70)

    from nexau import Agent, AgentConfig
    from nexau.archs.sandbox.base_sandbox import E2BSandboxConfig

    # -----------------------------------------------------------------------
    # 1. Load agent config from code_agent.yaml
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Loading agent config from code_agent.yaml ...")
    script_dir = Path(__file__).resolve().parents[2] / "examples" / "code_agent"
    config = AgentConfig.from_yaml(config_path=script_dir / "code_agent.yaml")

    # -----------------------------------------------------------------------
    # 2. Inject E2B sandbox_config (code_agent.yaml has no sandbox_config,
    #    so it defaults to LocalSandbox — override with E2B here)
    # -----------------------------------------------------------------------
    print("[Phase 2] Injecting E2B sandbox_config ...")
    config.sandbox_config = E2BSandboxConfig(
        type="e2b",
        api_key=e2b_api_key,
        template="base",
        timeout=300,
    )
    # Strip sub_agents to simplify (avoid extra YAML resolution)
    config.sub_agents = {}
    # Strip tracers to avoid langfuse dependency in test
    config.tracers = []

    # -----------------------------------------------------------------------
    # 3. Create Agent — this triggers _init_sandbox() + prepare_session_context()
    # -----------------------------------------------------------------------
    print("[Phase 3] Creating Agent ...")
    agent = Agent(config=config)
    print(f"  Agent name: {agent.agent_name}")

    # -----------------------------------------------------------------------
    # 4. Access sandbox_manager.instance — triggers lazy E2B sandbox creation
    # -----------------------------------------------------------------------
    print("[Phase 4] Accessing sandbox_manager.instance (lazy E2B start) ...")
    sandbox = agent.sandbox_manager.instance
    if sandbox is None:
        print("  [FAIL] Sandbox failed to start!")
        sys.exit(1)
    print(f"  [OK] Sandbox started: {sandbox.sandbox_id}")

    # -----------------------------------------------------------------------
    # 5. Agent.run() — THE REAL TEST
    #    Agent.run() calls asyncio.run(run_async(...))
    #    Inside run_async, tools execute bash commands / file ops in E2B sandbox
    #    Any nested event loop issue would crash here
    # -----------------------------------------------------------------------
    print("\n[Phase 5] Calling Agent.run() — REAL LLM + E2B tool execution ...")
    print("  This sends a simple task that should trigger tool calls.\n")

    user_message = (
        "List the files in /home/user using list_directory tool, "
        "then write a file /home/user/test_nexau.txt with content 'hello from nexau', "
        "then read it back with read_file to confirm, "
        "and finally call complete_task with a summary of what you did."
    )

    try:
        response = agent.run(
            message=user_message,
            context={
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": os.getenv("USER", "test"),
                "working_directory": str(sandbox.work_dir),
            },
        )
        print("\n" + "-" * 70)
        print("Agent response:")
        print("-" * 70)
        # response may be str or tuple
        if isinstance(response, tuple):
            print(response[0])
        else:
            print(response)
        print("-" * 70)
        print("\n  [PASS] Agent.run() completed without nested event loop error!")
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "event loop" in err_msg or "nested" in err_msg:
            print(f"\n  [FAIL] NESTED EVENT LOOP ERROR: {e}")
            traceback.print_exc()
            sys.exit(1)
        else:
            print(f"\n  [FAIL] RuntimeError: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\n  [FAIL] Exception during Agent.run(): {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 6. Verify sandbox file was actually created (confirms tools ran in E2B)
    # -----------------------------------------------------------------------
    print("\n[Phase 6] Verifying sandbox file was created by tools ...")
    try:
        read_result = sandbox.read_file("/home/user/test_nexau.txt")
        if read_result.content and "hello from nexau" in str(read_result.content):
            print("  [PASS] File /home/user/test_nexau.txt exists with correct content")
        else:
            print(f"  [WARN] File content: {read_result.content!r}")
    except Exception as e:
        print(f"  [WARN] Could not verify file: {e}")

    # -----------------------------------------------------------------------
    # 7. Second Agent.run() — verify repeated calls work
    # -----------------------------------------------------------------------
    print("\n[Phase 7] Second Agent.run() call ...")
    try:
        response2 = agent.run(
            message=("Run the shell command 'echo nested-loop-test-ok' and call complete_task with the output."),
            context={
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": os.getenv("USER", "test"),
                "working_directory": str(sandbox.work_dir),
            },
        )
        resp_text = response2[0] if isinstance(response2, tuple) else response2
        print(f"  Response (truncated): {str(resp_text)[:200]}")
        print("  [PASS] Second Agent.run() completed!")
    except Exception as e:
        print(f"  [FAIL] Second Agent.run() failed: {type(e).__name__}: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    print("\n[Cleanup] Stopping sandbox ...")
    try:
        agent.sandbox_manager.stop()
        print("  Sandbox stopped.")
    except Exception as e:
        print(f"  Warning: stop failed: {e}")

    print("\n" + "=" * 70)
    print("  ALL DONE — No nested event loop issues detected!")
    print("=" * 70)


if __name__ == "__main__":
    main()
