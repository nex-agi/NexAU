#!/usr/bin/env python3
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

"""Acceptance test script for E2BSandbox execute_bash output redirect.

需要设置环境变量：
    E2B_API_KEY    — E2B API key（必须）
    E2B_API_URL    — E2B API URL（可选, 自建集群时需要）
    E2B_TEMPLATE   — E2B sandbox template（可选, 默认 "base"）

验收项：
1. Foreground: 小输出 — output_dir 存在, stdout.txt/stderr.txt 正确, 不截断
2. Foreground: 大输出 — 智能截断, 文件中保留完整内容
3. Foreground: 命令自带重定向 — 不干扰用户重定向
4. Foreground: 命令失败 — output_dir 依然存在
5. Background: 启动后 output_dir 存在
6. Background: 完成后 stdout.txt/stderr.txt 有完整输出
7. Background: 大输出 — 智能截断
8. Background: kill — 进程终止

运行方式：
    E2B_API_KEY=xxx uv run python tests/scripts/acceptance_e2b_sandbox_output.py
"""

from __future__ import annotations

import os
import sys
import time

from nexau.archs.sandbox.base_sandbox import (
    E2BSandboxConfig,
    SandboxStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _create_e2b_sandbox():
    """Create an E2BSandbox via E2BSandboxManager."""
    from nexau.archs.sandbox.e2b_sandbox import E2B_AVAILABLE, E2BSandboxManager

    if not E2B_AVAILABLE:
        print("[SKIP] E2B SDK not installed. Install with: pip install e2b")
        sys.exit(0)

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        print("[SKIP] E2B_API_KEY not set. Set it to run E2B acceptance tests.")
        sys.exit(0)

    api_url = os.environ.get("E2B_API_URL")
    template = os.environ.get("E2B_TEMPLATE", "base")

    config = E2BSandboxConfig(
        api_key=api_key,
        api_url=api_url,
        template=template,
        timeout=300,
    )

    manager = E2BSandboxManager(
        api_key=api_key,
        api_url=api_url,
        template=template,
    )

    print(f"Starting E2B sandbox (template={template})...")
    sandbox = manager.start(
        session_manager=None,
        user_id="acceptance-test",
        session_id="acceptance-test",
        sandbox_config=config,
    )
    print(f"Sandbox started: {sandbox.sandbox_id}")
    return sandbox, manager


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_foreground_small_output(sandbox) -> None:
    """1. 小输出 — output_dir 存在, stdout.txt/stderr.txt 正确, 不截断"""
    _section("1. Foreground: 小输出")

    result = sandbox.execute_bash("echo hello && echo world >&2")

    _check("status == SUCCESS", result.status == SandboxStatus.SUCCESS)
    _check("exit_code == 0", result.exit_code == 0)
    _check("stdout contains 'hello'", "hello" in result.stdout)
    _check("stderr contains 'world'", "world" in result.stderr)
    _check("truncated == False", result.truncated is False)
    _check("output_dir is set", result.output_dir is not None)

    if result.output_dir:
        # 通过 sandbox 的 read_file 验证文件内容
        stdout_result = sandbox.read_file(f"{result.output_dir}/stdout.txt")
        stderr_result = sandbox.read_file(f"{result.output_dir}/stderr.txt")
        cmd_result = sandbox.read_file(f"{result.output_dir}/command.txt")
        _check("stdout.txt readable", stdout_result.status == SandboxStatus.SUCCESS)
        _check("stderr.txt readable", stderr_result.status == SandboxStatus.SUCCESS)
        _check("command.txt readable", cmd_result.status == SandboxStatus.SUCCESS)
        if stdout_result.content:
            _check(
                "stdout.txt content matches",
                "hello" in str(stdout_result.content),
                f"got: {str(stdout_result.content)[:100]!r}",
            )
        if stderr_result.content:
            _check(
                "stderr.txt content matches",
                "world" in str(stderr_result.content),
                f"got: {str(stderr_result.content)[:100]!r}",
            )


def test_foreground_large_output(sandbox) -> None:
    """2. 大输出 — 智能截断, 文件中保留完整内容"""
    _section("2. Foreground: 大输出 (>10k chars)")

    cmd = "python3 -c \"import sys; [print(f'line-{i:05d}-' + 'x'*80) for i in range(200)]\""
    result = sandbox.execute_bash(cmd)

    _check("status == SUCCESS", result.status == SandboxStatus.SUCCESS)
    _check("truncated == True", result.truncated is True)
    _check("output_dir is set", result.output_dir is not None)
    _check(
        "original_stdout_length is set",
        result.original_stdout_length is not None and result.original_stdout_length > 10000,
        f"got: {result.original_stdout_length}",
    )

    _check("stdout has truncation hint", "characters omitted" in result.stdout)
    _check("stdout has file path hint", "stdout.txt" in result.stdout)

    if result.output_dir:
        full_result = sandbox.read_file(f"{result.output_dir}/stdout.txt")
        if full_result.content:
            full_text = str(full_result.content)
            _check(
                "full file has all lines",
                "line-00000" in full_text and "line-00199" in full_text,
            )
            _check(
                "full file larger than returned stdout",
                len(full_text) > len(result.stdout),
                f"file={len(full_text)}, returned={len(result.stdout)}",
            )


def test_foreground_user_redirect(sandbox) -> None:
    """3. 命令自带重定向 — 不干扰用户重定向"""
    _section("3. Foreground: 命令自带重定向")

    user_file = "/tmp/user_redirect_test.txt"
    cmd = f'echo "to-file" > {user_file} && echo "to-stdout"'
    result = sandbox.execute_bash(cmd)

    _check("status == SUCCESS", result.status == SandboxStatus.SUCCESS)
    _check("stdout contains 'to-stdout'", "to-stdout" in result.stdout)
    _check("stdout does NOT contain 'to-file'", "to-file" not in result.stdout)

    # 验证用户重定向文件
    file_result = sandbox.read_file(user_file)
    _check("user file readable", file_result.status == SandboxStatus.SUCCESS)
    if file_result.content:
        _check(
            "user file content correct",
            "to-file" in str(file_result.content),
            f"got: {str(file_result.content)[:100]!r}",
        )


def test_foreground_failed_command(sandbox) -> None:
    """4. 命令失败 — output_dir 依然存在"""
    _section("4. Foreground: 命令失败")

    result = sandbox.execute_bash("echo 'error msg' >&2 && exit 42")

    _check("status == ERROR", result.status == SandboxStatus.ERROR)
    _check("exit_code == 42", result.exit_code == 42)
    _check("stderr contains 'error msg'", "error msg" in result.stderr)
    _check("output_dir is set", result.output_dir is not None)

    if result.output_dir:
        stderr_result = sandbox.read_file(f"{result.output_dir}/stderr.txt")
        _check("stderr.txt readable", stderr_result.status == SandboxStatus.SUCCESS)


def test_background_output_dir(sandbox) -> None:
    """5. Background: 启动后 output_dir 存在"""
    _section("5. Background: 启动后 output_dir")

    result = sandbox.execute_bash("echo bg-hello && sleep 1", background=True)

    _check("status == SUCCESS", result.status == SandboxStatus.SUCCESS)
    _check("background_pid is set", result.background_pid is not None)
    _check("output_dir is set", result.output_dir is not None)

    if result.output_dir:
        cmd_result = sandbox.read_file(f"{result.output_dir}/command.txt")
        _check("command.txt readable", cmd_result.status == SandboxStatus.SUCCESS)

    # 等待完成后清理
    if result.background_pid:
        time.sleep(2)
        sandbox.kill_background_task(result.background_pid)


def test_background_completed_output(sandbox) -> None:
    """6. Background: 完成后 stdout.txt/stderr.txt 有完整输出"""
    _section("6. Background: 完成后读取输出")

    result = sandbox.execute_bash(
        "echo 'bg-stdout' && echo 'bg-stderr' >&2",
        background=True,
    )
    pid = result.background_pid
    assert pid is not None

    for _ in range(30):
        status = sandbox.get_background_task_status(pid)
        if status.status != SandboxStatus.RUNNING:
            break
        time.sleep(0.3)

    _check("task finished", status.status != SandboxStatus.RUNNING)
    _check("stdout has 'bg-stdout'", "bg-stdout" in status.stdout)
    _check("stderr has 'bg-stderr'", "bg-stderr" in status.stderr)
    _check("output_dir is set", status.output_dir is not None)


def test_background_large_output(sandbox) -> None:
    """7. Background: 大输出 — 智能截断"""
    _section("7. Background: 大输出")

    cmd = "python3 -c \"import sys; [print(f'bg-line-{i:05d}-' + 'y'*80) for i in range(200)]\""
    result = sandbox.execute_bash(cmd, background=True)
    pid = result.background_pid
    assert pid is not None

    for _ in range(40):
        status = sandbox.get_background_task_status(pid)
        if status.status != SandboxStatus.RUNNING:
            break
        time.sleep(0.5)

    _check("task finished", status.status != SandboxStatus.RUNNING)
    _check("truncated == True", status.truncated is True)
    _check("truncation hint in stdout", "characters omitted" in status.stdout)


def test_background_kill(sandbox) -> None:
    """8. Background: kill"""
    _section("8. Background: kill")

    result = sandbox.execute_bash("echo 'kill-test' && sleep 60", background=True)
    pid = result.background_pid
    assert pid is not None

    time.sleep(1)

    kill_result = sandbox.kill_background_task(pid)
    _check("kill status == SUCCESS", kill_result.status == SandboxStatus.SUCCESS)
    _check("pid removed from tasks", pid not in sandbox._background_tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  E2BSandbox Output Redirect Acceptance Tests")
    print("=" * 60)

    sandbox, manager = _create_e2b_sandbox()

    try:
        test_foreground_small_output(sandbox)
        test_foreground_large_output(sandbox)
        test_foreground_user_redirect(sandbox)
        test_foreground_failed_command(sandbox)
        test_background_output_dir(sandbox)
        test_background_completed_output(sandbox)
        test_background_large_output(sandbox)
        test_background_kill(sandbox)
    finally:
        print("\nStopping sandbox...")
        try:
            manager.stop()
            print("Sandbox stopped.")
        except Exception as e:
            print(f"Warning: stop failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if _failed > 0 else 0)


if __name__ == "__main__":
    main()
