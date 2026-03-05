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

"""Acceptance test script for LocalSandbox execute_bash output redirect.

验收项：
1. Foreground: 小输出 — output_dir 存在, stdout.txt/stderr.txt 正确, 不截断
2. Foreground: 大输出 — 智能截断, 文件中保留完整内容
3. Foreground: 命令自带重定向 — 不干扰用户重定向
4. Foreground: 命令失败 — output_dir 依然存在
5. Foreground: 超时 — output_dir 依然存在, 已写入的输出可读
6. Background: 启动后 output_dir 存在
7. Background: 完成后 stdout.txt/stderr.txt 有完整输出
8. Background: 大输出 — 智能截断
9. Background: kill — 文件句柄正确关闭

运行方式：
    uv run python tests/scripts/acceptance_local_sandbox_output.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from nexau.archs.sandbox.base_sandbox import (
    BASH_TOOL_RESULTS_BASE_PATH,
    SandboxStatus,
)
from nexau.archs.sandbox.local_sandbox import LocalSandbox

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
# Setup / Teardown
# ---------------------------------------------------------------------------


def _create_sandbox(work_dir: str) -> LocalSandbox:
    return LocalSandbox(work_dir=work_dir)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_foreground_small_output(sandbox: LocalSandbox) -> None:
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
        stdout_file = Path(result.output_dir) / "stdout.txt"
        stderr_file = Path(result.output_dir) / "stderr.txt"
        command_file = Path(result.output_dir) / "command.txt"
        _check("stdout.txt exists", stdout_file.exists())
        _check("stderr.txt exists", stderr_file.exists())
        _check("command.txt exists", command_file.exists())
        _check(
            "stdout.txt content matches",
            stdout_file.read_text().strip() == "hello",
            f"got: {stdout_file.read_text()!r}",
        )
        _check(
            "stderr.txt content matches",
            stderr_file.read_text().strip() == "world",
            f"got: {stderr_file.read_text()!r}",
        )


def test_foreground_large_output(sandbox: LocalSandbox) -> None:
    """2. 大输出 — 智能截断, 文件中保留完整内容"""
    _section("2. Foreground: 大输出 (>10k chars)")

    # 生成 > 10k 字符的输出
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

    # 返回的 stdout 应包含截断提示
    _check("stdout has truncation hint", "characters omitted" in result.stdout)
    _check("stdout has file path hint", "stdout.txt" in result.stdout)

    if result.output_dir:
        full_stdout = Path(f"{result.output_dir}/stdout.txt").read_text()
        _check(
            "full file has all lines",
            "line-00000" in full_stdout and "line-00199" in full_stdout,
        )
        _check(
            "full file larger than returned stdout",
            len(full_stdout) > len(result.stdout),
            f"file={len(full_stdout)}, returned={len(result.stdout)}",
        )


def test_foreground_user_redirect(sandbox: LocalSandbox) -> None:
    """3. 命令自带重定向 — 不干扰用户重定向"""
    _section("3. Foreground: 命令自带重定向")

    user_file = f"{sandbox.work_dir}/user_output.txt"
    cmd = f'echo "to-file" > {user_file} && echo "to-stdout"'
    result = sandbox.execute_bash(cmd)

    _check("status == SUCCESS", result.status == SandboxStatus.SUCCESS)
    _check("stdout contains 'to-stdout'", "to-stdout" in result.stdout)
    _check("stdout does NOT contain 'to-file'", "to-file" not in result.stdout)

    user_content = Path(user_file).read_text().strip()
    _check(
        "user redirect file has correct content",
        user_content == "to-file",
        f"got: {user_content!r}",
    )


def test_foreground_failed_command(sandbox: LocalSandbox) -> None:
    """4. 命令失败 — output_dir 依然存在"""
    _section("4. Foreground: 命令失败")

    result = sandbox.execute_bash("echo 'error msg' >&2 && exit 42")

    _check("status == ERROR", result.status == SandboxStatus.ERROR)
    _check("exit_code == 42", result.exit_code == 42)
    _check("stderr contains 'error msg'", "error msg" in result.stderr)
    _check("output_dir is set", result.output_dir is not None)

    if result.output_dir:
        _check("stderr.txt exists", Path(f"{result.output_dir}/stderr.txt").exists())


def test_foreground_timeout(sandbox: LocalSandbox) -> None:
    """5. 超时 — output_dir 依然存在, 已写入的输出可读"""
    _section("5. Foreground: 超时")

    # 1 秒超时, 命令 sleep 10 秒
    cmd = "echo 'before-sleep' && sleep 10 && echo 'after-sleep'"
    result = sandbox.execute_bash(cmd, timeout=1000)

    _check("status == TIMEOUT", result.status == SandboxStatus.TIMEOUT)
    _check("output_dir is set", result.output_dir is not None)
    # echo 'before-sleep' 应该在超时前写入了文件
    _check("stdout contains 'before-sleep'", "before-sleep" in result.stdout)
    _check("stdout does NOT contain 'after-sleep'", "after-sleep" not in result.stdout)


def test_background_output_dir(sandbox: LocalSandbox) -> None:
    """6. Background: 启动后 output_dir 存在"""
    _section("6. Background: 启动后 output_dir")

    result = sandbox.execute_bash("echo bg-hello && sleep 1", background=True)

    _check("status == SUCCESS", result.status == SandboxStatus.SUCCESS)
    _check("background_pid is set", result.background_pid is not None)
    _check("output_dir is set", result.output_dir is not None)

    if result.output_dir:
        _check("output_dir exists", Path(result.output_dir).is_dir())
        _check("command.txt exists", Path(f"{result.output_dir}/command.txt").exists())

    # 等待完成后清理
    if result.background_pid:
        time.sleep(2)
        sandbox.kill_background_task(result.background_pid)


def test_background_completed_output(sandbox: LocalSandbox) -> None:
    """7. Background: 完成后 stdout.txt/stderr.txt 有完整输出"""
    _section("7. Background: 完成后读取输出")

    result = sandbox.execute_bash(
        "echo 'bg-stdout' && echo 'bg-stderr' >&2",
        background=True,
    )
    pid = result.background_pid
    assert pid is not None

    # 等待后台任务完成
    for _ in range(20):
        status = sandbox.get_background_task_status(pid)
        if status.status != SandboxStatus.RUNNING:
            break
        time.sleep(0.2)

    _check("task finished", status.status != SandboxStatus.RUNNING)
    _check("stdout has 'bg-stdout'", "bg-stdout" in status.stdout)
    _check("stderr has 'bg-stderr'", "bg-stderr" in status.stderr)
    _check("output_dir is set", status.output_dir is not None)

    if status.output_dir:
        full_stdout = Path(f"{status.output_dir}/stdout.txt").read_text()
        _check("file stdout matches", "bg-stdout" in full_stdout)


def test_background_large_output(sandbox: LocalSandbox) -> None:
    """8. Background: 大输出 — 智能截断"""
    _section("8. Background: 大输出")

    cmd = "python3 -c \"import sys; [print(f'bg-line-{i:05d}-' + 'y'*80) for i in range(200)]\""
    result = sandbox.execute_bash(cmd, background=True)
    pid = result.background_pid
    assert pid is not None

    # 等待完成
    for _ in range(30):
        status = sandbox.get_background_task_status(pid)
        if status.status != SandboxStatus.RUNNING:
            break
        time.sleep(0.3)

    _check("task finished", status.status != SandboxStatus.RUNNING)
    _check("truncated == True", status.truncated is True)
    _check("truncation hint in stdout", "characters omitted" in status.stdout)

    if status.output_dir:
        full = Path(f"{status.output_dir}/stdout.txt").read_text()
        _check("full file has all lines", "bg-line-00000" in full and "bg-line-00199" in full)


def test_background_kill(sandbox: LocalSandbox) -> None:
    """9. Background: kill — 文件句柄正确关闭"""
    _section("9. Background: kill")

    result = sandbox.execute_bash("echo 'kill-test' && sleep 60", background=True)
    pid = result.background_pid
    assert pid is not None
    output_dir = result.output_dir

    # 等待一下让 echo 输出写入
    time.sleep(0.5)

    kill_result = sandbox.kill_background_task(pid)
    _check("kill status == SUCCESS", kill_result.status == SandboxStatus.SUCCESS)
    _check("pid removed from tasks", pid not in sandbox._background_tasks)

    if output_dir:
        stdout_content = Path(f"{output_dir}/stdout.txt").read_text()
        _check("stdout.txt has 'kill-test'", "kill-test" in stdout_content)


def test_custom_threshold(work_dir: str) -> None:
    """10. 自定义阈值 — 小阈值触发截断, 大阈值不截断"""
    _section("10. 自定义阈值可配置")

    # 生成 ~500 字符的输出（5 行 * 100 字符）
    cmd = "python3 -c \"[print('a'*95 + f'{i:04d}') for i in range(5)]\""

    # 小阈值 (100) — 一定触发截断
    small_sandbox = LocalSandbox(
        work_dir=work_dir,
        output_char_threshold=100,
        truncate_head_chars=30,
        truncate_tail_chars=30,
    )
    result_small = small_sandbox.execute_bash(cmd)
    _check("small threshold: truncated == True", result_small.truncated is True)
    _check("small threshold: has omitted hint", "characters omitted" in result_small.stdout)

    # 大阈值 (100000) — 不会触发截断
    big_sandbox = LocalSandbox(
        work_dir=work_dir,
        output_char_threshold=100000,
        truncate_head_chars=50000,
        truncate_tail_chars=50000,
    )
    result_big = big_sandbox.execute_bash(cmd)
    _check("big threshold: truncated == False", result_big.truncated is False)
    _check("big threshold: no omitted hint", "characters omitted" not in result_big.stdout)
    _check("big threshold: all lines present", "0004" in result_big.stdout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  LocalSandbox Output Redirect Acceptance Tests")
    print("=" * 60)

    work_dir = tempfile.mkdtemp(prefix="nexau_accept_local_")
    print(f"\nWork dir: {work_dir}")

    # 清理可能残留的临时文件
    if Path(BASH_TOOL_RESULTS_BASE_PATH).exists():
        old_count = len(list(Path(BASH_TOOL_RESULTS_BASE_PATH).iterdir()))
        print(f"Existing output dirs: {old_count}")

    sandbox = _create_sandbox(work_dir)

    try:
        test_foreground_small_output(sandbox)
        test_foreground_large_output(sandbox)
        test_foreground_user_redirect(sandbox)
        test_foreground_failed_command(sandbox)
        test_foreground_timeout(sandbox)
        test_background_output_dir(sandbox)
        test_background_completed_output(sandbox)
        test_background_large_output(sandbox)
        test_background_kill(sandbox)
        test_custom_threshold(work_dir)
    finally:
        # 清理工作目录
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if _failed > 0 else 0)


if __name__ == "__main__":
    main()
