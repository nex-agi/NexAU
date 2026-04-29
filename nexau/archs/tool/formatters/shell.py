"""Shell tool specific formatters.

RFC-0017: Tool output flattening

Formats shell command results into a Claude Code-style transcript body so the
LLM sees stdout/stderr/background state directly instead of a metadata-heavy
dict shell.
"""

from __future__ import annotations

from typing import cast

from . import ToolFormatterContext


def format_run_shell_command_output(context: ToolFormatterContext) -> object:
    """Flatten run_shell_command output into a Claude Code-style text payload."""

    raw_output = context.tool_output
    if not isinstance(raw_output, dict):
        return str(raw_output)

    raw_output_dict = cast(dict[object, object], raw_output)
    output = {str(key): value for key, value in raw_output_dict.items()}

    stdout = _string_field(output.get("stdout"))
    stderr = _string_field(output.get("stderr"))
    content = _string_field(output.get("content"))
    processed_stdout = _normalize_stdout(stdout)
    error_text = _build_error_text(output, stderr=stderr)
    background_info = _build_background_info(output)

    parts = [part for part in (processed_stdout, error_text, background_info) if part]
    if parts:
        return "\n".join(parts)

    if content:
        return content

    return ""


def _normalize_stdout(stdout: str) -> str:
    if not stdout:
        return ""

    # Issue #498: defense-in-depth — strip ANSI escapes and resolve CR overwrites
    # for any output that may bypass the sandbox cleaning path
    from nexau.archs.sandbox.output_utils import resolve_cr, strip_ansi

    stdout = strip_ansi(stdout)
    stdout = resolve_cr(stdout)

    stripped_leading = stdout.lstrip("\n \t\r")
    return stripped_leading.rstrip()


def _build_error_text(output: dict[str, object], *, stderr: str) -> str:
    interrupted = output.get("interrupted") is True
    timed_out = output.get("timed_out") is True
    error_message = _extract_error_message(output.get("error"))
    exit_code = output.get("exit_code")

    parts: list[str] = []
    if stderr.strip():
        # Issue #498: defense-in-depth — clean stderr same as stdout
        from nexau.archs.sandbox.output_utils import resolve_cr, strip_ansi

        cleaned_stderr = resolve_cr(strip_ansi(stderr)).strip()
        parts.append(cleaned_stderr)

    if interrupted:
        parts.append("<error>Command was aborted before completion</error>")
    elif timed_out and error_message:
        if error_message not in parts:
            parts.append(error_message)
    elif error_message:
        if error_message not in parts:
            parts.append(error_message)
    elif isinstance(exit_code, int) and exit_code != 0:
        parts.append(f"Exit code: {exit_code}")

    return "\n".join(parts)


def _build_background_info(output: dict[str, object]) -> str:
    background_pids = output.get("backgroundPids")
    if not isinstance(background_pids, list) or not background_pids:
        return ""

    background_pid_list = cast(list[object], background_pids)
    task_id_value = background_pid_list[0]
    if not isinstance(task_id_value, (str, int)):
        return ""

    task_id = str(task_id_value)
    output_path = _string_field(output.get("stdout_file")) or _string_field(output.get("output_dir"))
    if output_path:
        return f"Command running in background with ID: {task_id}. Output is being written to: {output_path}"
    return f"Command running in background with ID: {task_id}."


def _extract_error_message(error_value: object) -> str:
    if isinstance(error_value, dict):
        error_dict = cast(dict[str, object], error_value)
        message = error_dict.get("message")
        if isinstance(message, str):
            return message.strip()

    if isinstance(error_value, str):
        return error_value.strip()

    return ""


def _string_field(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


__all__ = ["format_run_shell_command_output"]
