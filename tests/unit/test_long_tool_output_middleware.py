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

"""Unit tests for LongToolOutputMiddleware."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import AfterToolHookInput
from nexau.archs.main_sub.execution.middleware.long_tool_output import (
    LongToolOutputMiddleware,
)
from nexau.archs.sandbox.base_sandbox import (
    BaseSandbox,
    FileOperationResult,
    SandboxStatus,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Return a mock BaseSandbox whose write_file always succeeds."""
    sandbox = MagicMock(spec=BaseSandbox)
    sandbox.write_file.return_value = FileOperationResult(
        status=SandboxStatus.SUCCESS,
        file_path="",
    )
    return sandbox


def _make_hook_input(
    agent_state: AgentState,
    tool_output: object,
    sandbox: BaseSandbox | None = None,
    tool_name: str = "test_tool",
    tool_call_id: str = "call_abc12345",
) -> AfterToolHookInput:
    """Helper to construct an AfterToolHookInput."""
    return AfterToolHookInput(
        agent_state=agent_state,
        sandbox=sandbox,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_input={"query": "hello"},
        tool_output=tool_output,
    )


def _long_text(lines: int = 200, line_length: int = 80) -> str:
    """Generate a multi-line string guaranteed to be long."""
    return "\n".join(f"Line {i:04d}: {'x' * line_length}" for i in range(lines))


def _make_middleware(
    *,
    max_output_chars: int = 10_000,
    head_lines: int = 50,
    tail_lines: int = 30,
    head_chars: int = 0,
    tail_chars: int = 0,
    temp_dir: str | None = "/tmp/nexau_tool_outputs",
    bypass_tool_names: list[str] | None = None,
) -> LongToolOutputMiddleware:
    """Build middleware with explicit test char budgets unless overridden."""
    return LongToolOutputMiddleware(
        max_output_chars=max_output_chars,
        head_lines=head_lines,
        tail_lines=tail_lines,
        head_chars=head_chars,
        tail_chars=tail_chars,
        temp_dir=temp_dir,
        bypass_tool_names=bypass_tool_names,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_max_output_chars(self):
        with pytest.raises(ValueError, match="max_output_chars"):
            LongToolOutputMiddleware(max_output_chars=0)

    def test_invalid_head_lines(self):
        with pytest.raises(ValueError, match="head_lines"):
            LongToolOutputMiddleware(head_lines=-1)

    def test_invalid_tail_lines(self):
        with pytest.raises(ValueError, match="tail_lines"):
            LongToolOutputMiddleware(tail_lines=-1)

    def test_invalid_head_chars(self):
        with pytest.raises(ValueError, match="head_chars"):
            LongToolOutputMiddleware(head_chars=-1)

    def test_invalid_tail_chars(self):
        with pytest.raises(ValueError, match="tail_chars"):
            LongToolOutputMiddleware(tail_chars=-1)

    def test_invalid_char_budget(self):
        with pytest.raises(ValueError, match="head_chars \\+ tail_chars"):
            LongToolOutputMiddleware(max_output_chars=100, head_chars=51, tail_chars=50)

    def test_default_temp_dir(self):
        mw = LongToolOutputMiddleware()
        assert mw.temp_dir == "/tmp/nexau_tool_outputs"

    def test_custom_temp_dir(self):
        mw = LongToolOutputMiddleware(temp_dir="/custom/path")
        assert mw.temp_dir == "/custom/path"


# ---------------------------------------------------------------------------
# Short output – no truncation
# ---------------------------------------------------------------------------


class TestShortOutput:
    def test_string_output_passes_through(self, agent_state: AgentState, mock_sandbox):
        mw = _make_middleware(max_output_chars=1000)
        hook_input = _make_hook_input(agent_state, "short output", sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        assert not result.has_modifications()
        # Sandbox should not be called for short outputs
        mock_sandbox.write_file.assert_not_called()

    def test_dict_output_passes_through(self, agent_state: AgentState, mock_sandbox):
        mw = _make_middleware(max_output_chars=1000)
        output = {"content": "small result", "returnDisplay": "ok"}
        hook_input = _make_hook_input(agent_state, output, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        assert not result.has_modifications()
        mock_sandbox.write_file.assert_not_called()

    def test_large_return_display_excluded_from_length_measurement(self, agent_state: AgentState, mock_sandbox):
        """returnDisplay is a display-only field stripped before sending to the LLM.

        A large returnDisplay should not inflate the size check and trigger
        unnecessary truncation when the actual content is small.
        """
        small_content = "small result"
        huge_display = "x" * 50_000  # way over any reasonable threshold
        output = {"content": small_content, "returnDisplay": huge_display}
        mw = _make_middleware(max_output_chars=1000)
        hook_input = _make_hook_input(agent_state, output, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        # Should NOT truncate — content itself is tiny
        assert not result.has_modifications()
        mock_sandbox.write_file.assert_not_called()


# ---------------------------------------------------------------------------
# Long output – truncation
# ---------------------------------------------------------------------------


class TestLongOutput:
    def test_string_output_is_truncated(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        assert result.has_modifications()
        assert result.tool_output is not None
        truncated = result.tool_output
        assert isinstance(truncated, str)
        # Head lines present
        assert "Line 0000:" in truncated
        assert "Line 0001:" in truncated
        assert "Line 0002:" in truncated
        # Tail lines present
        assert "Line 0199:" in truncated
        assert "Line 0198:" in truncated
        # Middle omitted marker
        assert "lines omitted" in truncated
        # Hint present
        assert "LongToolOutputMiddleware" in truncated
        assert "read file tool" in truncated
        # Sandbox was called
        mock_sandbox.write_file.assert_called_once()

    def test_dict_with_content_key(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        output = {"content": long, "returnDisplay": "display text"}
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=5,
            tail_lines=5,
        )
        hook_input = _make_hook_input(agent_state, output, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        assert result.has_modifications()
        new_output = result.tool_output
        assert isinstance(new_output, dict)
        # returnDisplay should be preserved
        assert new_output["returnDisplay"] == "display text"
        # content should be truncated
        assert "lines omitted" in new_output["content"]
        assert "LongToolOutputMiddleware" in new_output["content"]

    def test_dict_with_large_return_display_does_not_distort_measurement(self, agent_state: AgentState, mock_sandbox):
        """When both content and returnDisplay are large, the truncation
        threshold and stats should be based on content alone, not on the
        combined serialization that includes returnDisplay.
        """
        long_content = _long_text(200)
        huge_display = "D" * 100_000
        output = {"content": long_content, "returnDisplay": huge_display}
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=5,
            tail_lines=5,
        )
        hook_input = _make_hook_input(agent_state, output, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        assert result.has_modifications()
        new_output = result.tool_output
        assert isinstance(new_output, dict)
        # returnDisplay must be preserved unchanged
        assert new_output["returnDisplay"] == huge_display
        # content should be truncated
        assert "lines omitted" in new_output["content"]
        assert "LongToolOutputMiddleware" in new_output["content"]

    def test_dict_with_result_key(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        output = {"result": long}
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=5,
            tail_lines=5,
        )
        hook_input = _make_hook_input(agent_state, output, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        assert result.has_modifications()
        new_output = result.tool_output
        assert isinstance(new_output, dict)
        assert "lines omitted" in new_output["result"]

    def test_dict_without_known_keys(self, agent_state: AgentState, mock_sandbox):
        # dict with lots of data but no "content" or "result" key
        output = {f"field_{i}": "x" * 500 for i in range(50)}
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
        )
        hook_input = _make_hook_input(agent_state, output, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        assert result.has_modifications()
        new_output = result.tool_output
        assert isinstance(new_output, dict)
        assert "content" in new_output
        assert "LongToolOutputMiddleware" in new_output["content"]


# ---------------------------------------------------------------------------
# Sandbox write_file interaction
# ---------------------------------------------------------------------------


class TestSandboxWriteFile:
    def test_write_file_called_with_correct_args(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            temp_dir="/my/custom/dir",
        )
        hook_input = _make_hook_input(
            agent_state,
            long,
            sandbox=mock_sandbox,
            tool_name="my_tool",
            tool_call_id="call_12345678",
        )

        mw.after_tool(hook_input)

        mock_sandbox.write_file.assert_called_once()
        call_kwargs = mock_sandbox.write_file.call_args
        # Check the file_path starts with the temp_dir
        assert call_kwargs.kwargs["file_path"].startswith("/my/custom/dir/")
        assert "my_tool" in call_kwargs.kwargs["file_path"]
        assert call_kwargs.kwargs["file_path"].endswith(".txt")
        # Full text is written
        assert call_kwargs.kwargs["content"] == long
        assert call_kwargs.kwargs["encoding"] == "utf-8"
        assert call_kwargs.kwargs["create_directories"] is True

    def test_file_path_appears_in_hint(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            temp_dir="/tmp/nexau_test",
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)

        truncated = result.tool_output
        assert isinstance(truncated, str)
        # The hint should contain the temp_dir path
        assert "/tmp/nexau_test/" in truncated
        assert ".txt" in truncated

    def test_no_sandbox_raises_error(self, agent_state: AgentState):
        """When no sandbox is available and temp_dir is set, the middleware should raise."""
        long = _long_text(200)
        mw = _make_middleware(max_output_chars=100, head_lines=3, tail_lines=2)
        hook_input = _make_hook_input(agent_state, long, sandbox=None)

        with pytest.raises(RuntimeError, match="No sandbox available"):
            mw.after_tool(hook_input)

    def test_sandbox_write_failure_raises_error(self, agent_state: AgentState):
        """When sandbox.write_file fails, the middleware should raise."""
        long = _long_text(200)
        failing_sandbox = MagicMock(spec=BaseSandbox)
        failing_sandbox.write_file.return_value = FileOperationResult(
            status=SandboxStatus.ERROR,
            file_path="/tmp/test.txt",
            error="disk full",
        )

        mw = _make_middleware(max_output_chars=100, head_lines=3, tail_lines=2)
        hook_input = _make_hook_input(agent_state, long, sandbox=failing_sandbox)

        with pytest.raises(RuntimeError, match="Failed to write temp file"):
            mw.after_tool(hook_input)

    def test_dict_output_full_json_saved_to_sandbox(self, agent_state: AgentState, mock_sandbox):
        """For dict outputs, the full JSON serialization is saved."""
        big_dict = {"data": ["item"] * 5000}
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=2,
            tail_lines=2,
        )
        hook_input = _make_hook_input(agent_state, big_dict, sandbox=mock_sandbox)

        mw.after_tool(hook_input)

        call_kwargs = mock_sandbox.write_file.call_args
        saved_text = call_kwargs.kwargs["content"]
        # Verify the saved content is valid JSON matching the original dict
        parsed = json.loads(saved_text)
        assert parsed == big_dict


# ---------------------------------------------------------------------------
# Truncation logic details
# ---------------------------------------------------------------------------


class TestTruncationLogic:
    def test_head_only_no_tail(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=5,
            tail_lines=0,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)

        assert "Line 0004:" in truncated
        assert "Line 0199:" not in truncated
        assert "lines omitted" in truncated

    def test_tail_only_no_head(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=0,
            tail_lines=5,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)

        assert "Line 0199:" in truncated
        assert "Line 0000:" not in truncated
        assert "lines omitted" in truncated

    def test_few_lines_but_long_chars(self, agent_state: AgentState, mock_sandbox):
        """When there are fewer lines than head+tail but chars exceed threshold."""
        # 3 very long lines
        long = "\n".join(["A" * 5000] * 3)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=5,
            tail_lines=5,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        # Still triggers (by char count) but _truncate returns full text since
        # 3 lines < 5+5 → truncated text equals original
        assert result.has_modifications()
        truncated = result.tool_output
        assert isinstance(truncated, str)
        # The hint is still appended even if line-based truncation didn't cut
        assert "LongToolOutputMiddleware" in truncated

    def test_omitted_count_is_correct(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(100)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=10,
            tail_lines=5,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)
        assert "[85 lines omitted]" in truncated

    def test_char_based_truncation(self, agent_state: AgentState, mock_sandbox):
        long = "BEGIN-" + ("x" * 500) + "-END"
        mw = LongToolOutputMiddleware(
            max_output_chars=100,
            head_lines=0,
            tail_lines=0,
            head_chars=10,
            tail_chars=8,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)

        assert "BEGIN-" in truncated
        assert "-END" in truncated
        assert "chars omitted" in truncated

    def test_uses_longer_char_based_truncation_when_both_configured(self, agent_state: AgentState, mock_sandbox):
        long = "A" * 400
        mw = LongToolOutputMiddleware(
            max_output_chars=100,
            head_lines=1,
            tail_lines=1,
            head_chars=30,
            tail_chars=20,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)

        assert "chars omitted" in truncated
        assert "lines omitted" not in truncated

    def test_uses_longer_line_based_truncation_when_both_configured(self, agent_state: AgentState, mock_sandbox):
        long = _long_text(100, line_length=20)
        mw = LongToolOutputMiddleware(
            max_output_chars=400,
            head_lines=5,
            tail_lines=5,
            head_chars=10,
            tail_chars=10,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)

        assert "lines omitted" in truncated
        assert "chars omitted" not in truncated


# ---------------------------------------------------------------------------
# Serialization edge cases
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_non_serializable_output(self, agent_state: AgentState, mock_sandbox):
        """Objects that can't be JSON-serialized fall back to str()."""

        class Weird:
            def __str__(self):
                return "w" * 20_000

        mw = _make_middleware(
            max_output_chars=100,
            head_lines=2,
            tail_lines=1,
        )
        hook_input = _make_hook_input(agent_state, Weird(), sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        # Because str(Weird()) is 20k chars, truncation should trigger
        assert result.has_modifications()


# ---------------------------------------------------------------------------
# Multiple calls
# ---------------------------------------------------------------------------


class TestMultipleCalls:
    def test_multiple_tool_calls_produce_separate_sandbox_writes(
        self,
        agent_state: AgentState,
        mock_sandbox,
    ):
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=2,
            tail_lines=2,
        )

        for i in range(3):
            long = _long_text(200)
            hook_input = _make_hook_input(
                agent_state,
                long,
                sandbox=mock_sandbox,
                tool_name=f"tool_{i}",
                tool_call_id=f"call_{i:08d}",
            )
            mw.after_tool(hook_input)

        # sandbox.write_file should have been called 3 times
        assert mock_sandbox.write_file.call_count == 3

        # Each call should have a different filepath containing the tool name
        filepaths = [call.kwargs["file_path"] for call in mock_sandbox.write_file.call_args_list]
        assert any("tool_0" in fp for fp in filepaths)
        assert any("tool_1" in fp for fp in filepaths)
        assert any("tool_2" in fp for fp in filepaths)


# ---------------------------------------------------------------------------
# Bypass tool names
# ---------------------------------------------------------------------------


class TestBypassToolNames:
    def test_bypassed_tool_skips_truncation(self, agent_state: AgentState, mock_sandbox: MagicMock):
        """Tools in bypass list should pass through without truncation."""
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            bypass_tool_names=["execute_bash", "run_code"],
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox, tool_name="execute_bash")

        result = mw.after_tool(hook_input)
        assert not result.has_modifications()
        mock_sandbox.write_file.assert_not_called()

    def test_non_bypassed_tool_still_truncated(self, agent_state: AgentState, mock_sandbox: MagicMock):
        """Tools NOT in bypass list should still be truncated."""
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            bypass_tool_names=["execute_bash"],
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox, tool_name="search_files")

        result = mw.after_tool(hook_input)
        assert result.has_modifications()
        truncated = result.tool_output
        assert isinstance(truncated, str)
        assert "lines omitted" in truncated

    def test_empty_bypass_list_no_effect(self, agent_state: AgentState, mock_sandbox: MagicMock):
        """An empty bypass list should have no effect — all tools truncated."""
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            bypass_tool_names=[],
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox, tool_name="execute_bash")

        result = mw.after_tool(hook_input)
        assert result.has_modifications()

    def test_default_bypass_is_empty(self):
        """By default, no tools are bypassed."""
        mw = LongToolOutputMiddleware()
        assert len(mw._bypass_tool_names) == 0


# ---------------------------------------------------------------------------
# temp_dir=None — truncate without file persistence
# ---------------------------------------------------------------------------


class TestTempDirNone:
    def test_truncates_without_saving(self, agent_state: AgentState, mock_sandbox: MagicMock):
        """When temp_dir is None, truncation happens but no file is written."""
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            temp_dir=None,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        assert result.has_modifications()
        truncated = result.tool_output
        assert isinstance(truncated, str)
        assert "lines omitted" in truncated
        assert "LongToolOutputMiddleware" in truncated
        # Sandbox should NOT be called
        mock_sandbox.write_file.assert_not_called()

    def test_hint_has_no_file_path(self, agent_state: AgentState, mock_sandbox: MagicMock):
        """When temp_dir is None, hint should not mention file path or read_file."""
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            temp_dir=None,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=mock_sandbox)

        result = mw.after_tool(hook_input)
        truncated = result.tool_output
        assert isinstance(truncated, str)
        assert "has been truncated" in truncated
        assert "read file tool" not in truncated
        assert "/tmp/" not in truncated

    def test_no_sandbox_required(self, agent_state: AgentState):
        """When temp_dir is None, sandbox=None should not raise."""
        long = _long_text(200)
        mw = _make_middleware(
            max_output_chars=100,
            head_lines=3,
            tail_lines=2,
            temp_dir=None,
        )
        hook_input = _make_hook_input(agent_state, long, sandbox=None)

        result = mw.after_tool(hook_input)
        assert result.has_modifications()
        truncated = result.tool_output
        assert isinstance(truncated, str)
        assert "has been truncated" in truncated
