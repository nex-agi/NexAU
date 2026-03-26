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

"""Additional coverage tests for tool.py.

Targets uncovered paths: execute with async impl in sync context,
_resolve_reserved_param_annotation failure, execute_async agent_state filtering,
_is_framework_context_annotation edge cases.
"""

from typing import Annotated, Any
from unittest.mock import Mock

import pytest

from nexau.archs.tool.tool import Tool

# ---------------------------------------------------------------------------
# Tool.execute — async impl from sync context
# ---------------------------------------------------------------------------


class TestToolExecuteAsyncFromSync:
    def test_async_impl_run_in_sync_no_loop(self):
        """When no running loop, async tool impl runs via asyncio.run()."""

        async def async_impl():
            return {"async_result": True}

        tool = Tool(
            name="async_sync",
            description="async tool called from sync",
            input_schema={"type": "object", "properties": {}},
            implementation=async_impl,
        )
        # Execute from a sync context (no running loop)
        result = tool.execute()
        assert result == {"async_result": True}

    def test_no_implementation_raises(self):
        tool = Tool(
            name="no_impl",
            description="no impl",
            input_schema={"type": "object", "properties": {}},
            implementation=None,
        )
        with pytest.raises(ValueError, match="no implementation"):
            tool.execute()


# ---------------------------------------------------------------------------
# Tool.execute — agent_state / global_storage / sandbox injection
# ---------------------------------------------------------------------------


class TestToolExecuteInjection:
    def test_agent_state_not_in_sig_filters_out(self):
        """agent_state is filtered out when not in function signature."""

        def impl(x: int = 1):
            return {"x": x}

        tool = Tool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            implementation=impl,
        )
        state = Mock()
        state.global_storage = Mock()
        result = tool.execute(agent_state=state, x=42)
        assert result == {"x": 42}

    def test_agent_state_converted_to_global_storage(self):
        """When impl has global_storage param but not agent_state, convert."""

        def impl(global_storage=None):
            return {"has_gs": global_storage is not None}

        tool = Tool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            implementation=impl,
        )
        state = Mock()
        state.global_storage = {"key": "value"}
        result = tool.execute(agent_state=state)
        assert result["has_gs"] is True

    def test_ctx_filtered_when_not_in_sig(self):
        """ctx is filtered out when not in function signature."""

        def impl():
            return {"ok": True}

        tool = Tool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            implementation=impl,
        )
        result = tool.execute(ctx=Mock())
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# Tool._validate_reserved_param_annotations — edge cases
# ---------------------------------------------------------------------------


class TestValidateReservedParamAnnotationsEdgeCases:
    def test_ctx_without_annotation_warns(self):
        def impl(ctx):
            pass

        # Should not raise, just warn
        tool = Tool(name="t", description="d", input_schema={}, implementation=impl)
        assert tool is not None

    def test_ctx_as_any_warns(self):
        def impl(ctx: Any):
            pass

        # Should not raise, just warn
        tool = Tool(name="t", description="d", input_schema={}, implementation=impl)
        assert tool is not None

    def test_ctx_unresolvable_annotation_warns(self):
        """When type hints can't be resolved, just warn."""

        def impl(ctx: "NonExistentType"):  # type: ignore[name-defined]  # noqa: F821
            pass

        # The annotation can't be resolved; tool should still be created
        tool = Tool(name="t", description="d", input_schema={}, implementation=impl)
        assert tool is not None


# ---------------------------------------------------------------------------
# Tool._is_framework_context_annotation — edge cases
# ---------------------------------------------------------------------------


class TestIsFrameworkContextAnnotation:
    def test_none_origin(self):
        assert Tool._is_framework_context_annotation(int) is False

    def test_union_with_multiple_types(self):
        """Union with multiple non-None types returns False."""
        assert Tool._is_framework_context_annotation(int | str) is False

    def test_annotated_with_non_framework_context(self):
        """Annotated with non-FrameworkContext returns False."""
        assert Tool._is_framework_context_annotation(Annotated[int, "meta"]) is False


# ---------------------------------------------------------------------------
# Tool.execute_async — agent_state injection paths
# ---------------------------------------------------------------------------


class TestToolExecuteAsyncInjection:
    @pytest.mark.anyio
    async def test_agent_state_filtered_for_async_impl(self):
        async def impl():
            return {"ok": True}

        tool = Tool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            implementation=impl,
        )
        state = Mock()
        state.global_storage = Mock()
        result = await tool.execute_async(agent_state=state)
        assert result == {"ok": True}

    @pytest.mark.anyio
    async def test_sandbox_filtered_for_async_impl(self):
        async def impl():
            return {"ok": True}

        tool = Tool(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            implementation=impl,
        )
        result = await tool.execute_async(sandbox=Mock(), agent_state=Mock())
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# Tool.from_yaml — reserved field detection
# ---------------------------------------------------------------------------


class TestToolFromYamlReservedFields:
    def test_ctx_in_schema_raises(self, tmp_path):
        yaml_content = """
name: test_tool
description: Test tool
input_schema:
  ctx:
    type: string
"""
        yaml_file = tmp_path / "tool.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="ctx"):
            Tool.from_yaml(str(yaml_file))

    def test_agent_state_in_schema_raises(self, tmp_path):
        yaml_content = """
name: test_tool
description: Test tool
input_schema:
  agent_state:
    type: string
"""
        yaml_file = tmp_path / "tool.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="agent_state"):
            Tool.from_yaml(str(yaml_file))

    def test_global_storage_in_schema_raises(self, tmp_path):
        yaml_content = """
name: test_tool
description: Test tool
input_schema:
  global_storage:
    type: string
"""
        yaml_file = tmp_path / "tool.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="global_storage"):
            Tool.from_yaml(str(yaml_file))


# ---------------------------------------------------------------------------
# Tool — validate_params with invalid input
# ---------------------------------------------------------------------------


class TestToolValidateParams:
    def test_invalid_param_type_raises(self):
        tool = Tool(
            name="t",
            description="d",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            implementation=lambda x: {"x": x},
        )
        with pytest.raises(Exception):
            tool.validate_params({"x": "not_an_integer"})
