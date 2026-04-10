import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from pytest import LogCaptureFixture

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.tool import ConfigError, Tool, normalize_structured_tool_definition


@pytest.fixture
def validator_tool() -> Tool:
    return Tool(
        name="validator",
        description="desc",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        implementation=lambda x: x,
    )


def test_from_yaml_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        Tool.from_yaml("/non/existent/tool.yaml", binding=None)


def test_from_yaml_uses_yaml_binding_when_agent_binding_missing(tmp_path: Path):
    yaml_path = tmp_path / "yaml_binding.tool.yaml"
    yaml_content = {
        "type": "tool",
        "name": "yaml_binding",
        "description": "desc",
        "binding": "pkg.module:func",
        "lazy": True,
        "input_schema": {"type": "object", "properties": {}},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    tool = Tool.from_yaml(str(yaml_path), binding=None)

    assert tool.implementation is None
    assert tool.implementation_import_path == "pkg.module:func"


def test_from_yaml_prefers_agent_binding_over_yaml_binding(tmp_path: Path):
    yaml_path = tmp_path / "yaml_binding_override.tool.yaml"
    yaml_content = {
        "type": "tool",
        "name": "yaml_binding_override",
        "description": "desc",
        "binding": "yaml.module:func",
        "input_schema": {"type": "object", "properties": {}},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    agent_binding = lambda: None  # noqa: E731
    tool = Tool.from_yaml(str(yaml_path), binding=agent_binding)

    assert tool.implementation is agent_binding
    assert tool.implementation_import_path is None


def test_from_yaml_reads_formatter_field(tmp_path: Path):
    yaml_path = tmp_path / "yaml_formatter.tool.yaml"
    yaml_content = {
        "type": "tool",
        "name": "yaml_formatter",
        "description": "desc",
        "formatter": "xml",
        "input_schema": {"type": "object", "properties": {}},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    tool = Tool.from_yaml(str(yaml_path), binding=lambda **_: {"result": "ok"})

    assert tool.formatter == "xml"


@pytest.mark.parametrize("reserved_field", ["global_storage", "agent_state", "ctx"])
def test_from_yaml_rejects_reserved_fields(tmp_path: Path, reserved_field):
    yaml_path = tmp_path / f"{reserved_field}.yaml"
    yaml_content = {
        "type": "tool",
        "name": "bad_tool",
        "description": "bad",
        "input_schema": {
            "type": "object",
            reserved_field: {"type": "string"},
        },
    }
    yaml_path.write_text(yaml.safe_dump(yaml_content))

    with pytest.raises(ValueError, match=reserved_field):
        Tool.from_yaml(str(yaml_path), binding=lambda **_: None)


def test_execute_without_implementation_raises():
    tool = Tool(name="no_impl", description="desc", input_schema={}, implementation=None)

    with pytest.raises(ValueError, match="no implementation"):
        tool.execute()


def test_execute_import_returns_none_reports_error(monkeypatch):
    with patch("nexau.archs.main_sub.utils.import_from_string", return_value=None):
        tool = Tool(
            name="import_none",
            description="desc",
            input_schema={},
            implementation="pkg:missing",
        )
        result = tool.execute()

    assert result["error_type"] == "ValueError"
    assert "no implementation" in result["error"]


def test_execute_maps_agent_state_to_global_storage(agent_state: AgentState):
    captured: dict[str, Any] = {}

    def impl(global_storage):
        captured["value"] = global_storage
        return {"global_storage": global_storage}

    tool = Tool(name="storage_tool", description="desc", input_schema={}, implementation=impl)

    result = tool.execute(agent_state=agent_state)

    assert result["global_storage"] is agent_state.global_storage
    assert captured["value"] is agent_state.global_storage


def test_tool_accepts_framework_context_annotation() -> None:
    def impl(ctx: FrameworkContext | None = None) -> dict[str, str]:
        return {"status": "ok"}

    tool = Tool(name="ctx_ok", description="desc", input_schema={}, implementation=impl)

    assert tool.execute(ctx=None) == {"status": "ok"}


def test_tool_rejects_invalid_ctx_annotation() -> None:
    def impl(ctx: str) -> dict[str, str]:
        return {"status": ctx}

    with pytest.raises(ConfigError, match="declares 'ctx' with incompatible type"):
        Tool(name="ctx_bad", description="desc", input_schema={}, implementation=impl)


def test_tool_warns_when_ctx_annotation_missing(caplog: LogCaptureFixture) -> None:
    def impl(ctx) -> dict[str, str]:
        return {"status": "ok"}

    with caplog.at_level(logging.WARNING, logger="nexau.archs.tool.tool"):
        Tool(name="ctx_untyped", description="desc", input_schema={}, implementation=impl)

    assert "declares 'ctx' without a FrameworkContext annotation" in caplog.text


def test_execute_invalid_params_raise_value_error(validator_tool: Tool):
    with pytest.raises(ValueError, match="Invalid parameters"):
        validator_tool.execute(x="not-an-int")


def test_validate_params_raises_value_error_with_detail(
    validator_tool: Tool,
):
    with pytest.raises(ValueError, match="Invalid parameters") as exc_info:
        validator_tool.validate_params({"x": "bad"})
    assert "'bad' is not of type 'integer'" in str(exc_info.value)


def test_validate_schema_invalid_schema_raises_value_error():
    with pytest.raises(ValueError):
        Tool(
            name="bad_schema",
            description="desc",
            input_schema={"type": "object", "properties": {"x": {"type": "not-a-type"}}},
            implementation=lambda x: x,
        )


def test_get_info_and_string_helpers():
    long_description = "d" * 60

    def impl():
        return {}

    tool = Tool(
        name="info_tool",
        description=long_description,
        skill_description="skill detail",
        input_schema={},
        implementation=impl,
        template_override="tmpl",
    )

    info = tool.get_info()
    assert info["name"] == "info_tool"
    assert info["template_override"] == "tmpl"

    assert "Tool 'info_tool'" in str(tool)
    assert "Skill description" in str(tool)
    assert "implementation=impl" in repr(tool)


def test_tool_defaults_to_xml_formatter() -> None:
    tool = Tool(
        name="xml_default",
        description="desc",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": "Hello from tool\nSecond line"},
    )

    formatted = tool.format_output_for_llm(
        tool_input={},
        tool_output={"result": "Hello from tool\nSecond line", "status": "success"},
        tool_call_id="call_1",
        is_error=False,
    )

    assert isinstance(formatted, str)
    assert "<tool_result>" in formatted
    assert '<body field="result">' in formatted
    assert "Hello from tool" in formatted


def test_tool_xml_formatter_unwraps_single_content_field() -> None:
    tool = Tool(
        name="content_only",
        description="desc",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"content": "Plain content for llm"},
    )

    formatted = tool.format_output_for_llm(
        tool_input={},
        tool_output={"content": "Plain content for llm"},
        tool_call_id="call_1",
        is_error=False,
    )

    assert formatted == "Plain content for llm"


def test_tool_xml_formatter_unwraps_single_result_field() -> None:
    tool = Tool(
        name="result_only",
        description="desc",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"result": "Plain result for llm"},
    )

    formatted = tool.format_output_for_llm(
        tool_input={},
        tool_output={"result": "Plain result for llm"},
        tool_call_id="call_1",
        is_error=False,
    )

    assert formatted == "Plain result for llm"


def test_tool_import_path_formatter_is_used() -> None:
    tool = Tool(
        name="agent_tool",
        description="desc",
        input_schema={"type": "object", "properties": {}},
        implementation=lambda: {"status": "success"},
        formatter="nexau.archs.tool.formatters.agent:format_agent_tool_output",
    )

    formatted = tool.format_output_for_llm(
        tool_input={},
        tool_output={
            "status": "success",
            "sub_agent_name": "explore",
            "sub_agent_id": "sub-123",
            "result": "## Answer\n\nDone.",
        },
        tool_call_id="call_1",
        is_error=False,
    )

    assert formatted == "Sub-agent finished (sub_agent_name: explore, sub_agent_id: sub-123).\n\n## Answer\n\nDone."


@pytest.mark.parametrize(
    "tool_kwargs, expected_props",
    [
        (
            {"name": "empty", "description": "desc", "input_schema": {}, "implementation": lambda: None},
            [],
        ),
        (
            {
                "name": "no_type",
                "description": "desc",
                "input_schema": {"properties": {"x": {"type": "string"}}},
                "implementation": lambda x: x,
            },
            ["x"],
        ),
    ],
)
def test_to_openai_and_anthropic_fill_missing_schema_fields(tool_kwargs, expected_props):
    tool = Tool(**tool_kwargs)

    openai_params = tool.to_openai()["function"]["parameters"]
    anthropic_params = tool.to_anthropic()["input_schema"]

    assert openai_params["type"] == anthropic_params["type"] == "object"

    for prop in expected_props:
        assert prop in openai_params.get("properties", {})
        assert prop in anthropic_params.get("properties", {})


def test_to_structured_definition_uses_skill_description_for_structured_models():
    tool = Tool(
        name="search",
        description="Full description",
        skill_description="Brief description",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        implementation=lambda query: query,
        as_skill=True,
    )

    structured_def = tool.to_structured_definition(description=tool.get_structured_description())

    assert structured_def == {
        "name": "search",
        "description": "Brief description",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "kind": "tool",
    }


def test_normalize_structured_tool_definition_accepts_legacy_openai_shape():
    openai_tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"properties": {"path": {"type": "string"}}},
        },
    }

    normalized = normalize_structured_tool_definition(openai_tool)

    assert normalized == {
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        "kind": "tool",
    }
