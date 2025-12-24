from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from diskcache import Cache
from pytest import CaptureFixture

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.tool import tool as tool_module
from nexau.archs.tool.tool import Tool


@pytest.fixture
def dummy_cache(monkeypatch, tmp_path: Path) -> Cache:
    cache = Cache(tmp_path / "cache")
    monkeypatch.setattr(tool_module, "cache", cache)
    yield cache
    cache.close()


@pytest.fixture
def validator_tool() -> Tool:
    return Tool(
        name="validator",
        description="desc",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        implementation=lambda x: x,
    )


def test_cache_result_skips_agent_state_in_cache_key(agent_state: AgentState, dummy_cache):
    class Counter:
        def __init__(self):
            self.calls = 0

        @tool_module.cache_result
        def add(self, value: int, agent_state: Any | None = None):
            self.calls += 1
            return self.calls

    counter = Counter()

    first = counter.add(1, agent_state=agent_state)
    second = counter.add(1, agent_state=agent_state)

    assert first == second == 1
    assert counter.calls == 1  # second call served from cache
    assert len(dummy_cache) == 1


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


@pytest.mark.parametrize("reserved_field", ["global_storage", "agent_state"])
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


def test_execute_dynamic_import_wrapped_with_cache(monkeypatch, dummy_cache):
    call_counter: dict[str, int] = {"count": 0}

    def imported_impl(x: int):
        call_counter["count"] += 1
        return {"value": x}

    with patch("nexau.archs.config.config_loader.import_from_string", return_value=imported_impl) as mock_import:
        tool = Tool(
            name="cached_tool",
            description="desc",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
            implementation="pkg:func",
            use_cache=True,
        )

        first = tool.execute(x=1)
        second = tool.execute(x=1)

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert call_counter["count"] == 1  # cached by cache_result wrapper
    assert mock_import.call_count == 1
    assert len(dummy_cache) == 1


def test_execute_without_implementation_raises():
    tool = Tool(name="no_impl", description="desc", input_schema={}, implementation=None)

    with pytest.raises(ValueError, match="no implementation"):
        tool.execute()


def test_execute_import_returns_none_reports_error(monkeypatch):
    with patch("nexau.archs.config.config_loader.import_from_string", return_value=None):
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


def test_execute_invalid_params_raise_value_error(validator_tool: Tool):
    with pytest.raises(ValueError, match="Invalid parameters"):
        validator_tool.execute(x="not-an-int")


def test_validate_params_returns_false_on_schema_error(
    capsys: CaptureFixture[str],
    validator_tool: Tool,
):
    assert not validator_tool.validate_params({"x": "bad"})
    captured = capsys.readouterr()
    assert "Invalid parameters" in captured.out


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
