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

"""Unit tests for configuration loading and normalization."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nexau.archs.main_sub.config import ConfigError as ConfigConfigError
from nexau.archs.main_sub.config.config import AgentConfig, AgentConfigBuilder, ExecutionConfig
from nexau.archs.main_sub.config.schema import (
    AgentConfigSchema,
    normalize_agent_config_dict,
)
from nexau.archs.main_sub.config.schema import (
    ConfigError as SchemaConfigError,
)
from nexau.archs.main_sub.utils.common import ConfigError as UtilsConfigError
from nexau.archs.main_sub.utils.common import load_yaml_with_vars
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.core import BaseTracer, Span, SpanType

MODULE_PATH = __name__


def sample_hook_fn(value: int = 0) -> int:
    """Simple hook used for import tests."""

    return value


class SampleHookClass:
    """Callable hook class to verify parameter instantiation."""

    def __init__(self, value: int):
        self.value = value

    def __call__(self, *args: Any, **kwargs: Any) -> int:  # pragma: no cover - trivial passthrough
        return self.value


class ConfigDummyTracer(BaseTracer):
    """Concrete tracer for testing."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self.started: list[Span] = []

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        span = Span(
            id=name,
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )
        self.started.append(span)
        return span

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        span.outputs = outputs or {}
        span.error = str(error) if error else None
        span.attributes.update(attributes or {})


def sample_token_counter(messages: list[dict[str, Any]], offset: int = 0) -> int:
    """Token counter used to validate token_counter wiring."""

    return len(messages) + offset


class ImportableTracer(ConfigDummyTracer):
    """Tracer referenced via import string."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__(label or "importable")


class StubTemplate:
    def __init__(self, template: str):
        self.template = template

    def render(self, context: dict[str, Any]) -> str:
        tool = context.get("tool")
        return f"{self.template}-{getattr(tool, 'name', 'unknown')}"


class StubPromptBuilder:
    """Lightweight PromptBuilder stand-in to avoid reading templates from disk."""

    def load_prompt_template(self, _: str) -> str:
        return "skill-detail"

    @property
    def jinja_env(self):
        class _Env:
            @staticmethod
            def from_string(template: str) -> StubTemplate:
                return StubTemplate(template)

        return _Env()


class TestLoadYamlWithVars:
    """Tests for YAML loading helpers."""

    def test_replaces_env_and_this_file_dir(self, temp_dir, monkeypatch):
        monkeypatch.setenv("TEST_ENV_VAR", "hello")
        content = textwrap.dedent(
            """
            system_prompt: "Here ${this_file_dir}"
            value: ${env.TEST_ENV_VAR}
            """,
        )
        path = Path(temp_dir) / "config.yaml"
        path.write_text(content)

        result = load_yaml_with_vars(path)

        assert "Here" in result["system_prompt"]
        assert temp_dir in result["system_prompt"]
        assert result["value"] == "hello"

    def test_yaml_variables_block_resolved_and_removed(self, temp_dir):
        content = textwrap.dedent(
            """
            variables:
              model_name: gpt-4o
            name: test_agent
            prompt: ${variables.model_name}
            llm_config:
              model: ${variables.model_name}
            """,
        )
        config_file = Path(temp_dir) / "config.yaml"
        config_file.write_text(content)

        result = load_yaml_with_vars(config_file)

        assert result["prompt"] == "gpt-4o"
        assert result["llm_config"]["model"] == "gpt-4o"
        assert "variables" not in result

    def test_missing_variable_raises(self, temp_dir):
        content = textwrap.dedent(
            """
            variables:
              foo: bar
            value: ${variables.unknown}
            """,
        )
        path = Path(temp_dir) / "missing.yaml"
        path.write_text(content)

        with pytest.raises(UtilsConfigError, match="unknown"):
            load_yaml_with_vars(path)

    def test_non_scalar_embedding_raises(self, temp_dir):
        content = textwrap.dedent(
            """
            variables:
              mapping:
                key: value
            value: "prefix-${variables.mapping}"
            """,
        )
        path = Path(temp_dir) / "non_scalar.yaml"
        path.write_text(content)

        with pytest.raises(UtilsConfigError, match="non-scalar value"):
            load_yaml_with_vars(path)

    def test_invalid_variables_block_raises(self, temp_dir):
        path = Path(temp_dir) / "bad.yaml"
        path.write_text("variables: true\nname: tester\n")

        with pytest.raises(UtilsConfigError, match="must be a mapping"):
            load_yaml_with_vars(path)


class TestSchemaNormalization:
    """Tests around AgentConfigSchema normalization."""

    def test_normalize_agent_config_dict_round_trips(self):
        config = {
            "name": "demo",
            "llm_config": {"model": "gpt-4o-mini"},
            "tools": [],
            "sub_agents": [],
            "max_iterations": 5,
        }

        normalized = normalize_agent_config_dict(config)

        assert normalized["name"] == "demo"
        assert normalized["llm_config"]["model"] == "gpt-4o-mini"
        assert normalized["max_iterations"] == 5
        assert "type" not in normalized or normalized.get("type") == "agent"

    def test_normalize_agent_config_dict_reports_invalid_tools(self):
        config = {"name": "demo", "llm_config": {"model": "gpt-4o-mini"}, "tools": "not-a-list"}

        with pytest.raises(SchemaConfigError, match="tools"):
            normalize_agent_config_dict(config)

    def test_normalize_agent_config_dict_reports_unknown_fields(self):
        config = {
            "name": "demo",
            "llm_config": {"model": "gpt-4o-mini"},
            "tools": [],
            "unexpected": True,
        }

        with pytest.raises(SchemaConfigError, match="unexpected"):
            normalize_agent_config_dict(config)

    def test_agent_config_schema_from_yaml_errors_on_missing_or_empty(self, temp_dir):
        missing_path = Path(temp_dir) / "missing.yaml"
        with pytest.raises(SchemaConfigError, match="not found"):
            AgentConfigSchema.from_yaml(str(missing_path))

        empty_path = Path(temp_dir) / "empty.yaml"
        empty_path.write_text("")
        with pytest.raises(SchemaConfigError, match="Empty or invalid"):
            AgentConfigSchema.from_yaml(str(empty_path))

    def test_agent_config_schema_from_yaml_invalid_yaml(self, temp_dir):
        bad_path = Path(temp_dir) / "bad.yaml"
        bad_path.write_text("name: test\nllm_config: [unterminated")

        with pytest.raises(SchemaConfigError, match="YAML parsing error"):
            AgentConfigSchema.from_yaml(str(bad_path))


class TestAgentConfigBuilderCore:
    """Tests for core builder behaviors."""

    def test_build_core_properties_sets_defaults(self, temp_dir):
        builder = AgentConfigBuilder({"name": "builder"}, Path(temp_dir))
        builder.build_core_properties()

        assert builder.agent_params["name"] == "builder"
        assert builder.agent_params["max_context_tokens"] == 128000
        assert builder.agent_params["max_running_subagents"] == 5
        assert builder.agent_params["stop_tools"] == set()

    def test_build_mcp_servers_validates_entries(self, temp_dir):
        config = {
            "mcp_servers": [
                {"name": "stdio", "type": "stdio", "command": "python"},
                {"name": "http", "type": "http", "url": "http://localhost"},
            ]
        }
        builder = AgentConfigBuilder(config, Path(temp_dir))
        builder.build_mcp_servers()

        assert len(builder.agent_params["mcp_servers"]) == 2
        assert builder.agent_params["mcp_servers"][0]["name"] == "stdio"
        assert builder.agent_params["mcp_servers"][1]["type"] == "http"

    def test_build_mcp_servers_rejects_non_list(self, temp_dir):
        builder = AgentConfigBuilder({"mcp_servers": "invalid"}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="must be a list"):
            builder.build_mcp_servers()

    def test_build_mcp_servers_http_requires_url(self, temp_dir):
        builder = AgentConfigBuilder({"mcp_servers": [{"name": "http_svr", "type": "http"}]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="missing 'url'"):
            builder.build_mcp_servers()

    def test_build_hooks_supports_strings_dicts_and_callables(self, temp_dir):
        hook_import = f"{MODULE_PATH}:sample_hook_fn"
        hook_dict = {"import": f"{MODULE_PATH}:SampleHookClass", "params": {"value": 5}}

        builder = AgentConfigBuilder(
            {
                "after_model_hooks": [hook_import, hook_dict, sample_hook_fn],
                "middlewares": [hook_import],
            },
            Path(temp_dir),
        )

        builder.build_hooks()

        assert len(builder.agent_params["after_model_hooks"]) == 3
        assert builder.agent_params["after_model_hooks"][0](value=2) == 2
        assert builder.agent_params["after_model_hooks"][1]() == 5
        assert builder.agent_params["middlewares"][0](value=1) == 1

    def test_build_hooks_rejects_bad_types(self, temp_dir):
        builder = AgentConfigBuilder({"after_model_hooks": "oops"}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="'after_model_hooks' must be a list"):
            builder.build_hooks()

        builder_bad_entry = AgentConfigBuilder({"before_tool_hooks": [123]}, Path(temp_dir))
        with pytest.raises(ConfigConfigError, match="before_tool_hooks entry 0"):
            builder_bad_entry.build_hooks()

    def test_build_hooks_requires_import_field(self, temp_dir):
        builder = AgentConfigBuilder({"after_tool_hooks": [{"params": {}}]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="missing 'import' field"):
            builder.build_hooks()

    def test_build_hooks_wraps_import_errors(self, temp_dir):
        builder = AgentConfigBuilder({"middlewares": ["nexau.invalid.module:missing"]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="Error loading middleware 0"):
            builder.build_hooks()

    def test_build_hooks_params_must_be_mapping(self, temp_dir):
        hook_dict = {"import": f"{MODULE_PATH}:sample_hook_fn", "params": "oops"}
        builder = AgentConfigBuilder({"after_tool_hooks": [hook_dict]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="params"):
            builder.build_hooks()

    def test_build_hooks_non_callable_with_params(self, temp_dir):
        class NotCallable:
            pass

        with patch("nexau.archs.main_sub.config.config.import_from_string", return_value=NotCallable()):
            hook_dict = {"import": "module:obj", "params": {"x": 1}}
            builder = AgentConfigBuilder({"before_model_hooks": [hook_dict]}, Path(temp_dir))

            with pytest.raises(ConfigConfigError, match="not callable"):
                builder.build_hooks()

    def test_build_tracers_accepts_instances_and_import_strings(self, temp_dir):
        tracer_import = f"{MODULE_PATH}:ImportableTracer"
        instance = ConfigDummyTracer()
        builder = AgentConfigBuilder({"tracers": [instance, tracer_import]}, Path(temp_dir))

        builder.build_tracers()

        tracers = builder.agent_params["tracers"]
        assert len(tracers) == 2
        assert isinstance(tracers[0], ConfigDummyTracer)
        assert isinstance(tracers[1], ImportableTracer)

    def test_build_tracers_rejects_invalid_entries(self, temp_dir):
        builder = AgentConfigBuilder({"tracers": "bad"}, Path(temp_dir))
        with pytest.raises(ConfigConfigError, match="'tracers' must be a list"):
            builder.build_tracers()

        builder_bad = AgentConfigBuilder({"tracers": [123]}, Path(temp_dir))
        with pytest.raises(ConfigConfigError, match="Tracer entry must be"):
            builder_bad.build_tracers()

        builder_not_tracer = AgentConfigBuilder({"tracers": [f"{MODULE_PATH}:sample_hook_fn"]}, Path(temp_dir))
        with pytest.raises(ConfigConfigError, match="Tracer must be an instance"):
            builder_not_tracer.build_tracers()

    def test_build_tracers_null_entry(self, temp_dir):
        builder = AgentConfigBuilder({"tracers": [None]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="cannot be null"):
            builder.build_tracers()

    def test_build_tracers_wraps_import_errors(self, temp_dir):
        with patch.object(AgentConfigBuilder, "_import_and_instantiate", side_effect=RuntimeError("boom")):
            builder = AgentConfigBuilder({"tracers": ["module:Tracer"]}, Path(temp_dir))

            with pytest.raises(ConfigConfigError, match="Error loading tracer"):
                builder.build_tracers()

    def test_build_llm_config_and_token_counter(self, temp_dir):
        builder = AgentConfigBuilder(
            {
                "llm_config": {"model": "gpt-4o-mini"},
                "token_counter": {"import": f"{MODULE_PATH}:sample_token_counter", "params": {"offset": 2}},
            },
            Path(temp_dir),
        )

        builder.build_llm_config()

        counter = builder.agent_params["token_counter"]
        assert callable(counter)
        assert counter([{}, {}]) == 4

    def test_build_llm_config_requires_llm_section(self, temp_dir):
        builder = AgentConfigBuilder({}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="'llm_config' is required"):
            builder.build_llm_config()

    def test_build_llm_config_rejects_bad_token_counter_params(self, temp_dir):
        builder = AgentConfigBuilder(
            {"llm_config": {"model": "gpt-4o-mini"}, "token_counter": {"import": f"{MODULE_PATH}:sample_token_counter", "params": "bad"}},
            Path(temp_dir),
        )

        with pytest.raises(ConfigConfigError, match="must be a mapping"):
            builder.build_llm_config()

    def test_build_system_prompt_path_resolves_relative(self, temp_dir):
        prompt_file = Path(temp_dir) / "prompt.txt"
        prompt_file.write_text("hello")
        builder = AgentConfigBuilder(
            {"name": "agent", "system_prompt": "prompt.txt", "system_prompt_type": "file"},
            Path(temp_dir),
        )

        builder.build_core_properties().build_system_prompt_path()

        assert builder.agent_params["system_prompt"] == str(prompt_file)

    def test_build_system_prompt_path_missing_file_raises(self, temp_dir):
        builder = AgentConfigBuilder(
            {"name": "agent", "system_prompt": "missing.txt", "system_prompt_type": "file"},
            Path(temp_dir),
        )

        builder.build_core_properties()
        with pytest.raises(ConfigConfigError, match="System prompt file not found"):
            builder.build_system_prompt_path()

    def test_build_tools_loads_yaml_and_overrides_name(self, temp_dir):
        tool_yaml = Path(temp_dir) / "tool.yaml"
        tool_yaml.write_text(
            textwrap.dedent(
                """
                name: yaml_tool
                description: Test tool
                input_schema:
                  type: object
                """,
            ),
        )
        builder = AgentConfigBuilder(
            {"tools": [{"name": "alias_tool", "yaml_path": str(tool_yaml), "binding": "builtins:print"}]},
            Path(temp_dir),
        )

        builder.build_tools()

        tool = builder.agent_params["tools"][0]
        assert isinstance(tool, Tool)
        assert tool.name == "alias_tool"
        assert getattr(tool, "source_name", None) == "yaml_tool"

    def test_build_tools_rejects_reserved_extra_kwargs(self, temp_dir):
        tool_yaml = Path(temp_dir) / "tool.yaml"
        tool_yaml.write_text(
            textwrap.dedent(
                """
                name: yaml_tool
                description: Test tool
                input_schema:
                  type: object
                """,
            ),
        )
        builder = AgentConfigBuilder(
            {
                "tools": [
                    {
                        "name": "alias_tool",
                        "yaml_path": str(tool_yaml),
                        "binding": "builtins:print",
                        "extra_kwargs": {"agent_state": "bad"},
                    }
                ]
            },
            Path(temp_dir),
        )

        with pytest.raises(ConfigConfigError, match="reserved keys"):
            builder.build_tools()

    def test_build_tools_missing_yaml_path(self, temp_dir):
        builder = AgentConfigBuilder({"tools": [{"name": "missing"}]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="missing 'yaml_path'"):
            builder.build_tools()

    def test_build_tools_wraps_load_errors(self, temp_dir):
        tool_yaml = Path(temp_dir) / "tool.yaml"
        tool_yaml.write_text(
            textwrap.dedent(
                """
                name: yaml_tool
                description: Test tool
                input_schema:
                  type: object
                """,
            ),
        )
        with patch("nexau.archs.main_sub.config.config.Tool.from_yaml", side_effect=ValueError("boom")):
            builder = AgentConfigBuilder({"tools": [{"name": "alias_tool", "yaml_path": str(tool_yaml)}]}, Path(temp_dir))

            with pytest.raises(ConfigConfigError, match="Error loading tool 'alias_tool'"):
                builder.build_tools()

    @patch("nexau.archs.main_sub.config.config.PromptBuilder", return_value=StubPromptBuilder())
    def test_build_skills_from_folders_and_tools(self, mock_prompt_builder: Mock, temp_dir):
        skill_folder = Path(temp_dir) / "skill"
        skill_folder.mkdir()
        skill_folder.joinpath("SKILL.md").write_text(
            "---\nname: folder-skill\ndescription: Folder skill\n---\n\nDetails here.\n",
        )

        tool = Tool(
            name="skill_tool",
            description="desc",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=True,
            skill_description="Use me as a skill",
        )
        builder = AgentConfigBuilder({"skills": [str(skill_folder)]}, Path(temp_dir))
        builder.agent_params["tools"] = [tool]

        builder.build_skills()

        skills = builder.agent_params["skills"]
        assert {s.name for s in skills} == {"folder-skill", "skill_tool"}
        tool_skill = next(s for s in skills if s.name == "skill_tool")
        assert tool_skill.detail.startswith("skill-detail")
        mock_prompt_builder.assert_called_once()

    def test_build_skills_invalid_folder_raises(self, temp_dir):
        builder = AgentConfigBuilder({"skills": [str(Path(temp_dir) / "missing")]}, Path(temp_dir))

        with pytest.raises(ConfigConfigError, match="Error loading skill"):
            builder.build_skills()

    def test_build_sub_agents_uses_agent_config_from_yaml(self, temp_dir):
        sub_path = Path(temp_dir) / "child.yaml"
        sub_path.write_text("name: child\nllm_config:\n  model: gpt-4o-mini\n")

        with patch("nexau.archs.main_sub.config.config.AgentConfig.from_yaml") as mock_from_yaml:
            mock_from_yaml.return_value = AgentConfig(name="child")
            builder = AgentConfigBuilder({"sub_agents": [{"name": "child", "config_path": str(sub_path)}]}, Path(temp_dir))

            builder.build_sub_agents()

            assert "child" in builder.agent_params["sub_agents"]
            mock_from_yaml.assert_called_once()

    def test_build_sub_agents_wraps_errors(self, temp_dir):
        sub_path = Path(temp_dir) / "child.yaml"
        sub_path.write_text("name: child\n")

        with patch("nexau.archs.main_sub.config.config.AgentConfig.from_yaml", side_effect=ConfigConfigError("boom")):
            builder = AgentConfigBuilder({"sub_agents": [{"name": "child", "config_path": str(sub_path)}]}, Path(temp_dir))
            with pytest.raises(ConfigConfigError, match="Error loading sub-agent 'child'"):
                builder.build_sub_agents()


class TestExecutionConfig:
    """Lightweight validation of ExecutionConfig wiring."""

    def test_execution_config_normalizes_tool_call_mode(self):
        cfg = ExecutionConfig(tool_call_mode="OPENAI")
        assert cfg.tool_call_mode == "openai"

    def test_execution_config_from_agent_config_copies_values(self):
        agent_cfg = AgentConfig(name="agent", max_iterations=5, max_context_tokens=10, max_running_subagents=1, retry_attempts=2, timeout=3)
        exec_cfg = ExecutionConfig.from_agent_config(agent_cfg)

        assert exec_cfg.max_iterations == 5
        assert exec_cfg.max_context_tokens == 10
        assert exec_cfg.max_running_subagents == 1
        assert exec_cfg.retry_attempts == 2
        assert exec_cfg.timeout == 3
