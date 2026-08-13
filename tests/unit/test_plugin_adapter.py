from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nexau.archs.main_sub.config import ConfigError
from nexau.archs.main_sub.config.base import AgentConfigLoadOptions
from nexau.archs.main_sub.config.config import AgentConfig
from nexau.archs.main_sub.plugin import Plugin, PluginManifest
from nexau.archs.main_sub.prompt_builder import PromptBuilder


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _write_tool_yaml(path: Path, *, with_query: bool = True) -> None:
    if with_query:
        _write(
            path,
            """
            name: lookup_source
            description: Lookup customer data.
            input_schema:
              type: object
              properties:
                query:
                  type: string
            """,
        )
        return

    _write(
        path,
        """
        name: lookup_source
        description: Lookup customer data.
        input_schema:
          type: object
          properties: {}
        """,
    )


def _write_basic_plugin(
    plugin_dir: Path,
    *,
    engine_spec: str = ">=0.1.0",
    tool_name: str = "lookup_customer",
    sub_agent_config_path: str = "agents/customer.yaml",
    extra_kwargs: str = 'project_id: "${config.project_id}"',
    variable_syntax: str = "${config.project_id}",
    top_level_config_key: str = "config",
    system_prompt_fragment: str | None = None,
) -> None:
    system_prompt_fragment_block = ""
    if system_prompt_fragment is not None:
        indented_fragment = textwrap.indent(textwrap.dedent(system_prompt_fragment).strip(), "          ")
        system_prompt_fragment_block = f"        system_prompt_fragment: |\n{indented_fragment}\n"

    _write(
        plugin_dir / "plugin.yaml",
        f"""
        type: plugin
        name: north.customer-service
        version: 0.1.0
        description: Customer service plugin.
        engines:
          nexau: "{engine_spec}"
        {top_level_config_key}:
          properties:
            project_id:
              type: string
              required: true
            retries:
              type: integer
              default: 2
{system_prompt_fragment_block.rstrip()}
        contributes:
          mcp_servers:
            - name: customer_mcp
              type: stdio
              command: python
              args:
                - "${{plugin.dir}}/server.py"
                - "--project={variable_syntax}"
                - "--retries=${{config.retries}}"
              permissions:
                allow:
                  - mcp__customer_mcp
                deny: []
              tool_permissions:
                lookup:
                  allow:
                    - mcp__customer_mcp__lookup
                  deny: []
          tools:
            - name: {tool_name}
              yaml_path: ${{plugin.dir}}/tools/lookup.yaml
              binding: ./handlers.py:lookup
              extra_kwargs:
                {extra_kwargs}
          skills:
            - name: customer-service
              path: skills/customer
          sub_agents:
            - name: customer_sub
              config_path: {sub_agent_config_path}
          middlewares:
            - name: customer_middleware
              import: ./middleware.py:CustomerMiddleware
              params:
                project_id: "{variable_syntax}"
        """,
    )
    _write_tool_yaml(plugin_dir / "tools" / "lookup.yaml")
    _write(
        plugin_dir / "handlers.py",
        """
        def lookup(query: str, project_id: str) -> dict:
            return {"query": query, "project_id": project_id}
        """,
    )
    _write(
        plugin_dir / "middleware.py",
        """
        from nexau.archs.main_sub.execution.hooks import Middleware

        class CustomerMiddleware(Middleware):
            def __init__(self, project_id: str):
                self.project_id = project_id
        """,
    )
    _write(
        plugin_dir / "skills" / "customer" / "SKILL.md",
        """
        ---
        name: customer-service-original
        description: Customer service runbook.
        ---

        Customer service details.
        """,
    )
    _write(
        plugin_dir / "agents" / "customer.yaml",
        """
        type: agent
        name: original_sub_name
        system_prompt: "project ${config.project_id}"
        llm_config:
          model: gpt-4o-mini
        plugins:
          - use: "path:./nested-plugin-that-must-be-ignored"
        """,
    )


def _write_agent_yaml(agent_path: Path, *, plugin_use: str = '"path:./plugins/north.customer-service"') -> None:
    _write_tool_yaml(agent_path.parent / "local_tool.yaml")
    _write(
        agent_path,
        f"""
        type: agent
        name: main
        llm_config:
          model: gpt-4o-mini
        mcp_servers:
          - name: local_mcp
            type: stdio
            command: python
        tools:
          - name: local_tool
            yaml_path: ./local_tool.yaml
            binding: builtins:print
        plugins:
          - use: {plugin_use}
            config:
              project_id: proj_123
        """,
    )


class TestPluginAdapterRFC0024:
    def test_mcp_auth_allows_non_secret_plugin_config_but_keeps_secret_reference(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.secure-mcp"
        _write(
            plugin_dir / "plugin.yaml",
            """
            type: plugin
            name: north.secure-mcp
            version: 0.1.0
            engines:
              nexau: ">=0.1.0"
            config:
              properties:
                service_client_id:
                  type: string
                  required: true
            contributes:
              mcp_servers:
                - name: secure_service
                  type: http
                  url: https://mcp.example.test/mcp
                  auth:
                    type: client_credentials
                    client_id: "${config.service_client_id}"
                    client_secret:
                      source: env
                      key: SERVICE_MCP_CLIENT_SECRET
            """,
        )
        agent_path = tmp_path / "agent.yaml"
        _write(
            agent_path,
            """
            type: agent
            name: main
            llm_config:
              model: gpt-4o-mini
            plugins:
              - use: "path:./plugins/north.secure-mcp"
                config:
                  service_client_id: service-client
            """,
        )

        config = AgentConfig.from_yaml(agent_path)

        auth = config.mcp_servers[0]["auth"]
        assert auth["client_id"] == "service-client"
        assert auth["client_secret"] == {
            "source": "env",
            "key": "SERVICE_MCP_CLIENT_SECRET",
        }

    def test_plugin_from_yaml_generates_agent_plugin_entry(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)

        plugin = Plugin.from_yaml(
            plugin_dir / "plugin.yaml",
            config={"project_id": "proj_123"},
            base_path=tmp_path,
        )

        assert plugin.use == "path:./plugins/north.customer-service"
        assert plugin.config == {"project_id": "proj_123"}
        assert PluginManifest.from_yaml(plugin_dir / "plugin.yaml").name == "north.customer-service"

        cfg = AgentConfig.from_dict(
            {
                "type": "agent",
                "name": "main",
                "llm_config": {"model": "gpt-4o-mini"},
                "plugins": [plugin],
            },
            base_path=tmp_path,
        )

        tools_by_name = {tool.name: tool for tool in cfg.tools}
        assert tools_by_name["lookup_customer"].source_id == "plugin:north.customer-service:tool:lookup_customer"

    def test_expands_path_plugin_resources_and_source_ids(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        cfg = AgentConfig.from_yaml(agent_path)

        tools_by_name = {tool.name: tool for tool in cfg.tools}
        assert tools_by_name["local_tool"].source_id == "local:tool:local_tool"
        assert tools_by_name["lookup_customer"].source_id == "plugin:north.customer-service:tool:lookup_customer"
        assert callable(tools_by_name["lookup_customer"].implementation)
        assert tools_by_name["lookup_customer"].extra_kwargs == {"project_id": "proj_123"}

        mcp_by_name = {server["name"]: server for server in cfg.mcp_servers}
        assert mcp_by_name["local_mcp"]["source_id"] == "local:mcp_server:local_mcp"
        assert mcp_by_name["customer_mcp"]["source_id"] == "plugin:north.customer-service:mcp_server:customer_mcp"
        assert mcp_by_name["customer_mcp"]["command"] == "python"
        assert mcp_by_name["customer_mcp"]["args"] == [
            f"{plugin_dir}/server.py",
            "--project=proj_123",
            "--retries=2",
        ]
        assert mcp_by_name["customer_mcp"]["permissions"] == {"allow": ["mcp__customer_mcp"], "deny": []}
        assert mcp_by_name["customer_mcp"]["tool_permissions"] == {"lookup": {"allow": ["mcp__customer_mcp__lookup"], "deny": []}}

        assert cfg.middlewares is not None
        middleware = cfg.middlewares[0]
        assert middleware.project_id == "proj_123"
        assert middleware.source_id == "plugin:north.customer-service:middleware:customer_middleware"

        skills_by_name = {skill.name: skill for skill in cfg.skills}
        assert skills_by_name["customer-service"].source_id == "plugin:north.customer-service:skill:customer-service"

        assert cfg.sub_agents is not None
        sub_agent = cfg.sub_agents["customer_sub"]
        assert sub_agent.source_id == "plugin:north.customer-service:sub_agent:customer_sub"
        assert sub_agent.system_prompt == "project proj_123"

    def test_system_prompt_fragment_combines_with_agent_prompt(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(
            plugin_dir,
            system_prompt_fragment="""
            Plugin guidance for ${config.project_id}.
            Keep customer replies audit-friendly.
            """,
        )
        agent_path = tmp_path / "agent.yaml"
        _write_tool_yaml(agent_path.parent / "local_tool.yaml")
        _write(
            agent_path,
            """
            type: agent
            name: main
            system_prompt: Host system prompt.
            system_prompt_suffix: Host suffix.
            llm_config:
              model: gpt-4o-mini
            plugins:
              - use: "path:./plugins/north.customer-service"
                config:
                  project_id: proj_123
            """,
        )

        cfg = AgentConfig.from_yaml(agent_path)

        assert cfg.system_prompt == "Host system prompt."
        assert cfg.system_prompt_suffix == "Plugin guidance for proj_123.\nKeep customer replies audit-friendly.\n\nHost suffix."

        prompt_parts = PromptBuilder().build_system_prompt(cfg, tools=[], include_tool_instructions=False)
        combined_prompt = "".join(part.text for part in prompt_parts)
        assert "Host system prompt." in combined_prompt
        assert "Plugin guidance for proj_123." in combined_prompt
        assert combined_prompt.index("Host system prompt.") < combined_prompt.index("Plugin guidance for proj_123.")
        assert combined_prompt.index("Plugin guidance for proj_123.") < combined_prompt.index("Host suffix.")

    def test_system_prompt_fragment_combines_with_default_prompt(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, system_prompt_fragment="Plugin default prompt guidance for ${config.project_id}.")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        cfg = AgentConfig.from_yaml(agent_path)

        prompt_parts = PromptBuilder().build_system_prompt(cfg, tools=[], include_tool_instructions=False)
        combined_prompt = "".join(part.text for part in prompt_parts)
        assert "Plugin default prompt guidance for proj_123." in combined_prompt

    def test_rejects_system_prompt_fragment_inside_contributes(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)
        manifest = plugin_dir / "plugin.yaml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "contributes:\n  mcp_servers:",
                "contributes:\n  system_prompt_fragment: Wrong location.\n  mcp_servers:",
            ),
            encoding="utf-8",
        )
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="system_prompt_fragment"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_multiple_system_prompt_fragments(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)
        manifest = plugin_dir / "plugin.yaml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "contributes:",
                textwrap.dedent(
                    """
                system_prompt_fragment:
                  - First fragment.
                  - Second fragment.
                contributes:
                    """,
                ).strip(),
            ),
            encoding="utf-8",
        )
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="system_prompt_fragment"):
            AgentConfig.from_yaml(agent_path)

    def test_phase1_rejects_non_path_plugin_scheme(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path, plugin_use='"registry:north.customer-service"')

        with pytest.raises(ConfigError, match="scheme 'registry' is not supported"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_legacy_config_colon_variable_syntax(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, variable_syntax="${config:project_id}")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match=r"\$\{config:project_id\}"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_legacy_configuration_manifest_key(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, top_level_config_key="configuration")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="configuration"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_incompatible_engine_version(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, engine_spec=">=999.0.0")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="requires NexAU"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_invalid_engine_spec(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, engine_spec="not-a-spec")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="invalid NexAU engine spec"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_invalid_config_property_name(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)
        manifest = plugin_dir / "plugin.yaml"
        manifest.write_text(manifest.read_text(encoding="utf-8").replace("project_id:", "project-id:"), encoding="utf-8")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="config property names"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_local_and_plugin_name_conflict(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, tool_name="local_tool")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="tool name conflict"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_plugin_path_escape(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)
        manifest = plugin_dir / "plugin.yaml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "yaml_path: ${plugin.dir}/tools/lookup.yaml",
                "yaml_path: ../lookup.yaml",
            ),
            encoding="utf-8",
        )
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="escapes plugin root"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_plugin_dir_inside_sub_agent_yaml(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir)
        _write(
            plugin_dir / "agents" / "customer.yaml",
            """
            type: agent
            name: original_sub_name
            system_prompt: "sub-agent should not render ${plugin.dir}"
            llm_config:
              model: gpt-4o-mini
            """,
        )
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match=r"Unsupported or undefined plugin variable '.+plugin\.dir"):
            AgentConfig.from_yaml(agent_path)

    def test_rejects_tool_extra_kwargs_schema_conflict(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, extra_kwargs='query: "forced"')
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="extra_kwargs conflicts"):
            AgentConfig.from_yaml(agent_path)

    def test_non_strict_skips_component_load_failure(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, sub_agent_config_path="agents/missing.yaml")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        cfg = AgentConfig.from_yaml(agent_path, options=AgentConfigLoadOptions(strict=False))

        assert cfg.sub_agents == {}
        assert any("Skipped sub-agent 'customer_sub'" in item for item in cfg.skipped_components)

    def test_strict_raises_component_load_failure(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "plugins" / "north.customer-service"
        _write_basic_plugin(plugin_dir, sub_agent_config_path="agents/missing.yaml")
        agent_path = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_path)

        with pytest.raises(ConfigError, match="Skipped sub-agent 'customer_sub'"):
            AgentConfig.from_yaml(agent_path)
