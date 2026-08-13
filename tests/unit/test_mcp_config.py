"""Configuration contract tests for RFC-0029 official MCP SDK integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from nexau.archs.main_sub.config.config import AgentConfig, AgentConfigBuilder, ConfigError
from nexau.archs.main_sub.config.schema import AgentConfigSchema
from nexau.archs.tool.builtin.mcp_client import MCPServerConfig as ProgrammaticMCPServerConfig


def _build(server: dict[str, object]) -> dict[str, object]:
    builder = AgentConfigBuilder({"mcp_servers": [server]}, Path.cwd())
    builder.build_mcp_servers()
    return builder.agent_params["mcp_servers"][0]


@pytest.mark.parametrize(
    ("server_type", "extra"),
    [
        ("stdio", {"command": "python", "args": ["server.py"], "env": {"EXPLICIT": "yes"}}),
        ("http", {"url": "http://127.0.0.1:8123/mcp", "headers": {"X-Tenant": "north"}}),
        ("sse", {"url": "http://localhost:8123/sse"}),
    ],
)
def test_legacy_transport_config_round_trips(server_type: str, extra: dict[str, object]) -> None:
    server = _build({"name": f"local_{server_type}", "type": server_type, **extra})

    assert server["type"] == server_type
    assert server["source_id"] == f"local:mcp_server:local_{server_type}"
    for key, value in extra.items():
        assert server[key] == value


@pytest.mark.parametrize(
    "auth",
    [
        {"type": "bearer", "token": {"source": "env", "key": "MCP_TOKEN"}},
        {
            "type": "authorization_code",
            "client_name": "NexAU test",
            "scopes": ["tools:call"],
        },
        {
            "type": "authorization_code",
            "client_name": "NexAU registered",
            "client_id": "client-id",
            "client_secret": {"source": "env", "key": "MCP_CLIENT_SECRET"},
        },
        {
            "type": "client_credentials",
            "client_id": "service-client",
            "client_secret": {"source": "env", "key": "MCP_CLIENT_SECRET"},
            "scopes": ["tools:read", "tools:call"],
        },
    ],
)
def test_http_auth_modes_are_available_from_agent_config(auth: dict[str, object]) -> None:
    server = _build({"name": "secured", "type": "http", "url": "https://mcp.example.test/mcp", "auth": auth})

    normalized_auth = server["auth"]
    assert isinstance(normalized_auth, dict)
    for key, value in auth.items():
        assert normalized_auth[key] == value


def test_auth_rejects_literal_secret() -> None:
    with pytest.raises(ConfigError, match="token"):
        _build(
            {
                "name": "secured",
                "type": "http",
                "url": "https://mcp.example.test/mcp",
                "auth": {"type": "bearer", "token": "plaintext-token"},
            }
        )


def test_authorization_code_rejects_client_secret_without_client_id() -> None:
    with pytest.raises(ConfigError, match="client_secret requires client_id"):
        _build(
            {
                "name": "oauth",
                "type": "http",
                "url": "https://mcp.example.test/mcp",
                "auth": {
                    "type": "authorization_code",
                    "client_secret": {"source": "env", "key": "MCP_CLIENT_SECRET"},
                },
            }
        )


def test_auth_rejects_authorization_header_conflict_without_leaking_value() -> None:
    secret = "Bearer should-not-appear-in-error"
    with pytest.raises(ConfigError) as error:
        _build(
            {
                "name": "secured",
                "type": "http",
                "url": "https://mcp.example.test/mcp",
                "headers": {"authorization": secret},
                "auth": {"type": "bearer", "token": {"source": "env", "key": "MCP_TOKEN"}},
            }
        )

    assert "auth cannot be combined" in str(error.value)
    assert secret not in str(error.value)


def test_remote_plain_http_is_rejected_but_loopback_is_allowed() -> None:
    with pytest.raises(ConfigError, match="must use HTTPS"):
        _build({"name": "remote", "type": "http", "url": "http://mcp.example.test/mcp"})

    assert _build({"name": "local", "type": "http", "url": "http://[::1]:8123/mcp"})["name"] == "local"


def test_duplicate_server_names_are_rejected() -> None:
    builder = AgentConfigBuilder(
        {
            "mcp_servers": [
                {"name": "duplicate", "type": "stdio", "command": "one"},
                {"name": "duplicate", "type": "stdio", "command": "two"},
            ]
        },
        Path.cwd(),
    )

    with pytest.raises(ConfigError, match="Duplicate MCP server name"):
        builder.build_mcp_servers()


def test_agent_config_repr_hides_mcp_headers() -> None:
    secret = "Bearer repr-secret"
    config = AgentConfig(
        name="repr-test",
        mcp_servers=[
            {
                "name": "legacy",
                "type": "http",
                "url": "https://mcp.example.test/mcp",
                "headers": {"Authorization": secret},
            }
        ],
    )

    assert secret not in repr(config)


def test_schema_repr_hides_sensitive_mcp_transport_fields() -> None:
    schema = AgentConfigSchema.model_validate(
        {
            "name": "repr-test",
            "llm_config": {},
            "mcp_servers": [
                {
                    "name": "remote",
                    "type": "http",
                    "url": "https://mcp.example.test/mcp?api_key=query-secret",
                    "headers": {"X-Api-Key": "header-secret"},
                },
                {
                    "name": "local",
                    "type": "stdio",
                    "command": "python",
                    "args": ["--token", "argument-secret"],
                    "env": {"MCP_TOKEN": "environment-secret"},
                },
            ],
        }
    )

    rendered = repr(schema)
    for secret in ("query-secret", "header-secret", "argument-secret", "environment-secret"):
        assert secret not in rendered


def test_direct_agent_config_uses_same_typed_mcp_contract() -> None:
    config = AgentConfig(
        name="direct",
        mcp_servers=[
            {
                "name": "secured",
                "type": "http",
                "url": "https://mcp.example.test/mcp",
                "auth": {
                    "type": "client_credentials",
                    "client_id": "service-client",
                    "client_secret": {"source": "env", "key": "SERVICE_MCP_CLIENT_SECRET"},
                },
            }
        ],
    )

    assert config.mcp_servers[0]["source_id"] == "local:mcp_server:secured"
    assert config.mcp_servers[0]["auth"]["type"] == "client_credentials"


def test_direct_agent_config_accepts_public_programmatic_server_dataclass() -> None:
    config = AgentConfig(
        name="direct",
        # Pydantic field annotations describe the normalized output (dicts),
        # while the before-validator intentionally also accepts this public
        # compatibility value object as input.
        mcp_servers=[cast(Any, ProgrammaticMCPServerConfig(name="stdio", type="stdio", command="python"))],
    )

    assert config.mcp_servers == [
        {
            "name": "stdio",
            "source_id": "local:mcp_server:stdio",
            "type": "stdio",
            "timeout": 30.0,
            "disable_parallel": False,
            "command": "python",
        }
    ]


def test_direct_agent_config_rejects_literal_auth_secret_without_leaking_it() -> None:
    secret = "literal-secret-must-not-leak"
    with pytest.raises(ValueError) as error:
        AgentConfig(
            name="direct",
            mcp_servers=[
                {
                    "name": "secured",
                    "type": "http",
                    "url": "https://mcp.example.test/mcp",
                    "auth": {"type": "bearer", "token": secret},
                }
            ],
        )

    assert secret not in str(error.value)


def test_all_modes_example_is_a_valid_agent_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    schema = AgentConfigSchema.from_yaml("examples/mcp/agent_all_modes.yaml")

    assert [server.type for server in schema.mcp_servers] == ["stdio", "http", "sse", "http", "http", "http"]
    auth_types = [getattr(server.auth, "type", None) for server in schema.mcp_servers if hasattr(server, "auth")]
    assert auth_types == [None, None, "bearer", "authorization_code", "client_credentials"]
