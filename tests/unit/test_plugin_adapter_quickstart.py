from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from examples.plugin_adapter import quickstart, quickstart_yaml
from nexau.archs.main_sub.config.config import AgentConfig
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer

_PLUGIN_DEFAULT_PROMPT_FRAGMENT = "Read the relevant code before editing, keep changes scoped, and verify behavior with focused tests."
# RFC-0028: web_search 是条件注入的内置工具——tests/conftest.py 全局提供了
# SERPER_API_KEY，因此在测试环境下它属于 runtime 内置工具集合。
_RUNTIME_BUILTIN_TOOL_NAMES = {
    "web_search",
    "ask_user",
    "complete_task",
    "read_file",
    "run_shell_command",
    "write_file",
    "write_todos",
}
_REQUIRED_ENV = {
    "LLM_MODEL": "test-model",
    "LLM_BASE_URL": "http://example.test/v1",
    "LLM_API_KEY": "test-key",
    "LLM_API_TYPE": "openai_chat_completion",
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
    "LANGFUSE_BASE_URL": "http://langfuse.example.test",
}


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


def _module_file(module: ModuleType) -> Path:
    if module.__file__ is None:
        raise AssertionError(f"module has no file: {module.__name__}")
    return Path(module.__file__)


def _assert_coding_plugin_config(config: AgentConfig) -> None:
    assert config.system_prompt_suffix == _PLUGIN_DEFAULT_PROMPT_FRAGMENT
    assert len(config.tracers) == 1
    assert isinstance(config.tracers[0], LangfuseTracer)
    assert len(config.mcp_servers) == 1
    mcp_server = config.mcp_servers[0]
    assert mcp_server["name"] == "coding_repo_context"
    assert mcp_server["source_id"] == "plugin:example.coding-agent:mcp_server:coding_repo_context"
    assert mcp_server["command"] == "python"
    assert mcp_server["env"] == {
        "CODING_AGENT_PROJECT_NAME": "nexau",
        "CODING_AGENT_DEFAULT_BRANCH": "main",
        "CODING_AGENT_PACKAGE_MANAGER": "uv",
        "CODING_AGENT_TEST_COMMAND": "uv run pytest",
    }
    mcp_server_path = Path(str(mcp_server["args"][0]))
    assert mcp_server_path.parts[-3:] == ("coding-agent", "mcp", "repo_context_server.py")

    assert config.sub_agents is not None
    explore_prompt = config.sub_agents["explore"].system_prompt
    worker_prompt = config.sub_agents["worker"].system_prompt
    assert isinstance(explore_prompt, str)
    assert isinstance(worker_prompt, str)
    assert _PLUGIN_DEFAULT_PROMPT_FRAGMENT in explore_prompt
    assert _PLUGIN_DEFAULT_PROMPT_FRAGMENT in worker_prompt
    explore_tool_names = {tool.name for tool in config.sub_agents["explore"].tools}
    assert explore_tool_names == _RUNTIME_BUILTIN_TOOL_NAMES | {
        "list_directory",
        "search_file_content",
    }
    worker_tool_names = {tool.name for tool in config.sub_agents["worker"].tools}
    assert worker_tool_names == _RUNTIME_BUILTIN_TOOL_NAMES | {
        "apply_patch",
        "list_directory",
        "replace",
        "search_file_content",
    }

    assert config.middlewares is not None
    middleware = config.middlewares[0]
    assert "prompt_fragment" not in middleware.context.as_dict()


def test_default_quickstart_builds_agent_programmatically() -> None:
    source = _module_file(quickstart).read_text(encoding="utf-8")
    assert "AgentConfig.from_dict" in source
    assert "AgentConfig.from_yaml" not in source
    assert "Plugin.from_yaml" in source
    assert '"use": "path:./plugins/coding-agent"' not in source


def test_yaml_quickstart_loads_agent_yaml() -> None:
    source = _module_file(quickstart_yaml).read_text(encoding="utf-8")
    assert "AgentConfig.from_yaml" in source


def test_programmatic_quickstart_uses_coding_agent_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    config = quickstart.build_agent_config()

    _assert_coding_plugin_config(config)


def test_yaml_example_uses_plugin_owned_prompt_fragment_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_env(monkeypatch)

    agent_yaml = _module_file(quickstart_yaml).with_name("agent.yaml")
    plugin_yaml = agent_yaml.parent / "plugins" / "coding-agent" / "plugin.yaml"
    sub_agent_yamls = [
        plugin_yaml.parent / "sub_agents" / "explore.yaml",
        plugin_yaml.parent / "sub_agents" / "worker.yaml",
    ]
    assert "prompt_fragment:" not in agent_yaml.read_text(encoding="utf-8")
    assert "${config.prompt_fragment}" not in plugin_yaml.read_text(encoding="utf-8")
    for sub_agent_yaml in sub_agent_yamls:
        assert "${plugin.dir}" not in sub_agent_yaml.read_text(encoding="utf-8")

    config = quickstart_yaml.build_agent_config()

    _assert_coding_plugin_config(config)


def test_quickstarts_require_live_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="Set LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_API_TYPE"):
        quickstart.main()

    with pytest.raises(RuntimeError, match="Set LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_API_TYPE"):
        quickstart_yaml.main()

    for module in (quickstart, quickstart_yaml):
        source = _module_file(module).read_text(encoding="utf-8")
        assert "_Fake" not in source
        assert "custom_llm_client_provider" not in source
