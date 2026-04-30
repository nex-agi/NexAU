from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import BeforeAgentHookInput
from nexau.archs.main_sub.execution.middleware.runtime_environment import RuntimeEnvironmentMiddleware
from nexau.core.messages import Message, Role, TextBlock


def _agent_state(context: dict[str, str | None]) -> AgentState:
    return cast(AgentState, SimpleNamespace(get_context_value=lambda key, default=None: context.get(key, default)))


def test_runtime_environment_middleware_appends_to_first_system_message() -> None:
    middleware = RuntimeEnvironmentMiddleware()
    system = Message(role=Role.SYSTEM, content=[TextBlock(text="Base prompt")], metadata={"cache": True})
    user = Message.user("hello")
    hook_input = BeforeAgentHookInput(
        agent_state=_agent_state(
            {
                "working_directory": r"C:\repo",
                "operating_system": "Windows-11",
                "shell_tool_backend": "Windows PowerShell backend",
            },
        ),
        messages=[system, user],
    )

    result = middleware.before_agent(hook_input)

    assert result.messages is not None
    updated_system = result.messages[0]
    assert updated_system.role == Role.SYSTEM
    assert "Base prompt" in updated_system.get_text_content()
    assert "# Runtime Environment" in updated_system.get_text_content()
    assert r"C:\repo" in updated_system.get_text_content()
    assert "Windows-11" in updated_system.get_text_content()
    assert "Windows PowerShell backend" in updated_system.get_text_content()
    assert updated_system.metadata["cache"] is True
    assert updated_system.metadata["runtime_environment_injected"] is True
    assert result.messages[1] is user


def test_runtime_environment_middleware_uses_custom_template_variables() -> None:
    middleware = RuntimeEnvironmentMiddleware(
        template="cwd={{ current_directory }} os={{ operating_system }} shell={{ shell_backend }}",
    )
    hook_input = BeforeAgentHookInput(
        agent_state=_agent_state(
            {
                "working_directory": "/workspace",
                "operating_system": "Linux",
                "shell_tool_backend": "bash backend",
            },
        ),
        messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="Base")])],
    )

    result = middleware.before_agent(hook_input)

    assert result.messages is not None
    assert "cwd=/workspace os=Linux shell=bash backend" in result.messages[0].get_text_content()


def test_runtime_environment_middleware_is_idempotent() -> None:
    middleware = RuntimeEnvironmentMiddleware()
    system = Message(
        role=Role.SYSTEM,
        content=[TextBlock(text="Base\n\n# Runtime Environment")],
        metadata={"runtime_environment_injected": True},
    )
    hook_input = BeforeAgentHookInput(
        agent_state=_agent_state(
            {
                "working_directory": "/workspace",
                "operating_system": "Linux",
                "shell_tool_backend": "bash backend",
            },
        ),
        messages=[system],
    )

    result = middleware.before_agent(hook_input)

    assert not result.has_modifications()


def test_runtime_environment_middleware_skips_without_runtime_values() -> None:
    middleware = RuntimeEnvironmentMiddleware()
    hook_input = BeforeAgentHookInput(
        agent_state=_agent_state({}),
        messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="Base")])],
    )

    result = middleware.before_agent(hook_input)

    assert not result.has_modifications()
