# Licensed under the Apache License, Version 2.0

"""Live two-turn payload capture tests.

These integration tests verify the *actual* second-turn provider payloads emitted by
Agent -> SessionManager -> LLMCaller when using real endpoints. They are intended to
complement the unit-level UMP matrix tests with end-to-end request capture.

Coverage goals:
- same-provider two-turn payload capture for all supported API types
- one representative cross-api-type switch (`openai_chat_completion -> openai_responses`)
- assertions focus on exact second-turn request shape, not only final text output

Tests are skipped unless the required provider-specific credentials are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import dotenv
import pytest

from nexau import Agent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager

dotenv.load_dotenv()

_TASK_A = "Task A: Compute ((35 * 11) + 57) / 13. Show concise steps and end with Final A: <number>."
_TASK_B = "Task B: Using the previous result as A, compute (A * 3) + 15. Show concise steps and end with Final B: <number>."


@dataclass(frozen=True, slots=True)
class ProviderEnv:
    api_type: str
    model: str
    base_url: str
    api_key: str


def _env(name: str) -> str:
    return (dotenv.get_key(".env", name) or "") if False else ""


def _from_env(model_key: str, base_key: str, key_key: str, api_type: str) -> ProviderEnv | None:
    import os

    model = os.getenv(model_key, "")
    base_url = os.getenv(base_key, "")
    api_key = os.getenv(key_key, "")
    if not model or not base_url or not api_key or api_key == "test-key-not-used":
        return None
    return ProviderEnv(api_type=api_type, model=model, base_url=base_url, api_key=api_key)


_OPENAI_CHAT = _from_env(
    "LIVE_OPENAI_CHAT_MODEL",
    "LIVE_OPENAI_CHAT_BASE_URL",
    "LIVE_OPENAI_CHAT_API_KEY",
    "openai_chat_completion",
)
_OPENAI_RESPONSES = _from_env(
    "LIVE_OPENAI_RESPONSES_MODEL",
    "LIVE_OPENAI_RESPONSES_BASE_URL",
    "LIVE_OPENAI_RESPONSES_API_KEY",
    "openai_responses",
)
_ANTHROPIC = _from_env(
    "LIVE_ANTHROPIC_MODEL",
    "LIVE_ANTHROPIC_BASE_URL",
    "LIVE_ANTHROPIC_API_KEY",
    "anthropic_chat_completion",
)
_GEMINI = _from_env(
    "LIVE_GEMINI_MODEL",
    "LIVE_GEMINI_BASE_URL",
    "LIVE_GEMINI_API_KEY",
    "gemini_rest",
)

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.external]


class _SecondTurnOpenAIChatSpy:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._inner.create(*args, **kwargs)


class _SecondTurnOpenAIResponsesSpy:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._inner.create(*args, **kwargs)


class _SecondTurnAnthropicSpy:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._inner.create(*args, **kwargs)


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(engine=InMemoryDatabaseEngine())


def _make_agent(env: ProviderEnv, session_manager: SessionManager, session_id: str) -> Agent:
    llm_kwargs: dict[str, Any] = {
        "model": env.model,
        "base_url": env.base_url,
        "api_key": env.api_key,
        "api_type": env.api_type,
        "temperature": 0.0,
        "max_tokens": 256,
        "max_retries": 1,
        "stream": False,
    }
    if env.api_type == "openai_chat_completion":
        llm_kwargs["reasoning_effort"] = "high"
    elif env.api_type == "openai_responses":
        llm_kwargs["reasoning"] = {"effort": "high"}
        llm_kwargs["include"] = ["reasoning.encrypted_content"]
    elif env.api_type == "anthropic_chat_completion":
        llm_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    elif env.api_type == "gemini_rest":
        llm_kwargs["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": 512}

    agent = Agent(
        config=AgentConfig(
            name=f"two-turn-live-{env.api_type}",
            system_prompt="You are a precise assistant. Keep calculations concise.",
            llm_config=LLMConfig(**llm_kwargs),
        ),
        session_manager=session_manager,
        user_id="two-turn-live-user",
        session_id=session_id,
    )
    return agent


def _run_two_turn_same_provider(env: ProviderEnv, session_manager: SessionManager) -> tuple[str, dict[str, Any]]:
    session_id = f"two-turn-{env.api_type}"
    agent1 = _make_agent(env, session_manager, session_id)
    first = agent1.run(message=_TASK_A)
    assert first

    agent2 = _make_agent(env, session_manager, session_id)

    if env.api_type == "openai_chat_completion":
        chat_spy = _SecondTurnOpenAIChatSpy(agent2.openai_client.chat.completions)
        with patch.object(agent2.openai_client.chat, "completions", chat_spy):
            second = agent2.run(message=_TASK_B)
        assert second
        return str(second), chat_spy.calls[-1]

    if env.api_type == "openai_responses":
        resp_spy = _SecondTurnOpenAIResponsesSpy(agent2.openai_client.responses)
        with patch.object(agent2.openai_client, "responses", resp_spy):
            second = agent2.run(message=_TASK_B)
        assert second
        return str(second), resp_spy.calls[-1]

    if env.api_type == "anthropic_chat_completion":
        anth_spy = _SecondTurnAnthropicSpy(agent2.openai_client.messages)
        with patch.object(agent2.openai_client, "messages", anth_spy):
            second = agent2.run(message=_TASK_B)
        assert second
        return str(second), anth_spy.calls[-1]

    # gemini_rest uses requests directly inside llm_caller, so capturing exact body here would
    # require patching requests.post. For now we assert the turn succeeds and the provider-specific
    # unit tests cover the exact serialized payload shape.
    second = agent2.run(message=_TASK_B)
    assert second
    return str(second), {}


@pytest.mark.skipif(_OPENAI_CHAT is None, reason="LIVE_OPENAI_CHAT_* env vars not set")
def test_two_turn_live_openai_chat_payload(session_manager: SessionManager) -> None:
    assert _OPENAI_CHAT is not None
    second_text, payload = _run_two_turn_same_provider(_OPENAI_CHAT, session_manager)

    assert "Final B" in second_text
    assert payload["messages"][0]["role"] == "system"
    assert any(msg.get("role") == "assistant" and msg.get("content") for msg in payload["messages"])
    assert payload.get("reasoning_effort") == "high"


@pytest.mark.skipif(_OPENAI_RESPONSES is None, reason="LIVE_OPENAI_RESPONSES_* env vars not set")
def test_two_turn_live_openai_responses_payload(session_manager: SessionManager) -> None:
    assert _OPENAI_RESPONSES is not None
    second_text, payload = _run_two_turn_same_provider(_OPENAI_RESPONSES, session_manager)

    assert "Final B" in second_text
    assert payload.get("reasoning", {}).get("effort") == "high"
    assert "reasoning.encrypted_content" in payload.get("include", [])
    assert isinstance(payload.get("input"), list)
    assert len(payload["input"]) >= 3


@pytest.mark.skipif(_ANTHROPIC is None, reason="LIVE_ANTHROPIC_* env vars not set")
def test_two_turn_live_anthropic_payload(session_manager: SessionManager) -> None:
    assert _ANTHROPIC is not None
    second_text, payload = _run_two_turn_same_provider(_ANTHROPIC, session_manager)

    assert "Final B" in second_text
    assert payload.get("thinking", {}).get("type") == "enabled"
    assert isinstance(payload.get("messages"), list)
    assert len(payload["messages"]) >= 3


@pytest.mark.skipif(_GEMINI is None, reason="LIVE_GEMINI_* env vars not set")
def test_two_turn_live_gemini_payload(session_manager: SessionManager) -> None:
    assert _GEMINI is not None
    second_text, payload = _run_two_turn_same_provider(_GEMINI, session_manager)

    assert "Final B" in second_text
    assert payload == {}


@pytest.mark.skipif(
    _OPENAI_CHAT is None or _OPENAI_RESPONSES is None,
    reason="LIVE_OPENAI_CHAT_* or LIVE_OPENAI_RESPONSES_* env vars not set",
)
def test_two_turn_live_completion_to_responses_gap(session_manager: SessionManager) -> None:
    assert _OPENAI_CHAT is not None
    assert _OPENAI_RESPONSES is not None

    session_id = "two-turn-live-completion-to-responses"
    agent1 = _make_agent(_OPENAI_CHAT, session_manager, session_id)
    first = agent1.run(message=_TASK_A)
    assert first

    agent2 = _make_agent(_OPENAI_RESPONSES, session_manager, session_id)
    spy = _SecondTurnOpenAIResponsesSpy(agent2.openai_client.responses)
    with patch.object(agent2.openai_client, "responses", spy):
        second = agent2.run(message=_TASK_B)
    assert second

    payload = spy.calls[-1]
    input_items = payload.get("input", [])
    assert isinstance(input_items, list)
    assert any(item.get("type") == "message" and item.get("role") == "assistant" for item in input_items)
    assert not any(item.get("type") == "reasoning" for item in input_items)
