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

"""Unit tests for /model runtime commands in CLI agent runner."""

from unittest.mock import Mock, patch

import pytest

from nexau.cli.agent_runner import CliAgentRuntime


def _build_runtime_stub() -> CliAgentRuntime:
    runtime = CliAgentRuntime.__new__(CliAgentRuntime)
    object.__setattr__(runtime, "is_busy", Mock(return_value=False))
    object.__setattr__(runtime, "_session_metadata", Mock(return_value={"session_id": "sess-1"}))
    object.__setattr__(runtime, "format_model_list", Mock(return_value="Available models:\n1. gpt-4o-mini [current]\n2. gpt-4.1"))
    object.__setattr__(runtime, "_resolve_model_inventory", Mock(return_value=("gpt-4o-mini", ("gpt-4o-mini", "gpt-4.1"))))
    object.__setattr__(
        runtime,
        "list_model_options",
        Mock(
            return_value=[
                {"index": 1, "name": "gpt-4o-mini", "current": True, "profile_alias": ""},
                {"index": 2, "name": "gpt-4.1", "current": False, "profile_alias": "nex"},
            ]
        ),
    )
    object.__setattr__(runtime, "switch_model", Mock())
    return runtime


class TestModelCommands:
    def test_handle_model_command_default(self):
        runtime = _build_runtime_stub()
        with patch("nexau.cli.agent_runner.send_message") as send_message:
            runtime.handle_model_command("/model")

        send_message.assert_any_call("response", "Available models:\n1. gpt-4o-mini [current]\n2. gpt-4.1")
        send_message.assert_any_call("ready", "", metadata={"session_id": "sess-1"})
        runtime.format_model_list.assert_called_once()

    @pytest.mark.parametrize(
        "command_text",
        [
            "/model list",
            "/model current",
            "/model use 2",
            "/model bad",
        ],
    )
    def test_handle_model_command_invalid_usage(self, command_text):
        runtime = _build_runtime_stub()
        with pytest.raises(RuntimeError, match="Usage: /model"):
            runtime.handle_model_command(command_text)

    def test_list_model_options_returns_structured_items(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        object.__setattr__(runtime, "is_busy", Mock(return_value=False))
        object.__setattr__(runtime, "_resolve_model_inventory", Mock(return_value=("gpt-4o-mini", ("gpt-4o-mini", "gpt-4.1"))))
        runtime._discover_llm_profiles = Mock(
            return_value={
                "nex": {
                    "alias": "nex",
                    "model": "gpt-4.1",
                    "base_url": "https://nex.example.com/v1",
                    "api_key": "nex-key",
                }
            }
        )

        result = runtime.list_model_options()

        assert result == [
            {"index": 1, "name": "gpt-4o-mini", "current": True, "profile_alias": ""},
            {"index": 2, "name": "gpt-4.1", "current": False, "profile_alias": "nex"},
        ]

    def test_list_model_options_includes_profile_only_models(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        object.__setattr__(runtime, "is_busy", Mock(return_value=False))
        object.__setattr__(runtime, "_resolve_model_inventory", Mock(return_value=("kimi-model", ("kimi-model",))))
        runtime._discover_llm_profiles = Mock(
            return_value={
                "nex": {
                    "alias": "nex",
                    "model": "nex-model",
                    "base_url": "https://nex.example.com/v1",
                    "api_key": "nex-key",
                }
            }
        )

        result = runtime.list_model_options()

        assert result == [
            {"index": 1, "name": "kimi-model", "current": True, "profile_alias": ""},
            {"index": 2, "name": "nex-model", "current": False, "profile_alias": "nex"},
        ]

    def test_switch_model_updates_override_and_rebuilds_agent(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        object.__setattr__(runtime, "is_busy", Mock(return_value=False))
        object.__setattr__(runtime, "_resolve_model_inventory", Mock(return_value=("gpt-4o-mini", ("gpt-4o-mini", "gpt-4.1"))))
        runtime._discover_llm_profiles = Mock(
            return_value={
                "nex": {
                    "alias": "nex",
                    "model": "gpt-4.1",
                    "base_url": "https://nex.example.com/v1",
                    "api_key": "nex-key",
                }
            }
        )
        runtime._persist_state = Mock()
        runtime._activate_agent = Mock()
        runtime._session_metadata = Mock(return_value={"session_id": "sess-1"})
        runtime._session_store = Mock()
        runtime._session_store.load_session.return_value = {"session_id": "sess-1"}
        runtime._agent = Mock()
        runtime._agent._session_id = "sess-1"
        runtime._agent.config = Mock()
        runtime._agent.config.llm_config = Mock()
        runtime._agent.config.llm_config.base_url = "https://kimi.example.com/v1"
        runtime._agent.config.llm_config.api_key = "kimi-key"
        runtime._llm_override = None

        with patch("nexau.cli.agent_runner.send_message") as send_message:
            runtime.switch_model("2")

        assert runtime._llm_override == {
            "model": "gpt-4.1",
            "base_url": "https://nex.example.com/v1",
            "api_key": "nex-key",
        }
        runtime._activate_agent.assert_called_once_with(
            restored_snapshot={"session_id": "sess-1"},
            session_id="sess-1",
        )
        send_message.assert_any_call("status", "Switching model to gpt-4.1 (profile: nex)...")
        send_message.assert_any_call("response", "Switched model to gpt-4.1 (profile: nex).")
        send_message.assert_any_call("ready", "", metadata={"session_id": "sess-1"})

    def test_switch_model_rejects_unknown_model_name(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        object.__setattr__(runtime, "is_busy", Mock(return_value=False))
        object.__setattr__(runtime, "_resolve_model_inventory", Mock(return_value=("gpt-4o-mini", ("gpt-4o-mini", "gpt-4.1"))))
        runtime._discover_llm_profiles = Mock(return_value={})
        runtime._agent = Mock()
        runtime._agent._session_id = "sess-1"
        runtime._agent.config = Mock()
        runtime._agent.config.llm_config = Mock()
        runtime._agent.config.llm_config.base_url = "https://kimi.example.com/v1"
        runtime._agent.config.llm_config.api_key = "kimi-key"

        with pytest.raises(RuntimeError, match="Unknown model: unknown-model"):
            runtime.switch_model("unknown-model")

    def test_switch_model_by_profile_alias_switches_model_and_credentials(self):
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        object.__setattr__(runtime, "is_busy", Mock(return_value=False))
        object.__setattr__(runtime, "_resolve_model_inventory", Mock(return_value=("kimi-model", ("kimi-model", "nex-model"))))
        runtime._discover_llm_profiles = Mock(
            return_value={
                "nex": {
                    "alias": "nex",
                    "model": "nex-model",
                    "base_url": "https://nex.example.com/v1",
                    "api_key": "nex-key",
                }
            }
        )
        runtime._persist_state = Mock()
        runtime._activate_agent = Mock()
        runtime._session_metadata = Mock(return_value={"session_id": "sess-1"})
        runtime._session_store = Mock()
        runtime._session_store.load_session.return_value = {"session_id": "sess-1"}
        runtime._agent = Mock()
        runtime._agent._session_id = "sess-1"
        runtime._agent.config = Mock()
        runtime._agent.config.llm_config = Mock()
        runtime._agent.config.llm_config.base_url = "https://kimi.example.com/v1"
        runtime._agent.config.llm_config.api_key = "kimi-key"
        runtime._llm_override = None

        with patch("nexau.cli.agent_runner.send_message"):
            runtime.switch_model("nex")

        assert runtime._llm_override == {
            "model": "nex-model",
            "base_url": "https://nex.example.com/v1",
            "api_key": "nex-key",
        }
