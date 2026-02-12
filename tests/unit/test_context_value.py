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

"""Unit tests for ContextValue model and its integration across the stack."""

import shutil
import tempfile
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.context_value import ContextValue
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.sandbox.base_sandbox import SandboxStatus
from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.transports.http.models import AgentRequest

# ---------------------------------------------------------------------------
# Part 1: ContextValue model
# ---------------------------------------------------------------------------


class TestContextValue:
    def test_default_empty(self):
        cv = ContextValue()
        assert cv.template == {}
        assert cv.runtime_vars == {}
        assert cv.sandbox_env == {}

    def test_with_all_fields(self):
        cv = ContextValue(
            template={"project": "nexau"},
            runtime_vars={"api_key": "sk-xxx"},
            sandbox_env={"TOKEN": "ghp_xxx"},
        )
        assert cv.template == {"project": "nexau"}
        assert cv.runtime_vars == {"api_key": "sk-xxx"}
        assert cv.sandbox_env == {"TOKEN": "ghp_xxx"}

    def test_partial_fields(self):
        cv = ContextValue(runtime_vars={"key": "val"})
        assert cv.template == {}
        assert cv.runtime_vars == {"key": "val"}
        assert cv.sandbox_env == {}

    def test_serialization_roundtrip(self):
        cv = ContextValue(
            template={"a": "1"},
            runtime_vars={"b": "2"},
            sandbox_env={"c": "3"},
        )
        data = cv.model_dump()
        restored = ContextValue.model_validate(data)
        assert restored == cv

    def test_from_json(self):
        raw = '{"template": {"x": "1"}, "runtime_vars": {}, "sandbox_env": {"Y": "2"}}'
        cv = ContextValue.model_validate_json(raw)
        assert cv.template == {"x": "1"}
        assert cv.sandbox_env == {"Y": "2"}


# ---------------------------------------------------------------------------
# Part 2: AgentState variables accessors
# ---------------------------------------------------------------------------


class TestAgentStateVariables:
    @pytest.fixture
    def mock_executor(self):
        executor = Mock()
        executor.add_tool = Mock()
        return cast(Executor, executor)

    @pytest.fixture
    def make_state(self, mock_executor):
        def _make(variables: ContextValue | None = None) -> AgentState:
            return AgentState(
                agent_name="test",
                agent_id="test_123",
                run_id="run_1",
                root_run_id="run_1",
                context=AgentContext({}),
                global_storage=GlobalStorage(),
                executor=mock_executor,
                variables=variables,
            )

        return _make

    def test_default_empty_variables(self, make_state):
        state = make_state()
        assert state.get_variable("any_key") is None
        assert state.get_sandbox_env("any_key") is None
        assert state.all_variables == {}
        assert state.all_sandbox_env == {}

    def test_get_variable(self, make_state):
        cv = ContextValue(runtime_vars={"api_key": "sk-xxx", "region": "us-east-1"})
        state = make_state(cv)
        assert state.get_variable("api_key") == "sk-xxx"
        assert state.get_variable("region") == "us-east-1"
        assert state.get_variable("missing") is None
        assert state.get_variable("missing", "fallback") == "fallback"

    def test_get_sandbox_env(self, make_state):
        cv = ContextValue(sandbox_env={"TOKEN": "ghp_xxx"})
        state = make_state(cv)
        assert state.get_sandbox_env("TOKEN") == "ghp_xxx"
        assert state.get_sandbox_env("MISSING") is None
        assert state.get_sandbox_env("MISSING", "default") == "default"

    def test_all_variables_returns_copy(self, make_state):
        cv = ContextValue(runtime_vars={"k": "v"})
        state = make_state(cv)
        all_vars = state.all_variables
        all_vars["k"] = "mutated"
        # Original should be unchanged
        assert state.get_variable("k") == "v"

    def test_all_sandbox_env_returns_copy(self, make_state):
        cv = ContextValue(sandbox_env={"K": "V"})
        state = make_state(cv)
        all_env = state.all_sandbox_env
        all_env["K"] = "mutated"
        assert state.get_sandbox_env("K") == "V"


# ---------------------------------------------------------------------------
# Part 3: BaseSandbox._merge_envs
# ---------------------------------------------------------------------------


class TestBaseSandboxMergeEnvs:
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_both_empty_returns_none(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir)
        assert sb._merge_envs(None) is None
        assert sb._merge_envs({}) is None

    def test_instance_envs_only(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"A": "1"})
        result = sb._merge_envs(None)
        assert result == {"A": "1"}

    def test_per_call_envs_only(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir)
        result = sb._merge_envs({"B": "2"})
        assert result == {"B": "2"}

    def test_per_call_overrides_instance(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"K": "old", "A": "1"})
        result = sb._merge_envs({"K": "new", "B": "2"})
        assert result == {"K": "new", "A": "1", "B": "2"}


# ---------------------------------------------------------------------------
# Part 4: LocalSandbox._build_local_envs and envs injection
# ---------------------------------------------------------------------------


class TestLocalSandboxEnvs:
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_build_local_envs_none_when_no_envs(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir)
        assert sb._build_local_envs(None) is None

    def test_build_local_envs_includes_os_environ(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"MY_VAR": "hello"})
        result = sb._build_local_envs(None)
        assert result is not None
        assert result["MY_VAR"] == "hello"
        # os.environ keys should be present (PATH is almost always there)
        assert "PATH" in result

    def test_build_local_envs_per_call_overrides(self, temp_dir):
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"K": "instance"})
        result = sb._build_local_envs({"K": "per_call"})
        assert result is not None
        assert result["K"] == "per_call"

    def test_execute_bash_with_instance_envs(self, temp_dir):
        """Instance envs are available inside executed commands."""
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"MY_TEST_VAR": "hello_nexau"})
        result = sb.execute_bash("echo $MY_TEST_VAR")
        assert result.status == SandboxStatus.SUCCESS
        assert "hello_nexau" in result.stdout

    def test_execute_bash_preserves_path(self, temp_dir):
        """Instance envs don't clobber PATH."""
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"MY_VAR": "val"})
        result = sb.execute_bash("echo $PATH")
        assert result.status == SandboxStatus.SUCCESS
        assert "/" in result.stdout  # PATH should contain at least one slash

    def test_execute_bash_per_call_overrides_instance(self, temp_dir):
        """Per-call envs override instance envs."""
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir, envs={"K": "instance_val"})
        result = sb.execute_bash("echo $K", envs={"K": "per_call_val"})
        assert result.status == SandboxStatus.SUCCESS
        assert "per_call_val" in result.stdout

    def test_execute_bash_no_envs_inherits_parent(self, temp_dir):
        """Without any envs, subprocess inherits parent environment normally."""
        sb = LocalSandbox(sandbox_id="t", _work_dir=temp_dir)
        result = sb.execute_bash("echo $PATH")
        assert result.status == SandboxStatus.SUCCESS
        assert "/" in result.stdout


# ---------------------------------------------------------------------------
# Part 5: AgentRequest.variables field
# ---------------------------------------------------------------------------


class TestAgentRequestVariables:
    def test_default_none(self):
        req = AgentRequest(messages="hi")
        assert req.variables is None

    def test_with_variables(self):
        cv = ContextValue(template={"p": "v"}, sandbox_env={"T": "1"})
        req = AgentRequest(messages="hi", variables=cv)
        assert req.variables is not None
        assert req.variables.template == {"p": "v"}
        assert req.variables.sandbox_env == {"T": "1"}

    def test_from_dict(self):
        raw = {
            "messages": "hi",
            "variables": {
                "template": {"a": "1"},
                "runtime_vars": {"b": "2"},
                "sandbox_env": {"c": "3"},
            },
        }
        req = AgentRequest.model_validate(raw)
        assert req.variables is not None
        assert req.variables.template == {"a": "1"}


# ---------------------------------------------------------------------------
# Part 6: TransportBase passes variables through
# ---------------------------------------------------------------------------


class TestTransportBaseVariables:
    @pytest.fixture
    def transport(self):
        from dataclasses import dataclass as dc

        from nexau.archs.llm.llm_config import LLMConfig
        from nexau.archs.main_sub.config import AgentConfig
        from nexau.archs.session import InMemoryDatabaseEngine
        from nexau.archs.transports.base import TransportBase

        @dc
        class _Cfg:
            host: str = "localhost"
            port: int = 8080

        class _Transport(TransportBase[_Cfg]):
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        engine = InMemoryDatabaseEngine()
        agent_config = AgentConfig(
            name="test_agent",
            system_prompt="test",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )
        return _Transport(engine=engine, config=_Cfg(), default_agent_config=agent_config)

    def test_handle_request_passes_variables(self, transport):
        import asyncio

        cv = ContextValue(template={"k": "v"}, runtime_vars={"secret": "s"})

        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_async = AsyncMock(return_value="ok")
            mock_agent_cls.return_value = mock_agent

            asyncio.run(
                transport.handle_request(
                    message="hi",
                    user_id="u1",
                    variables=cv,
                )
            )

            # Agent constructor should receive variables
            init_kwargs = mock_agent_cls.call_args[1]
            assert init_kwargs["variables"] is cv

            # run_async should also receive variables
            run_kwargs = mock_agent.run_async.call_args[1]
            assert run_kwargs["variables"] is cv

    def test_handle_streaming_request_passes_variables(self, transport):
        import asyncio

        cv = ContextValue(sandbox_env={"T": "1"})

        with patch("nexau.archs.transports.base.Agent") as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_async = AsyncMock(return_value="ok")
            mock_agent_cls.return_value = mock_agent

            async def run():
                collected = []
                async for event in transport.handle_streaming_request(
                    message="hi",
                    user_id="u1",
                    variables=cv,
                ):
                    collected.append(event)

            asyncio.run(run())

            init_kwargs = mock_agent_cls.call_args[1]
            assert init_kwargs["variables"] is cv

            run_kwargs = mock_agent.run_async.call_args[1]
            assert run_kwargs["variables"] is cv
