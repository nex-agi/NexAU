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

"""Unit tests for Agent.run_async run_id parameter."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import pytest

from nexau import Agent


@pytest.fixture
def _agent(agent_config):
    """Create an Agent with mocked internals for run_id testing."""
    with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
        mock_openai.OpenAI.return_value = Mock()
        agent = Agent(config=agent_config)
        yield agent


class TestRunAsyncRunId:
    """Test run_id parameter on Agent.run_async."""

    def test_custom_run_id_passed_to_inner(self, _agent):
        """When run_id is provided, _run_async_inner should receive it unchanged."""
        captured: dict = {}

        async def fake_inner(**kwargs):
            captured.update(kwargs)
            return "ok"

        with patch.object(_agent, "_run_async_inner", side_effect=fake_inner):
            asyncio.run(_agent.run_async(message="hello", run_id="run_MY_CUSTOM_ID"))

        assert captured["run_id"] == "run_MY_CUSTOM_ID"

    def test_auto_generated_run_id_when_none(self, _agent):
        """When run_id is not provided, a run_id starting with 'run_' should be generated."""
        captured: dict = {}

        async def fake_inner(**kwargs):
            captured.update(kwargs)
            return "ok"

        with patch.object(_agent, "_run_async_inner", side_effect=fake_inner):
            asyncio.run(_agent.run_async(message="hello"))

        assert captured["run_id"].startswith("run_")
        assert len(captured["run_id"]) == 4 + 26  # "run_" + 26-char ULID

    def test_auto_generated_run_id_unique(self, _agent):
        """Each call without explicit run_id should produce a unique ID."""
        ids: list[str] = []

        async def fake_inner(**kwargs):
            ids.append(kwargs["run_id"])
            return "ok"

        with patch.object(_agent, "_run_async_inner", side_effect=fake_inner):
            asyncio.run(_agent.run_async(message="a"))
            asyncio.run(_agent.run_async(message="b"))

        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_explicit_none_generates_run_id(self, _agent):
        """Passing run_id=None explicitly should auto-generate."""
        captured: dict = {}

        async def fake_inner(**kwargs):
            captured.update(kwargs)
            return "ok"

        with patch.object(_agent, "_run_async_inner", side_effect=fake_inner):
            asyncio.run(_agent.run_async(message="hello", run_id=None))

        assert captured["run_id"].startswith("run_")
