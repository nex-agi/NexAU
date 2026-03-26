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

"""Coverage improvement tests for TeamSSEMultiplexer.

Targets uncovered paths in:
- nexau/archs/main_sub/team/sse/multiplexer.py
"""

import asyncio
from unittest.mock import Mock

import pytest

from nexau.archs.llm.llm_aggregators.events import TextMessageStartEvent
from nexau.archs.main_sub.team.sse.envelope import TeamStreamEnvelope
from nexau.archs.main_sub.team.sse.multiplexer import TeamSSEMultiplexer


def _make_event() -> TextMessageStartEvent:
    return TextMessageStartEvent(message_id="m1", run_id="r1")


class TestTeamSSEMultiplexer:
    @pytest.mark.anyio
    async def test_create_event_handler_and_stream(self):
        mux = TeamSSEMultiplexer(team_id="team-1")
        handler = mux.create_event_handler("agent-1", "coder")

        handler(_make_event())
        mux.close()

        envelopes = []
        async for envelope in mux.stream():
            envelopes.append(envelope)

        assert len(envelopes) == 1
        assert envelopes[0].agent_id == "agent-1"
        assert envelopes[0].role_name == "coder"

    @pytest.mark.anyio
    async def test_emit_with_role_name(self):
        mux = TeamSSEMultiplexer(team_id="team-2")
        mux.emit("agent-2", _make_event(), role_name="reviewer")
        mux.close()

        envelopes = []
        async for envelope in mux.stream():
            envelopes.append(envelope)

        assert len(envelopes) == 1
        assert envelopes[0].role_name == "reviewer"

    @pytest.mark.anyio
    async def test_on_envelope_callback(self):
        collected: list[object] = []
        mux = TeamSSEMultiplexer(team_id="team-3", on_envelope=lambda e: collected.append(e))

        handler = mux.create_event_handler("agent-3", "planner")
        handler(_make_event())
        mux.close()

        async for _ in mux.stream():
            pass

        assert len(collected) == 1

    @pytest.mark.anyio
    async def test_close_sends_sentinel(self):
        mux = TeamSSEMultiplexer(team_id="team-4")
        mux.close()

        envelopes = []
        async for envelope in mux.stream():
            envelopes.append(envelope)

        assert envelopes == []

    def test_put_with_no_loop(self):
        """Test _put when no event loop is captured."""
        mux = TeamSSEMultiplexer.__new__(TeamSSEMultiplexer)
        mux._team_id = "t"
        mux._queue = asyncio.Queue()
        mux._on_envelope = None
        mux._loop = None

        envelope = TeamStreamEnvelope(
            team_id="t",
            agent_id="a",
            role_name="r",
            event=_make_event(),
        )
        mux._put(envelope)
        assert not mux._queue.empty()

    def test_put_with_closed_loop(self):
        """Test _put when loop is closed."""
        mux = TeamSSEMultiplexer.__new__(TeamSSEMultiplexer)
        mux._team_id = "t"
        mux._queue = asyncio.Queue()
        mux._on_envelope = None
        mock_loop = Mock()
        mock_loop.is_closed.return_value = True
        mux._loop = mock_loop

        envelope = TeamStreamEnvelope(
            team_id="t",
            agent_id="a",
            role_name="r",
            event=_make_event(),
        )
        mux._put(envelope)
        assert not mux._queue.empty()
