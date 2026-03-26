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

"""Coverage improvement tests for nexau/core/utils.py.

Targets uncovered paths: get_running_loop_or_none, run_async_function_sync,
schedule_coroutine_from_sync.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from nexau.core.utils import (
    get_running_loop_or_none,
    run_async_function_sync,
    schedule_coroutine_from_sync,
)


class TestGetRunningLoopOrNone:
    def test_returns_none_when_no_loop(self):
        result = get_running_loop_or_none()
        assert result is None

    @pytest.mark.anyio
    async def test_returns_loop_when_running(self):
        result = get_running_loop_or_none()
        assert result is not None
        assert isinstance(result, asyncio.AbstractEventLoop)


class TestRunAsyncFunctionSync:
    def test_runs_coroutine_when_no_loop(self):
        async def my_coro():
            return 42

        result = run_async_function_sync(my_coro)
        assert result == 42

    @pytest.mark.anyio
    async def test_raises_when_loop_is_running(self):
        async def my_coro():
            return 42

        with pytest.raises(RuntimeError, match="cannot be called from within a running event loop"):
            run_async_function_sync(my_coro)


class TestScheduleCoroutineFromSync:
    def test_dispatch_to_running_loop(self):
        """When loop is provided and running, uses run_coroutine_threadsafe."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.is_running.return_value = True

        called = False

        async def my_coro():
            nonlocal called
            called = True

        coro = my_coro()
        with patch("nexau.core.utils.asyncio.run_coroutine_threadsafe") as mock_dispatch:
            schedule_coroutine_from_sync(coro, loop)
            mock_dispatch.assert_called_once_with(coro, loop)
        # Close the coroutine that was never awaited (mock intercepted it)
        coro.close()

    def test_falls_back_to_asyncio_run_when_no_loop(self):
        """When loop is None, falls back to asyncio.run."""
        results = []

        async def my_coro():
            results.append(True)

        schedule_coroutine_from_sync(my_coro(), None)
        assert results == [True]

    def test_handles_runtime_error_on_fallback(self):
        """When asyncio.run raises RuntimeError, logs warning and doesn't crash."""

        async def my_coro():
            pass

        coro = my_coro()
        with patch("nexau.core.utils.asyncio.run", side_effect=RuntimeError("no loop")):
            # Should not raise
            schedule_coroutine_from_sync(coro, None)
        # Close the coroutine that was never awaited (mock raised instead)
        coro.close()

    def test_dispatch_to_non_running_loop_falls_back(self):
        """When loop is provided but not running, falls back to asyncio.run."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.is_running.return_value = False

        results = []

        async def my_coro():
            results.append(True)

        schedule_coroutine_from_sync(my_coro(), loop)
        assert results == [True]
