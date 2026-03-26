"""Shared utility helpers for NexAU core."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)


def get_running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    """Return the running loop or None if not in async context."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def run_async_function_sync[T](
    func: Callable[[], Coroutine[object, object, T]],
) -> T:
    """Run an async function from a sync context.

    - If no event loop is running: uses ``asyncio.run()`` to create a temporary loop.
    - If a loop IS running: raises RuntimeError — the caller should use
      ``await`` directly, or ``run_coroutine_threadsafe`` for cross-thread dispatch.

    The old implementation used ``nest_asyncio`` to monkey-patch a running loop,
    which is fragile and incompatible with uvloop/TaskGroup.  This version
    intentionally does NOT support nested loop execution.

    Args:
        func: Zero-argument callable returning a coroutine.

    Returns:
        The coroutine's return value.

    Raises:
        RuntimeError: If called from within a running event loop.
    """
    loop = get_running_loop_or_none()
    if loop is not None:
        raise RuntimeError(
            "run_async_function_sync() cannot be called from within a running "
            "event loop. Use `await func()` directly, or dispatch via "
            "`asyncio.run_coroutine_threadsafe(func(), loop)` from a worker thread."
        )
    return asyncio.run(func())


def schedule_coroutine_from_sync(
    coro: Coroutine[object, object, object],
    loop: asyncio.AbstractEventLoop | None,
) -> None:
    """Fire-and-forget schedule a coroutine from sync context (thread-safe).

    Dispatches *coro* to *loop* via ``run_coroutine_threadsafe`` if the loop
    is still running; falls back to ``asyncio.run()`` when no loop is available
    (pure CLI / script entry points).

    Args:
        coro: The coroutine to schedule.
        loop: The owning event loop captured at object-creation time, or None.
    """
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(coro, loop)
    else:
        try:
            asyncio.run(coro)
        except RuntimeError:
            logger.warning("schedule_coroutine_from_sync: no event loop available, coroutine dropped")
