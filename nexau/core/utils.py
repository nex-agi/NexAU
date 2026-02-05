"""Shared utility helpers for NexAU core."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from importlib import import_module
from typing import Protocol, cast

from asyncer import syncify


class _NestAsyncioModule(Protocol):
    def apply(self, loop: asyncio.AbstractEventLoop | None = None) -> None: ...


def apply_nest_asyncio(loop: asyncio.AbstractEventLoop) -> None:
    """Apply nest_asyncio to an existing event loop."""
    nest_asyncio = cast(_NestAsyncioModule, import_module("nest_asyncio"))
    nest_asyncio.apply(loop)


def get_running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    """Return the running loop or None if not in async context."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def run_async_function_sync[T](
    func: Callable[[], Coroutine[object, object, T]],
    *,
    raise_sync_error: bool = False,
) -> T:
    """Run an async function from sync context, reusing any running loop."""
    loop = get_running_loop_or_none()
    if loop is not None:
        apply_nest_asyncio(loop)
        return loop.run_until_complete(func())
    return syncify(func, raise_sync_error=raise_sync_error)()
