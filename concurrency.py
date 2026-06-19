"""Gestion de la concurrence multi-clients."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proxai.config import Settings

_SEMAPHORE: asyncio.Semaphore | None = None
_ACTIVE_REQUESTS = 0
_ACTIVE_LOCK: asyncio.Lock | None = None


def _get_active_lock() -> asyncio.Lock:
    global _ACTIVE_LOCK
    if _ACTIVE_LOCK is None:
        _ACTIVE_LOCK = asyncio.Lock()
    return _ACTIVE_LOCK


def init_concurrency(settings: Settings) -> None:
    global _SEMAPHORE
    _SEMAPHORE = asyncio.Semaphore(settings.max_concurrent_requests)


@asynccontextmanager
async def acquire_request_slot():
    global _ACTIVE_REQUESTS
    if _SEMAPHORE is None:
        from proxai.config import get_settings

        init_concurrency(get_settings())

    async with _SEMAPHORE:
        async with _get_active_lock():
            _ACTIVE_REQUESTS += 1
        try:
            yield
        finally:
            async with _get_active_lock():
                _ACTIVE_REQUESTS -= 1


async def wrap_stream_with_slot(
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Maintient le slot de concurrence pendant tout le streaming."""
    async with acquire_request_slot():
        async for chunk in stream:
            yield chunk


async def active_request_count() -> int:
    async with _get_active_lock():
        return _ACTIVE_REQUESTS