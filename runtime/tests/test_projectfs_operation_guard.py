"""Cancellation cannot hand off a workspace while a worker thread owns it."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from contractor_runtime.projectfs import operation_guard
from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.projectfs.operation_guard import WorkspaceOperationGuard


@pytest.mark.parametrize("cancel", [True, False])
def test_cancelled_or_timed_out_mutation_is_owned_until_cleanup(cancel: bool) -> None:
    async def scenario() -> None:
        guard = WorkspaceOperationGuard()
        started = threading.Event()
        finish = threading.Event()
        events: list[str] = []

        def mutation() -> None:
            started.set()
            assert finish.wait(3)
            events.append("mutated")

        task = asyncio.create_task(
            guard.run(
                mutation,
                deadline=time.monotonic() + (2 if cancel else 0.05),
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(WorkspaceStorageError):
                    await task
            assert guard.fenced
            with pytest.raises(WorkspaceStorageError):
                await guard.run(lambda: events.append("unsafe"), deadline=time.monotonic() + 1)
            with pytest.raises(WorkspaceStorageError):
                await guard.close(
                    lambda: events.append("cleanup"), deadline=time.monotonic() + 0.02
                )
            assert events == []
            # The blocked syscall has not blocked this event loop.
            for _ in range(5):
                await asyncio.sleep(0)
            finish.set()
            await guard.close(lambda: events.append("duplicate"), deadline=time.monotonic() + 2)
            await guard.close(lambda: events.append("duplicate"), deadline=time.monotonic() + 2)
            assert events == ["mutated", "cleanup"]
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_deadline_expiring_just_after_launch_fences_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        guard = WorkspaceOperationGuard()
        calls = 0
        expires = time.monotonic() + 2

        def monotonic() -> float:
            nonlocal calls
            calls += 1
            return expires - 1 if calls < 3 else expires + 1

        with monkeypatch.context() as patch:
            patch.setattr(operation_guard, "time", SimpleNamespace(monotonic=monotonic))
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await guard.run(lambda: None, deadline=expires)
        assert guard.fenced
        await guard.close(lambda: None, deadline=time.monotonic() + 2)

    asyncio.run(scenario())


def test_cancelled_waiter_never_launches_its_mutation() -> None:
    async def scenario() -> None:
        guard = WorkspaceOperationGuard()
        started = threading.Event()
        finish = threading.Event()
        calls: list[str] = []

        def first() -> None:
            started.set()
            assert finish.wait(3)
            calls.append("first")

        owner = asyncio.create_task(guard.run(first, deadline=time.monotonic() + 2))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            waiter = asyncio.create_task(
                guard.run(
                    lambda: calls.append("cancelled"),
                    deadline=time.monotonic() + 2,
                )
            )
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            finish.set()
            await owner
            assert not guard.fenced
            assert await guard.run(lambda: 42, deadline=time.monotonic() + 2) == 42
            await guard.close(lambda: calls.append("cleanup"), deadline=time.monotonic() + 2)
            assert calls == ["first", "cleanup"]
        finally:
            finish.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


def test_failed_cleanup_remains_fenced_and_is_not_duplicated() -> None:
    async def scenario() -> None:
        guard = WorkspaceOperationGuard()
        calls: list[str] = []

        def cleanup() -> None:
            calls.append("cleanup")
            raise OSError("private detail")

        for _ in range(2):
            with pytest.raises(WorkspaceStorageError, match=r"^workspace_unavailable$"):
                await guard.close(cleanup, deadline=time.monotonic() + 1)
        assert guard.fenced
        assert calls == ["cleanup"]

    asyncio.run(scenario())
