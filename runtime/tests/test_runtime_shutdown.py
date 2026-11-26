"""Real allocation teardown keeps neighbors owned through failure/cancellation."""

import asyncio

import pytest
from test_allocation import make_service, make_spec


@pytest.mark.parametrize("operation", ["close", "expire_control_lease"])
@pytest.mark.parametrize("cancel", [False, True])
def test_shutdown_joins_neighbors_before_raising(tmp_path, runtime_capabilities, operation, cancel):
    async def scenario():
        _, service = await make_service(tmp_path, runtime_capabilities, allocation_slots=4)
        for index in range(2):
            await service.prepare(make_spec(allocation_id=f"allocation-{index}"))
        first, second = service._allocations.values()
        original_release = first._cleanup.release
        original_first_stop = first._stop_worker_for_fence
        original_second_stop = second._stop_worker_for_fence
        entered, resume, first_settled, second_finished = [asyncio.Event() for _ in range(4)]

        async def failing(*args, **kwargs):
            await entered.wait()
            task = asyncio.current_task()
            task.add_done_callback(lambda _: first_settled.set())
            raise OSError("injected cleanup failure")

        async def blocked(*args, **kwargs):
            entered.set()
            await resume.wait()
            await original_second_stop(*args, **kwargs)
            second_finished.set()

        if operation == "close":
            first._cleanup.release = failing
        else:
            first._stop_worker_for_fence = failing
        second._stop_worker_for_fence = blocked
        shutdown = asyncio.create_task(getattr(service, operation)(30))
        try:
            await first_settled.wait()
            assert not shutdown.done()
            assert second._context.worker is not None
            if cancel:
                shutdown.cancel()
            resume.set()
            result = (await asyncio.gather(shutdown, return_exceptions=True))[0]
            assert second_finished.is_set()
            assert second._context is None or second._context.worker is None
            assert isinstance(result, asyncio.CancelledError if cancel else Exception)
        finally:
            resume.set()
            await asyncio.gather(shutdown, return_exceptions=True)
            first._cleanup.release = original_release
            first._stop_worker_for_fence = original_first_stop
            second._stop_worker_for_fence = original_second_stop
            await service.close(30)
        assert not service._allocations

    asyncio.run(scenario())
