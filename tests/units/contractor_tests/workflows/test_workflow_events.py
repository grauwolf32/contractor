from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from contractor.workflows import Workflow


class _ProbeWorkflow(Workflow):
    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__(MagicMock())
        self.error = error
        self.impl_calls = 0
        self.cleanup_calls = 0

    async def _run_impl(self, *, user_id, on_event):
        self.impl_calls += 1
        if self.error is not None:
            raise self.error
        return "RESULT"

    async def _cleanup(self, *, user_id):
        self.cleanup_calls += 1


@pytest.mark.asyncio
async def test_start_event_failure_is_best_effort():
    workflow = _ProbeWorkflow()

    async def broken_handler(event):
        if event.type == "workflow_started":
            raise OSError("metrics unavailable")

    assert await workflow.run(on_event=broken_handler) == "RESULT"
    assert workflow.impl_calls == 1
    assert workflow.cleanup_calls == 1


@pytest.mark.asyncio
async def test_finish_event_failure_does_not_replace_result():
    workflow = _ProbeWorkflow()

    async def broken_handler(event):
        if event.type == "workflow_finished":
            raise OSError("disk full")

    assert await workflow.run(on_event=broken_handler) == "RESULT"
    assert workflow.cleanup_calls == 1


@pytest.mark.asyncio
async def test_finish_event_failure_does_not_mask_workflow_error():
    original = ValueError("workflow failed")
    workflow = _ProbeWorkflow(error=original)

    async def broken_handler(event):
        if event.type == "workflow_finished":
            raise OSError("disk full")

    with pytest.raises(ValueError, match="workflow failed") as caught:
        await workflow.run(on_event=broken_handler)
    assert caught.value is original
    assert workflow.cleanup_calls == 1


@pytest.mark.asyncio
async def test_event_handler_cancellation_still_propagates():
    workflow = _ProbeWorkflow()

    async def cancelled_handler(event):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await workflow.run(on_event=cancelled_handler)
    assert workflow.impl_calls == 0
