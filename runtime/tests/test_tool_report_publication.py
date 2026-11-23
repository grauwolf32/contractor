"""Ordinary report notifications are optional and cannot replace tool outcomes."""

import asyncio
import json

import pytest
from test_tool_worker import ArtifactStore, FixtureTool, request, tool_spec, worker

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.report_publications import notify_report_publication


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("subscriber", ["accept", "fail", "cancel"])
def test_shared_tool_notifies_exact_report_after_publication_without_changing_result(
    tmp_path, status, subscriber
):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        tool.result = {"status": status, "results": [{"kept": True}]}
        runtime = await worker(tmp_path, store, tool, spec=spec)
        received = []

        class Sink:
            async def receive_report(self, name, artifact):
                saved = store.revisions.get((artifact.namespace, artifact.name, artifact.revision))
                received.append(
                    (name, artifact, json.loads(saved[0])["observation"] if saved else None)
                )
                if subscriber == "fail":
                    raise RuntimeError("optional subscriber failed")
                if subscriber == "cancel":
                    raise asyncio.CancelledError

        runtime._resources.report_publication_sink = Sink()
        result = await runtime.invoke(request(spec))
        if status == "completed":
            assert result.failure is None
        else:
            assert result.failure.code == "tool_execution_failed"
        assert received == [
            (
                "scan_nuclei",
                ArtifactRef(namespace=spec.namespace, name="report", revision="revision-2"),
                tool.result,
            )
        ]
        assert await runtime.invoke(request(spec)) == result
        assert len(received) == 1 and len(tool.calls) == 1
        await runtime.finalize(spec.lease_expires_at)

    asyncio.run(scenario())


def test_shared_failed_report_publication_never_notifies_unavailable_bytes(tmp_path):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        store.fail_report = True
        runtime = await worker(tmp_path, store, tool, spec=spec)
        received = []

        class Sink:
            async def receive_report(self, name, artifact):
                received.append((name, artifact))

        runtime._resources.report_publication_sink = Sink()
        result = await runtime.invoke(request(spec))
        assert result.failure.code == "tool_report_failed" and not received
        await runtime.finalize(spec.lease_expires_at)

    asyncio.run(scenario())


def test_report_notification_joins_subscriber_before_preserving_producer_cancellation():
    async def scenario():
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Sink:
            async def receive_report(self, _name, _artifact):
                entered.set()
                await release.wait()
                finished.set()

        task = asyncio.create_task(
            notify_report_publication(
                Sink(),
                "scan_ffuf",
                ArtifactRef(namespace="research", name="report", revision="exact"),
            )
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())
