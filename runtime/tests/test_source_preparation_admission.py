"""Shared source preparation admits aggregate bytes before any artifact writes."""

import asyncio
import base64
from types import SimpleNamespace

import pytest
from fakes.worker_state import WorkerState
from test_http_capture import capture

from contractor_runtime.toolsets.common.http_capture import HTTPObservation
from contractor_runtime.toolsets.common.sources import SourceArgumentError, SourcePreparer


def preparer():
    state = WorkerState()
    writes = []

    async def write_artifact(*args, **kwargs):
        writes.append((args, kwargs))
        raise RuntimeError("oversized selection reached publication")

    client = SimpleNamespace(write_artifact=write_artifact)
    return SourcePreparer(client, state.resources, "run-1", "worker", 5), state.resources, writes


async def add_capture(resources, request_id, *, owner="invocation-1", body_bytes=0):
    captures = resources.http_captures
    assert captures.new_http_request() == request_id
    value = capture(str(request_id))
    if body_bytes:
        attempt = value["attempts"][0]
        del attempt["error"]
        attempt["response"] = {
            "status": 200,
            "complete": True,
            "message": {
                "kind": "structured",
                "headers": [],
                "body_base64": base64.b64encode(b"x" * body_bytes).decode(),
            },
        }
    await captures.put(value, owner, observation=HTTPObservation("2026-11-22T12:00:00Z"))


def test_real_sized_http_selections_share_one_prepublication_byte_budget():
    async def scenario():
        selected, resources, writes = preparer()
        for request_id in (1, 2):
            await add_capture(resources, request_id, body_bytes=13 * 1024 * 1024 // 4)
        with pytest.raises(SourceArgumentError, match="budget"):
            await selected.prepare(
                [{"kind": "http", "request_id": request_id} for request_id in (1, 2)],
                "invocation-1",
                "call-1",
            )
        assert not writes

    asyncio.run(scenario())


def test_code_proofs_share_the_http_preparation_allowance():
    async def scenario():
        selected, resources, writes = preparer()
        await add_capture(resources, 1, body_bytes=4 * 1024 * 1024)
        captures = []

        async def source(file, start, end):
            captures.append(file)
            return {
                "source": {"artifact": {"namespace": "inputs", "name": "source", "revision": "r1"}},
                "selected": "x" * (1024 * 1024 - 1024),
            }

        resources.source_evidence = source
        sources = [
            {"kind": "http", "request_id": 1},
            *(
                {"kind": "code", "file": f"file-{index}.py", "start_line": 1, "end_line": 1}
                for index in range(8)
            ),
        ]
        with pytest.raises(SourceArgumentError, match="budget"):
            await selected.prepare(sources, "invocation-1", "call-1")
        assert len(captures) == 3
        assert not writes

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["missing", "foreign", "evicted"])
def test_unavailable_http_capture_is_a_safe_source_argument_error(kind):
    async def scenario():
        selected, resources, writes = preparer()
        if kind != "missing":
            await add_capture(
                resources, 1, owner="private-owner-canary" if kind == "foreign" else "invocation-1"
            )
        if kind == "evicted":
            resources.http_captures.clear("http")
        with pytest.raises(SourceArgumentError) as failure:
            await selected.prepare([{"kind": "http", "request_id": 1}], "invocation-1", "call-1")
        assert failure.value.model_details["field"] == "sources[0].request_id"
        assert "private-owner-canary" not in str(failure.value)
        assert not writes

    asyncio.run(scenario())


def test_adapter_schema_failures_remain_python_failures():
    async def scenario():
        from pydantic import ValidationError

        selected, resources, writes = preparer()

        class BrokenCapture:
            def select_observation(self, *args):
                return b'{"source":"unsupported-adapter-value"}', None

        selected.http.captures = BrokenCapture()
        with pytest.raises(ValidationError):
            await selected.prepare([{"kind": "http", "request_id": 1}], "invocation-1", "call-1")
        assert not writes

    asyncio.run(scenario())
