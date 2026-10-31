"""Retained Artifact history answers the Runtime shares with the Go Server.

internal/artifacts, internal/httpapi/privateartifacts and internal/memory read
the same api/testdata fixture; make test-wire-cross-language runs both sides.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from shared_config_defaults import MAX_ARTIFACT_BYTES
from test_memory_toolset import (
    ArtifactRequestAttempt,
    FakeArtifactClient,
    FaultingArtifactTransport,
    create_tools,
)

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import (
    ARTIFACT_HISTORY_MAINTENANCE_PENDING,
    ArtifactAPIError,
    ArtifactClient,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.document_write import write_document_exact
from contractor_runtime.toolsets.memory.tools import MemoryToolError, MemoryToolsetFactory

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"
ERROR_KEYS = {"code", "status", "retryable", "retryAfter", "resend", "memoryCode"}


def _cases() -> dict[str, Any]:
    cases = json.loads((FIXTURES / "artifact-history-cases.json").read_text(encoding="utf-8"))
    if set(cases) != {"description", "policy", "errors"} or not cases["errors"]:
        raise ValueError("artifact-history-cases.json has an unexpected shape")
    for error in cases["errors"]:
        if set(error) != ERROR_KEYS:
            raise ValueError(f"history error {error} needs exactly {sorted(ERROR_KEYS)}")
    return cases


CASES = _cases()
ERRORS = CASES["errors"]


class _RejectingTransport(FaultingArtifactTransport):
    """Answer the first PUTs with one private Artifact API error."""

    def __init__(self, backend: FakeArtifactClient, error: dict[str, Any], rejections: int):
        super().__init__(backend)
        self._error = error
        self._rejections = rejections

    async def request(self, method, path, **kwargs):
        if method == "PUT" and self._rejections > 0:
            self._rejections -= 1
            self.put_attempts.append(
                ArtifactRequestAttempt(path, dict(kwargs["headers"]), kwargs["body"])
            )
            return self._error_response(
                ArtifactAPIError(
                    self._error["status"], self._error["code"], self._error["retryable"]
                )
            )
        # The base transport records an accepted PUT itself.
        return await super().request(method, path, **kwargs)


def test_shared_artifact_history_pending_is_the_only_resendable_answer() -> None:
    pending = [error for error in ERRORS if error["resend"]]
    if [error["code"] for error in pending] != [ARTIFACT_HISTORY_MAINTENANCE_PENDING]:
        raise AssertionError(f"resendable history answers {pending}")
    for error in ERRORS:
        if error["resend"] and not (error["retryable"] and error["retryAfter"]):
            raise AssertionError(f"{error['code']} is resendable without Retry-After")
        if error["memoryCode"] == "memory_namespace_full" and error["code"].startswith(
            "artifact_history_"
        ):
            raise AssertionError("history pressure never reports a full Memory Namespace")


@pytest.mark.parametrize("error", ERRORS, ids=[error["code"] for error in ERRORS])
def test_shared_artifact_history_memory_codes(error: dict[str, Any]) -> None:
    async def scenario() -> None:
        backend = FakeArtifactClient()
        transport = _RejectingTransport(backend, error, rejections=1)
        client = ArtifactClient("allocation-1", transport)
        tools = await create_tools(
            MemoryToolsetFactory(lambda *_: client), WorkerState(), ["write_memory"]
        )
        with pytest.raises(MemoryToolError) as failure:
            await tools["write_memory"]("shared", "body")
        if failure.value.code != error["memoryCode"]:
            raise AssertionError(f"{error['code']} maps to {failure.value.code}")
        # Every answer is definite: the tool neither replays nor reconciles it.
        if len(transport.put_attempts) != 1 or backend.semantic_writes != 0:
            raise AssertionError(f"{error['code']} sent {len(transport.put_attempts)} writes")

    asyncio.run(scenario())


@pytest.mark.parametrize("error", ERRORS, ids=[error["code"] for error in ERRORS])
def test_shared_artifact_history_document_writes(error: dict[str, Any]) -> None:
    async def scenario() -> None:
        backend = FakeArtifactClient()
        transport = _RejectingTransport(backend, error, rejections=1)
        client = ArtifactClient("allocation-1", transport)
        target = ArtifactRef(namespace="builder", name="document")

        async def write() -> object:
            return await write_document_exact(
                client,
                target,
                data=b"document",
                media_type="text/plain",
                expected_revision=None,
                max_bytes=MAX_ARTIFACT_BYTES,
                reload_tool="reload_document",
                read_tool="read_document",
            )

        if error["resend"]:
            # Pending maintenance stored nothing, so the one resend may store it.
            written = await write()
            if len(transport.put_attempts) != 2 or written.artifact.revision is None:
                raise AssertionError(f"{error['code']} resend wrote {written}")
            return
        with pytest.raises(ArtifactAPIError) as failure:
            await write()
        decoded = failure.value
        if (decoded.status_code, decoded.code, decoded.retryable) != (
            error["status"],
            error["code"],
            error["retryable"],
        ):
            raise AssertionError(f"decoded {decoded!r} for {error}")
        if len(transport.put_attempts) != 1:
            raise AssertionError(f"{error['code']} is final, yet sent {transport.put_attempts}")

    asyncio.run(scenario())


def test_shared_artifact_history_pending_twice_is_not_a_conflict() -> None:
    (pending,) = [error for error in ERRORS if error["resend"]]

    async def scenario() -> None:
        backend = FakeArtifactClient()
        transport = _RejectingTransport(backend, pending, rejections=2)
        client = ArtifactClient("allocation-1", transport)
        with pytest.raises(ArtifactAPIError) as failure:
            await write_document_exact(
                client,
                ArtifactRef(namespace="builder", name="document"),
                data=b"document",
                media_type="text/plain",
                expected_revision=None,
                max_bytes=MAX_ARTIFACT_BYTES,
                reload_tool="reload_document",
                read_tool="read_document",
            )
        if failure.value.code != pending["code"] or len(transport.put_attempts) != 2:
            raise AssertionError(f"second pending answer became {failure.value!r}")

    asyncio.run(scenario())
