from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from fakes.allocation_rpc import AllocationRPCFixture
from fakes.artifact_responses import ArtifactHTTPResponse
from fakes.model import json_result, scripted_model, text_result
from fakes.worker_state import WorkerState

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    AllocationWorkspaceExport,
    ArtifactRef,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    RuntimeSettings,
    SandboxProfileRef,
    StageContentRequest,
    WorkerObservations,
    WorkerResult,
    WorkerRuntimeRef,
    WorkerSessionMode,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.projectfs import (
    WORKSPACE_DIFF_MEDIA_TYPE,
    WORKSPACE_OVERLAY_MEDIA_TYPE,
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
    WorkspaceAutoExporter,
    WorkspaceExportError,
    decode_workspace_state,
)
from contractor_runtime.projectfs.config import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.projectfs.exporter import MAX_CAS_ATTEMPTS
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from contractor_runtime.worker.runtime import AdkWorkerRuntime
from contractor_runtime.workspace import AllocationWorkspace


@pytest.mark.parametrize("limit_offset", [0, -1])
def test_export_respects_a_lower_transport_budget_without_partial_checkpoint(
    monkeypatch: pytest.MonkeyPatch, limit_offset: int
) -> None:
    async def scenario() -> None:
        session = await overlay("transport-budget")
        await session.write_text("source.txt", "changed\n")
        bundle = await session.prepare_export()
        assert len(bundle.state) > len(bundle.diff)
        monkeypatch.setattr(
            "contractor_runtime.projectfs.exporter.artifact_payload_max_bytes",
            lambda: len(bundle.state) + limit_offset,
        )
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)

        if limit_offset < 0:
            with pytest.raises(WorkspaceExportError) as rejected:
                await exporter.export(worker_result())
            assert rejected.value.cause == "workspace_limit_exceeded"
            assert client.calls == []
            assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)
        else:
            await exporter.export(worker_result())
            restored = decode_workspace_state(
                client.binding("workspace_state").data, session._source, session.limits
            )
            assert restored.snapshot() == await session.snapshot()
            assert tuple(entry.path for entry in await session.change_entries()) == ()

    asyncio.run(scenario())


def test_export_persists_exact_cumulative_state_and_checkpoint_diff(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session = await overlay("complete")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")

        first = await exporter.export(worker_result())

        assert set(first.result.artifacts) == {"workspace_state", "workspace_diff"}
        assert first.result.artifacts["workspace_state"].revision == "revision-1"
        assert first.result.artifacts["workspace_diff"].revision == "revision-1"
        state = client.binding("workspace_state")
        diff = client.binding("workspace_diff")
        assert state.media_type == WORKSPACE_OVERLAY_MEDIA_TYPE
        assert diff.media_type == WORKSPACE_DIFF_MEDIA_TYPE
        assert b"first result" in diff.data
        assert json.loads(state.data)["resultWorkspaceDigest"] == first.result_workspace_digest
        reconstructed = decode_workspace_state(
            state.data,
            session._source,
            session.limits,
        )
        assert reconstructed.snapshot() == await session.snapshot()
        assert tuple(entry.path for entry in await session.change_entries()) == ()

        await session.write_text("source.txt", "second result\n")
        second = await exporter.export(worker_result())

        assert second.result.artifacts["workspace_state"].revision == "revision-2"
        assert second.result.artifacts["workspace_diff"].revision == "revision-2"
        assert b"first result" in client.binding("workspace_diff").data
        assert b"second result" in client.binding("workspace_diff").data
        cumulative = client.binding("workspace_state").data
        reconstructed = decode_workspace_state(
            cumulative,
            session._source,
            session.limits,
        )
        assert reconstructed.snapshot() == await session.snapshot()
        assert tuple(entry.path for entry in await session.change_entries()) == ()

    asyncio.run(scenario())


def test_partial_write_and_lost_response_never_advance_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("retry")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        client.fail_before("workspace_diff")

        with pytest.raises(WorkspaceExportError) as partial:
            await exporter.export(worker_result())
        assert partial.value.retryable
        assert client.binding("workspace_state").revision == "revision-1"
        assert "workspace_diff" not in client.bindings
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)

        completed = await exporter.export(worker_result())
        assert completed.result.artifacts["workspace_state"].revision == "revision-1"
        assert completed.result.artifacts["workspace_diff"].revision == "revision-1"
        assert tuple(entry.path for entry in await session.change_entries()) == ()

        await session.write_text("source.txt", "second result\n")
        client.fail_after("workspace_state")
        with pytest.raises(WorkspaceExportError):
            await exporter.export(worker_result())
        assert client.binding("workspace_state").revision == "revision-2"
        assert client.binding("workspace_diff").revision == "revision-1"
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)

        recovered = await exporter.export(worker_result())
        assert recovered.result.artifacts["workspace_state"].revision == "revision-2"
        assert recovered.result.artifacts["workspace_diff"].revision == "revision-2"
        assert tuple(entry.path for entry in await session.change_entries()) == ()

    asyncio.run(scenario())


def test_reserved_result_collision_is_rejected_without_artifact_io(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("collision")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        result = worker_result().model_copy(
            update={
                "artifacts": {
                    "workspace_state": ArtifactRef(
                        namespace="editor", name="invented", revision="revision-invented"
                    )
                }
            }
        )

        with pytest.raises(WorkspaceExportError) as collision:
            await exporter.export(result)
        assert collision.value.cause == "reserved_result_slot"
        assert not collision.value.retryable
        assert client.calls == []

    asyncio.run(scenario())


def test_adk_maps_partial_export_to_stable_retryable_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("runtime-partial")
        client = MemoryArtifactClient()
        client.fail_before("workspace_diff")
        runtime = await runtime_with_export(
            tmp_path,
            session,
            client,
            [structured_response("Model completed")],
        )
        await session.write_text("source.txt", "partial export\n")

        result = await runtime.invoke(stage_request())

        assert result.failure is not None
        assert result.failure.code == "workspace_export_failed"
        assert result.failure.retryable
        assert set(client.bindings) == {"workspace_state"}
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)
        assert runtime._metrics.counters["workspace_exports.failed"] == 1
        assert runtime._metrics.errors[-1].code == "workspace_export_failed"
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_treats_protocol_shaped_and_free_text_as_opaque_before_export(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        failed_session = await overlay("semantic-failure")
        failed_client = MemoryArtifactClient()
        failed_runtime = await runtime_with_export(
            tmp_path,
            failed_session,
            failed_client,
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "failed",
                        "summary": "Useful partial analysis",
                        "artifacts": {},
                        "error": {
                            "code": "analysis_incomplete",
                            "message": "Analysis is incomplete",
                            "retryable": True,
                        },
                    }
                )
            ],
        )
        await failed_session.write_text("source.txt", "useful partial state\n")
        failed = await failed_runtime.invoke(stage_request())
        assert failed.result is not None
        assert failed.failure is None
        assert '"outcome":"failed"' in failed.result.result
        assert set(failed.result.artifacts) == {"workspace_state", "workspace_diff"}
        assert tuple(entry.path for entry in await failed_session.change_entries()) == ()
        await failed_runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        malformed_session = await overlay("malformed")
        malformed_client = MemoryArtifactClient()
        malformed_runtime = await runtime_with_export(
            tmp_path,
            malformed_session,
            malformed_client,
            [text_result("ordinary summary")],
        )
        await malformed_session.write_text("source.txt", "must stay private\n")
        malformed = await malformed_runtime.invoke(stage_request())
        assert malformed.result is not None
        assert malformed.failure is None
        assert malformed.result.result == "ordinary summary"
        assert set(malformed.result.artifacts) == {"workspace_state", "workspace_diff"}
        assert tuple(entry.path for entry in await malformed_session.change_entries()) == ()
        await malformed_runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        reserved_session = await overlay("reserved")
        reserved_client = MemoryArtifactClient()
        reserved_runtime = await runtime_with_export(
            tmp_path,
            reserved_session,
            reserved_client,
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Invented export",
                        "artifacts": {
                            "workspace_diff": {
                                "namespace": "editor",
                                "name": "workspace_diff",
                                "revision": "invented",
                            }
                        },
                    }
                )
            ],
        )
        reserved = await reserved_runtime.invoke(stage_request())
        assert reserved.result is not None
        assert reserved.failure is None
        assert '"revision":"invented"' in reserved.result.result
        assert set(reserved.result.artifacts) == {"workspace_state", "workspace_diff"}
        await reserved_runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_invocation_cancellation_during_export_writes_nothing_and_keeps_checkpoint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session = await overlay("cancelled")
        client = MemoryArtifactClient()
        client.block_before("workspace_state")
        runtime = await runtime_with_export(
            tmp_path,
            session,
            client,
            [structured_response("Would export")],
        )
        await session.write_text("source.txt", "cancelled result\n")
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(client.blocked.wait(), timeout=1)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert client.bindings == {}
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_factory_attaches_runtime_owned_exporter(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("factory")
        client = MemoryArtifactClient()
        state = WorkerState()
        model = scripted_model([structured_response("Factory result")])
        factory = AdkWorkerRuntimeFactory(
            model_factory=lambda _: model,
            artifact_client_factory=lambda *_: client,
        )
        runtime = await factory.create(build_context(tmp_path, session, state))
        await session.write_text("source.txt", "factory export\n")

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert set(result.result.artifacts) == {"workspace_state", "workspace_diff"}
        assert runtime._workspace_exporter is not None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_metadata_that_proves_a_difference_skips_the_prior_payload() -> None:
    async def scenario() -> None:
        session = await overlay("metadata-difference")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        await exporter.export(worker_result())
        assert client.calls == [
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
            ("metadata", "workspace_diff"),
            ("write", "workspace_diff"),
        ]

        # Different sizes prove both payloads changed.
        previous = {name: len(client.binding(name).data) for name in client.bindings}
        await session.write_text("source.txt", "a considerably longer second result\n")
        client.calls.clear()
        second = await exporter.export(worker_result())
        assert {name: len(client.binding(name).data) for name in client.bindings} != previous
        assert all(len(client.binding(name).data) != previous[name] for name in previous)
        assert client.calls == [
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
            ("metadata", "workspace_diff"),
            ("write", "workspace_diff"),
        ]
        assert second.result.artifacts["workspace_state"].revision == "revision-2"

        # The same bytes under another media type are different content too.
        state = client.binding("workspace_state")
        client.commit("workspace_state", "application/json", state.data)
        client.calls.clear()
        third = await exporter.export(worker_result())
        assert ("read", "workspace_state") not in client.calls
        assert client.calls[:2] == [
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
        ]
        assert client.binding("workspace_state").media_type == WORKSPACE_OVERLAY_MEDIA_TYPE
        assert client.binding("workspace_state").data == state.data
        assert third.result.artifacts["workspace_state"].revision == "revision-4"
        assert client.reads == []

    asyncio.run(scenario())


def test_repeated_unchanged_export_reuses_the_confirmed_exact_revision() -> None:
    async def scenario() -> None:
        session = await overlay("confirmed")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        await exporter.export(worker_result())

        # The cumulative state is unchanged; the checkpoint diff became empty.
        client.calls.clear()
        second = await exporter.export(worker_result())
        assert client.calls == [
            ("metadata", "workspace_state"),
            ("metadata", "workspace_diff"),
            ("write", "workspace_diff"),
        ]
        client.calls.clear()
        third = await exporter.export(worker_result())
        assert client.calls == [
            ("metadata", "workspace_state"),
            ("metadata", "workspace_diff"),
        ]
        for result in (second, third):
            assert result.result.artifacts["workspace_state"].revision == "revision-1"
            assert result.result.artifacts["workspace_diff"].revision == "revision-2"
        assert client.reads == []
        assert set(exporter._confirmed) == {"workspace_state", "workspace_diff"}

        # Equal size at the confirmed revision but other bytes: the recorded
        # digest proves the difference, so the write needs no download.
        before = client.binding("workspace_state")
        await session.write_text("source.txt", "first resulT\n")
        bundle = await session.prepare_export()
        assert len(bundle.state) == len(before.data) and bundle.state != before.data
        client.calls.clear()
        changed = await exporter.export(worker_result())
        assert client.calls[:2] == [
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
        ]
        assert client.binding("workspace_state").data == bundle.state
        assert changed.result.artifacts["workspace_state"].revision == "revision-2"
        assert client.reads == []
        assert set(exporter._confirmed) == {"workspace_state", "workspace_diff"}

    asyncio.run(scenario())


def test_equal_size_revisions_this_exporter_never_confirmed_are_read_exactly() -> None:
    async def scenario() -> None:
        session = await overlay("unknown-revision")
        client = MemoryArtifactClient()
        await session.write_text("source.txt", "result\n")
        bundle = await session.prepare_export()
        # Another writer, or an earlier allocation, left the same content.
        client.commit("workspace_state", WORKSPACE_OVERLAY_MEDIA_TYPE, bundle.state)
        client.commit("workspace_diff", WORKSPACE_DIFF_MEDIA_TYPE, bundle.diff)
        exporter = make_exporter(session, client)

        first = await exporter.export(worker_result())
        assert client.calls == [
            ("metadata", "workspace_state"),
            ("read", "workspace_state"),
            ("metadata", "workspace_diff"),
            ("read", "workspace_diff"),
        ]
        # Each read asks for the exact observed revision, bounded by its size.
        assert client.reads == [
            ("workspace_state", "revision-1", len(bundle.state)),
            ("workspace_diff", "revision-1", len(bundle.diff)),
        ]
        assert first.result.artifacts["workspace_state"].revision == "revision-1"
        assert first.result.artifacts["workspace_diff"].revision == "revision-1"

        # Another writer replaces the state with different equal-size bytes.
        # The new revision is unknown, so the cached comparison cannot apply.
        forged = bytes([bundle.state[0] ^ 1]) + bundle.state[1:]
        client.commit("workspace_state", WORKSPACE_OVERLAY_MEDIA_TYPE, forged)
        client.calls.clear()
        client.reads.clear()
        second = await exporter.export(worker_result())
        assert client.calls[:3] == [
            ("metadata", "workspace_state"),
            ("read", "workspace_state"),
            ("write", "workspace_state"),
        ]
        assert client.reads[0] == ("workspace_state", "revision-2", len(bundle.state))
        assert client.binding("workspace_state").data == bundle.state
        assert second.result.artifacts["workspace_state"].revision == "revision-3"

    asyncio.run(scenario())


def test_lost_write_response_is_reconciled_by_exact_comparison() -> None:
    async def scenario() -> None:
        session = await overlay("lost-response")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        await exporter.export(worker_result())
        await session.write_text("source.txt", "second result\n")
        client.fail_after("workspace_state")
        with pytest.raises(WorkspaceExportError) as lost:
            await exporter.export(worker_result())
        assert lost.value.cause == "artifact_transport" and lost.value.retryable
        assert client.binding("workspace_state").revision == "revision-2"

        client.calls.clear()
        recovered = await exporter.export(worker_result())
        # The committed revision is new to this exporter, so it is compared,
        # found equal and reused without a second write.
        assert client.calls[:2] == [
            ("metadata", "workspace_state"),
            ("read", "workspace_state"),
        ]
        assert ("write", "workspace_state") not in client.calls
        assert recovered.result.artifacts["workspace_state"].revision == "revision-2"
        assert tuple(entry.path for entry in await session.change_entries()) == ()

    asyncio.run(scenario())


def test_cas_race_after_metadata_is_retried_against_fresh_metadata() -> None:
    async def scenario() -> None:
        session = await overlay("race")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        await exporter.export(worker_result())
        await session.write_text("source.txt", "second result\n")
        bundle = await session.prepare_export()
        races = 1

        def concurrent_writer(name: str) -> None:
            nonlocal races
            if name == "workspace_state" and races:
                races -= 1
                client.commit(name, WORKSPACE_OVERLAY_MEDIA_TYPE, b"{}")

        client.after_metadata = concurrent_writer
        client.calls.clear()
        raced = await exporter.export(worker_result())
        assert client.calls[:4] == [
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
        ]
        assert client.binding("workspace_state").data == bundle.state
        assert raced.result.artifacts["workspace_state"].revision == "revision-3"

        # A binding that keeps moving exhausts the bounded CAS attempts.
        races = 10
        await session.write_text("source.txt", "third result\n")
        with pytest.raises(WorkspaceExportError) as conflict:
            await exporter.export(worker_result())
        assert conflict.value.cause == "artifact_conflict" and conflict.value.retryable
        assert client.calls.count(("write", "workspace_state")) == 2 + MAX_CAS_ATTEMPTS
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)

    asyncio.run(scenario())


def test_pending_history_maintenance_is_retried_within_the_cas_attempts() -> None:
    """A 503 artifact_history_maintenance_pending stored nothing (ART-R127).

    The Server committed one maintenance pass before it answered, so the
    exporter sends the write again within its bounded CAS attempts, and a
    binding that stays pending fails retryably with the Server's code.
    """

    async def scenario() -> None:
        session = await overlay("history")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        await session.write_text("source.txt", "first result\n")
        await exporter.export(worker_result())
        await session.write_text("source.txt", "second result\n")
        bundle = await session.prepare_export()
        pending = private_error(503, "artifact_history_maintenance_pending", retryable=True)
        client.fault("write", "workspace_state", pending)
        client.calls.clear()
        exported = await exporter.export(worker_result())
        assert client.calls[:4] == [
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
            ("metadata", "workspace_state"),
            ("write", "workspace_state"),
        ]
        assert client.binding("workspace_state").data == bundle.state
        assert exported.result.artifacts["workspace_state"].revision == "revision-2"

        await session.write_text("source.txt", "third result\n")
        attempts = MAX_CAS_ATTEMPTS

        def still_pending(name: str) -> None:
            nonlocal attempts
            if name == "workspace_state" and attempts:
                attempts -= 1
                client.fault("write", name, pending)

        client.after_metadata = still_pending
        client.calls.clear()
        with pytest.raises(WorkspaceExportError) as failure:
            await exporter.export(worker_result())
        assert failure.value.cause == "artifact_history_maintenance_pending"
        assert failure.value.retryable
        assert client.calls.count(("write", "workspace_state")) == MAX_CAS_ATTEMPTS
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("kind", "status", "cause", "retryable"),
    [
        ("metadata", 503, "storage_unavailable", True),
        ("metadata", 403, "artifact_forbidden", False),
        ("metadata", 404, "allocation_not_found", False),
        ("metadata", None, "artifact_transport", True),
        ("read", None, "artifact_transport", True),
        ("read", 404, "artifact_not_found", False),
    ],
)
def test_failed_resolution_keeps_its_classification_and_never_overwrites(
    kind: str, status: int | None, cause: str, retryable: bool
) -> None:
    outcome: ArtifactHTTPResponse | Exception = (
        ArtifactTransportError("synthetic lost response")
        if status is None
        else private_error(status, cause, retryable=retryable)
    )

    async def scenario() -> None:
        session = await overlay(f"resolution-{kind}-{cause}")
        client = MemoryArtifactClient()
        await session.write_text("source.txt", "result\n")
        bundle = await session.prepare_export()
        client.commit("workspace_state", WORKSPACE_OVERLAY_MEDIA_TYPE, bundle.state)
        exporter = make_exporter(session, client)
        client.fault(kind, "workspace_state", outcome)

        with pytest.raises(WorkspaceExportError) as failure:
            await exporter.export(worker_result())
        assert (failure.value.cause, failure.value.retryable) == (cause, retryable)
        assert all(call[0] != "write" for call in client.calls)
        assert client.binding("workspace_state").revision == "revision-1"
        assert "workspace_diff" not in client.bindings
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)

        # The failure confirmed nothing: the retry compares the revision again.
        client.calls.clear()
        retried = await exporter.export(worker_result())
        assert client.calls[:2] == [
            ("metadata", "workspace_state"),
            ("read", "workspace_state"),
        ]
        assert retried.result.artifacts["workspace_state"].revision == "revision-1"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["metadata", "read"])
def test_cancelled_resolution_writes_nothing_and_confirms_nothing(kind: str) -> None:
    async def scenario() -> None:
        session = await overlay(f"cancel-{kind}")
        client = MemoryArtifactClient()
        await session.write_text("source.txt", "result\n")
        bundle = await session.prepare_export()
        client.commit("workspace_state", WORKSPACE_OVERLAY_MEDIA_TYPE, bundle.state)
        exporter = make_exporter(session, client)
        client.block(kind, "workspace_state")

        export = asyncio.create_task(exporter.export(worker_result()))
        await asyncio.wait_for(client.blocked.wait(), timeout=1)
        export.cancel()
        with pytest.raises(asyncio.CancelledError):
            await export
        assert all(call[0] != "write" for call in client.calls)
        assert set(client.bindings) == {"workspace_state"}
        assert exporter._confirmed == {}
        assert tuple(entry.path for entry in await session.change_entries()) == ("source.txt",)

        client.calls.clear()
        completed = await exporter.export(worker_result())
        assert ("read", "workspace_state") in client.calls
        assert completed.result.artifacts["workspace_state"].revision == "revision-1"

    asyncio.run(scenario())


def test_worker_release_forgets_confirmed_revisions(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await overlay("release")
        client = MemoryArtifactClient()
        exporter = make_exporter(session, client)
        runtime = AdkWorkerRuntime(
            build_context(tmp_path, session, WorkerState()),
            scripted_model([structured_response("Exported")]),
            workspace_exporter=exporter,
        )
        await runtime.start()
        await session.write_text("source.txt", "released result\n")
        result = await runtime.invoke(stage_request())
        assert result.result is not None
        assert set(exporter._confirmed) == {"workspace_state", "workspace_diff"}

        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        assert runtime._workspace_exporter is None
        assert exporter._confirmed == {}
        client.calls.clear()
        await exporter.export(worker_result())
        assert ("read", "workspace_state") in client.calls

    asyncio.run(scenario())


async def overlay(name: str) -> OverlayWorkspaceSession:
    limits = WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=1 << 20,
        max_managed_text_bytes=1 << 19,
        max_file_bytes=1 << 18,
    )
    provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=limits))
    storage = await provider.create(name)
    source = ManagedWorkspaceTree(text_files={"source.txt": "source\n"})
    return OverlayWorkspaceSession(
        storage=storage,
        limits=limits,
        directories=source.directories,
        text_files=source.text_files,
        binary_paths=source.binary_paths,
    )


def make_exporter(
    session: OverlayWorkspaceSession, client: MemoryArtifactClient
) -> WorkspaceAutoExporter:
    return WorkspaceAutoExporter(
        workspace=session,
        client=client,
        namespace="editor",
        slots=AllocationWorkspaceExport(state="workspace_state", diff="workspace_diff"),
    )


def worker_result() -> WorkerResult:
    return WorkerResult(
        subtaskId="0",
        result="Worker completed gracefully",
        observations=WorkerObservations(profile="lean@1", tools={}, truncated=False),
        artifacts={},
        summarized=False,
    )


def structured_response(result: str) -> object:
    return text_result(result)


async def runtime_with_export(
    tmp_path: Path,
    session: OverlayWorkspaceSession,
    client: MemoryArtifactClient,
    responses: list[Any],
) -> AdkWorkerRuntime:
    state = WorkerState()
    context = build_context(tmp_path, session, state)
    runtime = AdkWorkerRuntime(
        context,
        scripted_model(responses),
        workspace_exporter=make_exporter(session, client),
    )
    await runtime.start()
    return runtime


def build_context(
    tmp_path: Path, session: OverlayWorkspaceSession, state: WorkerState
) -> WorkerBuildContext:
    template = agent_template()
    return WorkerBuildContext(
        allocation_id=f"allocation-{session.storage.owner_token}",
        run_id="run-export",
        stage_execution_id="stage-export",
        logical_agent_name="editor",
        namespace="editor",
        worker_session_mode=WorkerSessionMode.SHARED,
        description=template.description,
        instruction=template.instructions.text,
        card_version=template.ref.version,
        model_policy=worker_policy(),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        tools={},
        state=state,
        resources=state.resources,
        a2a_base_url="https://runtime.example",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="test-token",
            artifactApiUrl="https://control.example",
            requestTimeoutSeconds=30,
        ),
        project_workspace=session,
        workspace_export=AllocationWorkspaceExport(state="workspace_state", diff="workspace_diff"),
    )


def agent_template() -> ResolvedAgentTemplate:
    return ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="workspace_editor", version="1", digest="sha256:" + "0" * 64
        ),
        description="Edit a project workspace",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/workspace.md",
            digest="sha256:" + "1" * 64,
            text="Analyze the workspace.",
        ),
        toolsets=[],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        subtaskId="0",
        objective="Analyze source",
        instructions="Return a result.",
        parameters={},
        artifacts={},
    )


class MemoryArtifactTransport(AllocationRPCFixture):
    def __init__(self, owner):
        self.owner = owner

    async def request(self, *args, **kwargs):
        return await self.owner._request(*args, **kwargs)


class MemoryArtifactClient(ArtifactClient):
    """The real ArtifactClient over an in-memory private Artifact API.

    A separate transport adapter owns named RPC calls. Bindings are keyed by name, and answers
    name the Namespace of the request. A write needs
    ``If-None-Match: *`` or one strong ``If-Match`` (ART-R45), and a stale one
    is a retryable ``409 artifact_conflict``. Every committed revision stays
    readable by its exact revision (ART-R49, ART-R53), the metadata route
    answers revision, media type and size without the payload (ART-R116), and a
    payload above the caller's byte limit fails as the mTLS transport does.
    Faults and blocks are injected per call kind and binding name.
    """

    def __init__(self) -> None:
        super().__init__("allocation-memory", MemoryArtifactTransport(self))
        self.bindings: dict[str, StoredArtifact] = {}
        self.history: dict[tuple[str, str], StoredArtifact] = {}
        self.calls: list[tuple[str, str]] = []
        self.reads: list[tuple[str, str | None, int]] = []
        self.after_metadata: Callable[[str], None] | None = None
        self.blocked = asyncio.Event()
        self._fail_before: set[str] = set()
        self._fail_after: set[str] = set()
        self._blocks: set[tuple[str, str]] = set()
        self._faults: dict[tuple[str, str], ArtifactHTTPResponse | Exception] = {}

    def fail_before(self, name: str) -> None:
        self._fail_before.add(name)

    def fail_after(self, name: str) -> None:
        self._fail_after.add(name)

    def block_before(self, name: str) -> None:
        self.block("write", name)

    def block(self, kind: str, name: str) -> None:
        """Hold the next ``kind`` call (metadata, read or write) of ``name``."""

        self._blocks.add((kind, name))

    def fault(self, kind: str, name: str, outcome: ArtifactHTTPResponse | Exception) -> None:
        """Answer the next ``kind`` call of ``name`` with ``outcome``."""

        self._faults[(kind, name)] = outcome

    def binding(self, name: str) -> StoredArtifact:
        return self.bindings[name]

    def commit(self, name: str, media_type: str, data: bytes) -> StoredArtifact:
        """Commit the next revision of ``name``, as any writer of the Run would."""

        current = self.bindings.get(name)
        number = 1 if current is None else int(current.revision.rsplit("-", 1)[1]) + 1
        if current is not None:
            self.history[(name, current.revision)] = current
        stored = StoredArtifact(f"revision-{number}", media_type, data)
        self.bindings[name] = stored
        return stored

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        max_response_bytes: int,
    ) -> ArtifactHTTPResponse:
        parsed = urlsplit(path)
        resource = parsed.path.split("/artifacts/", 1)[1]
        metadata = resource.endswith("/metadata")
        namespace, name = map(unquote, resource.removesuffix("/metadata").split("/"))
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=bool(parsed.query))
        revision = query.pop("revision", [None])[0]
        if query or (method == "PUT" and (metadata or revision is not None)):
            raise AssertionError(f"unexpected private Artifact API request {method} {path}")
        kind = "write" if method == "PUT" else "metadata" if metadata else "read"
        self.calls.append((kind, name))
        if kind == "read":
            self.reads.append((name, revision, max_response_bytes))
        if (kind, name) in self._blocks:
            self._blocks.remove((kind, name))
            self.blocked.set()
            await asyncio.Event().wait()
        fault = self._faults.pop((kind, name), None)
        if isinstance(fault, Exception):
            raise fault
        if fault is not None:
            return fault
        if kind == "write":
            return self._write(namespace, name, headers, body)
        current = self.bindings.get(name)
        record = (
            current
            if revision is None or (current is not None and current.revision == revision)
            else self.history.get((name, revision))
        )
        if record is None:
            return private_error(404, "artifact_not_found", retryable=False)
        if kind == "metadata":
            response = json_response(
                200,
                {**record.read_result(namespace, name), "contentCacheKey": "test_metadata_version"},
            )
            if self.after_metadata is not None:
                self.after_metadata(name)
            return response
        if len(record.data) > max_response_bytes:
            raise ArtifactResponseLimitError("Artifact API response exceeds the read byte limit")
        return ArtifactHTTPResponse(
            200,
            {"content-type": record.media_type, "etag": f'"{record.revision}"', **TIMESTAMPS},
            record.data,
        )

    def _write(
        self, namespace: str, name: str, headers: Mapping[str, str], body: bytes
    ) -> ArtifactHTTPResponse:
        if name in self._fail_before:
            self._fail_before.remove(name)
            raise ArtifactTransportError("synthetic failure before commit")
        if "If-None-Match" in headers:
            if headers["If-None-Match"] != "*" or "If-Match" in headers:
                raise AssertionError("a create must carry only If-None-Match: *")
            expected = None
        else:
            match = headers["If-Match"]
            if len(match) < 3 or match[0] != '"' or match[-1] != '"':
                raise AssertionError("an update must carry one strong If-Match")
            expected = match[1:-1]
        current = self.bindings.get(name)
        if (None if current is None else current.revision) != expected:
            return private_error(409, "artifact_conflict", retryable=True)
        stored = self.commit(name, headers["Content-Type"], body)
        if name in self._fail_after:
            self._fail_after.remove(name)
            raise ArtifactTransportError("synthetic lost response")
        return json_response(201 if expected is None else 200, stored.read_result(namespace, name))


class StoredArtifact:
    def __init__(self, revision: str, media_type: str, data: bytes) -> None:
        self.revision = revision
        self.media_type = media_type
        self.data = data

    def read_result(self, namespace: str, name: str) -> dict[str, Any]:
        return {
            "apiVersion": API_VERSION,
            "artifact": {"namespace": namespace, "name": name, "revision": self.revision},
            "mediaType": self.media_type,
            "size": len(self.data),
        }


TIMESTAMPS = {
    "x-contractor-binding-created-at": "2026-10-23T10:00:00Z",
    "x-contractor-revision-created-at": "2026-10-23T10:00:01Z",
}


def json_response(status: int, value: object) -> ArtifactHTTPResponse:
    """A JSON private API answer; a revision result also carries its ETag."""

    headers = {"content-type": "application/json", **TIMESTAMPS}
    if isinstance(value, dict) and "artifact" in value:
        headers["etag"] = f'"{value["artifact"]["revision"]}"'
    return ArtifactHTTPResponse(status, headers, json.dumps(value).encode())


def private_error(status: int, code: str, *, retryable: bool) -> ArtifactHTTPResponse:
    return ArtifactHTTPResponse(
        status,
        {"content-type": "application/json"},
        json.dumps(
            {
                "code": code,
                "message": "synthetic private Artifact API error",
                "retryable": retryable,
                "requestId": "export-test-1",
            }
        ).encode(),
    )


def worker_policy() -> ResolvedModelPolicy:
    return ResolvedModelPolicy(
        ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "2" * 64),
        model="worker-model",
        maxOutputTokens=4096,
        maxModelCalls=8,
        maxToolCalls=16,
        maxTotalTokens=32768,
        temperature=0.1,
    )
