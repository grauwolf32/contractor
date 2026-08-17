from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from google.genai import types

import contractor.workflows.trace_verify.workflow as trace_verify_module
from contractor.workflows.trace_annotation import OpenApiPath
from contractor.workflows.trace_verify.workflow import (
    MissingVerificationError,
    TraceVerifyWorkflow,
)


def _workflow(artifact_service: AsyncMock) -> TraceVerifyWorkflow:
    workflow = object.__new__(TraceVerifyWorkflow)
    workflow.ctx = SimpleNamespace(
        app_name="app",
        artifact_service=artifact_service,
        checkpoint_path=None,
        folder_name="src",
        fs=MemoryFileSystem(),
    )
    workflow.llm = object()
    return workflow


@pytest.mark.asyncio
async def test_trace_verify_rejects_runner_success_without_verification_artifact(
    monkeypatch,
):
    class SuccessfulRunnerWithoutArtifact:
        def __init__(self, **kwargs):
            pass

        def add_variable(self, **kwargs):
            pass

        def add_task(self, **kwargs):
            pass

        async def run(self, **kwargs):
            return []

    monkeypatch.setattr(
        trace_verify_module,
        "TaskRunner",
        SuccessfulRunnerWithoutArtifact,
    )
    service = AsyncMock()
    service.load_artifact = AsyncMock(return_value=None)
    workflow = _workflow(service)

    with pytest.raises(MissingVerificationError, match="finding-one"):
        await workflow._verify_namespace_findings(
            api_path=OpenApiPath(path="/items"),
            source_namespace="trace:openapi:items",
            findings=[{"name": "finding-one", "summary": "upstream"}],
            user_id="user",
            on_event=None,
        )


@pytest.mark.asyncio
async def test_trace_verify_accepts_persisted_verification_record():
    artifact = types.Part.from_text(
        text=(
            "finding-one:\n"
            "  name: finding-one\n"
            "  verdict: inconclusive\n"
        )
    )
    service = AsyncMock()
    service.load_artifact = AsyncMock(return_value=artifact)
    workflow = _workflow(service)

    await workflow._assert_verifications_persisted(
        source_namespace="trace:openapi:items",
        expected_names={"finding-one"},
        user_id="user",
    )

    service.load_artifact.assert_awaited_once_with(
        app_name="app",
        user_id="user",
        filename=(
            "user:vulnerability-verifications/trace:openapi:items"
        ),
    )


@pytest.mark.asyncio
async def test_trace_verify_rejects_unchanged_stale_verification_record():
    record = {
        "name": "finding-one",
        "verdict": "inconclusive",
        "verified_at": "2026-01-01T00:00:00+00:00",
    }
    artifact = types.Part.from_text(
        text=(
            "finding-one:\n"
            "  name: finding-one\n"
            "  verdict: inconclusive\n"
            "  verified_at: '2026-01-01T00:00:00+00:00'\n"
        )
    )
    service = AsyncMock()
    service.load_artifact = AsyncMock(return_value=artifact)
    workflow = _workflow(service)

    with pytest.raises(MissingVerificationError, match="without updating"):
        await workflow._assert_verifications_persisted(
            source_namespace="trace:openapi:items",
            expected_names={"finding-one"},
            user_id="user",
            previous_records={"finding-one": record},
            require_updated_names={"finding-one"},
        )


@pytest.mark.asyncio
async def test_trace_verify_run_propagates_missing_verification_invariant(
    monkeypatch,
):
    service = AsyncMock()
    service.load_artifact = AsyncMock(
        return_value=types.Part.from_text(text="openapi: 3.1.0\npaths: {}\n")
    )
    workflow = _workflow(service)

    async def _no_seed(*args, **kwargs):
        return None

    async def _missing(*args, **kwargs):
        raise MissingVerificationError("verdict artifact missing")

    monkeypatch.setattr(trace_verify_module, "persist_seed_artifact", _no_seed)
    monkeypatch.setattr(
        trace_verify_module,
        "extract_openapi_paths",
        lambda openapi: [OpenApiPath(path="/items")],
    )
    monkeypatch.setattr(workflow, "_verify_path_findings", _missing)

    with pytest.raises(MissingVerificationError, match="artifact missing"):
        await workflow._run_impl(user_id="user", on_event=None)
