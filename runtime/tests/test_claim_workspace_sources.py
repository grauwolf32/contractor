"""Allocation composition exposes only optional source capture to claim tools."""

import asyncio

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

from contractor_runtime.allocation.errors import AllocationError
from contractor_runtime.allocation.preparation import AllocationPreparation
from contractor_runtime.projectfs import MemoryWorkspaceProvider, hydrate_workspace
from contractor_runtime.toolsets.audit_results.v2 import AuditResultsToolsetFactory
from contractor_runtime.toolsets.security_findings.facades import (
    CodeFindingsToolsetFactory,
    GeneralFindingsToolsetFactory,
    HTTPFindingsToolsetFactory,
)
from contractor_runtime.toolsets.security_findings.trace import TraceFindingsToolsetFactory

FACTORIES = (
    GeneralFindingsToolsetFactory,
    CodeFindingsToolsetFactory,
    HTTPFindingsToolsetFactory,
    AuditResultsToolsetFactory,
)


@pytest.mark.parametrize("factory_type", FACTORIES)
@pytest.mark.parametrize("mode", ["overlay", "direct", "absent"])
@pytest.mark.parametrize("source_path", ["app.py", "schemas/App::Identity::handle/input.json"])
def test_claim_source_view_uses_hydrated_workspace_without_requiring_one(
    factory_type, mode, source_path
):
    async def scenario():
        factory = factory_type(lambda *_: None)
        if mode == "absent":
            assert AllocationPreparation._workspace_tool_view(factory, None) is None
            return
        original = "def handle():\n    return external_client.get('/items')\n"
        spec, reader = workspace_inputs([("source", "", archive({source_path: original.encode()}))])
        spec.mode = mode
        session = await hydrate_workspace(
            provider=MemoryWorkspaceProvider(settings("memory")),
            spec=spec,
            artifact_reader=reader,
            allocation_id="allocation-1",
            timeout_seconds=5,
        )
        try:
            assert await session.reader_view().read_text(source_path) == original
            view = AllocationPreparation._workspace_tool_view(factory, session)
            if mode == "direct":
                assert view is None
                return
            capture = await view.capture_source_file(source_path)
            assert capture.original == original
            assert capture.current == original
            assert capture.source_path == source_path
            assert capture.source_ref == spec.sources[0].artifact
            assert not hasattr(view, "write_text")
            assert not hasattr(view, "read_text")
        finally:
            await session.close()

    asyncio.run(scenario())


def test_mandatory_trace_capture_still_rejects_direct_workspace():
    async def scenario():
        spec, reader = workspace_inputs([("source", "", archive({"app.py": b"x=1\n"}))])
        session = await hydrate_workspace(
            provider=MemoryWorkspaceProvider(settings("memory")),
            spec=spec,
            artifact_reader=reader,
            allocation_id="allocation-1",
            timeout_seconds=5,
        )
        try:
            with pytest.raises(AllocationError) as caught:
                AllocationPreparation._workspace_tool_view(TraceFindingsToolsetFactory(), session)
            assert caught.value.code == "workspace_mode_unsupported"
        finally:
            await session.close()

    asyncio.run(scenario())
