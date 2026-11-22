"""Runtime-owned model write bases keep stale and sibling changes explicit."""

import asyncio
from types import SimpleNamespace

import pytest
from fakes.worker_state import WorkerState
from test_text_artifacts_toolset import MemoryArtifactClient, make_tools

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.versions import MAX_SELECTED_VERSIONS, SelectedVersions
from contractor_runtime.worker.function_tools import ContractorFunctionTool


def test_write_bases_are_frozen_per_response_and_recovered_only_by_content_reads(tmp_path):
    async def scenario():
        client, state = MemoryArtifactClient(), WorkerState()
        tools = await make_tools(tmp_path, client, state)
        versions = state.resources.selected_versions
        write, read = tools["write_text_artifact"], tools["read_artifact"]
        first = await write("report", "first", "text/plain")
        versions.begin_response()
        second = await write("report", "second", "text/plain")
        with pytest.raises(ValueError, match="CAS"):
            await write("report", "sibling", "text/plain")
        versions.begin_response()
        client.seed("worker-space", "report", "text/plain", b"external")
        with pytest.raises(ValueError, match="CAS"):
            await write("report", "stale", "text/plain")
        assert versions.base("worker-space", "report") == second["artifact"]["revision"]
        assert second["artifact"]["revision"] != first["artifact"]["revision"]
        fresh = await read("worker-space", "report")
        versions.begin_response()
        result = await write("report", "reviewed", "text/plain")
        assert result["artifact"]["revision"] != fresh["artifact"]["revision"]
        tool = ContractorFunctionTool(write, logical_invocation_id=lambda _: "logical")
        assert (
            "expected_revision" not in tool._get_declaration().parameters_json_schema["properties"]
        )
        with pytest.raises(ToolInputError, match="undeclared"):
            await tool.run_async(
                args={
                    "name": "report",
                    "text": "bad",
                    "media_type": "text/plain",
                    "expected_revision": None,
                },
                tool_context=SimpleNamespace(invocation_id="sdk"),
            )
        versions.clear()
        assert versions.base("worker-space", "report") is None
        with pytest.raises(ValueError, match="CAS"):
            await write("report", "new invocation", "text/plain")

    asyncio.run(scenario())


def test_selected_versions_bound_preserves_previously_admitted_bases():
    versions = SelectedVersions()
    for index in range(MAX_SELECTED_VERSIONS):
        versions.observe(ArtifactRef(namespace="worker", name=f"n{index}", revision="r1"))
    with pytest.raises(ToolInputError, match="limit"):
        versions.admit_write(ArtifactRef(namespace="worker", name="overflow"))
    versions.begin_response()
    versions.observe(ArtifactRef(namespace="worker", name="n0", revision="r2"))
    assert versions.base("worker", "n0") == "r1"
    versions.begin_response()
    assert versions.base("worker", "n0") == "r2"


def test_material_selectors_reject_versions_and_discovery_survives_response_boundaries():
    versions = SelectedVersions()
    ref = ArtifactRef(namespace="inputs", name="proof", revision="pinned")
    versions.assign([ref])
    versions.discover_findings([{"findingId": "finding-1", "revision": 3}])
    for _ in range(3):
        versions.begin_response()
        assert versions.resolve({"namespace": "inputs", "name": "proof"}) == ref
        assert versions.finding_revision("finding-1") == 3
    with pytest.raises(ToolInputError, match="namespace/name only"):
        versions.resolve(ref.model_dump())
    with pytest.raises(ToolInputError, match="not selected"):
        versions.resolve({"namespace": "inputs", "name": "unknown"})
    versions.assign([ref, ref.model_copy(update={"revision": "other"})])
    with pytest.raises(ToolInputError, match="Ambiguous"):
        versions.resolve({"namespace": "inputs", "name": "proof"})
    versions.clear()
    with pytest.raises(ToolInputError, match="discovered"):
        versions.finding_revision("finding-1")
