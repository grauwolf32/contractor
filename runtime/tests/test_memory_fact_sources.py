"""Free-form facts attach selected original memories, without quotation checks."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import jcs
import pytest
from fakes.source_selectors import artifact_sources
from fakes.worker_state import WorkerState
from test_finding_facades import context
from test_security_findings_toolset import FakeFindingClient
from test_text_artifacts_toolset import MemoryArtifactClient

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.sources import SourcePreparer
from contractor_runtime.toolsets.security_findings.facades import FactTool
from contractor_runtime.toolsets.security_findings.memories import select_memory
from contractor_runtime.toolsets.security_findings.publisher import FindingPublisher


def snapshot():
    return {
        "schemaVersion": "contractor.memory-snapshot/v1",
        "namespace": "writer",
        "notes": [
            {"name": "ignored", "content": "Working chatter", "description": "", "tags": []},
            {
                "name": "technology",
                "content": "A banner reports nginx. Version has not been independently checked.",
                "description": "Observed technology",
                "tags": ["http", "reported"],
            },
        ],
    }


class MemorySources(MemoryArtifactClient):
    def __init__(self, *, lose_reply=False):
        super().__init__()
        self.lose_reply = lose_reply

    async def write_artifact(self, *args, **kwargs):
        try:
            result = await super().write_artifact(*args, **kwargs)
        except ValueError:
            raise ArtifactAPIError(409, "artifact_conflict", False) from None
        if self.lose_reply:
            self.lose_reply = False
            raise ArtifactTransportError("source reply lost")
        return result


def make_tool(artifacts, source):
    state = WorkerState()
    state.resources.invocation_artifacts = {"memories": source}
    client = FakeFindingClient()
    publisher = FindingPublisher(
        client,
        state.resources.metrics,
        (),
        state.resources,
        artifacts,
        namespace="facts",
        source_preparer=SourcePreparer(artifacts, state.resources, "run-test", "facts", 10),
    )
    return FactTool(publisher), client, state


@pytest.mark.parametrize("lose_reply", [False, True])
@pytest.mark.parametrize("extra_sources", [0, 3])
def test_original_memory_is_exact_free_form_and_reconciles_lost_reply(lose_reply, extra_sources):
    async def scenario():
        artifacts = MemorySources(lose_reply=lose_reply)
        original = snapshot()
        source = artifacts.seed(
            "snapshot", "selected", "application/json", jcs.canonicalize(original)
        )
        tool, client, state = make_tool(artifacts, source)
        other_evidence = [
            artifacts.seed("source", f"proof_{i}", "text/plain", b"Evidence")
            for i in range(extra_sources)
        ]
        for ref in other_evidence:
            state.resources.selected_versions.observe(ref)
        # The mutable binding changes, while the assigned exact revision stays pinned.
        changed = deepcopy(original)
        changed["notes"][1]["content"] = "New current memory must not replace the original"
        artifacts.seed("snapshot", "selected", "application/json", jcs.canonicalize(changed))
        for call in ["first", "second"]:
            result = await tool(
                title="Технология сервиса",
                description="Наблюдение своими словами, без дословной цитаты.",
                memory_name="technology",
                sources=artifact_sources(other_evidence),
                tool_context=context(call),
            )
            assert result["finding_id"]
            assert len(result["evidence_refs"]) == 1 + extra_sources
            assert len(client.requests[-1]["proposal"]["evidence_ids"]) == 1 + extra_sources
            request = client.requests[-1]
            assert request["incidental"] is True and "itemKey" not in request
            assert request["proposal"]["description"].startswith("Наблюдение своими словами")
            ref = ArtifactRef.model_validate(request["evidenceRefs"][0])
            evidence = await artifacts.read_artifact(ref)
            assert json.loads(evidence.data) == {**original, "notes": [original["notes"][1]]}
        assert client.requests[0]["evidenceRefs"] == client.requests[1]["evidenceRefs"]
        assert len([key for key in artifacts.bindings if key[0] == "facts"]) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kind", ["missing-name", "no-input", "inexact", "bad-source", "wrong-media"]
)
def test_invalid_memory_selection_never_submits_a_finding(kind):
    async def scenario():
        artifacts = MemorySources()
        payload = b"{}" if kind == "bad-source" else jcs.canonicalize(snapshot())
        source = artifacts.seed(
            "snapshot",
            "selected",
            "text/plain" if kind == "wrong-media" else "application/json",
            payload,
        )
        tool, client, state = make_tool(artifacts, source)
        if kind == "no-input":
            state.resources.invocation_artifacts = {}
        elif kind == "inexact":
            state.resources.invocation_artifacts = {
                "memories": ArtifactRef(namespace="snapshot", name="selected")
            }
        with pytest.raises(ToolInputError):
            await tool(
                title="Observation",
                description="Free text",
                memory_name="absent" if kind == "missing-name" else "technology",
                tool_context=context(),
            )
        assert not client.requests and artifacts.write_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["title", "description", "source-limit", "invalid-ref", "audit"])
def test_invalid_fact_arguments_do_not_materialize_original_memory(kind):
    async def scenario():
        artifacts = MemorySources()
        source = artifacts.seed(
            "snapshot", "selected", "application/json", jcs.canonicalize(snapshot())
        )
        tool, client, state = make_tool(artifacts, source)
        arguments = {
            "title": "Observation",
            "description": "Reported technology",
            "memory_name": "technology",
        }
        if kind == "title":
            arguments["title"] = ""
        elif kind == "description":
            arguments["description"] = "x" * (32 * 1024)
        elif kind == "source-limit":
            arguments["sources"] = [
                artifacts.seed("source", f"evidence_{i}", "text/plain", b"Evidence")
                for i in range(4)
            ]
        elif kind == "invalid-ref":
            arguments["sources"] = [ArtifactRef(namespace="source", name="inexact")]
        else:
            from audit_fixtures import assigned
            from test_audit_result_collector import tool_for

            _, inputs = assigned()
            collector, _, _ = await tool_for(inputs)
            state.resources.proposal_collector = lambda: collector
            arguments.update(item_key="unknown-item", incidental=False)
        if "sources" in arguments:
            for ref in arguments["sources"]:
                if ref.revision is not None:
                    state.resources.selected_versions.observe(ref)
            arguments["sources"] = artifact_sources(arguments["sources"])
        with pytest.raises(ToolInputError):
            await tool(**arguments, tool_context=context())
        assert artifacts.write_calls == 0 and not client.requests

    asyncio.run(scenario())


def test_intake_rejection_keeps_source_and_retry_reuses_it(monkeypatch):
    async def scenario():
        artifacts = MemorySources()
        source = artifacts.seed(
            "snapshot", "selected", "application/json", jcs.canonicalize(snapshot())
        )
        tool, client, _state = make_tool(artifacts, source)
        submit = client.submit
        rejection = AsyncMock(side_effect=ArtifactAPIError(400, "invalid_finding_request", False))
        monkeypatch.setattr(client, "submit", rejection)
        arguments = {"title": "Technology", "description": "Reported", "memory_name": "technology"}
        with pytest.raises(ArtifactAPIError):
            await tool(**arguments, tool_context=context())
        saved = [key for key in artifacts.bindings if key[0] == "facts"]
        assert len(saved) == 1 and not client.requests
        monkeypatch.setattr(client, "submit", submit)
        result = await tool(**arguments, tool_context=context())
        assert result["finding_id"]
        assert [key for key in artifacts.bindings if key[0] == "facts"] == saved

    asyncio.run(scenario())


def test_extraction_quota_counts_distinct_calls_and_resets_for_a_new_invocation():
    async def scenario():
        artifacts = MemorySources()
        source = artifacts.seed(
            "snapshot", "selected", "application/json", jcs.canonicalize(snapshot())
        )
        tool, client, _state = make_tool(artifacts, source)
        for index in range(8):
            await tool(
                title=f"Claim {index}",
                description="Reported",
                memory_name="technology",
                tool_context=context(str(index)),
            )
        before = artifacts.write_calls
        with pytest.raises(ToolInputError, match="eight"):
            await tool(
                title="Ninth",
                description="Reported",
                memory_name="technology",
                tool_context=context("ninth"),
            )
        assert len(client.requests) == 8 and artifacts.write_calls == before
        # Reconciliation of an existing tool call does not consume another slot.
        await tool(
            title="Claim 0",
            description="Reported",
            memory_name="technology",
            tool_context=context("0"),
        )
        await tool(
            title="New invocation",
            description="Reported",
            memory_name="technology",
            tool_context=context("0", invocation_id="invocation-new"),
        )
        assert len(client.requests) == 10

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unknown", "tags", "namespace", "noncanonical", "deep"]
)
def test_snapshot_selection_is_strict_without_interpreting_the_claim(mutation):
    value = snapshot()
    if mutation == "duplicate":
        value["notes"].append(deepcopy(value["notes"][1]))
    elif mutation == "unknown":
        value["extra"] = True
    elif mutation == "tags":
        value["notes"][1]["tags"] = None
    elif mutation == "namespace":
        value["namespace"] = "skills"
    data = jcs.canonicalize(value)
    if mutation == "noncanonical":
        data += b"\n"
    elif mutation == "deep":
        data = b"[" * 2000 + b"0" + b"]" * 2000
    with pytest.raises((ValueError, RecursionError, TypeError)):
        select_memory(data, "technology")


def test_shared_finding_memory_sources():
    from pathlib import Path

    from contractor_runtime.toolsets.security_findings.memories import (
        MAX_MEMORY_FACT_CALLS,
        MAX_SNAPSHOT_BYTES,
        MAX_SNAPSHOT_MEMORIES,
    )

    fixture = json.loads(
        (Path(__file__).parents[2] / "api/testdata/finding-memory-sources.json").read_text()
    )
    assert fixture["limits"] == {
        "snapshotBytes": MAX_SNAPSHOT_BYTES,
        "snapshotMemories": MAX_SNAPSHOT_MEMORIES,
        "factCalls": MAX_MEMORY_FACT_CALLS,
    }
    assert fixture["cases"]
    for case in fixture["cases"]:
        if case["selectionValid"]:
            selected = json.loads(select_memory(case["snapshotJson"].encode(), case["memoryName"]))
            assert selected["notes"][0]["name"] == case["memoryName"]
        else:
            with pytest.raises((ValueError, TypeError, RecursionError)):
                select_memory(case["snapshotJson"].encode(), case["memoryName"])
