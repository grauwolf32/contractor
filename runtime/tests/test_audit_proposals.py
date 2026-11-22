"""Exercise the producer's real Runtime assembler with the shared wire cases."""

import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from contractor_runtime.contracts.audit_documents import decode_audit_document
from contractor_runtime.toolsets.audit_results.proposals import assemble_proposals


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parents[2] / "api/testdata/audit-common/proposals.json").read_text()
    )["cases"],
    ids=lambda case: case["name"],
)
def test_shared_producer_assembler_cases(case):
    def assemble():
        kind = json.loads(case["document"])["schema"]
        value = decode_audit_document(case["document"].encode(), schema=kind)
        if kind == "contractor.audit.proposal-context.v1":
            return case["document"].encode()
        return assemble_proposals(
            json.dumps(value["hypotheses"]),
            json.dumps(value["work"]),
            case["context"].encode(),
            selected_materials_json=json.dumps(value.get("selected_materials", {})),
            selected_metadata=case.get("selected_metadata"),
        )

    if case["valid"]:
        result = json.loads(assemble())
        assert result["schema"] == json.loads(case["document"])["schema"]
    else:
        with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
            assemble()


def test_shared_tool_resolves_configured_roles_cas_and_cumulative_selections():
    import asyncio
    from datetime import UTC, datetime

    import jcs
    from fakes.worker_state import WorkerState

    from contractor_runtime.artifacts import ArtifactAPIError, ArtifactValue
    from contractor_runtime.contracts import (
        ArtifactMetadataResult,
        ArtifactRef,
        ArtifactWriteResult,
    )
    from contractor_runtime.core.input_errors import ToolInputError
    from contractor_runtime.toolsets.audit_results.proposals import SubmitHypothesisProposalsTool

    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/audit-common/proposals.json").read_text()
    )["cases"]
    context = json.loads(cases[0]["context"])
    target = ArtifactRef(namespace="explorer", name="drafts")
    roles = {
        "schema": "contractor.audit.proposal-binding.v1",
        "run_id": "run-1",
        "context": {"namespace": "inputs", "name": "research", "revision": "r1"},
        "result": target.model_dump(exclude_none=True),
    }

    class Client:
        allocation_id = "allocation-1"

        def __init__(self):
            self.data = None
            self.revision = None
            self.writes = 0

        async def read_artifact(self, ref, *, max_bytes):
            if ref.namespace == "audit-context":
                data = jcs.canonicalize(roles)
            elif ref.name == "research":
                data = jcs.canonicalize(context)
            elif self.data is not None:
                data = self.data
                ref = ref.model_copy(update={"revision": self.revision})
            else:
                raise ArtifactAPIError(404, "artifact_not_found", False)
            assert len(data) <= max_bytes
            return ArtifactValue(
                ref, "application/json", data, datetime.now(UTC), datetime.now(UTC)
            )

        async def read_artifact_metadata(self, ref):
            return ArtifactMetadataResult(
                apiVersion="contractor/v1alpha1",
                artifact=ref,
                mediaType="application/zip",
                size=42,
                contentCacheKey="opaque-key",
            )

        async def write_artifact(self, ref, *, data, media_type, expected_revision):
            assert ref == target
            if expected_revision != self.revision:
                raise ArtifactAPIError(409, "artifact_revision_conflict", False)
            self.writes += 1
            self.revision = f"r{self.writes}"
            self.data = data
            return ArtifactWriteResult(
                apiVersion="contractor/v1alpha1",
                artifact=ref.model_copy(update={"revision": self.revision}),
                mediaType=media_type,
                size=len(data),
            )

    async def scenario():
        client = Client()
        tool = SubmitHypothesisProposalsTool(
            client, WorkerState().resources.metrics, (), "explorer", "run-1"
        )
        refs = {
            f"choice-{i}": {"namespace": "explorer", "name": f"snapshot-{i}", "revision": "r1"}
            for i in range(30)
        }
        first = await tool("{}", "[]", selected_materials_json=json.dumps(refs))
        assert first["artifact"]["name"] == "drafts"
        await tool("{}", "[]", selected_materials_json=json.dumps(refs))
        assert client.writes == 1
        one = {"choice-30": {"namespace": "explorer", "name": "snapshot-30", "revision": "r1"}}
        await tool("{}", "[]", selected_materials_json=json.dumps(one))
        assert len(json.loads(client.data)["selected_materials"]) == 31
        overflow = {"choice-31": {"namespace": "explorer", "name": "snapshot-31", "revision": "r1"}}
        with pytest.raises(ToolInputError):
            await tool("{}", "[]", selected_materials_json=json.dumps(overflow))
        for ref in [
            {"namespace": "skills", "name": "private", "revision": "r1"},
            {"namespace": "inputs", "name": "context", "revision": "r1"},
            {"namespace": "explorer", "name": "drafts", "revision": "r1"},
            {"namespace": "elsewhere", "name": "missing-version"},
        ]:
            with pytest.raises(ToolInputError):
                await tool("{}", "[]", selected_materials_json=json.dumps({"forbidden": ref}))
        seed = {
            "candidate": {
                "statement": "A selected actor can cross a boundary",
                "scope": {},
                "grounds": "Exact selected snapshot",
                "sources": ["choice-0"],
            }
        }
        client.revision = "external-r3"
        with pytest.raises(ArtifactAPIError, match="409"):
            await tool(json.dumps(seed), "[]")
        client.revision = "r2"
        await tool(json.dumps(seed), "[]")
        assert client.writes == 3
        roles["run_id"] = "foreign-run"
        with pytest.raises(ToolInputError):
            await tool("{}", "[]")
        assert client.writes == 3

    asyncio.run(scenario())
