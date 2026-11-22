"""Common completion tools through ADK and exact allocation preparation."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from audit_fixtures import accept_fixture_sources, arguments, assigned
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from google.genai import types

from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import ArtifactRef, WorkerCompletionContract
from contractor_runtime.llm.openai import _tools
from contractor_runtime.toolsets.audit_results.completion import PreparedAuditCompletion
from contractor_runtime.toolsets.audit_results.encoding import AuditResultError
from contractor_runtime.toolsets.audit_results.v2 import AuditResultsToolsetFactory


def contract():
    return WorkerCompletionContract(
        kind="audit-check-results@1",
        context=ArtifactRef(namespace="inputs", name="context", revision="r1"),
        resultArtifact=ArtifactRef(namespace="check", name="result"),
    )


class ExactClient:
    allocation_id = "allocation-1"

    def __init__(self, data):
        self.data = data
        self.reads = []

    async def read_artifact(self, ref, *, max_bytes):
        self.reads.append(ref)
        return ArtifactValue(
            ref, "application/json", self.data, datetime.now(UTC), datetime.now(UTC)
        )


def test_adk_schema_exposes_only_judgment_and_selected_evidence():
    async def scenario():
        _, inputs = assigned()
        state = WorkerState()
        client = ExactClient(inputs.context)
        settings = type("Settings", (), {"request_timeout_seconds": 10})()
        tools = await AuditResultsToolsetFactory(lambda *_: client).create_selected(
            selected=["read_audit_task", "submit_result", "submit_results", "evidence"],
            allocation_id="allocation-1",
            namespace="check",
            runtime_settings=settings,
            resources=state.resources,
            completion_contract=contract(),
        )
        declaration = FunctionTool(tools["submit_result"])._get_declaration()
        wire = _tools([types.Tool(function_declarations=[declaration])])[0]["function"]
        fields = set(wire["parameters"]["properties"])
        assert fields == {
            "item_key",
            "rationale",
            "limitations",
            "verdict",
        }
        assert set(wire["parameters"]["required"]) == {
            "item_key",
            "rationale",
            "limitations",
        }
        batch = FunctionTool(tools["submit_results"])._get_declaration()
        batch_wire = _tools([types.Tool(function_declarations=[batch])])[0]["function"]
        assert set(batch_wire["parameters"]["properties"]) == {"items"}
        assert batch_wire["parameters"]["required"] == ["items"]
        record = batch_wire["parameters"]["$defs"]["BatchResultArgument"]
        assert set(record["required"]) == {
            "item_key",
            "rationale",
            "limitations",
        }
        assert "expected_revision" not in record["properties"]
        assert len(client.reads) == 1 and client.reads[0] == contract().context
        binding = tools["read_audit_task"].completion_binding
        binding.begin("invocation-1")
        context = type("Context", (), {"invocation_id": "invocation-1"})()
        read = await tools["read_audit_task"](context)
        assert read["tasks"][0]["materials"][0]["source"] == {
            "kind": "artifact",
            "namespace": "inputs",
            "name": "source",
        }
        await accept_fixture_sources(binding.current())
        assert (
            await tools["submit_result"](
                context, **arguments(assigned()[0].items[0].value, keyed=True)
            )
        )["complete"]
        await binding.end()

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["alias", "media", "kind", "extra", "duplicate"])
def test_preparation_rejects_nonexact_or_malformed_context_before_model(change):
    async def scenario():
        _, inputs = assigned()
        client = ExactClient(inputs.context)
        original = client.read_artifact

        async def read(ref, *, max_bytes):
            value = await original(ref, max_bytes=max_bytes)
            if change == "alias":
                return replace(value, artifact=ref.model_copy(update={"revision": "foreign"}))
            if change == "media":
                return replace(value, media_type="application/zip")
            if change == "kind":
                return replace(value, data=b'{"schema":"contractor.audit.task.v1"}')
            if change == "extra":
                return replace(value, data=value.data[:-1] + b',"Task":{}}')
            return replace(
                value, data=value.data.replace(b'"schema":', b'"schema":"x","schema":', 1)
            )

        client.read_artifact = read
        with pytest.raises(AuditResultError):
            await PreparedAuditCompletion.prepare(
                contract=contract(),
                allocation_id="allocation-1",
                client=client,
                timeout=10,
            )

    asyncio.run(scenario())
