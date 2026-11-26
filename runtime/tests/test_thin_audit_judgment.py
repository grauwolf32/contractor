"""Thin model judgments assemble only the explicitly accepted check prefix."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from audit_fixtures import accept_fixture_sources, arguments, assigned
from fakes.audit_judgments import acknowledged_judgments
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from google.genai import types

from contractor_runtime.llm.request import _tools
from contractor_runtime.toolsets.audit_results.collector import InvocationAuditCollector
from contractor_runtime.toolsets.audit_results.encoding import CanonicalAuditResultEncoder
from contractor_runtime.toolsets.audit_results.v2 import SubmitResultsTool, SubmitResultTool

CONTEXT = SimpleNamespace(invocation_id="invocation-1")


def setup(*, batch=False):
    expected, inputs = assigned(batch=batch)
    collector = InvocationAuditCollector(
        inputs, CanonicalAuditResultEncoder(), persist=acknowledged_judgments(inputs)
    )
    metrics = WorkerState().metrics
    return (
        expected,
        inputs,
        collector,
        SubmitResultTool(collector, metrics),
        SubmitResultsTool(collector, metrics),
    )


def test_declaration_has_only_judgment_fields():
    _, _, _, tool, _ = setup()
    declaration = FunctionTool(tool)._get_declaration()
    wire = _tools([types.Tool(function_declarations=[declaration])])[0]["function"]
    assert set(wire["parameters"]["properties"]) == {
        "item_key",
        "rationale",
        "limitations",
        "verdict",
    }
    assert set(wire["parameters"]["required"]) == {"item_key", "rationale", "limitations"}


def test_reading_grounds_never_promotes_them_to_check_evidence():
    async def scenario():
        expected, _, collector, tool, _ = setup()
        task = (await collector.read_tasks(CONTEXT.invocation_id))["tasks"][0]
        assert "evidence" not in task and task["materials"][0]["source"]["kind"] == "artifact"
        result = await tool(CONTEXT, **arguments(expected.items[0].value, keyed=True))
        assert result["status"] == "error" and result["error"]["field"] == "verdict"
        assert not (await collector.snapshot()).items

    asyncio.run(scenario())


def test_accepted_notes_are_item_bound_and_snapshots_stay_frozen():
    async def scenario():
        expected, inputs, collector, tool, _ = setup(batch=True)
        one, two = (record.value for record in expected.items)
        await accept_fixture_sources(
            collector,
            item_key=one.item_key,
            conditions=("control inspected",),
            limitations=("Source inference",),
        )
        assert (await tool(CONTEXT, **arguments(one, keyed=True)))["acceptedCount"] == 1
        assert (await tool(CONTEXT, **arguments(two, keyed=True)))["status"] == "error"
        first = await collector.snapshot()
        before = CanonicalAuditResultEncoder().encode(first, inputs=inputs).data
        await accept_fixture_sources(
            collector,
            item_key=one.item_key,
            conditions=("postcondition inspected",),
            suffix="later",
        )
        assert CanonicalAuditResultEncoder().encode(first, inputs=inputs).data == before
        collector.begin_response()
        result = await tool(
            CONTEXT, **arguments(replace(one, rationale="Refined interpretation"), keyed=True)
        )
        assert result["revisions"][0]["revision"] == 2
        item = json.loads(
            CanonicalAuditResultEncoder().encode(await collector.snapshot(), inputs=inputs).data
        )["items"][0]
        assert item["tested_conditions"] == ["control inspected", "postcondition inspected"]
        assert item["limitations"] == ["Source inference"]
        assert len(item["evidence"]) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "removed",
    [
        "tested_conditions",
        "evidence",
        "http_request_ids",
        "revision",
        "expected_revision",
        "replace",
    ],
)
def test_removed_batch_fields_including_null_reject_without_mutation(removed):
    async def scenario():
        expected, _, collector, _, batch = setup()
        await accept_fixture_sources(collector)
        value = {**arguments(expected.items[0].value, keyed=True), removed: None}
        response = await batch(CONTEXT, items=[value])
        assert response["status"] == "error" and response["error"]["code"] == "audit_result_invalid"
        assert not (await collector.snapshot()).items

    asyncio.run(scenario())


def test_same_response_conflicts_and_batch_cannot_replace_a_judgment():
    async def scenario():
        expected, _, collector, tool, batch = setup(batch=True)
        await accept_fixture_sources(collector)
        one, two = (record.value for record in expected.items)
        assert (await tool(CONTEXT, **arguments(one, keyed=True)))["acceptedCount"] == 1
        collector.begin_response()
        a, b = await asyncio.gather(
            *(
                tool(CONTEXT, **arguments(replace(one, rationale=text), keyed=True))
                for text in ("A", "B")
            )
        )
        assert sum(result["status"] == "recorded" for result in (a, b)) == 1
        before = await collector.snapshot()
        assert (
            await batch(CONTEXT, items=[arguments(one, keyed=True), arguments(two, keyed=True)])
        )["status"] == "error"
        assert await collector.snapshot() == before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "removed",
    [
        "tested_conditions",
        "evidence",
        "http_request_ids",
        "revision",
        "expected_revision",
        "replace",
        "audit_id",
        "task_id",
        "run_id",
        "hypothesis_id",
    ],
)
def test_actual_scalar_adk_boundary_rejects_removed_fields_even_with_null(removed):
    async def scenario():
        from contractor_runtime.core.input_errors import ToolInputError
        from contractor_runtime.worker.function_tools import ContractorFunctionTool

        expected, _, collector, tool, _ = setup()
        declared = ContractorFunctionTool(
            tool, logical_invocation_id=lambda _: CONTEXT.invocation_id
        )
        with pytest.raises(ToolInputError, match="undeclared field"):
            await declared.run_async(
                args={**arguments(expected.items[0].value, keyed=True), removed: None},
                tool_context=CONTEXT,
            )
        assert not (await collector.snapshot()).items

    asyncio.run(scenario())
