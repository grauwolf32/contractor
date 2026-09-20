"""Model-facing repair errors and canonical input normalization."""

import asyncio
from types import SimpleNamespace

import pytest
from test_audit_results_toolset import digest, fixture_inputs

from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.audit_results.collector import InvocationAuditCollector
from contractor_runtime.toolsets.audit_results.contracts import (
    AuditInvocationOwner,
    AuditTrustedInputs,
)
from contractor_runtime.toolsets.audit_results.v2 import SubmitCheckResultTool

CONTEXT = SimpleNamespace(invocation_id="invocation-1")
VALID = {
    "assessment": "satisfied",
    "summary": "Verified the requested control.",
    "completed": ["source-trace"],
    "gaps": [],
    "evidence": [{"kind": "source-trace", "summary": "The guard precedes object access."}],
}


def tools():
    task, execution = fixture_inputs()
    metrics = MetricsState()
    owner = AuditInvocationOwner(
        "allocation-1", CONTEXT.invocation_id, digest(task), ("check-authz",)
    )
    collector = InvocationAuditCollector(AuditTrustedInputs(owner, task, execution))
    return SubmitCheckResultTool(collector, metrics), collector, metrics


def test_identifiers_are_canonicalized_without_changing_model_arguments():
    async def scenario():
        tool, collector, _ = tools()
        supplied = {
            **VALID,
            "completed": ["source-trace", "source-trace"],
            "gaps": ["z-gap", "a-gap", "z-gap"],
            "proposal_keys": ["z-proposal", "a-proposal", "z-proposal"],
        }
        result = await tool(tool_context=CONTEXT, **supplied)
        assert "error" not in result
        item = (await collector.snapshot()).items[0].value
        assert item.completed == ("source-trace",)
        assert item.gaps == ("a-gap", "z-gap")
        assert item.proposal_keys == ("a-proposal", "z-proposal")
        assert (await tool(tool_context=CONTEXT, **supplied))["revisions"] == result["revisions"]
        assert supplied["completed"] == ["source-trace", "source-trace"]
        assert supplied["gaps"] == ["z-gap", "a-gap", "z-gap"]

    asyncio.run(scenario())


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("assessment", {}),
        ("summary", "\ud800"),
        ("completed", '["source-trace"]'),
        ("completed", [{}]),
        ("completed", ["\ud800"]),
        ("gaps", ["gap"] * 513),
        ("proposal_keys", ["proposal"] * 129),
        ("evidence", "[]"),
        ("evidence", [{}]),
    ],
)
def test_invalid_arguments_return_field_errors_without_writes(batch, field, value):
    async def scenario():
        tool, collector, metrics = tools()
        supplied = {**VALID, field: value}
        result = await tool(
            tool_context=CONTEXT, **({"results": [supplied]} if batch else supplied)
        )
        assert result["error"]["code"] == "audit_result_invalid"
        assert result["error"]["field"] == field
        assert not (await collector.snapshot()).items
        assert metrics.tool_calls[-1].error.message.startswith(field + ":")

    asyncio.run(scenario())


def test_unknown_coverage_returns_allowed_values_and_can_be_repaired():
    async def scenario():
        tool, collector, metrics = tools()
        result = await tool(tool_context=CONTEXT, **{**VALID, "completed": ["invented-step"]})
        error = result["error"]
        assert error["field"] == "completed"
        assert error["index"] == 0
        assert error["invalidValue"] == "invented-step"
        assert error["allowedValues"] == ["source-trace"]
        assert error["itemKey"] == "check-authz"
        assert "invented-step" not in repr(metrics.tool_calls) + repr(metrics.errors)
        assert not (await collector.snapshot()).items
        assert "error" not in await tool(tool_context=CONTEXT, **VALID)

    asyncio.run(scenario())
