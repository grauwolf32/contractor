"""Actual invocation collection: strict common context, authority and concurrency."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import jcs
import pytest
from audit_fixtures import arguments, assigned, context_document, digest
from fakes.worker_state import WorkerState
from test_audit_result_publication import BlockingEncode

from contractor_runtime.toolsets.audit_results.collector import (
    AuditCollectionError,
    InvocationAuditCollector,
)
from contractor_runtime.toolsets.audit_results.contracts import AuditTrustedInputs
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditResultEncoder,
)
from contractor_runtime.toolsets.audit_results.v2 import (
    SubmitCheckResultsTool,
    SubmitCheckResultTool,
)

CONTEXT = SimpleNamespace(invocation_id="invocation-1")


def tool_for(inputs):
    collector = InvocationAuditCollector(inputs, CanonicalAuditResultEncoder())
    metrics = WorkerState().metrics
    return collector, SubmitCheckResultTool(collector, metrics), metrics


def test_exploration_and_hypothesis_use_the_same_collector_without_fabricated_verdict():
    async def scenario():
        for checking in (False, True):
            expected, inputs = assigned(checking=checking)
            collector, tool, _ = tool_for(inputs)
            value = expected.items[0].value
            result = await tool(CONTEXT, **arguments(value, keyed=True))
            assert result["complete"] and result["acceptedCount"] == 1
            sealed = await collector.seal()
            output = json.loads(CanonicalAuditResultEncoder().encode(sealed, inputs=inputs).data)
            assert ("verdict" in output["items"][0]) is checking
            assert "origin" not in output["items"][0]
            assert "finding_receipts" not in output["items"][0]
            assert (
                output["items"][0]["evidence"] == json.loads(inputs.context)["items"][0]["evidence"]
            )
            with pytest.raises(AuditCollectionError):
                await collector.record(value)
            await collector.discard()
            with pytest.raises(AuditCollectionError):
                await collector.read_tasks(CONTEXT.invocation_id)

    asyncio.run(scenario())


def test_revision_replay_and_concurrent_updates_have_one_winner():
    async def scenario():
        expected, inputs = assigned()
        value = expected.items[0].value
        collector, _, _ = tool_for(inputs)
        first = await collector.record(value)
        assert (await collector.record(value)) == first
        a, b = replace(value, rationale="Trace A"), replace(value, rationale="Trace B")
        results = await asyncio.gather(
            collector.record(a, expected_revision=1),
            collector.record(b, expected_revision=1),
            return_exceptions=True,
        )
        assert sum(isinstance(result, AuditCollectionError) for result in results) == 1
        current = (await collector.snapshot()).items[0]
        assert current.revision == 2
        assert (await collector.record(current.value, expected_revision=1)).revisions[0][1] == 2
        with pytest.raises(AuditCollectionError):
            await collector.record(value, expected_revision=1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"rationale": "private\x00marker"},
        {"evidence": ["foreign-handle"]},
        {"verdict": "satisfied"},
        {"tested_conditions": "[]"},
        {"limitations": [None]},
        {"evidence": []},
        {"tested_conditions": []},
        {"verdict": "inconclusive", "limitations": []},
    ],
)
def test_invalid_arguments_leave_collection_unchanged_without_echo(changes):
    async def scenario():
        expected, inputs = assigned()
        collector, tool, _ = tool_for(inputs)
        result = await tool(
            CONTEXT, **{**arguments(expected.items[0].value, keyed=True), **changes}
        )
        assert result["status"] == "error"
        assert "private" not in json.dumps(result)
        assert not (await collector.snapshot()).items

    asyncio.run(scenario())


def test_parallel_64_members_use_trusted_order_and_read_copies():
    async def scenario():
        expected, inputs = assigned(count=64)
        collector, _, _ = tool_for(inputs)
        read = await collector.read_tasks(CONTEXT.invocation_id)
        read["tasks"][0]["objective"] = "untrusted change"
        read["tasks"][0]["evidence"][0]["material"]["digest"] = "forged"
        await asyncio.gather(*(collector.record(item.value) for item in reversed(expected.items)))
        sealed = await collector.seal()
        assert tuple(item.value.item_key for item in sealed.items) == expected.owner.item_keys
        data = json.loads(CanonicalAuditResultEncoder().encode(sealed, inputs=inputs).data)
        assert data["items"][0]["evidence"][0]["digest"] != "forged"
        assert len(data["items"]) == 64

    asyncio.run(scenario())


def test_batch_invalid_member_is_atomic_and_changed_replay_requires_revision():
    async def scenario():
        expected, inputs = assigned(batch=True)
        collector, tool, _ = tool_for(inputs)
        values = [arguments(item.value, keyed=True) for item in reversed(expected.items)]
        tool = SubmitCheckResultsTool(collector, WorkerState().metrics)
        broken = [values[0], {**values[1], "evidence": ["foreign"]}]
        assert (await tool(CONTEXT, items=broken))["status"] == "error"
        assert not (await collector.snapshot()).items
        receipt = await tool(CONTEXT, items=values)
        assert receipt["complete"]
        assert await tool(CONTEXT, items=values) == receipt
        values[0]["rationale"] = "Updated trace"
        assert (await tool(CONTEXT, items=values))["status"] == "error"
        assert (await collector.snapshot()).items[0].revision == 1

    asyncio.run(scenario())


def test_context_and_invocation_authority_are_checked_before_admission():
    async def scenario():
        expected, inputs = assigned()
        collector, tool, _ = tool_for(inputs)
        wrong = SimpleNamespace(invocation_id="foreign")
        assert (await tool(wrong, **arguments(expected.items[0].value, keyed=True)))[
            "status"
        ] == "error"
        context = context_document()
        context["items"].append(context["items"][0])
        data = jcs.canonicalize(context)
        owner = replace(inputs.owner, task_set_sha256=digest(data))
        with pytest.raises(AuditResultError):
            CanonicalAuditResultEncoder().trusted_inputs(AuditTrustedInputs(owner, data))
        with pytest.raises(AuditResultError):
            CanonicalAuditResultEncoder().trusted_inputs(
                replace(inputs, owner=replace(inputs.owner, item_keys=("foreign",)))
            )
        assert not (await collector.snapshot()).items

    asyncio.run(scenario())


def test_cancellation_joins_admission_before_discard_and_keeps_prior_revision(monkeypatch):
    async def scenario():
        expected, inputs = assigned()
        collector, _, _ = tool_for(inputs)
        await collector.record(expected.items[0].value)
        blocking = BlockingEncode(monkeypatch)
        task = asyncio.create_task(
            collector.record(
                replace(expected.items[0].value, rationale="changed"),
                expected_revision=1,
            )
        )
        await blocking.wait_entered()
        task.cancel()
        discard = asyncio.create_task(collector.discard())
        for _ in range(4):
            await asyncio.sleep(0)
        assert not task.done() and not discard.done()
        blocking.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await discard
        assert blocking.exited.is_set()

    asyncio.run(scenario())


def test_real_eight_mib_prospective_limit_preserves_accepted_revision():
    async def scenario():
        expected, inputs = assigned()
        collector, _, _ = tool_for(inputs)
        await collector.record(expected.items[0].value)
        large = replace(
            expected.items[0].value,
            tested_conditions=tuple(str(i) + "x" * 65530 for i in range(129)),
        )
        with pytest.raises(AuditCollectionError):
            await collector.record(large, expected_revision=1)
        assert (await collector.snapshot()).items[0].revision == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [1, 2, 64])
def test_keyed_batch_accepts_reversed_members_and_preserves_trusted_order(count):
    async def scenario():
        expected, inputs = assigned(count=count, checking=False)
        collector, _, _ = tool_for(inputs)
        tool = SubmitCheckResultsTool(collector, WorkerState().metrics)
        values = [
            {**arguments(item.value, keyed=True), "rationale": f"Observed member {index}"}
            for index, item in enumerate(expected.items)
        ]
        receipt = await tool(CONTEXT, items=list(reversed(values)))
        assert receipt["complete"]
        assert [item["itemKey"] for item in receipt["revisions"]] == list(expected.owner.item_keys)
        assert await tool(CONTEXT, items=values) == receipt
        snapshot = await collector.snapshot()
        assert [record.value.rationale for record in snapshot.items] == [
            f"Observed member {index}" for index in range(count)
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "missing-member"])
def test_invalid_batch_keys_preserve_all_revisions(kind):
    async def scenario():
        expected, inputs = assigned(batch=True)
        collector, _, _ = tool_for(inputs)
        tool = SubmitCheckResultsTool(collector, WorkerState().metrics)
        values = [arguments(item.value, keyed=True) for item in expected.items]
        await collector.record(expected.items[0].value)
        if kind == "missing":
            del values[0]["item_key"]
        elif kind == "duplicate":
            values[1]["item_key"] = values[0]["item_key"]
        elif kind == "foreign":
            values[0]["item_key"] = "unassigned"
        else:
            values.pop()
        assert (await tool(CONTEXT, items=values))["status"] == "error"
        snapshot = await collector.snapshot()
        assert len(snapshot.items) == 1 and snapshot.items[0].revision == 1

    asyncio.run(scenario())
