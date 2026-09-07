"""Offline collection tests; shared task data belongs under api/testdata."""

import asyncio
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import jcs
import pytest
from test_audit_result_publication import assigned
from test_audit_results_toolset import digest, package

from contractor_runtime.allocation import WorkerState
from contractor_runtime.audit_completion_contracts import (
    MAX_SUMMARY_BYTES,
    AuditEvidence,
    AuditInvocationOwner,
    AuditSnapshot,
    AuditTrustedInputs,
    RecordedAuditItem,
)
from contractor_runtime.audit_result_collector import AuditCollectionError, InvocationAuditCollector
from contractor_runtime.audit_result_encoding import AuditResultError, CanonicalAuditPackageEncoder
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.audit_results_v2 import ReadAuditTaskTool, SubmitCheckResultTool

FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "api/testdata/audit-completion/task-local-validation.json"
    ).read_text()
)
CONTEXT = SimpleNamespace(invocation_id="invocation-1")


def fixture_assignment(task_name):
    task = FIXTURE["tasks"][task_name]
    payload = package(
        "task-fixture",
        "item-task",
        [
            ("task-document", "task.json", "application/json", jcs.canonicalize(task)),
        ],
    )
    manifest = jcs.canonicalize(
        {
            "schema": "contractor.audit.execution-manifest.v1",
            "items": [
                {
                    "item_key": task["item_key"],
                    "ordinal": 0,
                    "subject_key": task["subject_key"],
                    "task_package_id": "task-fixture",
                    "task_package_digest": digest(payload),
                    "inputs": [],
                    "task_ref": {"namespace": "inputs", "name": "task", "revision": "r1"},
                }
            ],
        }
    )
    owner = AuditInvocationOwner(
        "allocation-1", CONTEXT.invocation_id, digest(payload), (task["item_key"],)
    )
    return AuditTrustedInputs(owner, payload, manifest)


def tool_for(inputs):
    collector = InvocationAuditCollector(inputs)
    metrics = WorkerState().metrics
    return collector, SubmitCheckResultTool(collector, metrics), metrics


def arguments(item):
    return {
        "assessment": item.assessment,
        "summary": item.summary,
        "completed": list(item.completed),
        "gaps": list(item.gaps),
        "evidence": [{"kind": value.kind, "summary": value.summary} for value in item.evidence],
        "proposal_keys": list(item.proposal_keys),
    }


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_shared_task_local_acceptance(case):
    async def scenario():
        collector, tool, _ = tool_for(fixture_assignment(case["task"]))
        result = await tool(CONTEXT, **case["result"])
        assert (result["status"] == "recorded") == case["valid"], result
        assert len((await collector.snapshot()).items) == int(case["valid"])
        if case["valid"]:
            CanonicalAuditPackageEncoder().encode(await collector.seal(), inputs=collector.inputs)
        else:
            assert result["error"]["field"]
            assert len(json.dumps(result)) < 2048

    asyncio.run(scenario())


def test_reordered_parallel_recording_and_truthful_content_free_receipts():
    async def scenario():
        expected, inputs = assigned(batch=True)
        collector, tool, metrics = tool_for(inputs)
        first, second = [entry.value for entry in expected.items]
        result = await tool(CONTEXT, item_key=second.item_key, **arguments(second))
        assert result == {
            "status": "recorded",
            "revisions": [{"itemKey": second.item_key, "revision": 1}],
            "acceptedCount": 1,
            "totalCount": 2,
            "missingItemKeys": [first.item_key],
            "complete": False,
        }
        replies = await asyncio.gather(
            *[tool(CONTEXT, item_key=first.item_key, **arguments(first)) for _ in range(20)]
        )
        assert all(reply == replies[0] for reply in replies)
        assert replies[0]["complete"] and replies[0]["acceptedCount"] == 2
        assert await collector.seal() == expected
        diagnostics = repr(metrics.tool_calls)
        assert first.summary not in diagnostics and first.evidence[0].summary not in diagnostics
        assert all("artifact" not in json.dumps(reply).lower() for reply in replies)

    asyncio.run(scenario())


def test_revision_updates_and_only_immediate_update_replay():
    async def scenario():
        expected, inputs = assigned()
        collector = InvocationAuditCollector(inputs)
        initial = expected.items[0].value
        updated = replace(initial, summary="Corrected result.")
        latest = replace(initial, summary="Final correction.")
        with pytest.raises(AuditCollectionError):
            await collector.record(initial, expected_revision=1)
        await collector.record(initial)
        with pytest.raises(AuditCollectionError):
            await collector.record(updated)
        results = await asyncio.gather(
            *[collector.record(updated, expected_revision=1) for _ in range(20)]
        )
        assert all(result.revisions == ((initial.item_key, 2),) for result in results)
        assert (await collector.record(updated)).revisions == results[0].revisions
        await collector.record(latest, expected_revision=2)
        for value, revision in [(updated, 1), (latest, 1), (initial, 2), (latest, True)]:
            with pytest.raises(AuditCollectionError):
                await collector.record(value, expected_revision=revision)
        assert (await collector.record(latest, expected_revision=2)).revisions == (
            (initial.item_key, 3),
        )
        assert (await collector.snapshot()).items[0].value == latest
        # Canonical set ordering replays without a revision.
        latest = replace(latest, gaps=("gap-b", "gap-a"))
        await collector.record(latest, expected_revision=3)
        assert (await collector.record(replace(latest, gaps=("gap-a", "gap-b")))).revisions == (
            (initial.item_key, 4),
        )

    asyncio.run(scenario())


def test_parallel_conflicting_updates_choose_one_winner():
    async def scenario():
        expected, inputs = assigned()
        collector = InvocationAuditCollector(inputs)
        item = expected.items[0].value
        await collector.record(item)
        outcomes = await asyncio.gather(
            *[
                collector.record(replace(item, summary=f"Correction {index}"), expected_revision=1)
                for index in range(20)
            ],
            return_exceptions=True,
        )
        assert sum(isinstance(result, AuditCollectionError) for result in outcomes) == 19
        assert (await collector.snapshot()).items[0].revision == 2

    asyncio.run(scenario())


def test_atomic_batch_create_replay_conflict_and_scalar_mixing():
    async def scenario():
        expected, inputs = assigned(batch=True)
        collector, tool, _ = tool_for(inputs)
        first, second = [entry.value for entry in expected.items]
        await collector.record(second)
        before = await collector.snapshot()
        invalid = [arguments(first), arguments(replace(second, summary="Changed result."))]
        assert (await tool(CONTEXT, results=invalid))["status"] == "error"
        assert await collector.snapshot() == before
        batch = [arguments(first), arguments(second)]
        for mixing in ({"item_key": first.item_key}, {"expected_revision": 1}, {"evidence": []}):
            assert (await tool(CONTEXT, results=batch, **mixing))["status"] == "error"
            assert await collector.snapshot() == before
        assert (await tool(CONTEXT, results=batch[:1]))["status"] == "error"
        invalid = [arguments(first), {**arguments(second), "subject_key": "invented"}]
        assert (await tool(CONTEXT, results=invalid))["status"] == "error"
        assert await collector.snapshot() == before
        assert (await tool(CONTEXT, results=batch))["complete"]
        assert (await tool(CONTEXT, results=batch))["revisions"] == [
            {"itemKey": first.item_key, "revision": 1},
            {"itemKey": second.item_key, "revision": 1},
        ]
        assert await collector.seal() == expected

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"assessment": "secret-credential"}, "assessment"),
        ({"item_key": "secret-credential"}, "item_key"),
        ({"summary": ""}, "summary"),
        ({"summary": "x" * (MAX_SUMMARY_BYTES + 1)}, "summary"),
        ({"summary": "\ud800"}, "summary"),
        ({"completed": [{}]}, "completed"),
        ({"completed": ["source-trace"] * 2}, "completed"),
        ({"gaps": [str(index) for index in range(513)]}, "gaps"),
        ({"proposal_keys": [str(index) for index in range(129)]}, "proposal_keys"),
        ({"evidence": [{}]}, "evidence"),
        (
            {"evidence": [{"kind": "secret/credential", "summary": "secret-credential"}]},
            "evidence.kind",
        ),
        ({"evidence": [{"kind": "source-trace", "summary": ""}]}, "evidence.summary"),
        ({"expected_revision": "secret-credential"}, "expected_revision"),
    ],
)
def test_invalid_fields_never_mutate_or_echo_payload(changes, field):
    async def scenario():
        expected, inputs = assigned()
        collector, tool, metrics = tool_for(inputs)
        item = expected.items[0].value
        await collector.record(item)
        result = await tool(CONTEXT, **{**arguments(item), **changes})
        assert result["status"] == "error" and result["error"]["field"] == field
        assert "secret-credential" not in repr(result) + repr(metrics.tool_calls)
        assert await collector.snapshot() == AuditSnapshot(expected.owner, expected.items)

    asyncio.run(scenario())


def test_absent_arguments_batch_item_key_and_missing_evidence():
    async def scenario():
        for batch in (False, True):
            expected, inputs = assigned(batch=batch)
            collector, tool, _ = tool_for(inputs)
            assert (await tool(CONTEXT))["status"] == "error"
            item = expected.items[0].value
            if batch:
                assert (await tool(CONTEXT, **arguments(item)))["error"]["field"] == "item_key"
            result = await tool(
                CONTEXT, item_key=item.item_key, **arguments(replace(item, evidence=()))
            )
            assert result["error"]["field"] == "evidence"
            assert "source-trace" in result["error"]["message"]
            assert not (await collector.snapshot()).items

    asyncio.run(scenario())


def test_sealed_discarded_cancelled_and_reused_invocation():
    async def scenario():
        expected, inputs = assigned()
        collector, tool, _ = tool_for(inputs)
        item = expected.items[0].value
        with pytest.raises(AuditCollectionError):
            await collector.seal()
        await collector.record(item)
        snapshot = await collector.seal()
        with pytest.raises(FrozenInstanceError):
            snapshot.items[0].revision = 8
        for call in (collector.record(item), collector.record_batch((item,))):
            with pytest.raises(AuditCollectionError):
                await call
        reused = SimpleNamespace(invocation_id="invocation-2")
        assert (await tool(reused, **arguments(item)))["error"]["field"] == "invocation"
        _, next_inputs = assigned(invocation="invocation-2")
        assert not (await InvocationAuditCollector(next_inputs).snapshot()).items
        await collector.discard()
        await collector.discard()
        for call in (collector.record(item), collector.snapshot(), collector.seal()):
            with pytest.raises(AuditCollectionError):
                await call
        assert snapshot == expected
        # Invocation cleanup can discard drafts even when its task is cancelled.
        cancelled = InvocationAuditCollector(inputs)
        started = asyncio.Event()

        async def invocation():
            try:
                await cancelled.record(item)
                started.set()
                await asyncio.Event().wait()
            finally:
                await cancelled.discard()

        task = asyncio.create_task(invocation())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(AuditCollectionError):
            await cancelled.record(item)

    asyncio.run(scenario())


def test_read_and_submit_use_only_pinned_bytes_and_return_safe_copies(monkeypatch):
    async def scenario():
        expected, inputs = assigned()
        collector, submit, metrics = tool_for(inputs)
        ref = ArtifactRef(namespace="inputs", name="task", revision="r1")
        read = ReadAuditTaskTool(collector, ref, metrics)
        first = await read(CONTEXT)
        first["tasks"][0]["checklist"]["required_evidence"].clear()
        ref.revision = "r2"

        # Any reread of current aliases (or publication) after pinning is a bug.
        async def forbidden(*args, **kwargs):
            pytest.fail("collection tools must only use the pinned snapshot")

        monkeypatch.setattr("contractor_runtime.artifacts.ArtifactClient.read_artifact", forbidden)
        monkeypatch.setattr("contractor_runtime.artifacts.ArtifactClient.write_artifact", forbidden)
        second = await read(CONTEXT)
        assert second["task"]["checklist"]["required_evidence"] == ["source-trace"]
        assert second["taskArtifact"]["revision"] == "r1"
        assert second["executionManifestDigest"] == inputs.execution_manifest_sha256
        item = replace(expected.items[0].value, evidence=())
        assert (await submit(CONTEXT, **arguments(item)))["status"] == "error"
        stale = await read(SimpleNamespace(invocation_id="invocation-2"))
        assert stale["error"]["field"] == "invocation"
        assert FIXTURE["tasks"]["checklist"]["checklist"]["statement"] not in repr(
            metrics.tool_calls
        )
        await collector.discard()
        assert (await read(CONTEXT))["status"] == "error"

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["MAX_COLLECTED_BYTES", "MAX_MEMBER_BYTES", "MAX_PACKAGE_BYTES"])
def test_real_encoder_prospective_replacement_bound_is_atomic(monkeypatch, limit):
    async def scenario():
        expected, inputs = assigned()
        collector = InvocationAuditCollector(inputs)
        item = expected.items[0].value
        await collector.record(item)
        updated = replace(item, summary="x" * MAX_SUMMARY_BYTES)
        prospective = replace(expected, items=(RecordedAuditItem(updated, 2),))
        encoded = CanonicalAuditPackageEncoder().encode(prospective, inputs=inputs)
        exact = {
            "MAX_COLLECTED_BYTES": encoded.collected_bytes,
            "MAX_MEMBER_BYTES": max(encoded.member_bytes),
            "MAX_PACKAGE_BYTES": len(encoded.data),
        }[limit]
        monkeypatch.setattr(f"contractor_runtime.audit_result_encoding.{limit}", exact - 1)
        with pytest.raises(AuditResultError):
            CanonicalAuditPackageEncoder().encode(prospective, inputs=inputs)
        with pytest.raises(AuditCollectionError):
            await collector.record(updated, expected_revision=1)
        assert (await collector.snapshot()).items == expected.items
        monkeypatch.setattr(f"contractor_runtime.audit_result_encoding.{limit}", exact)
        await collector.record(updated, expected_revision=1)
        assert await collector.seal() == prospective

    asyncio.run(scenario())


def test_collection_evidence_budget_and_per_list_limits_are_distinct():
    async def scenario():
        expected, inputs = assigned(batch=True)
        collector = InvocationAuditCollector(inputs)
        first, second = [entry.value for entry in expected.items]
        first = replace(
            first,
            evidence=first.evidence * 128,
            gaps=tuple(f"gap-{i:03d}" for i in range(512)),
            proposal_keys=tuple(f"proposal-{i:03d}" for i in range(128)),
        )
        second = replace(
            second,
            evidence=second.evidence * 128,
            gaps=first.gaps,
            proposal_keys=first.proposal_keys,
        )
        await collector.record_batch((first, second))
        before = await collector.snapshot()
        with pytest.raises(AuditCollectionError, match="256"):
            await collector.record(
                replace(first, evidence=first.evidence + first.evidence[:1]), expected_revision=1
            )
        assert await collector.snapshot() == before
        await collector.record(replace(first, evidence=first.evidence[:1]), expected_revision=1)
        await collector.record(
            replace(second, evidence=second.evidence + second.evidence[:1]), expected_revision=1
        )
        assert len((await collector.seal()).items) == 2

    asyncio.run(scenario())


def test_actual_eight_mib_limit_rejects_large_replacement_and_preserves_revision():
    async def scenario():
        expected, inputs = assigned()
        collector = InvocationAuditCollector(inputs)
        item = expected.items[0].value
        await collector.record(item)
        large = replace(
            item, evidence=(AuditEvidence("source-trace", "x" * MAX_SUMMARY_BYTES),) * 256
        )
        with pytest.raises(AuditCollectionError, match="bounds"):
            await collector.record(large, expected_revision=1)
        assert (await collector.snapshot()).items == expected.items

    asyncio.run(scenario())
