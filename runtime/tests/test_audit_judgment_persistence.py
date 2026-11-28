"""Frozen private mutations and honest local state across acknowledgement loss."""

import asyncio
import json
from dataclasses import replace

import grpc
import pytest
from audit_fixtures import assigned
from fakes.audit_judgments import JudgmentTransport

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient, ArtifactTransportError
from contractor_runtime.audit_judgments import AuditJudgmentClient, JudgmentPersistenceUnknown
from contractor_runtime.toolsets.audit_results.collector import (
    AuditCollectionError,
    InvocationAuditCollector,
)
from contractor_runtime.toolsets.audit_results.encoding import CanonicalAuditResultEncoder


class FaultTransport(JudgmentTransport):
    def __init__(self, inputs, *, lost=0, block=False):
        super().__init__(inputs)
        self.lost, self.block = lost, block
        self.committed = asyncio.Event()

    async def submit_audit_judgments(self, *args, **kwargs):
        response = await super().submit_audit_judgments(*args, **kwargs)
        self.committed.set()
        if self.block:
            await asyncio.Event().wait()
        if self.lost:
            self.lost -= 1
            raise ArtifactTransportError("private provider canary must not leak")
        return response


def setup(*, lost=0, block=False):
    expected, inputs = assigned(batch=True, checking=False)
    transport = FaultTransport(inputs, lost=lost, block=block)
    encoder = CanonicalAuditResultEncoder()
    collector = InvocationAuditCollector(
        inputs,
        encoder,
        persist=AuditJudgmentClient(ArtifactClient("allocation-1", transport), encoder, inputs),
    )
    return collector, transport, expected.items[0].value


def test_lost_ack_reconciles_identical_body_without_another_semantic_revision():
    async def scenario():
        collector, transport, item = setup(lost=1)
        receipt = await collector.record(item)
        assert receipt.accepted_count == 1 and len(receipt.materials) == 1
        assert len(transport.requests) == 2
        assert transport.requests[0] == transport.requests[1]
        assert len(transport.receipts) == 1
        assert transport.items[item.item_key][0] == 1
        assert (await collector.snapshot()).items[0].revision == 1

    asyncio.run(scenario())


def test_unknown_commit_then_later_response_recovers_server_base_without_old_retry_rollback():
    async def scenario():
        collector, transport, item = setup(lost=2)
        with pytest.raises(JudgmentPersistenceUnknown) as unknown:
            await collector.record(item)
        assert "canary" not in str(unknown.value)
        assert not (await collector.snapshot()).items
        assert transport.items[item.item_key][0] == 1
        original = transport.requests[0]
        correction = replace(item, rationale="New later correction B")
        collector.begin_response()
        with pytest.raises(AuditCollectionError):
            await collector.record(correction)
        sends = len(transport.requests)
        # The trusted stale response updates only the next response's base.
        # A sibling of the rejected call cannot use it to overwrite a result.
        with pytest.raises(AuditCollectionError):
            await collector.record(correction)
        assert len(transport.requests) == sends
        collector.begin_response()
        receipt = await collector.record(correction)
        assert receipt.revisions == ((item.item_key, 2),)
        await transport.submit_audit_judgments("allocation-1", json.loads(original))
        assert transport.items[item.item_key][0] == 2
        assert transport.items[item.item_key][1]["rationale"] == correction.rationale
        assert (await collector.snapshot()).items[0].value.rationale == correction.rationale

    asyncio.run(scenario())


def test_cancel_after_commit_retains_owner_value_without_claiming_local_acceptance():
    async def scenario():
        collector, transport, item = setup(block=True)
        task = asyncio.create_task(collector.record(item))
        await transport.committed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not (await collector.snapshot()).items
        assert transport.items[item.item_key][0] == 1
        await collector.discard()
        assert transport.items[item.item_key][0] == 1

    asyncio.run(scenario())


def test_definite_rejection_is_not_retried_or_locally_accepted():
    async def scenario():
        collector, transport, item = setup()

        async def reject(allocation_id, value):
            transport.requests.append(value)
            raise ArtifactAPIError(
                grpc.StatusCode.FAILED_PRECONDITION, "audit_result_limit_exceeded", False, "fixture"
            )

        transport.submit_audit_judgments = reject
        with pytest.raises(ArtifactAPIError) as error:
            await collector.record(item)
        assert error.value.code == "audit_result_limit_exceeded"
        assert not error.value.retryable
        assert len(transport.requests) == 1
        assert not transport.items and not (await collector.snapshot()).items

    asyncio.run(scenario())


@pytest.mark.parametrize("bad_reply", [b"{}", b"{", b'{"apiVersion":"wrong"}'])
@pytest.mark.parametrize("persistent", [False, True])
def test_invalid_success_receipt_reconciles_same_committed_mutation(bad_reply, persistent):
    async def scenario():
        collector, transport, item = setup()
        request = transport.submit_audit_judgments

        async def malformed_once(*args, **kwargs):
            response = await request(*args, **kwargs)
            if persistent or len(transport.requests) == 1:
                return {"receipt": json.loads(bad_reply)} if bad_reply != b"{" else {}
            return response

        transport.submit_audit_judgments = malformed_once
        if persistent:
            with pytest.raises(JudgmentPersistenceUnknown):
                await collector.record(item)
            assert not (await collector.snapshot()).items
        else:
            receipt = await collector.record(item)
            assert receipt.accepted_count == 1
        assert len(transport.requests) == 2
        assert transport.requests[0] == transport.requests[1]
        assert transport.items[item.item_key][0] == 1

    asyncio.run(scenario())


def test_unknown_correction_clears_readiness_without_guessing_owner_generation():
    async def scenario():
        collector, transport, item = setup()
        expected, _ = assigned(batch=True, checking=False)
        for record in expected.items:
            await collector.record(record.value)
        transport.lost = 2
        correction = replace(item, rationale="Corrected statement")
        with pytest.raises(JudgmentPersistenceUnknown):
            await collector.record(correction)
        assert transport.items[item.item_key][0] == 2
        assert len((await collector.snapshot()).items) == 1
        with pytest.raises(AuditCollectionError):
            await collector.seal()
        assert collector.item_base(item.item_key) == 1
        collector.begin_response()
        # An identical committed correction can acknowledge its current value.
        await collector.record(correction)
        sealed = await collector.seal()
        assert sealed.items[0].revision == 2
        assert sealed.items[0].value.rationale == correction.rationale

    asyncio.run(scenario())
