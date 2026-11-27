"""Offline ordinary JSON publication through the production ArtifactClient."""

import asyncio
import json
import threading
from dataclasses import replace

import pytest
from audit_fixtures import assigned, digest
from fakes.allocation_rpc import AllocationRPCFixture
from fakes.clock import advance_clock, run_with_advanceable_clock
from test_artifacts import TIMESTAMP_HEADERS, json_response

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_DOCUMENT_BYTES,
    AuditSnapshot,
    RecordedAuditItem,
)
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditResultEncoder,
)
from contractor_runtime.toolsets.audit_results.publication import DeterministicAuditResultPublisher


class ScriptedArtifactTransport(AllocationRPCFixture):
    """Exercise real API CAS/receipt checks with controlled lost replies and fencing."""

    def __init__(
        self,
        steps=(),
        *,
        existing=None,
        media_type="application/json",
        on_io=None,
        hang_advances=None,
    ):
        self.steps = list(steps)
        self.existing = existing
        self.media_type = media_type
        self.requests = []
        self.on_io = on_io
        # With hang_advances, a hanging request moves the loop clock
        # (fakes.clock) that far, so the timer that must end it fires at
        # once, and fails the test if nothing has ended it after HANG_GUARD.
        self.hang_advances = hang_advances

    async def request(self, method, path, *, headers, body, max_response_bytes):
        self.requests.append((method, path, dict(headers), body, max_response_bytes))
        if self.on_io:
            self.on_io()
        action = self.steps.pop(0) if self.steps else "normal"
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, ArtifactHTTPResponse):
            return action
        if action == "hang":
            if self.hang_advances is None:
                await asyncio.Event().wait()
            advance_clock(self.hang_advances)
            await asyncio.sleep(HANG_GUARD)
            raise AssertionError("no timer ended the hanging request")
        if action == "lost":
            raise ArtifactTransportError("private request must not leak")
        if method == "PUT":
            assert headers["If-None-Match"] == "*" and "If-Match" not in headers
            if self.existing is not None:
                return api_error(409, "artifact_conflict")
            self.existing = body
            if action == "write-then-lost":
                raise ArtifactTransportError("private committed reply lost")
            namespace, name = path.split("/")[-2:]
            return json_response(
                201,
                {
                    "apiVersion": "contractor/v1alpha1",
                    "artifact": {"namespace": namespace, "name": name, "revision": "exact-r1"},
                    "mediaType": "application/json",
                    "size": len(body),
                },
                etag='"exact-r1"',
            )
        assert method == "GET"
        if self.existing is None:
            return api_error(404, "artifact_not_found")
        return ArtifactHTTPResponse(
            200,
            {
                "content-type": self.media_type,
                "etag": '"exact-r1"',
                **TIMESTAMP_HEADERS,
            },
            self.existing,
        )


# Loop-clock seconds a hanging request waits after moving the clock. Only a
# broken timeout lets it pass, so it bounds a failing test, never a passing one.
HANG_GUARD = 30


def api_error(status, code, retryable=True):
    return json_response(
        status,
        {
            "code": code,
            "message": "private unsafe error text",
            "retryable": retryable,
            "requestId": "req-publication-1",
        },
    )


def publisher(snapshot, transport, *, target=None, check_active=lambda: None):
    return DeterministicAuditResultPublisher(
        owner=snapshot.owner,
        result_artifact=target or ArtifactRef(namespace="check", name="configured-result"),
        client=ArtifactClient(snapshot.owner.allocation_id, transport),
        check_active=check_active,
        encoder=CanonicalAuditResultEncoder(),
    )


@pytest.mark.parametrize(
    "steps,methods",
    [
        ([], ["PUT"]),
        (["write-then-lost"], ["PUT", "GET"]),
        (["lost"], ["PUT", "GET", "PUT"]),
        (["lost", "lost", "write-then-lost"], ["PUT", "GET", "PUT", "GET"]),
        ([api_error(503, "artifact_transfer_capacity")], ["PUT", "GET", "PUT"]),
    ],
)
def test_publication_success_and_lost_reply_reconciliation(steps, methods):
    async def scenario():
        snapshot, inputs = assigned(batch=True)
        transport = ScriptedArtifactTransport(steps)
        receipt = await publisher(snapshot, transport).publish(
            snapshot,
            inputs=inputs,
            deadline=asyncio.get_running_loop().time() + 5,
        )
        assert receipt.owner == snapshot.owner and receipt.revision == "exact-r1"
        assert receipt.sha256 == digest(transport.existing)
        assert receipt.size_bytes == len(transport.existing)
        assert receipt.namespace == "check" and receipt.name == "configured-result"
        assert [request[0] for request in transport.requests] == methods
        for method, path, _, body, limit in transport.requests:
            assert path == "/allocations/allocation-1/artifacts/check/configured-result"
            if method == "GET":
                assert limit == MAX_DOCUMENT_BYTES
            else:
                assert body == transport.existing

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", [None, "rationale", "media"])
def test_prior_attempt_requires_independent_complete_set_and_never_overwrites(changed):
    async def scenario():
        old, old_inputs = assigned(invocation="old")
        original = CanonicalAuditResultEncoder().encode(old, inputs=old_inputs).data
        snapshot, inputs = assigned(invocation="new")
        if changed == "rationale":
            snapshot = replace(
                snapshot,
                items=(
                    replace(
                        snapshot.items[0],
                        value=replace(snapshot.items[0].value, rationale="Another finding."),
                    ),
                ),
            )
        transport = ScriptedArtifactTransport(
            existing=original,
            media_type="text/plain" if changed == "media" else "application/json",
        )
        impl = publisher(snapshot, transport)
        with pytest.raises(AuditResultError, match="audit_result_invalid"):
            await impl.publish(AuditSnapshot(snapshot.owner, ()), inputs=inputs, deadline=10**12)
        assert not transport.requests
        if changed:
            with pytest.raises(
                AuditResultError, match="audit_result_publication_conflict"
            ) as caught:
                await impl.publish(snapshot, inputs=inputs, deadline=10**12)
            assert caught.value.retryable is False
        else:
            assert (
                await impl.publish(snapshot, inputs=inputs, deadline=10**12)
            ).revision == "exact-r1"
        assert [r[0] for r in transport.requests] == ["PUT", "GET"]
        assert transport.existing == original
        # A new child Run has its own allocation-scoped artifact storage.
        fresh_owner = replace(snapshot.owner, allocation_id="new-child-allocation")
        fresh = replace(snapshot, owner=fresh_owner)
        fresh_inputs = replace(inputs, owner=fresh_owner)
        fresh_transport = ScriptedArtifactTransport()
        await publisher(fresh, fresh_transport).publish(fresh, inputs=fresh_inputs, deadline=10**12)
        assert fresh_transport.requests[0][1].startswith("/allocations/new-child-allocation/")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "unauthorized"),
        (403, "artifact_access_denied"),
        (409, "allocation_write_fenced"),
        (404, "allocation_not_found"),
        (503, "allocation_write_fenced"),
        (400, "invalid_request"),
    ],
)
@pytest.mark.parametrize("phase", ["write", "read"])
def test_authority_and_contract_rejections_stop_without_retry_despite_flag(status, code, phase):
    async def scenario():
        snapshot, inputs = assigned()
        steps = (["lost"] if phase == "read" else []) + [api_error(status, code)]
        transport = ScriptedArtifactTransport(steps)
        with pytest.raises(AuditResultError) as caught:
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=10**12)
        assert caught.value.code == "audit_result_publication_failed"
        assert not caught.value.retryable
        assert str(caught.value) == "audit_result_publication_failed"
        assert len(transport.requests) == (2 if phase == "read" else 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "steps",
    [
        ["lost"] * 4,
        ["write-then-lost", "lost", "normal", "lost"],
        [api_error(503, "internal_error"), "lost"] * 2,
    ],
)
def test_repeated_ambiguity_is_bounded_and_never_claims_success(steps):
    async def scenario():
        snapshot, inputs = assigned()
        transport = ScriptedArtifactTransport(steps)
        with pytest.raises(AuditResultError) as caught:
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=10**12)
        assert caught.value.code == "audit_result_publication_failed" and caught.value.retryable
        assert [r[0] for r in transport.requests] == ["PUT", "GET", "PUT", "GET"]
        assert transport.requests[0][3] == transport.requests[2][3]

    asyncio.run(scenario())


@pytest.mark.parametrize("reported", [False, True])
def test_oversized_reconciliation_fails_even_with_noncompliant_transport(reported):
    async def scenario():
        snapshot, inputs = assigned()
        steps = ["normal", ArtifactResponseLimitError("private content")] if reported else []
        transport = ScriptedArtifactTransport(steps, existing=b"x" * (MAX_DOCUMENT_BYTES + 1))
        with pytest.raises(AuditResultError, match="audit_result_size_exceeded"):
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=10**12)
        assert len(transport.requests) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["before", "write", "read", "after-write"])
def test_cancellation_never_becomes_success_or_retry(phase):
    async def scenario():
        snapshot, inputs = assigned()
        cancelled = phase == "before"

        def check():
            if cancelled:
                raise asyncio.CancelledError

        def cancel_after_write():
            nonlocal cancelled
            cancelled = True

        steps = {
            "before": [],
            "write": [asyncio.CancelledError()],
            "read": ["write-then-lost", asyncio.CancelledError()],
            "after-write": [],
        }[phase]
        transport = ScriptedArtifactTransport(
            steps,
            on_io=cancel_after_write if phase == "after-write" else None,
        )
        with pytest.raises(asyncio.CancelledError):
            await publisher(snapshot, transport, check_active=check).publish(
                snapshot,
                inputs=inputs,
                deadline=10**12,
            )
        assert (
            len(transport.requests) == {"before": 0, "write": 1, "read": 2, "after-write": 1}[phase]
        )
        if phase == "after-write":
            assert transport.existing  # Diagnostic artifact does not imply completion.

    asyncio.run(scenario())


# Deadlines and request timeouts machine load cannot exhaust before the
# scripted request hangs; the hang moves the loop clock past the one under test.
PUBLICATION_DEADLINE = 600
REQUEST_TIMEOUT = 60


@pytest.mark.parametrize("phase", ["before", "write", "read"])
def test_deadline_bounds_each_io_and_reconciliation(phase):
    async def scenario():
        snapshot, inputs = assigned()
        transport = ScriptedArtifactTransport(
            ["lost", "hang"] if phase == "read" else ["hang"],
            hang_advances=PUBLICATION_DEADLINE + 1,
        )
        now = asyncio.get_running_loop().time()
        deadline = now - 1 if phase == "before" else now + PUBLICATION_DEADLINE
        with pytest.raises(AuditResultError, match="audit_result_publication_failed"):
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=deadline)
        assert len(transport.requests) == {"before": 0, "write": 1, "read": 2}[phase]

    run_with_advanceable_clock(scenario())


def test_each_publication_attempt_has_its_own_request_timeout():
    async def scenario():
        snapshot, inputs = assigned()
        # Each hang outlasts one request timeout but not the publication
        # deadline: a timeout shared by the attempts would end the whole
        # publication at the first hang, and a missing one would leave the hang
        # to its guard.
        transport = ScriptedArtifactTransport(["hang", "hang"], hang_advances=REQUEST_TIMEOUT + 1)
        started = asyncio.get_running_loop().time()
        receipt = await publisher(snapshot, transport).publish(
            snapshot,
            inputs=inputs,
            deadline=started + PUBLICATION_DEADLINE,
            request_timeout=REQUEST_TIMEOUT,
        )
        assert receipt.revision == "exact-r1"
        assert [request[0] for request in transport.requests] == ["PUT", "GET", "PUT"]

        exhausted = ScriptedArtifactTransport(["hang"] * 4, hang_advances=REQUEST_TIMEOUT + 1)
        with pytest.raises(AuditResultError, match="audit_result_publication_failed"):
            await publisher(snapshot, exhausted).publish(
                snapshot,
                inputs=inputs,
                deadline=asyncio.get_running_loop().time() + PUBLICATION_DEADLINE,
                request_timeout=REQUEST_TIMEOUT,
            )
        assert [request[0] for request in exhausted.requests] == ["PUT", "GET", "PUT", "GET"]

    run_with_advanceable_clock(scenario())


def test_publisher_pins_binding_rejects_foreign_owner_and_preserves_fatal_failure():
    async def scenario():
        snapshot, inputs = assigned()
        target = ArtifactRef(namespace="check", name="pinned")
        transport = ScriptedArtifactTransport()
        impl = publisher(snapshot, transport, target=target)
        target.name = "mutated"
        receipt = await impl.publish(snapshot, inputs=inputs, deadline=10**12)
        assert receipt.name == "pinned"
        with pytest.raises(AuditResultError):
            await impl.publish(
                replace(snapshot, owner=replace(snapshot.owner, invocation_id="foreign")),
                inputs=inputs,
                deadline=10**12,
            )
        for invalid in (
            ArtifactRef(namespace="inputs", name="task"),
            ArtifactRef(namespace="check", name="result", revision="already-exact"),
        ):
            with pytest.raises(AuditResultError):
                publisher(snapshot, transport, target=invalid)
        with pytest.raises(AuditResultError):
            DeterministicAuditResultPublisher(
                owner=snapshot.owner,
                result_artifact=target,
                client=ArtifactClient("foreign-allocation", transport),
                check_active=lambda: None,
                encoder=CanonicalAuditResultEncoder(),
            )

        class FatalBudget(RuntimeError):
            pass

        def fail():
            raise FatalBudget("budget exhausted")

        with pytest.raises(FatalBudget):
            await publisher(snapshot, transport, check_active=fail).publish(
                snapshot, inputs=inputs, deadline=10**12
            )
        assert len(transport.requests) == 1

    asyncio.run(scenario())


def test_pending_cancellation_after_inline_write_is_delivered_before_receipt():
    async def scenario():
        snapshot, inputs = assigned()
        task = asyncio.current_task()

        def schedule_cancel():
            asyncio.get_running_loop().call_soon(task.cancel)

        transport = ScriptedArtifactTransport(on_io=schedule_cancel)
        with pytest.raises(asyncio.CancelledError):
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=10**12)
        assert transport.existing and len(transport.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("revision", ["x" * 129, "\ud800"])
@pytest.mark.parametrize("phase", ["write", "read"])
def test_malformed_exact_receipts_never_produce_unbounded_or_invented_revision(phase, revision):
    async def scenario():
        snapshot, inputs = assigned()
        payload = CanonicalAuditResultEncoder().encode(snapshot, inputs=inputs).data
        bad_write = json_response(
            201,
            {
                "apiVersion": "contractor/v1alpha1",
                "artifact": {
                    "namespace": "check",
                    "name": "configured-result",
                    "revision": revision,
                },
                "mediaType": "application/json",
                "size": len(payload),
            },
            etag=json.dumps(revision),
        )
        bad_read = ArtifactHTTPResponse(
            200,
            {
                "content-type": "application/json",
                "etag": json.dumps(revision),
                **TIMESTAMP_HEADERS,
            },
            payload,
        )
        if phase == "write":
            transport = ScriptedArtifactTransport([bad_write], existing=payload)
            receipt = await publisher(snapshot, transport).publish(
                snapshot, inputs=inputs, deadline=10**12
            )
            assert receipt.revision == "exact-r1" and len(transport.requests) == 2
        else:
            transport = ScriptedArtifactTransport(["lost", bad_read] * 2)
            with pytest.raises(AuditResultError, match="audit_result_publication_failed"):
                await publisher(snapshot, transport).publish(
                    snapshot, inputs=inputs, deadline=10**12
                )
            assert len(transport.requests) == 4

    asyncio.run(scenario())


def test_oversized_sealed_set_fails_before_any_artifact_io():
    async def scenario():
        snapshot, inputs = assigned()
        value = replace(
            snapshot.items[0].value,
            tested_conditions=tuple(str(i) + "x" * 65530 for i in range(129)),
        )
        snapshot = replace(snapshot, items=(RecordedAuditItem(value, 1),))
        transport = ScriptedArtifactTransport()
        with pytest.raises(AuditResultError, match="audit_result_size_exceeded"):
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=10**12)
        assert not transport.requests

    asyncio.run(scenario())


class BlockingEncode:
    """Hold the real encoder inside its worker thread until the test releases it."""

    def __init__(self, monkeypatch):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.threads = []
        real = CanonicalAuditResultEncoder.encode

        def encode(encoder, snapshot, *, inputs):
            self.threads.append(threading.get_ident())
            self.entered.set()
            self.release.wait()
            try:
                return real(encoder, snapshot, inputs=inputs)
            finally:
                self.exited.set()

        monkeypatch.setattr(CanonicalAuditResultEncoder, "encode", encode)

    async def wait_entered(self):
        await asyncio.to_thread(self.entered.wait)


def test_publication_encodes_off_loop_and_cancellation_joins_before_any_write(monkeypatch):
    async def scenario():
        snapshot, inputs = assigned()
        transport = ScriptedArtifactTransport()
        blocking = BlockingEncode(monkeypatch)
        try:
            publication = asyncio.create_task(
                publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=10**12)
            )
            await blocking.wait_entered()
            assert threading.get_ident() not in blocking.threads
            publication.cancel()
            for _ in range(10):
                await asyncio.sleep(0)
            assert not publication.done()
        finally:
            blocking.release.set()
        with pytest.raises(asyncio.CancelledError):
            await publication
        assert blocking.exited.is_set() and not transport.requests

    asyncio.run(scenario())
