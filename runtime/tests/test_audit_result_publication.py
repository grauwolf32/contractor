"""Offline publication through the production allocation ArtifactClient."""

import asyncio
import io
import json
import stat
import threading
import zipfile
from dataclasses import replace

import jcs
import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from fakes.worker_state import WorkerState
from test_artifacts import TIMESTAMP_HEADERS, json_response
from test_audit_results_toolset import digest, fixture_batch_inputs, fixture_inputs

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_PACKAGE_BYTES,
    AuditEvidence,
    AuditInvocationOwner,
    AuditSnapshot,
    AuditTrustedInputs,
    NormalizedAuditItem,
    RecordedAuditItem,
    SealedAuditSnapshot,
)
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditPackageEncoder,
)
from contractor_runtime.toolsets.audit_results.packages import (
    _build_result_package,
    _decode_task_input,
)
from contractor_runtime.toolsets.audit_results.publication import DeterministicAuditResultPublisher


def assigned(*, batch=False, invocation="invocation-1", proposals=()):
    task, execution = fixture_batch_inputs() if batch else fixture_inputs()
    keys = tuple(item["item_key"] for item in json.loads(execution)["items"])
    owner = AuditInvocationOwner("allocation-1", invocation, digest(task), keys)
    inputs = AuditTrustedInputs(owner, task, execution)
    values = tuple(
        RecordedAuditItem(
            NormalizedAuditItem(
                key,
                "satisfied",
                "Verified control.",
                ("source-trace",),
                (),
                (AuditEvidence("source-trace", "Guard at app.py:12."),),
                proposals,
            ),
            1,
        )
        for key in keys
    )
    return SealedAuditSnapshot(owner, values), inputs


def test_encoder_matches_legacy_bytes_and_excludes_revision_and_acceptance_history():
    snapshot, inputs = assigned(batch=True, proposals=("candidate",))
    encoder = CanonicalAuditPackageEncoder()
    encoded = encoder.encode(snapshot, inputs=inputs)
    # Captured from the original @1 implementation at f17476a7 in a separate checkout.
    assert (
        digest(encoded.data)
        == "sha256:50b0e966a0f6c01477464a0c16ca4668385ea5697e5df8759f70fa45f208c5ea"
    )
    # Simulate completion from opposite arrival orders; snapshot uses trusted order.
    reversed_arrival = dict(reversed([(item.value.item_key, item) for item in snapshot.items]))
    rebuilt = replace(
        snapshot,
        items=tuple(
            replace(reversed_arrival[key], revision=91) for key in snapshot.owner.item_keys
        ),
    )
    assert encoded.data == encoder.encode(rebuilt, inputs=inputs).data
    tasks = [record[0] for record in _decode_task_input(inputs.task_package, "application/zip")]
    legacy = _build_result_package(
        tasks=tasks,
        execution_digest=digest(inputs.execution_manifest),
        results=[
            {
                "assessment": item.value.assessment,
                "summary": item.value.summary,
                "completed": list(item.value.completed),
                "gaps": [],
                "evidence": [{"kind": "source-trace", "summary": "Guard at app.py:12."}],
                "proposals": [{"invocation_id": "invocation-1", "client_key": "candidate"}],
            }
            for item in snapshot.items
        ],
    )
    assert encoded.data == legacy
    with zipfile.ZipFile(io.BytesIO(encoded.data)) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "check-results.json",
            "evidence.json",
            "evidence/ev-1-1.txt",
            "evidence/ev-2-1.txt",
        ]
        assert encoded.member_bytes == tuple(info.file_size for info in archive.infolist())
        assert encoded.collected_bytes == sum(encoded.member_bytes[1:])
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.create_system == 3
            assert info.external_attr >> 16 == stat.S_IFREG | 0o644
            if info.filename.endswith(".json"):
                raw = archive.read(info)
                assert jcs.canonicalize(json.loads(raw)) == raw
        manifest = json.loads(archive.read("manifest.json"))
        for member in manifest["members"]:
            content = archive.read(member["path"])
            assert member["digest"] == digest(content)
            assert member["size"] == len(content)


def test_proposals_retain_invocation_identity_without_inventing_other_nondeterminism():
    encoder = CanonicalAuditPackageEncoder()
    left, left_inputs = assigned(invocation="first")
    right, right_inputs = assigned(invocation="second")
    assert (
        encoder.encode(left, inputs=left_inputs).data
        == encoder.encode(right, inputs=right_inputs).data
    )
    left, left_inputs = assigned(invocation="first", proposals=("candidate",))
    right, right_inputs = assigned(invocation="second", proposals=("candidate",))
    assert (
        encoder.encode(left, inputs=left_inputs).data
        != encoder.encode(right, inputs=right_inputs).data
    )


def test_encoder_prospective_partial_and_empty_do_not_fabricate_missing_items():
    snapshot, inputs = assigned(batch=True)
    encoder = CanonicalAuditPackageEncoder()
    for items in ((), snapshot.items[1:]):
        encoded = encoder.encode(AuditSnapshot(snapshot.owner, items), inputs=inputs)
        with zipfile.ZipFile(io.BytesIO(encoded.data)) as archive:
            results = json.loads(archive.read("check-results.json"))["results"]
        assert [item["item_key"] for item in results] == [item.value.item_key for item in items]


@pytest.mark.parametrize("case", ["owner", "manifest", "membership", "coverage", "sort"])
def test_encoder_rejects_mismatched_inputs_and_unencodable_content(case):
    snapshot, inputs = assigned()
    if case == "owner":
        inputs = replace(inputs, owner=replace(inputs.owner, invocation_id="another"))
    elif case == "manifest":
        inputs = replace(inputs, execution_manifest=b"{}")
    elif case == "membership":
        owner = replace(inputs.owner, item_keys=("another",))
        inputs = replace(inputs, owner=owner)
        snapshot = AuditSnapshot(owner, ())
    else:
        item = snapshot.items[0].value
        if case == "coverage":
            item = replace(item, completed=("foreign-coverage",))
        elif case == "sort":
            item = replace(item, gaps=("z", "a"))
        snapshot = replace(snapshot, items=(RecordedAuditItem(item, 1),))
    with pytest.raises(AuditResultError, match="audit_result_invalid"):
        CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs)


def test_existing_importer_proposal_limit_is_distinct_from_coverage_list_limit():
    snapshot, inputs = assigned(proposals=tuple(f"p{i:03d}" for i in range(128)))
    assert CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs).data
    with pytest.raises(ValueError, match="proposal keys"):
        replace(snapshot.items[0].value, proposal_keys=tuple(f"p{i:03d}" for i in range(129)))
    value = replace(snapshot.items[0].value, gaps=tuple(f"g{i:03d}" for i in range(512)))
    assert len(value.gaps) == 512


@pytest.mark.parametrize("limit", ["MAX_COLLECTED_BYTES", "MAX_MEMBER_BYTES", "MAX_PACKAGE_BYTES"])
def test_encoder_enforces_actual_sizes_including_zip_overhead(monkeypatch, limit):
    from contractor_runtime.toolsets.audit_results import encoding

    snapshot, inputs = assigned()
    encoded = CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs)
    actual = {
        "MAX_COLLECTED_BYTES": encoded.collected_bytes,
        "MAX_MEMBER_BYTES": max(encoded.member_bytes),
        "MAX_PACKAGE_BYTES": len(encoded.data),
    }[limit]
    monkeypatch.setattr(encoding, limit, actual)
    assert CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs) == encoded
    monkeypatch.setattr(encoding, limit, actual - 1)
    with pytest.raises(AuditResultError, match="audit_result_size_exceeded"):
        CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs)


def test_actual_evidence_amplification_hits_collected_limit_without_large_individual_members():
    snapshot, inputs = assigned()
    evidence = (AuditEvidence("source-trace", "x" * (16 * 1024)),) * 256
    item = replace(snapshot.items[0].value, evidence=evidence)
    snapshot = replace(snapshot, items=(RecordedAuditItem(item, 1),))
    with pytest.raises(AuditResultError, match="audit_result_size_exceeded"):
        CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs)


class ScriptedArtifactTransport:
    """Exercise real API CAS/receipt checks with controlled lost replies and fencing."""

    def __init__(
        self,
        steps=(),
        *,
        existing=None,
        media_type="application/zip",
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
                    "mediaType": "application/zip",
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
        encoder=CanonicalAuditPackageEncoder(),
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
                assert limit == MAX_PACKAGE_BYTES
            else:
                assert body == transport.existing

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", [None, "summary", "proposal-invocation", "media"])
def test_prior_attempt_requires_independent_complete_set_and_never_overwrites(changed):
    async def scenario():
        old, old_inputs = assigned(
            invocation="old", proposals=("candidate",) if changed == "proposal-invocation" else ()
        )
        original = CanonicalAuditPackageEncoder().encode(old, inputs=old_inputs).data
        snapshot, inputs = assigned(
            invocation="new", proposals=("candidate",) if changed == "proposal-invocation" else ()
        )
        if changed == "summary":
            snapshot = replace(
                snapshot,
                items=(
                    replace(
                        snapshot.items[0],
                        value=replace(snapshot.items[0].value, summary="Another finding."),
                    ),
                ),
            )
        transport = ScriptedArtifactTransport(
            existing=original,
            media_type="text/plain" if changed == "media" else "application/zip",
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
        transport = ScriptedArtifactTransport(steps, existing=b"x" * (MAX_PACKAGE_BYTES + 1))
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
                encoder=CanonicalAuditPackageEncoder(),
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
        payload = CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs).data
        bad_write = json_response(
            201,
            {
                "apiVersion": "contractor/v1alpha1",
                "artifact": {
                    "namespace": "check",
                    "name": "configured-result",
                    "revision": revision,
                },
                "mediaType": "application/zip",
                "size": len(payload),
            },
            etag=json.dumps(revision),
        )
        bad_read = ArtifactHTTPResponse(
            200,
            {
                "content-type": "application/zip",
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
            snapshot.items[0].value, evidence=(AuditEvidence("trace", "x" * 16384),) * 256
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
        real = CanonicalAuditPackageEncoder.encode

        def encode(encoder, snapshot, *, inputs):
            self.threads.append(threading.get_ident())
            self.entered.set()
            self.release.wait()
            try:
                return real(encoder, snapshot, inputs=inputs)
            finally:
                self.exited.set()

        monkeypatch.setattr(CanonicalAuditPackageEncoder, "encode", encode)

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


def test_shared_audit_http_observation_metadata():
    from pathlib import Path
    from types import SimpleNamespace

    from contractor_runtime.toolsets.audit_results.collector import InvocationAuditCollector
    from contractor_runtime.toolsets.audit_results.v2 import SubmitCheckResultTool
    from contractor_runtime.toolsets.common.http_capture import HTTPObservation, RecentHTTPCaptures

    fixture = json.loads(
        (
            Path(__file__).parents[2] / "api/testdata/audit-completion/evidence-observations.json"
        ).read_text()
    )
    _, inputs = assigned()

    async def scenario():
        for case in fixture["cases"]:
            value = case["evidence"]
            try:
                captures = RecentHTTPCaptures()
                observation = HTTPObservation(value["observed_at"], value["deployment_marker"])
                await captures.put(
                    fixture["capture"], inputs.owner.invocation_id, observation=observation
                )
                request_id = value["http_request_id"]
                selector = int(request_id) if str(int(request_id)) == request_id else request_id
                evidence = {
                    "kind": "source-trace",
                    "summary": value["summary"],
                    "http_request_id": selector,
                }
                if value["http_request_tag"] != fixture["capture"]["request_tag"]:
                    # Tags are Runtime-owned facts, never accepted model arguments.
                    evidence["http_request_tag"] = value["http_request_tag"]
                collector = InvocationAuditCollector(inputs, CanonicalAuditPackageEncoder())
                tool = SubmitCheckResultTool(collector, WorkerState().metrics, captures)
                reply = await tool(
                    SimpleNamespace(invocation_id=inputs.owner.invocation_id),
                    assessment="satisfied",
                    summary="Selected capture.",
                    completed=["source-trace"],
                    gaps=[],
                    evidence=[evidence],
                )
                accepted = reply["status"] == "recorded"
                if accepted:
                    result = CanonicalAuditPackageEncoder().encode(
                        await collector.seal(), inputs=inputs
                    )
                    with zipfile.ZipFile(io.BytesIO(result.data)) as archive:
                        encoded = json.loads(archive.read("evidence.json"))["evidence"][0]
                        assert encoded["observed_at"] == value["observed_at"]
                        assert encoded["http_request_tag"] == fixture["capture"]["request_tag"]
                        assert json.loads(archive.read("evidence/ev-1.json")) == fixture["capture"]
            except ValueError:
                accepted = False
            assert accepted == case["valid"], case["name"]

    asyncio.run(scenario())


def test_selected_http_evidence_maximum_batch_and_early_byte_rejection():
    from test_audit_results_toolset import fixture_task_set

    async def scenario():
        from contractor_runtime.toolsets.audit_results.collector import (
            AuditCollectionError,
            InvocationAuditCollector,
        )

        task, manifest = fixture_task_set(64)
        keys = tuple(item["item_key"] for item in json.loads(manifest)["items"])
        inputs = AuditTrustedInputs(
            AuditInvocationOwner("allocation-1", "invocation-1", digest(task), keys), task, manifest
        )
        capture = json.dumps(
            {
                "source": "http",
                "request_id": "1",
                "request_tag": "r-test-h1",
                "attempts": [
                    {
                        "request": {
                            "method": "GET",
                            "url": "https://target.example/",
                            "message": {"kind": "structured", "headers": [], "body_base64": ""},
                        },
                        "error": "timeout",
                    }
                ],
            }
        ).encode()
        evidence = AuditEvidence(
            "source-trace", "Selected observation.", capture, "2026-10-27T12:00:00Z"
        )
        values = tuple(
            NormalizedAuditItem(
                key, "satisfied", "Checked.", ("source-trace",), (), (evidence,) * 4
            )
            for key in keys
        )
        collector = InvocationAuditCollector(inputs, CanonicalAuditPackageEncoder())
        receipt = await collector.record_batch(values)
        assert receipt.complete
        snapshot = await collector.seal()
        encoded = CanonicalAuditPackageEncoder().encode(snapshot, inputs=inputs)
        with zipfile.ZipFile(io.BytesIO(encoded.data)) as archive:
            rows = json.loads(archive.read("evidence.json"))["evidence"]
            assert len(rows) == 256 and all(
                row["observed_at"] == evidence.observed_at for row in rows
            )
        # Immutable bytes are shared; reject the sum before parsing/encoding a large package.
        huge = replace(evidence, http_capture=b"x" * (64 * 1024))
        fresh = InvocationAuditCollector(inputs, CanonicalAuditPackageEncoder())
        with pytest.raises(AuditCollectionError, match="collection bytes"):
            await fresh.record_batch(tuple(replace(item, evidence=(huge,) * 4) for item in values))
        assert (await fresh.snapshot()).items == ()

    asyncio.run(scenario())
