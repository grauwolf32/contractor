"""Offline publication through the production allocation ArtifactClient."""

import asyncio
import io
import json
import stat
import zipfile
from dataclasses import replace

import jcs
import pytest
from test_artifacts import TIMESTAMP_HEADERS, json_response
from test_audit_results_toolset import digest, fixture_batch_inputs, fixture_inputs

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.audit_completion_contracts import (
    MAX_PACKAGE_BYTES,
    AuditEvidence,
    AuditInvocationOwner,
    AuditSnapshot,
    AuditTrustedInputs,
    NormalizedAuditItem,
    RecordedAuditItem,
    SealedAuditSnapshot,
)
from contractor_runtime.audit_packages import _build_result_package, _decode_task_input
from contractor_runtime.audit_result_encoding import AuditResultError, CanonicalAuditPackageEncoder
from contractor_runtime.audit_result_publication import DeterministicAuditResultPublisher
from contractor_runtime.contracts import ArtifactRef


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
        execution_bytes=inputs.execution_manifest,
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
    from contractor_runtime import audit_result_encoding as encoding

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

    def __init__(self, steps=(), *, existing=None, media_type="application/zip", on_io=None):
        self.steps = list(steps)
        self.existing = existing
        self.media_type = media_type
        self.requests = []
        self.on_io = on_io

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
            await asyncio.Event().wait()
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


@pytest.mark.parametrize("phase", ["before", "write", "read"])
def test_deadline_bounds_each_io_and_reconciliation(phase):
    async def scenario():
        snapshot, inputs = assigned()
        transport = ScriptedArtifactTransport(
            ["lost", "hang"] if phase == "read" else ["hang"],
        )
        deadline = asyncio.get_running_loop().time() + (-1 if phase == "before" else 0.01)
        with pytest.raises(AuditResultError, match="audit_result_publication_failed"):
            await publisher(snapshot, transport).publish(snapshot, inputs=inputs, deadline=deadline)
        assert len(transport.requests) == {"before": 0, "write": 1, "read": 2}[phase]

    asyncio.run(scenario())


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
