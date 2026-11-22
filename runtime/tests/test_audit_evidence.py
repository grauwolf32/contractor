import json
from pathlib import Path

import pytest

from contractor_runtime.audit_evidence import evidence_snapshot_ref
from contractor_runtime.contracts import ArtifactRef


def test_shared_audit_evidence_snapshot_refs():
    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/audit-evidence-snapshot-refs.json").read_text()
    )
    assert cases
    for case in cases:
        assert (
            evidence_snapshot_ref(ArtifactRef.model_validate(case["source"])).model_dump(
                exclude_none=True
            )
            == case["snapshot"]
        )


def evidence_receipt():
    source = ArtifactRef(namespace="worker", name="source", revision="source-r1")
    snapshot = evidence_snapshot_ref(source).model_copy(update={"revision": "snapshot-r1"})

    def material(kind, scope, ref):
        return {
            "scope_kind": kind,
            "scope_id": scope,
            "ref": ref.model_dump(exclude_none=True),
            "digest": "sha256:" + "a" * 64,
            "media_type": "text/plain",
            "size_bytes": 3,
        }

    receipt = {
        "apiVersion": "contractor/v1alpha1",
        "material": material(
            "audit", "audit-1", ArtifactRef(namespace="audit-a", name="note", revision="note-r1")
        ),
        "sources": [
            material(
                "audit",
                "audit-1",
                ArtifactRef(namespace="audit-a", name="source", revision="retained-r1"),
            )
        ],
        "selectedSources": [material("run", "run-1", snapshot)],
        "itemKey": "item-00",
        "replayed": True,
    }
    return source, receipt


def test_evidence_lost_ack_reconciles_identical_bytes_without_traffic():
    import asyncio
    from types import SimpleNamespace

    from contractor_runtime.artifacts import ArtifactHTTPResponse, ArtifactTransportError
    from contractor_runtime.audit_evidence import AuditEvidenceClient

    source, receipt = evidence_receipt()

    class Transport:
        def __init__(self):
            self.bodies = []

        async def request(self, method, path, *, headers, body, max_response_bytes):
            assert method == "POST" and path.endswith("/audit-evidence")
            self.bodies.append(body)
            if len(self.bodies) == 1:
                raise ArtifactTransportError("lost acknowledgement")
            return ArtifactHTTPResponse(
                200, {"content-type": "application/json"}, json.dumps(receipt).encode()
            )

    transport = Transport()
    client = AuditEvidenceClient(
        SimpleNamespace(transport=transport, allocation_id="allocation-1"),
        run_id="run-1",
        audit_id="audit-1",
    )
    result = asyncio.run(client.submit({"itemKey": "item-00", "sources": [source.model_dump()]}))
    assert result["replayed"] is True
    assert len(transport.bodies) == 2 and transport.bodies[0] == transport.bodies[1]


def test_evidence_unconfirmed_tail_is_safe_unknown_persistence():
    import asyncio
    from types import SimpleNamespace

    import pytest

    from contractor_runtime.artifacts import ArtifactTransportError
    from contractor_runtime.audit_evidence import AuditEvidenceClient, EvidencePersistenceUnknown
    from contractor_runtime.worker.instrumentation import _safe_tool_response

    class Transport:
        calls = 0

        async def request(self, *args, **kwargs):
            self.calls += 1
            raise ArtifactTransportError("provider-secret-canary")

    transport = Transport()
    client = AuditEvidenceClient(
        SimpleNamespace(transport=transport, allocation_id="allocation-1"),
        run_id="run-1",
        audit_id="audit-1",
    )
    with pytest.raises(EvidencePersistenceUnknown) as failure:
        asyncio.run(client.submit({"itemKey": "item-00", "sources": []}))
    response = _safe_tool_response("evidence", failure.value)
    assert transport.calls == 2
    assert response["error"]["persistence"] == "unknown"
    assert "provider-secret-canary" not in json.dumps(response)


def test_source_error_paths_never_echo_unknown_member_names():
    import pytest

    from contractor_runtime.toolsets.common.sources import SourceArgumentError, parse_sources
    from contractor_runtime.worker.instrumentation import _safe_tool_response

    with pytest.raises(SourceArgumentError) as failure:
        parse_sources(
            [
                {
                    "kind": "artifact",
                    "namespace": "inputs",
                    "name": "source",
                    "secret-canary": "value-canary",
                }
            ]
        )
    response = _safe_tool_response("evidence", failure.value)
    assert "secret-canary" not in json.dumps(response)
    assert "value-canary" not in json.dumps(response)
    assert response["error"]["field"].startswith("sources[0]")


@pytest.mark.parametrize("persistent", [False, True])
def test_invalid_evidence_receipt_reconciles_identical_bytes(persistent):
    import asyncio
    from types import SimpleNamespace

    from contractor_runtime.artifacts import ArtifactHTTPResponse
    from contractor_runtime.audit_evidence import AuditEvidenceClient, EvidencePersistenceUnknown

    source, receipt = evidence_receipt()

    class Transport:
        def __init__(self):
            self.bodies = []

        async def request(self, method, path, *, headers, body, max_response_bytes):
            self.bodies.append(body)
            return ArtifactHTTPResponse(
                201,
                {"content-type": "application/json"},
                b"{}" if persistent or len(self.bodies) == 1 else json.dumps(receipt).encode(),
            )

    transport = Transport()
    client = AuditEvidenceClient(
        SimpleNamespace(transport=transport, allocation_id="allocation-1"),
        run_id="run-1",
        audit_id="audit-1",
    )
    if persistent:
        with pytest.raises(EvidencePersistenceUnknown):
            asyncio.run(client.submit({"itemKey": "item-00", "sources": [source.model_dump()]}))
    else:
        result = asyncio.run(
            client.submit({"itemKey": "item-00", "sources": [source.model_dump()]})
        )
        assert result["replayed"] is True
    assert len(transport.bodies) == 2 and transport.bodies[0] == transport.bodies[1]
