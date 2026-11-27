"""Actual HTTP result-tool selection over native sockets and ordinary Artifact CAS."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from audit_fixtures import assigned
from fakes.allocation_rpc import AllocationRPCFixture
from fakes.audit_judgments import acknowledged_judgments
from fakes.engagement_socket import SocketTarget, preflight
from fakes.worker_state import WorkerState
from test_artifacts import TIMESTAMP_HEADERS, json_response

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.audit_evidence import evidence_snapshot_ref
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.digests import sha256_digest
from contractor_runtime.toolsets.audit_results.collector import InvocationAuditCollector
from contractor_runtime.toolsets.audit_results.encoding import CanonicalAuditResultEncoder
from contractor_runtime.toolsets.audit_results.evidence import EvidenceTool
from contractor_runtime.toolsets.audit_results.v2 import SubmitResultTool
from contractor_runtime.toolsets.common.http_capture import HTTPObservation, RecentHTTPCaptures
from contractor_runtime.toolsets.http.controlled import ControlledHTTPSession
from contractor_runtime.worker.instrumentation import _safe_tool_response


class EvidenceTransport(AllocationRPCFixture):
    def __init__(self, *, lose_reply=False):
        self.values = {}
        self.requests = []
        self.lose_reply = lose_reply
        self.receipts = {}

    async def request(self, method, path, *, headers, body, max_response_bytes):
        self.requests.append((method, path))
        if method == "POST" and path.endswith("/audit-evidence"):
            value = json.loads(body)
            identity = value["submissionId"]
            if identity in self.receipts:
                return json_response(200, {**self.receipts[identity], "replayed": True})
            sources, selected = [], []
            for ref in value["sources"]:
                key = next(
                    key
                    for key in self.values
                    if key.endswith("/" + ref["namespace"] + "/" + ref["name"])
                )
                raw = self.values[key]
                material = {
                    "scope_kind": "audit",
                    "scope_id": "audit-1",
                    "ref": {**ref, "namespace": "audit-fixture"},
                    "digest": sha256_digest(raw),
                    "media_type": "application/vnd.contractor.http-exchange+json",
                    "size_bytes": len(raw),
                }
                sources.append(material)
                snapshot = evidence_snapshot_ref(ArtifactRef.model_validate(ref))
                selected.append(
                    {
                        **material,
                        "scope_kind": "run",
                        "scope_id": "run-1",
                        "ref": {
                            **snapshot.model_dump(exclude_none=True),
                            "revision": "snapshot-r1",
                        },
                    }
                )
            receipt = {
                "apiVersion": "contractor/v1alpha1",
                "itemKey": value["itemKey"],
                "sources": sources,
                "selectedSources": selected,
                "material": {
                    "scope_kind": "audit",
                    "scope_id": "audit-1",
                    "ref": {
                        "namespace": "audit-fixture",
                        "name": "note-" + sha256_digest(body).removeprefix("sha256:"),
                        "revision": "note-r1",
                    },
                    "digest": sha256_digest(body),
                    "media_type": "application/json",
                    "size_bytes": len(body),
                },
                "replayed": False,
            }
            assert (
                sum(len(row["sources"]) + 1 for row in self.receipts.values()) + len(sources) + 1
                <= 256
            )
            self.receipts[identity] = receipt
            return json_response(201, receipt)
        if method == "PUT":
            if path in self.values:
                return json_response(
                    409,
                    {
                        "code": "artifact_conflict",
                        "message": "conflict",
                        "retryable": True,
                        "requestId": "test-http",
                    },
                )
            self.values[path] = body
            if self.lose_reply:
                self.lose_reply = False
                raise ArtifactTransportError("committed response lost")
            namespace, name = path.split("/")[-2:]
            return json_response(
                201,
                {
                    "apiVersion": "contractor/v1alpha1",
                    "artifact": {"namespace": namespace, "name": name, "revision": "r1"},
                    "mediaType": headers["Content-Type"],
                    "size": len(body),
                },
                etag='"r1"',
            )
        return ArtifactHTTPResponse(
            200,
            {
                "content-type": "application/vnd.contractor.http-exchange+json",
                "etag": '"r1"',
                **TIMESTAMP_HEADERS,
            },
            self.values[path],
        )


def result_tool(captures, *, lose_reply=False, batch=False):
    _, inputs = assigned(batch=batch)
    transport = EvidenceTransport(lose_reply=lose_reply)
    state = WorkerState()
    state.resources.http_captures = captures
    client = ArtifactClient("allocation-1", transport)
    encoder = CanonicalAuditResultEncoder()
    collector = InvocationAuditCollector(
        inputs, encoder, persist=acknowledged_judgments(inputs, encoder)
    )
    tool = SubmitResultTool(collector, state.metrics)
    tool.evidence = EvidenceTool(lambda: collector, client, state.resources, "run-1", "worker", 5)
    return collector, tool, transport, encoder, inputs


async def select_and_judge(
    tool, context, *, item_key, rationale, conditions, limitations, verdict, request_ids
):
    """Run two explicit model operations through their actual production facades."""
    evidence_context = SimpleNamespace(
        invocation_id=context.invocation_id,
        function_call_id="select-" + item_key + "-" + "-".join(map(str, request_ids)),
    )
    try:
        await tool.evidence(
            item_key=item_key,
            sources=[{"kind": "http", "request_id": key} for key in request_ids],
            interpretation=rationale,
            tested_conditions=conditions,
            limitations=limitations,
            tool_context=evidence_context,
        )
    except ToolInputError as error:
        return _safe_tool_response("evidence", error)
    return await tool(
        context, item_key=item_key, rationale=rationale, limitations=limitations, verdict=verdict
    )


async def request(session, url):
    return await session.request(
        url=url,
        method="GET",
        headers=None,
        query=None,
        body_type="none",
        body=None,
        timeout_seconds=5,
        follow_redirects=True,
        invocation_id="invocation-1",
    )


def test_selected_redirect_survives_reply_loss_eviction_and_cleanup_without_finding():
    async def scenario():
        async def respond(target, reader, writer, message):
            if message.startswith(b"GET /allowed HTTP/"):
                writer.write(
                    b"HTTP/1.1 302 Found\r\nLocation: /allowed/final\r\nContent-Length: 0\r\n\r\n"
                )
            else:
                writer.write(
                    b"HTTP/1.1 403 Forbidden\r\nContent-Type: application/octet-stream\r\n"
                    b"Content-Length: 3\r\n\r\n\x00\xffx"
                )
            await writer.drain()

        captures = RecentHTTPCaptures()
        collector, tool, transport, encoder, inputs = result_tool(captures, lose_reply=True)
        async with SocketTarget(respond) as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
            session = ControlledHTTPSession(
                owner, 5, captures=captures, deployment_marker="fixture-deployment"
            )
            await request(session, target.origin + "/allowed/noise")
            selected = await request(session, target.origin + "/allowed")
            args = dict(
                rationale="Access was denied in the tested conditions.",
                conditions=["Selected anonymous request"],
                limitations=[],
                verdict="refuted",
                request_ids=[selected["request_id"]],
            )
            context = SimpleNamespace(invocation_id="invocation-1")
            receipt = await select_and_judge(tool, context, item_key="check-authz", **args)
            assert receipt["status"] == "recorded", receipt
            assert len(transport.values) == 1
            payload = next(iter(transport.values.values()))
            document = json.loads(payload)
            assert set(document) == {"schema", "origin", "capture"}
            assert document["origin"] == {"run_id": "run-1", "invocation_id": "invocation-1"}
            capture = document["capture"]
            assert capture["request_id"] == "2"
            assert capture["deployment_marker"] == "fixture-deployment"
            assert capture["observed_at"].endswith("Z")
            assert len(capture["attempts"]) == 2
            assert [attempt["response"]["status"] for attempt in capture["attempts"]] == [
                302,
                403,
            ]
            assert not capture["attempts"][0]["response"]["complete"]
            assert capture["attempts"][1]["response"]["complete"]
            assert capture["attempts"][1]["response"]["message"]["body_base64"] == "AP94"
            assert capture["request_tag"].encode() in target.requests[1]
            captures.clear("http")
            calls = len(transport.requests)
            assert (
                await tool(
                    context,
                    item_key="check-authz",
                    **{key: args[key] for key in ("rationale", "limitations", "verdict")},
                )
                == receipt
            )
            assert len(transport.requests) == calls
            sealed = await collector.seal()
            result = json.loads(encoder.encode(sealed, inputs=inputs).data)
            assert all(
                material["scope_kind"] == "audit" for material in result["items"][0]["evidence"]
            )
            assert len(result["items"][0]["evidence"]) == 2
            assert len(target.requests) == 3
            await session.close()
            assert not captures._entries
            assert next(iter(transport.values.values())) == payload
        await collector.discard()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["foreign", "evicted", "missing"])
def test_unavailable_selection_has_no_result_mutation_or_artifact(kind):
    async def scenario():
        from test_http_capture import capture

        captures = RecentHTTPCaptures()
        collector, tool, transport, _, _ = result_tool(captures)
        if kind != "missing":
            captures.new_http_request()
            await captures.put(
                capture("1"),
                "other" if kind == "foreign" else "invocation-1",
                observation=HTTPObservation("2026-11-06T18:20:00Z"),
            )
        if kind == "evicted":
            captures.clear("http")
        response = await select_and_judge(
            tool,
            SimpleNamespace(invocation_id="invocation-1"),
            item_key="check-authz",
            rationale="Unavailable observation",
            conditions=["probe"],
            limitations=[],
            verdict="refuted",
            request_ids=[1],
        )
        assert "error" in response
        assert not (await collector.snapshot()).items
        assert not transport.requests

    asyncio.run(scenario())


def test_full_recent_buffer_selection_and_two_member_evidence_bound():
    async def scenario():
        from test_http_capture import capture

        captures = RecentHTTPCaptures()
        collector, tool, transport, encoder, inputs = result_tool(captures, batch=True)
        context = SimpleNamespace(invocation_id="invocation-1")
        for batch, key in enumerate(collector.owner.item_keys):
            ids = list(range(1, 129)) if batch == 0 else list(range(129, 255))
            for request_id in ids:
                assert captures.new_http_request() == request_id
                await captures.put(
                    capture(str(request_id)),
                    "invocation-1",
                    observation=HTTPObservation("2026-11-06T18:20:00Z"),
                )
            result = await select_and_judge(
                tool,
                context,
                item_key=key,
                rationale="Bounded failed probes",
                conditions=["Explicit target attempts"],
                limitations=["Transport failure leaves the hypothesis unresolved"],
                verdict="inconclusive",
                request_ids=ids,
            )
            assert result["status"] == "recorded", result
        assert len(transport.values) == 254
        output = json.loads(encoder.encode(await collector.seal(), inputs=inputs).data)
        assert [len(item["evidence"]) for item in output["items"]] == [129, 127]
        assert len(captures._entries) == 128
        await collector.discard()

    asyncio.run(scenario())


def test_mixed_foreign_capture_selection_rejects_before_any_publication():
    async def scenario():
        from test_http_capture import capture

        captures = RecentHTTPCaptures()
        for request_id, invocation in [(1, "invocation-1"), (2, "foreign")]:
            assert captures.new_http_request() == request_id
            await captures.put(
                capture(str(request_id)),
                invocation,
                observation=HTTPObservation("2026-11-06T18:20:00Z"),
            )
        collector, tool, transport, _, _ = result_tool(captures, batch=True)
        with pytest.raises(ToolInputError):
            await tool.evidence(
                item_key="check-authz",
                sources=[{"kind": "http", "request_id": key} for key in (1, 2)],
                interpretation="Select actual probes",
                tested_conditions=["actual probes"],
                limitations=[],
                tool_context=SimpleNamespace(
                    invocation_id="invocation-1", function_call_id="mixed"
                ),
            )
        assert not (await collector.snapshot()).items
        assert not transport.requests

    asyncio.run(scenario())


def test_selected_bytes_are_bounded_before_publication(monkeypatch):
    async def scenario():
        from test_http_capture import capture

        from contractor_runtime.toolsets.common import sources as module

        monkeypatch.setattr(module, "MAX_PREPARED_SOURCE_BYTES", 64)
        captures = RecentHTTPCaptures()
        captures.new_http_request()
        await captures.put(
            capture("1"), "invocation-1", observation=HTTPObservation("2026-11-06T18:20:00Z")
        )
        collector, tool, transport, _, _ = result_tool(captures)
        result = await select_and_judge(
            tool,
            SimpleNamespace(invocation_id="invocation-1"),
            item_key="check-authz",
            rationale="Oversized selection",
            conditions=[],
            limitations=["Evidence budget"],
            verdict="inconclusive",
            request_ids=[1],
        )
        assert "error" in result and "budget" in result["error"]["message"]
        assert not transport.requests
        assert not (await collector.snapshot()).items

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("bearer", {"token": "private-assigned-secret"}),
        ("basic", {"username": "fixture", "password": "private-assigned-secret"}),
        ("cookies", {"session": "private-assigned-secret"}),
    ],
)
def test_controlled_capture_redacts_only_runtime_injected_credentials(kind, fields):
    async def scenario():
        from test_engagement_identity import identity_spec

        captures = RecentHTTPCaptures()
        async with SocketTarget() as target:
            spec = identity_spec(target, [("account-a", kind, fields, None)])
            async with preflight(target, spec=spec) as (owner, _):
                owner.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
                session = ControlledHTTPSession(owner, 5, captures=captures)
                result = await session.request(
                    url=target.origin + "/allowed",
                    method="GET",
                    headers=None,
                    query=None,
                    body_type="none",
                    body=None,
                    timeout_seconds=5,
                    follow_redirects=True,
                    invocation_id="invocation-1",
                    credential_id="credential-account-a",
                )
                captured = await captures.get(
                    "http", str(result["request_id"]), invocation_id="invocation-1"
                )
                headers = captured["attempts"][0]["request"]["message"]["headers"]
                private = [
                    row for row in headers if row["name"].lower() in {"authorization", "cookie"}
                ]
                assert private and all(row["value"] == "[runtime-credential]" for row in private)
                assert "private-assigned-secret" not in json.dumps(captured)
                assert len(target.requests) == 1
                await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "header_name, secret", [("X-API-Key", "k"), ("x-PaRtNeR-ToKeN", "private-header-canary")]
)
@pytest.mark.parametrize("redirect", [False, True])
def test_selected_api_key_evidence_masks_injected_headers(header_name, secret, redirect):
    async def scenario():
        from test_engagement_identity import identity_spec

        async def respond(target, reader, writer, message):
            if redirect and message.startswith(b"GET /allowed?"):
                writer.write(
                    b"HTTP/1.1 302 Found\r\nLocation: /allowed/final\r\nContent-Length: 0\r\n\r\n"
                )
            else:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

        captures = RecentHTTPCaptures()
        collector, tool, published, _, _ = result_tool(captures)
        async with SocketTarget(respond) as target:
            fields = {
                "header:" + header_name: secret,
                "header:X-Second-Key": "private-second-canary",
                "query:api_key": "private-query-canary",
                "cookie:session": "private-cookie-canary",
            }
            spec = identity_spec(target, [("account-a", "api-keys", fields, None)])
            async with preflight(target, spec=spec) as (owner, _):
                owner.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
                session = ControlledHTTPSession(owner, 5, captures=captures)
                options = dict(
                    url=target.origin + "/allowed",
                    method="GET",
                    headers={"X-Public": "caller-value"},
                    query=None,
                    body_type="none",
                    body=None,
                    timeout_seconds=5,
                    follow_redirects=True,
                    invocation_id="invocation-1",
                )
                result = await session.request(**options, credential_id="credential-account-a")
                selected = await select_and_judge(
                    tool,
                    SimpleNamespace(invocation_id="invocation-1"),
                    item_key="check-authz",
                    rationale="Record the selected exchange.",
                    conditions=["Selected credential"],
                    limitations=["A single exchange does not establish a verdict."],
                    verdict="inconclusive",
                    request_ids=[result["request_id"]],
                )
                assert selected["status"] == "recorded", selected
                assert len(published.values) == 1
                payload = next(iter(published.values.values()))
                document = json.loads(payload)
                attempts = document["capture"]["attempts"]
                assert len(attempts) == (2 if redirect else 1)
                for attempt, wire in zip(attempts, target.requests, strict=True):
                    recorded = {
                        row["name"].lower(): row["value"]
                        for row in attempt["request"]["message"]["headers"]
                    }
                    for private in (header_name.lower(), "x-second-key", "cookie"):
                        assert recorded[private] == "[runtime-credential]"
                    assert recorded["x-public"] == "caller-value"
                    assert (header_name + ": " + secret).lower().encode() in wire.lower()
                    assert b"api_key=private-query-canary" in wire
                    assert b"session=private-cookie-canary" in wire
                for value in list(fields.values())[1:]:
                    assert value.encode() not in payload
                if len(secret) > 1:
                    assert secret.encode() not in payload
                # The same arbitrary name supplied by the caller is ordinary
                # evidence when no Runtime credential was selected.
                plain = await session.request(
                    **{**options, "headers": {header_name: "caller-owned"}}
                )
                captured = await captures.get(
                    "http", str(plain["request_id"]), invocation_id="invocation-1"
                )
                headers = captured["attempts"][0]["request"]["message"]["headers"]
                assert {row["name"].lower(): row["value"] for row in headers}[
                    header_name.lower()
                ] == "caller-owned"
                await session.close()
        await collector.discard()

    asyncio.run(scenario())


def test_failed_and_truncated_actual_exchanges_remain_selectable_as_inconclusive():
    async def scenario():
        from contractor_runtime.contracts.engagement import HTTPLimits
        from contractor_runtime.toolsets.http.controlled import ControlledHTTPError

        for failed in (False, True):

            async def respond(target, reader, writer, message, failed=failed):
                if failed:
                    writer.close()
                    await writer.wait_closed()
                else:
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n" + b"x" * 20)
                    await writer.drain()

            captures = RecentHTTPCaptures()
            _collector, tool, transport, _, _ = result_tool(captures)
            async with (
                SocketTarget(respond) as target,
                preflight(target, limits=HTTPLimits(responseBodyBytes=4)) as (owner, _),
            ):
                owner.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
                session = ControlledHTTPSession(owner, 5, captures=captures)
                if failed:
                    with pytest.raises(ControlledHTTPError):
                        await request(session, target.origin + "/allowed")
                    assert session._history[-1].request_id == 1 and session._history[-1].error
                else:
                    result = await request(session, target.origin + "/allowed")
                    assert result["body_truncated"]
                result = await select_and_judge(
                    tool,
                    SimpleNamespace(invocation_id="invocation-1"),
                    item_key="check-authz",
                    rationale="A bounded probe did not establish the claim.",
                    conditions=["Explicit request"],
                    limitations=["Unavailable or partial response"],
                    verdict="inconclusive",
                    request_ids=[1],
                )
                assert result["status"] == "recorded", result
                capture = json.loads(next(iter(transport.values.values())))["capture"]
                assert capture["observed_at"]
                attempt = capture["attempts"][-1]
                assert attempt["error"]
                assert "response" not in attempt or not attempt["response"]["complete"]
                assert len(target.requests) == 1
                await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("commit_before_cancel", [False, True])
def test_cancellation_joins_publication_and_never_claims_result_acceptance(commit_before_cancel):
    async def scenario():
        from test_http_capture import capture

        captures = RecentHTTPCaptures()
        captures.new_http_request()
        await captures.put(
            capture("1"), "invocation-1", observation=HTTPObservation("2026-11-06T18:20:00Z")
        )
        collector, tool, transport, _, _ = result_tool(captures)
        entered = asyncio.Event()
        finished = asyncio.Event()
        original = transport.request

        async def blocked(method, path, **options):
            try:
                if commit_before_cancel:
                    await original(method, path, **options)
                entered.set()
                await asyncio.Event().wait()
            finally:
                finished.set()

        transport.request = blocked
        task = asyncio.create_task(
            select_and_judge(
                tool,
                SimpleNamespace(invocation_id="invocation-1"),
                item_key="check-authz",
                rationale="Interrupted persistence",
                conditions=[],
                limitations=["Publication interrupted"],
                verdict="inconclusive",
                request_ids=[1],
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not (await collector.snapshot()).items
        assert bool(transport.values) == commit_before_cancel
        await collector.discard()

    asyncio.run(scenario())
