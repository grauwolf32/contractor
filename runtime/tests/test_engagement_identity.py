"""Explicit Audit credential selection and isolated cookie state."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fakes.engagement_engine import transport as HTTPTransport
from fakes.engagement_socket import SocketTarget, socket_spec
from test_engagement_target import Network, policy_config

from contractor_runtime.contracts import AllocationSpec
from contractor_runtime.contracts.engagement import HTTPLimits


def wire_response(status=200, headers=()):
    return (
        f"HTTP/1.1 {status} Result\r\nContent-Length: 0\r\n".encode()
        + b"".join((f"{key}: {value}\r\n".encode() for key, value in headers))
        + b"\r\n"
    )


def test_cookie_identity_jars_are_distinct_even_when_initial_credential_is_shared():
    from types import SimpleNamespace

    from contractor_runtime.adapters.http_session import IdentitySession, PrivateResponse

    target = SimpleNamespace(origin="http://target.example", address="10.20.0.8")
    spec = identity_spec(
        target,
        [
            ("account-a", "cookies", {"session": "private-cookie"}, None),
            ("account-b", "cookies", {"session": "private-cookie"}, None),
        ],
    )
    identities = spec.engagement.http.identities
    identities[2].credential = identities[1].credential.model_copy(deep=True)
    spec.runtime_settings.http_identities["account-b"].credential = identities[
        1
    ].credential.model_copy(deep=True)
    a = IdentitySession(identities[1], spec.runtime_settings.http_identities["account-a"])
    b = IdentitySession(identities[2], spec.runtime_settings.http_identities["account-b"])
    try:
        a.receive(
            PrivateResponse(
                target.origin + "/", 200, [(b"Set-Cookie", b"session=changed-for-a; Path=/")]
            ),
            a.epoch,
        )
        assert (b"Cookie", b"session=changed-for-a") in a.request(target.origin + "/").headers
        assert (b"Cookie", b"session=private-cookie") in b.request(target.origin + "/").headers
        assert a.cookies[target.origin] is not b.cookies[target.origin]
    finally:
        a.close()
        b.close()


def test_supplied_cookie_limit_covers_all_origins_before_request():
    from types import SimpleNamespace

    from fakes.engagement_engine import transport as HTTPTransport
    from test_engagement_target import Network, policy_config

    target = SimpleNamespace(origin="http://target.example", address="10.20.0.8")
    spec = identity_spec(
        target,
        [
            ("account-a", "cookies", {"session": "private-cookie"}, None),
            ("account-b", "cookies", {f"c{i}": "v" for i in range(65)}, None),
        ],
    )
    identity = spec.engagement.http.identities[2]
    identity.origins.append("http://other.example")
    materials = list(spec.runtime_settings.http_identities.values())
    network, _ = (Network([], []), None)
    with pytest.raises(ValueError, match="cookies exceed bound"):
        HTTPTransport(spec, resolver=policy_config().resolver, backend=network)
    assert spec.runtime_settings.http_identities is None
    assert all(material.fields == {} for material in materials)
    assert not network.streams and (not network.streams)


@pytest.mark.parametrize(
    "origin", ["http://target.example", "http://10.20.0.8", "http://[2001:db8::1]"]
)
def test_deleted_session_cookie_blocks_next_request(origin):
    from types import SimpleNamespace

    from contractor_runtime.adapters.http_session import IdentitySession, PrivateResponse

    target = SimpleNamespace(origin=origin, address="10.20.0.8")
    spec = identity_spec(target, [("account-a", "cookies", {"session": "private-cookie"}, None)])
    session = IdentitySession(
        spec.engagement.http.identities[1],
        spec.runtime_settings.http_identities["account-a"],
    )
    try:
        assert (b"Cookie", b"session=private-cookie") in session.request(
            target.origin + "/"
        ).headers
        session.receive(
            PrivateResponse(
                target.origin + "/", 200, [(b"Set-Cookie", b"session=; Max-Age=0; Path=/")]
            ),
            session.epoch,
        )
        with pytest.raises(ValueError, match="no sendable cookies"):
            session.request(target.origin + "/next")
        assert session.state == "blocked"
    finally:
        session.close()


def identity_spec(target, identities, *, concurrency=2):
    spec = socket_spec(target)
    raw = spec.model_dump(mode="json", by_alias=True, exclude_none=True)
    assignment = raw["engagement"]["http"]
    materials = {}
    for identity, kind, fields, _ in identities:
        reference = {
            "credentialId": "credential-" + identity,
            "kind": {
                "basic": "target-basic@1",
                "bearer": "target-bearer@1",
                "cookies": "target-cookies@1",
            }[kind],
        }
        assignment["identities"].append(
            {"id": identity, "kind": kind, "origins": [target.origin], "credential": reference}
        )
        materials[identity] = {"credential": reference, "fields": fields}
    raw["runtimeSettings"]["httpIdentities"] = materials
    return AllocationSpec.model_validate_json(json.dumps(raw))


def test_private_material_binding_and_repr_are_strict():

    async def scenario():
        async with SocketTarget() as target:
            spec = identity_spec(
                target, [("account-a", "bearer", {"token": "canary-private-bearer"}, None)]
            )
            assert "canary-private-bearer" not in repr(spec)
            data = spec.model_dump(mode="json", by_alias=True, exclude_none=True)
            for change in (
                lambda d: d["runtimeSettings"]["httpIdentities"].clear(),
                lambda d: d["runtimeSettings"]["httpIdentities"]["account-a"]["credential"].update(
                    credentialId="foreign"
                ),
                lambda d: d["runtimeSettings"]["httpIdentities"].update(
                    extra=d["runtimeSettings"]["httpIdentities"]["account-a"]
                ),
            ):
                candidate = json.loads(json.dumps(data))
                change(candidate)
                with pytest.raises(ValueError):
                    AllocationSpec.model_validate_json(json.dumps(candidate))

    asyncio.run(scenario())


def owner_for(*, responses=None, **network_options):
    spec = identity_spec(
        SimpleNamespace(origin="http://target.example", address="10.20.0.8"),
        [
            ("account-a", "cookies", {"session": "private-cookie"}, None),
            ("account-b", "bearer", {"token": "private-token"}, None),
        ],
    )
    events = []
    _, network = (None, Network(events, responses, **network_options))
    spec.engagement.http.limits = HTTPLimits(requestBodyBytes=65536)
    owner = HTTPTransport(spec, resolver=policy_config().resolver, backend=network)
    owner.configure(authority_deadline=lambda: time.monotonic() + 60)
    return (owner, None, network)


def test_catalog_and_value_read_are_passive_and_audit_scoped():

    async def scenario():
        owner, _, network = owner_for()
        assert [c["credential_id"] for c in owner.credentials()] == [
            "credential-account-a",
            "credential-account-b",
        ]
        assert "private-cookie" not in str(owner.credentials())
        assert owner.read_credential("credential-account-a")["values"] == {
            "session": "private-cookie"
        }
        for credential in ("foreign", "", None):
            with pytest.raises(ValueError):
                owner.read_credential(credential)
        assert network.streams == network.streams == []
        await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "credential,expected",
    [
        (None, None),
        ("credential-account-a", b"Cookie: session=private-cookie"),
        ("credential-account-b", b"Authorization: Bearer private-token"),
    ],
)
def test_only_explicit_selected_credential_is_sent(credential, expected):

    async def scenario():
        owner, _, network = owner_for()
        try:
            outcome, response = await owner.request(
                "http://target.example/model-chosen?x=1", credential_id=credential
            )
            assert outcome.kind == "completed" and response.body == b"ok"
            wire = bytes(network.streams[0].written)
            if expected is None:
                assert b"Authorization:" not in wire and b"Cookie:" not in wire
            else:
                assert expected in wire
        finally:
            await owner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"credential_id": "foreign"},
        {"credential_id": "credential-account-a", "headers": [(b"Authorization", b"manual")]},
        {"credential_id": "credential-account-b", "headers": [(b"Cookie", b"manual=value")]},
        {"body": b"x" * 65537},
    ],
)
def test_invalid_credential_or_body_emits_no_target_action(options):

    async def scenario():
        owner, _, network = owner_for()
        with pytest.raises(ValueError):
            await owner.request("http://target.example/", **options)
        assert network.streams == network.streams == []
        await owner.close()

    asyncio.run(scenario())


def test_repeated_http_statuses_remain_responses_with_no_retry_or_auth_inference():

    async def scenario():
        owner, _, network = owner_for(
            responses=[wire_response(code) for code in [401, 403, 403, 503]]
        )
        for index, code in enumerate([401, 403, 403, 503]):
            outcome, response = await owner.request(
                "http://target.example/path" + str(index), credential_id="credential-account-a"
            )
            assert outcome.kind == "completed" and response.status == code
        assert len(network.streams) == len(network.streams) == 4
        assert all(b"Cookie: session=private-cookie" in s.written for s in network.streams)
        await owner.close()

    asyncio.run(scenario())


def test_expired_learned_cookies_are_removed_before_next_request():

    async def scenario():
        owner, _, network = owner_for(
            responses=[
                wire_response(headers=[("Set-Cookie", "learned=temporary; Max-Age=1; Path=/")]),
                wire_response(),
            ]
        )
        now = [1000.0]
        with patch("time.time", side_effect=lambda: now[0]):
            await owner.request("http://target.example/first", credential_id="credential-account-a")
            now[0] = 1002.0
            await owner.request(
                "http://target.example/second", credential_id="credential-account-a"
            )
        assert b"learned=" not in network.streams[1].written
        assert b"Cookie: session=private-cookie" in network.streams[1].written
        await owner.close()

    asyncio.run(scenario())


def test_deleted_cookie_blocks_selected_credential_without_affecting_anonymous():

    async def scenario():
        owner, _, network = owner_for(
            responses=[
                wire_response(headers=[("Set-Cookie", "session=; Max-Age=0; Path=/")]),
                wire_response(),
            ]
        )
        first, _ = await owner.request(
            "http://target.example/first", credential_id="credential-account-a"
        )
        assert first.kind == "completed"
        denied, _ = await owner.request(
            "http://target.example/second", credential_id="credential-account-a"
        )
        assert denied.kind == "denied"
        completed, _ = await owner.request("http://target.example/third")
        assert completed.kind == "completed" and len(network.streams) == 2
        assert b"Cookie:" not in network.streams[-1].written
        await owner.close()

    asyncio.run(scenario())


def test_stop_joins_simultaneous_explicit_credentials_and_erases_material():

    async def scenario():
        owner, _, network = owner_for(hang=True)
        tasks = [
            asyncio.create_task(owner.request("http://target.example/", credential_id=c))
            for c in (None, "credential-account-a")
        ]
        while len(network.streams) < 2:
            await asyncio.sleep(0)
        await asyncio.gather(*(s.entered_read.wait() for s in network.streams))
        await owner.close()
        assert all(t.done() and t.cancelled() for t in tasks)
        assert all(s.closed for s in network.streams) and (not owner._sessions)

    asyncio.run(scenario())
