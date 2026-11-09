"""Combined API credentials are injected only at the private HTTP boundary."""

import asyncio
import json
import time
from urllib.parse import quote_plus

import pytest
from fakes.engagement_socket import SocketTarget, resolver_for, tls_contexts
from test_engagement_identity import identity_spec

from contractor_runtime.adapters.bridge import HTTPProxyBridge
from contractor_runtime.adapters.http_transport import HTTPTransport

FIELDS = {
    "header:X-Api-Key": "header-canary",
    "query:api_key": "query-canary +/ключ",
    "cookie:client": "cookie-canary",
}


@pytest.mark.parametrize("https", [False, True])
def test_combined_keys_on_real_wire_keep_logical_url_and_clear(tmp_path, https):
    async def scenario():
        server, client, _ = tls_contexts(tmp_path) if https else (None, None, [])
        async with SocketTarget(ssl_context=server) as target:
            spec = identity_spec(target, [("api", "api-keys", FIELDS, None)])
            owner = HTTPTransport(spec, resolver=resolver_for(target).resolver, ssl_context=client)
            owner.configure(authority_deadline=lambda: time.monotonic() + 30)
            try:
                logical = target.origin + "/allowed?keep=%2Ffixed"
                outcome, response = await owner.request(
                    logical,
                    credential_id="credential-api",
                    headers=[(b"Cookie", b"selected_parameter=42")],
                )
                assert outcome.kind == "completed"
                assert response.url == logical
                wire = target.requests[0]
                assert wire.startswith(
                    (
                        "GET /allowed?keep=%2Ffixed&api_key=" + quote_plus(FIELDS["query:api_key"])
                    ).encode()
                )
                assert b"x-api-key: header-canary" in wire.lower()
                assert b"selected_parameter=42; client=cookie-canary" in wire
                # Scanner diagnostics use this same boundary's redactor, including
                # the URL-encoded representation a target may echo.
                bridge = HTTPProxyBridge(
                    owner, credential_id="credential-api", deadline=time.monotonic() + 5
                )
                for value in (*FIELDS.values(), quote_plus(FIELDS["query:api_key"])):
                    assert value.encode() not in bridge.redact(value.encode())
                anonymous, _ = await owner.request(logical)
                assert anonymous.kind == "completed"
                assert all(value.encode() not in target.requests[1] for value in FIELDS.values())
                assert b"api_key=" not in target.requests[1]
                await owner.clear_cookies()
                with pytest.raises(ValueError, match="credential"):
                    await owner.request(logical, credential_id="credential-api")
                assert len(target.requests) == 2
            finally:
                await owner.close()
            assert spec.runtime_settings.http_identities is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "path,headers",
    [
        ("/allowed?api_key=caller", []),
        ("/allowed?%61pi_key=caller", []),
        ("/allowed", [(b"x-API-key", b"caller")]),
        ("/allowed", [(b"Cookie", b"client=caller")]),
        ("/allowed", [(b"Cookie", b"a=1; a=2")]),
    ],
)
def test_conflicting_keys_fail_before_target_bytes(path, headers):
    async def scenario():
        async with SocketTarget() as target:
            spec = identity_spec(target, [("api", "api-keys", FIELDS, None)])
            owner = HTTPTransport(spec, resolver=resolver_for(target).resolver)
            owner.configure(authority_deadline=lambda: time.monotonic() + 30)
            try:
                outcome, response = await owner.request(
                    target.origin + path, credential_id="credential-api", headers=headers
                )
                assert outcome.kind == "denied" and response is None
                assert owner._request_count == 0
                assert not target.connections
                assert not any(value in json.dumps(outcome.reason) for value in FIELDS.values())
            finally:
                await owner.close()

    asyncio.run(scenario())
