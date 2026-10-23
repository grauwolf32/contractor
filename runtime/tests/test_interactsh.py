from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from contractor_runtime.adapters.host import AdapterHandles, RuntimeAdapterMetricsState
from contractor_runtime.adapters.interactsh import (
    MAX_EVENT_BYTES,
    MAX_EVENTS,
    MAX_REGISTRATIONS,
    MAX_RESPONSE_BYTES,
    REGISTRATION_TTL_SECONDS,
    InteractshClient,
    InteractshError,
)
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import ArtifactRef, InteractshSettings, RuntimeSettings
from contractor_runtime.toolsets.interactsh.tools import (
    EVIDENCE_MEDIA_TYPE,
    MAX_EVIDENCE_BYTES,
    InteractshToolsetFactory,
    _Session,
)
from contractor_runtime.workspace import AllocationWorkspace

TOKEN = "recognizable-interactsh-private-token"


class Collector:
    def __init__(self):
        self.sessions = {}
        self.events = []
        self.calls = []
        self.lose_registration = False
        self.poll_response = None
        self.block_poll = None
        self.poll_started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.live_item = None

    async def request(self, request):
        self.calls.append(request.url.path)
        assert request.headers["authorization"] == TOKEN
        if request.url.path == "/register":
            payload = json.loads(request.content)
            self.sessions[payload["correlation-id"]] = payload
            if self.lose_registration:
                self.lose_registration = False
                raise httpx.ReadError("private provider detail", request=request)
            return httpx.Response(200, json={"message": "registration successful"})
        if request.url.path == "/deregister":
            payload = json.loads(request.content)
            self.sessions.pop(payload["correlation-id"], None)
            return httpx.Response(200, json={"message": "deregistration successful"})
        self.poll_started.set()
        if self.block_poll is not None:
            try:
                await self.block_poll.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.poll_response is not None:
            return self.poll_response
        identity = request.url.params["id"]
        if identity not in self.sessions:
            return httpx.Response(400, text="could not get correlation-id from cache")
        registration = self.sessions[identity]
        assert request.url.params["secret"] == registration["secret-key"]
        public = base64.b64decode(registration["public-key"]).replace(
            b"RSA PUBLIC KEY", b"PUBLIC KEY"
        )
        key = serialization.load_pem_public_key(public)
        aes = b"k" * 32
        encrypted_key = key.encrypt(
            aes,
            padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
        encrypted = []
        for raw in self.events:
            iv = b"i" * 16
            encryptor = Cipher(algorithms.AES(aes), modes.CTR(iv)).encryptor()
            encrypted.append(
                base64.b64encode(iv + encryptor.update(raw) + encryptor.finalize()).decode()
            )
        self.events = []
        return httpx.Response(
            200,
            json={
                "aes_key": base64.b64encode(encrypted_key).decode(),
                "data": encrypted,
                "extra": None,
            },
        )


class Artifacts:
    def __init__(self):
        self.rows = {}
        self.calls = 0
        self.lose_write = False
        self.unavailable = False

    async def write_artifact(self, target, *, data, media_type, expected_revision):
        self.calls += 1
        assert expected_revision is None
        assert len(data) <= MAX_EVIDENCE_BYTES
        assert media_type == EVIDENCE_MEDIA_TYPE
        if self.unavailable:
            raise RuntimeError("private Artifact provider detail")
        if target.name in self.rows:
            raise RuntimeError("already exists")
        exact = ArtifactRef(namespace=target.namespace, name=target.name, revision="r1")
        value = SimpleNamespace(artifact=exact, data=data, media_type=media_type)
        self.rows[target.name] = value
        if self.lose_write:
            self.lose_write = False
            raise RuntimeError("response lost")
        return value

    async def read_artifact(self, ref, *, max_bytes):
        if self.unavailable:
            raise RuntimeError("private Artifact provider detail")
        value = self.rows[ref.name]
        assert len(value.data) <= max_bytes
        return value


def settings():
    return InteractshSettings(
        adapter="interactsh-http@1",
        endpoint="https://oast.example",
        token=TOKEN,
        requestTimeoutSeconds=30,
    )


def client(collector, clock=None):
    return InteractshClient(
        settings(),
        timeout_seconds=30,
        metrics=RuntimeAdapterMetricsState(),
        transport=httpx.MockTransport(collector.request),
        monotonic=clock or (lambda: 0),
    )


def event(item, protocol="http", *, size=0, unique=None):
    return json.dumps(
        {
            "protocol": protocol,
            "unique-id": unique or item.correlation + item.nonce,
            "full-id": item.correlation + item.nonce,
            "timestamp": "2026-10-23T10:00:00Z",
            "remote-address": "192.0.2.10",
            "raw-request": "x" * size,
        },
        separators=(",", ":"),
    ).encode()


def test_exact_evidence_ownership_empty_and_lost_artifact_response():
    async def scenario():
        collector, artifacts = Collector(), Artifacts()
        handle = client(collector)
        session = _Session(handle, artifacts, "worker", (TOKEN,))
        registration = await session.register("http-request-42: authorized SSRF check")
        item = handle.registration(registration["handle"])
        assert registration["state"] == "registered"
        assert len(registration["payload"].split(".")[0]) == 33
        raw = event(item)
        collector.events = [raw, event(item, unique="someone-else")]
        artifacts.lose_write = True
        polled = await session.poll(item.handle)
        assert polled["truncated"] and polled["discarded"] == 1
        assert polled["attribution"] == "unverified"
        exact = polled["events"][0]["evidence_ref"]
        assert exact["revision"] == "r1"
        evidence = json.loads(artifacts.rows[exact["name"]].data)
        assert base64.b64decode(evidence["events_base64"][0]) == raw
        assert evidence["trigger"]["action"].startswith("http-request-42")
        assert TOKEN not in repr(polled) + repr(evidence)
        assert item.secret not in repr(polled) + repr(evidence)
        assert artifacts.calls == 1
        with pytest.raises(InteractshError, match="interactsh_handle_invalid"):
            await session.poll("another-allocation")
        with pytest.raises(InteractshError, match="interactsh_rate_limited"):
            await session.poll(item.handle)
        await session.close()
        await handle.close()
        assert not collector.sessions
        assert item.key is None and not item.secret

    asyncio.run(scenario())


def test_lost_registration_response_reconciles_same_identity_and_empty_is_explicit():
    async def scenario():
        collector = Collector()
        collector.lose_registration = True
        handle = client(collector)
        item = await handle.register()
        assert item.state == "registered"
        assert collector.calls == ["/register", "/poll"]
        assert len(collector.sessions) == 1
        session = _Session(handle, Artifacts(), "worker", ())
        session.actions[item.handle] = "intended check"
        result = await session.poll(item.handle)
        assert result["state"] == "no_interaction" and not result["coverage_gap"]
        await handle.close()

    asyncio.run(scenario())


def test_unknown_registration_is_reconciled_later_without_new_register():
    async def scenario():
        now = [0]
        collector = Collector()
        collector.lose_registration = True
        collector.poll_response = httpx.Response(503, text="private collector failure")
        handle = client(collector, lambda: now[0])
        item = await handle.register()
        assert handle.public_registration(item) == {
            "handle": item.handle,
            "state": "outcome_unknown",
        }
        collector.poll_response = None
        await handle.poll(item.handle)
        assert item.state == "registered" and collector.calls.count("/register") == 1
        now[0] = REGISTRATION_TTL_SECONDS
        with pytest.raises(InteractshError, match="interactsh_registration_expired"):
            await handle.poll(item.handle)
        await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "code", "retryable"),
    [
        (httpx.Response(401, text=TOKEN), "interactsh_auth_failed", False),
        (httpx.Response(503, text=TOKEN), "interactsh_request_failed", True),
        (httpx.Response(200, content=b"not-json"), "interactsh_response_invalid", False),
        (
            httpx.Response(200, content=b" " * (MAX_RESPONSE_BYTES + 1)),
            "interactsh_response_too_large",
            False,
        ),
        (
            httpx.Response(200, json={"aes_key": "invalid", "data": []}),
            "interactsh_response_invalid",
            False,
        ),
        (
            httpx.Response(400, text="could not get correlation-id from cache"),
            "interactsh_registration_expired",
            False,
        ),
    ],
)
def test_poll_failure_preserves_uncertainty_and_never_leaks_provider_text(
    response, code, retryable
):
    async def scenario():
        now = [0]
        collector = Collector()
        handle = client(collector, lambda: now[0])
        item = await handle.register()
        collector.poll_response = response
        with pytest.raises(InteractshError) as caught:
            await handle.poll(item.handle)
        assert (caught.value.code, caught.value.retryable) == (code, retryable)
        assert TOKEN not in repr(caught.value)
        assert item.coverage_gap
        assert collector.calls.count("/poll") == 1
        collector.poll_response = None
        now[0] += 2
        await handle.poll(item.handle)
        assert item.coverage_gap
        await handle.close()

    asyncio.run(scenario())


def test_maximum_event_prefix_and_event_size_are_bounded():
    async def scenario():
        now = [0]
        collector = Collector()
        handle = client(collector, lambda: now[0])
        item = await handle.register()
        raw = event(item)
        collector.events = [raw] * (MAX_EVENTS + 20)
        batch = await handle.poll(item.handle)
        assert len(batch.events) == MAX_EVENTS and batch.discarded == 20
        now[0] += 2
        prefix = len(event(item))
        collector.events = [
            event(item, size=MAX_EVENT_BYTES - prefix),
            event(item, size=MAX_EVENT_BYTES),
        ]
        batch = await handle.poll(item.handle)
        assert len(batch.events[0]) == MAX_EVENT_BYTES and batch.discarded == 1
        await handle.close()

    asyncio.run(scenario())


def test_registration_lifetime_cap_and_close_join_poll_before_deregister():
    async def scenario():
        collector = Collector()
        handle = client(collector)
        for _ in range(MAX_REGISTRATIONS):
            item = await handle.register()
        with pytest.raises(InteractshError, match="interactsh_limit_exceeded"):
            await handle.register()
        collector.block_poll = asyncio.Event()
        poll = asyncio.create_task(handle.poll(item.handle))
        await collector.poll_started.wait()
        await handle.close()
        assert poll.cancelled() and collector.cancelled.is_set()
        assert collector.calls[-MAX_REGISTRATIONS:] == ["/deregister"] * MAX_REGISTRATIONS
        assert not collector.sessions and item.key is None

    asyncio.run(scenario())


def test_pending_evidence_is_retried_before_another_destructive_poll():
    async def scenario():
        collector, artifacts = Collector(), Artifacts()
        handle = client(collector)
        session = _Session(handle, artifacts, "worker", ())
        registration = await session.register("check")
        item = handle.registration(registration["handle"])
        collector.events = [event(item)]
        artifacts.unavailable = True
        with pytest.raises(InteractshError, match="interactsh_evidence_unconfirmed"):
            await session.poll(item.handle)
        artifacts.unavailable = False
        result = await session.poll(item.handle)
        assert result["events"] and collector.calls.count("/poll") == 1
        assert len(artifacts.rows) == 1 and not session.pending
        await session.close()
        await handle.close()

    asyncio.run(scenario())


def test_factory_requires_private_channel_and_exposes_only_selected_tools(tmp_path):
    async def scenario():
        collector, artifacts = Collector(), Artifacts()
        handle = client(collector)
        factory = InteractshToolsetFactory(lambda *_: artifacts)
        runtime = RuntimeSettings(
            artifactApiUrl="https://server.example/private/v1", requestTimeoutSeconds=30
        )
        args = dict(
            selected=["interactsh_poll"],
            allocation_id="allocation",
            run_id="run",
            namespace="worker",
            runtime_settings=runtime,
            workspace=AllocationWorkspace(tmp_path, tmp_path),
            state=WorkerState(),
        )
        with pytest.raises(InteractshError, match="interactsh_not_configured"):
            await factory.create_selected(**args)
        selected = await factory.create_selected(
            **args, adapter_handles=AdapterHandles(interactsh=handle)
        )
        assert set(selected) == {"interactsh_poll"}
        assert selected["interactsh_poll"].contractor_sensitive_output
        assert factory.sensitive_output_tools == factory.exported_tools
        await selected["interactsh_poll"].close()
        await handle.close()

    asyncio.run(scenario())


def test_cancelled_evidence_encoding_keeps_owned_batch_for_retry(monkeypatch):
    from contractor_runtime.toolsets.interactsh import tools as tool_module

    async def scenario():
        collector, artifacts = Collector(), Artifacts()
        handle = client(collector)
        session = _Session(handle, artifacts, "worker", ())
        registered = await session.register("authorized request 42")
        item = handle.registration(registered["handle"])
        collector.events = [event(item)]
        started = asyncio.Event()
        real_thread = tool_module.threads.to_thread_until_done

        async def interrupted(function, *args, name):
            if name == "interactsh-evidence":
                started.set()
                await asyncio.Event().wait()
            return await real_thread(function, *args, name=name)

        monkeypatch.setattr(tool_module.threads, "to_thread_until_done", interrupted)
        pending = asyncio.create_task(session.poll(item.handle))
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert item.handle in session.pending
        monkeypatch.setattr(tool_module.threads, "to_thread_until_done", real_thread)
        result = await session.poll(item.handle)
        assert result["events"] and collector.calls.count("/poll") == 1
        await session.close()
        await handle.close()

    asyncio.run(scenario())


def test_model_errors_are_bounded_and_retry_after_is_preserved(tmp_path):
    async def scenario():
        collector, artifacts = Collector(), Artifacts()
        handle = client(collector)
        runtime = RuntimeSettings(
            artifactApiUrl="https://server.example/private/v1", requestTimeoutSeconds=30
        )
        factory = InteractshToolsetFactory(lambda *_: artifacts)
        tools = await factory.create_selected(
            selected=["interactsh_register", "interactsh_poll"],
            allocation_id="allocation",
            run_id="run",
            namespace="worker",
            runtime_settings=runtime,
            workspace=AllocationWorkspace(tmp_path, tmp_path),
            state=WorkerState(),
            adapter_handles=AdapterHandles(interactsh=handle),
        )
        invalid = await tools["interactsh_register"]("\ud800")
        assert invalid == {
            "state": "error",
            "error": {"code": "interactsh_request_invalid", "retryable": False},
        }
        registered = await tools["interactsh_register"]("authorized check")
        collector.poll_response = httpx.Response(429, headers={"Retry-After": "9999"}, text=TOKEN)
        limited = await tools["interactsh_poll"](registered["handle"])
        assert limited["error"] == {
            "code": "interactsh_rate_limited",
            "retryable": True,
            "retry_after_seconds": 120,
        }
        assert TOKEN not in repr(limited)
        await tools["interactsh_poll"].close()
        await handle.close()

    asyncio.run(scenario())


def test_maximum_batch_publication_and_escaped_secret_rejection():
    async def scenario():
        collector, artifacts = Collector(), Artifacts()
        now = [0]
        handle = client(collector, lambda: now[0])
        session = _Session(handle, artifacts, "worker", (TOKEN,))
        registered = await session.register("authorized check")
        item = handle.registration(registered["handle"])
        collector.events = [event(item)] * MAX_EVENTS
        result = await session.poll(item.handle)
        assert len(result["events"]) == MAX_EVENTS
        assert len(json.dumps(result).encode()) < 64 * 1024
        now[0] += 2
        raw = event(item).decode().replace('"raw-request":""', '"raw-request":"' + TOKEN + '"')
        collector.events = [
            raw.replace(TOKEN, "\\u" + format(ord(TOKEN[0]), "04x") + TOKEN[1:], 1).encode()
        ]
        with pytest.raises(InteractshError, match="interactsh_response_invalid"):
            await session.poll(item.handle)
        assert item.coverage_gap
        await session.close()
        await handle.close()

    asyncio.run(scenario())
