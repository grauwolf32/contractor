from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from test_cli import ConfigTransport, make_settings

from contractor_runtime import _https, cli, shared_config
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, decode_private
from contractor_runtime.contracts.shared_config import SharedConfig
from contractor_runtime.control_client import (
    ControlClient,
    ControlClientError,
    ControlHTTPError,
    SharedConfigError,
)
from contractor_runtime.state import RuntimeState
from contractor_runtime.toolsets.common.artifact_read_cache import artifact_read_cache_max_bytes


def document(cap: int = 17) -> dict:
    limits = json.loads((Path(__file__).parent / "testdata/shared-config.json").read_text())[
        "limits"
    ]
    limits["artifactPayloadMaxBytes"] = cap
    projection = {"limits": limits, "version": 1}
    digest = hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"apiVersion": "contractor/v1alpha1", **projection, "digest": "sha256:" + digest}


def test_snapshot_is_required_immutable_and_controls_artifact_operations(monkeypatch):
    monkeypatch.setattr(shared_config, "_snapshot", None)
    with pytest.raises(RuntimeError, match="has not been loaded"):
        shared_config.artifact_payload_max_bytes()
    config = decode_private(SharedConfig, json.dumps(document()).encode())
    shared_config.install_shared_config(config.snapshot())
    shared_config.install_shared_config(config.snapshot())
    assert shared_config.artifact_payload_max_bytes() == 17
    assert artifact_read_cache_max_bytes() == 17 + 2 * 8
    with pytest.raises(ValidationError):
        config.limits.artifact_payload_max_bytes = 19
    other = decode_private(SharedConfig, json.dumps(document(18)).encode())
    with pytest.raises(RuntimeError, match="restart"):
        shared_config.install_shared_config(other.snapshot())

    async def scenario():
        transport = AsyncMock()
        client = ArtifactClient("allocation-test", transport)
        ref = ArtifactRef(namespace="analysis", name="report")
        with pytest.raises(ValueError, match="configured Artifact limit"):
            await client.write_artifact(
                ref, data=b"x" * 18, media_type="text/plain", expected_revision=None
            )
        with pytest.raises(ValueError, match="read byte limit"):
            await client.read_artifact(ref, max_bytes=18)
        transport.request.assert_not_called()
        # The default is evaluated per operation, not frozen at module import.
        transport.request.side_effect = OSError("observed transport")
        with pytest.raises(OSError):
            await client.read_artifact(ref)
        assert transport.request.call_args.kwargs["max_response_bytes"] == 17

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(version=2),
        lambda d: d.update(digest="sha256:" + "0" * 64),
        lambda d: d["limits"].update(artifactPayloadMaxBytes=0),
        lambda d: d["limits"].update(artifactPayloadMaxBytes=True),
        lambda d: d.update(extra="secret"),
    ],
)
def test_invalid_config_never_prepares_or_registers(tmp_path, monkeypatch, change):
    async def scenario():
        invalid = document()
        change(invalid)
        transport = AsyncMock()
        transport.get_json.return_value = invalid
        prepare = AsyncMock()
        monkeypatch.setattr(cli, "_serve_configured", prepare)
        monkeypatch.setattr(shared_config, "_snapshot", None)
        with pytest.raises(SharedConfigError):
            await cli.serve(
                make_settings(tmp_path), transport=transport, install_signal_handlers=False
            )
        prepare.assert_not_called()
        transport.post_json.assert_not_called()
        assert shared_config._snapshot is None

    asyncio.run(scenario())


def test_startup_fetch_precedes_preparation(tmp_path, monkeypatch):
    async def scenario():
        order = []
        monkeypatch.setattr(shared_config, "_snapshot", None)

        class Transport(ConfigTransport):
            async def get_json(self, path):
                order.append("config")
                return document()

        async def prepare(*args, **kwargs):
            assert shared_config.artifact_payload_max_bytes() == 17
            order.append("prepare")

        monkeypatch.setattr(cli, "_serve_configured", prepare)
        await cli.serve(
            make_settings(tmp_path), transport=Transport(), install_signal_handlers=False
        )
        assert order == ["config", "prepare"]

    asyncio.run(scenario())


def test_retryable_fetch_uses_existing_backoff_and_permanent_rejection_stops(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(shared_config, "_snapshot", None)
        transport = AsyncMock()
        transport.get_json.side_effect = [
            OSError(),
            TimeoutError(),
            ControlClientError("truncated response"),
            ControlHTTPError(503),
            document(),
        ]
        sleep = AsyncMock()
        client = ControlClient(
            make_settings(tmp_path), RuntimeState(), transport, sleep=sleep, jitter=lambda: 0
        )
        assert await client.configure_until_stopped(asyncio.Event())
        assert transport.get_json.await_count == 5
        assert len(sleep.await_args_list) == 4
        assert all(0 < call.args[0] <= 5 for call in sleep.await_args_list)
        transport.get_json.reset_mock(side_effect=True)
        transport.get_json.side_effect = ControlHTTPError(403)
        with pytest.raises(SharedConfigError):
            await client.configure_until_stopped(asyncio.Event())
        assert transport.get_json.await_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_task", [False, True])
def test_shutdown_joins_inflight_config_before_preparation(tmp_path, monkeypatch, cancel_task):
    async def scenario():
        started = asyncio.Event()
        joined = asyncio.Event()
        stop = asyncio.Event()

        async def fetch(path):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()

        transport = AsyncMock()
        transport.get_json.side_effect = fetch
        prepare = AsyncMock()
        monkeypatch.setattr(cli, "_serve_configured", prepare)
        task = asyncio.create_task(
            cli.serve(
                make_settings(tmp_path),
                transport=transport,
                stop_requested=stop,
                install_signal_handlers=False,
            )
        )
        await started.wait()
        if cancel_task:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            stop.set()
            await task
        assert joined.is_set()
        prepare.assert_not_called()
        transport.post_json.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", [None, "oversized", "interrupted"])
def test_get_transport_is_bounded_idempotent_and_uses_safe_headers(fault):
    from types import SimpleNamespace

    from contractor_runtime.control_client import MAX_CONTROL_RESPONSE_BYTES, MTLSJSONTransport

    async def scenario():
        connections = AsyncMock()
        connections.exchange.return_value = SimpleNamespace(status_code=200, body=b"{}")
        if fault == "oversized":
            connections.exchange.side_effect = _https.HTTPResponseTooLargeError()
        elif fault == "interrupted":
            connections.exchange.side_effect = _https.HTTPResponseNotStartedError()
        transport = MTLSJSONTransport("https://control.test:8443", connections, 7)
        if fault == "oversized":
            with pytest.raises(SharedConfigError):
                await transport.get_json("/private/v1/config")
        elif fault == "interrupted":
            with pytest.raises(ControlClientError) as error:
                await transport.get_json("/private/v1/config")
            assert not isinstance(error.value, SharedConfigError)
        else:
            assert await transport.get_json("/private/v1/config") == b"{}"
        call = connections.exchange.call_args
        assert call.kwargs == {
            "idempotent": True,
            "timeout_seconds": 7,
            "max_response_bytes": MAX_CONTROL_RESPONSE_BYTES,
        }
        request = call.args[2]
        assert request.startswith(b"GET /private/v1/config HTTP/1.1\r\n")
        assert b"Host: control.test:8443\r\n" in request
        assert request.endswith(b"\r\n\r\n")

    asyncio.run(scenario())


def test_small_payload_policy_keeps_metadata_transport_budget(monkeypatch):
    from contractor_runtime.artifacts import MAX_ARTIFACT_JSON_BYTES, MTLSArtifactTransport

    async def scenario():
        monkeypatch.setattr(shared_config, "_snapshot", None)
        config = decode_private(SharedConfig, json.dumps(document()).encode())
        shared_config.install_shared_config(config.snapshot())
        connections = AsyncMock()
        connections.exchange.return_value = _https.HTTPResponse(200, {}, b"{}")
        transport = MTLSArtifactTransport(
            "https://control.test:8443/private/v1", connections, 7, "runtime-test"
        )
        await transport.request(
            "GET",
            "/allocations/a/artifacts/n/name/metadata",
            headers={},
            body=b"",
            max_response_bytes=MAX_ARTIFACT_JSON_BYTES,
        )
        assert (
            connections.exchange.call_args.kwargs["max_response_bytes"] == MAX_ARTIFACT_JSON_BYTES
        )

    asyncio.run(scenario())


def test_fetched_artifact_list_ceiling_controls_client_before_transport(monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(
        shared_config, "_snapshot", replace(shared_config._snapshot, artifact_list_max_items=2)
    )

    async def scenario():
        transport = AsyncMock()
        client = ArtifactClient("allocation-test", transport)
        with pytest.raises(ValueError, match="list limit"):
            await client.list_artifact_page(limit=3)
        with pytest.raises(ValueError, match="list limit"):
            await client.list_artifact_prefix("analysis", "report", limit=3)
        transport.request.assert_not_called()
        # Boundary and continuation use the fetched limit without rewriting the cursor.
        transport.request.side_effect = OSError("observed request")
        with pytest.raises(OSError):
            await client.list_artifact_page(limit=2, cursor="next")
        assert transport.request.call_args.args[1].endswith("limit=2&cursor=next")

    asyncio.run(scenario())
