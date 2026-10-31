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


def test_shared_received_worker_policy_controls_post_import_models(monkeypatch):
    from dataclasses import replace

    from contractor_runtime.contracts import (
        AgentStateSnapshot,
        WorkerCompletion,
        WorkerFailure,
        WorkerObservations,
        WorkerResult,
        WorkspaceObservationSummary,
    )
    from contractor_runtime.contracts.worker import WorkerStateWorkspaceObservation

    original = shared_config.get_shared_config()
    monkeypatch.setattr(
        shared_config,
        "_snapshot",
        replace(
            original,
            worker_result_max_bytes=7,
            worker_failure_message_max_bytes=5,
            worker_result_max_artifacts=1,
            worker_observation_max_tools=1,
            worker_workspace_max_files_read=1,
            worker_state_workspace_max_paths=1,
        ),
    )
    observations = {"profile": "lean@1", "tools": {}, "truncated": False}
    result = {
        "subtaskId": "0",
        "result": "éééx",
        "observations": observations,
        "artifacts": {},
        "summarized": False,
    }
    WorkerResult.model_validate(result)
    with pytest.raises(ValidationError):
        WorkerResult.model_validate({**result, "result": "éééxx"})
    WorkerFailure(code="failed", message="ééx", retryable=False)
    with pytest.raises(ValidationError):
        WorkerFailure(code="failed", message="ééxx", retryable=False)
    artifact = {"namespace": "analysis", "name": "report", "revision": "r1"}
    WorkerResult.model_validate({**result, "artifacts": {"one": artifact}})
    with pytest.raises(ValidationError):
        WorkerResult.model_validate({**result, "artifacts": {"one": artifact, "two": artifact}})
    count = {"calls": 1, "failures": 0}
    WorkerObservations.model_validate({**observations, "tools": {"one": count}})
    with pytest.raises(ValidationError):
        WorkerObservations.model_validate({**observations, "tools": {"one": count, "two": count}})
    workspace = dict(
        scopedFiles=2,
        scopeComplete=False,
        discoveredFiles=2,
        readFiles=2,
        matchedFiles=0,
        modifiedFiles=0,
        detailComplete=False,
        filesRead=["a.py"],
        filesReadTruncated=True,
    )
    WorkspaceObservationSummary.model_validate(workspace)
    with pytest.raises(ValidationError):
        WorkspaceObservationSummary.model_validate(
            {**workspace, "filesRead": ["a.py", "b.py"], "filesReadTruncated": False}
        )
    state_workspace = dict(
        scopePaths=["a.py"], scopeComplete=False, interactions=[], detailComplete=False
    )
    WorkerStateWorkspaceObservation.model_validate(state_workspace)
    with pytest.raises(ValidationError):
        WorkerStateWorkspaceObservation.model_validate(
            {**state_workspace, "scopePaths": ["a.py", "b.py"]}
        )
    cap = len(b'["a.py"]')
    monkeypatch.setattr(
        shared_config,
        "_snapshot",
        replace(shared_config.get_shared_config(), worker_state_workspace_paths_max_bytes=cap),
    )
    WorkerStateWorkspaceObservation.model_validate(state_workspace)
    with pytest.raises(ValidationError):
        WorkerStateWorkspaceObservation.model_validate({**state_workspace, "scopePaths": ["aa.py"]})

    completion = WorkerCompletion(
        apiVersion="contractor/v1alpha1",
        invocationId="invocation-1",
        stateRevision=1,
        result=result,
    )
    cap = len(completion.model_dump_json(by_alias=True, exclude_none=True).encode())
    monkeypatch.setattr(
        shared_config,
        "_snapshot",
        replace(shared_config.get_shared_config(), worker_completion_max_bytes=cap),
    )
    WorkerCompletion.model_validate(completion.model_dump(exclude_none=True))
    with pytest.raises(ValidationError):
        WorkerCompletion.model_validate(
            {**completion.model_dump(exclude_none=True), "invocation_id": "invocation-11"}
        )
    # State envelope checking reads the received cap at validation time as well.
    fixture = (
        Path(__file__).resolve().parents[2]
        / "api/testdata/v1alpha1/valid/agent-state-snapshot.json"
    )
    monkeypatch.setattr(shared_config, "_snapshot", original)
    state = AgentStateSnapshot.model_validate_json(fixture.read_bytes())
    cap = len(state.model_dump_json(by_alias=True, exclude_none=True).encode())
    monkeypatch.setattr(shared_config, "_snapshot", replace(original, worker_state_max_bytes=cap))
    AgentStateSnapshot.model_validate_json(fixture.read_bytes())
    monkeypatch.setattr(
        shared_config, "_snapshot", replace(original, worker_state_max_bytes=cap - 1)
    )
    with pytest.raises(ValidationError):
        AgentStateSnapshot.model_validate_json(fixture.read_bytes())


def test_received_http_policy_controls_post_import_models_and_tools(monkeypatch):
    import base64
    from dataclasses import replace

    from contractor_runtime.toolsets.common.http_capture import (
        CapturedHTTPRequest,
        CapturedHTTPResponse,
        StructuredHTTPMessage,
    )
    from contractor_runtime.toolsets.http import tools as http_tools

    monkeypatch.setattr(
        shared_config,
        "_snapshot",
        replace(
            shared_config.get_shared_config(),
            http_request_body_max_bytes=3,
            http_response_body_max_bytes=5,
            http_header_block_max_bytes=6,
        ),
    )

    def message(body):
        return dict(kind="structured", headers=[], body_base64=base64.b64encode(body).decode())

    CapturedHTTPRequest(method="POST", url="https://example.test/", message=message(b"123"))
    with pytest.raises(ValidationError):
        CapturedHTTPRequest(method="POST", url="https://example.test/", message=message(b"1234"))
    CapturedHTTPResponse(status=200, message=message(b"12345"), complete=True)
    with pytest.raises(ValidationError):
        CapturedHTTPResponse(status=200, message=message(b"123456"), complete=True)
    StructuredHTTPMessage(kind="structured", headers=[dict(name="X", value="ééx")], body_base64="")
    with pytest.raises(ValidationError):
        StructuredHTTPMessage(
            kind="structured", headers=[dict(name="X", value="ééxx")], body_base64=""
        )
    assert http_tools._request_body("text", "éx")[0] == "éx".encode()
    with pytest.raises(http_tools.HTTPToolError):
        http_tools._request_body("text", "éxx")
    assert http_tools.decoded_body_cache_max_bytes() == 4 * 5 + 128
    assert http_tools.request_header_max_bytes() == 6


def test_received_metadata_policy_controls_wire_validation_and_telemetry(monkeypatch):
    from dataclasses import replace

    from contractor_runtime.adapters.otlp_http import (
        RUN_METADATA_LABEL_ATTRIBUTE_PREFIX,
        _run_metadata_attributes,
    )
    from contractor_runtime.contracts.base import (
        _require_runtime_labels,
        normalize_run_metadata_labels,
    )

    monkeypatch.setattr(
        shared_config,
        "_snapshot",
        replace(
            shared_config.get_shared_config(),
            run_metadata_max_labels=1,
            run_metadata_key_max_bytes=4,
            run_metadata_value_max_bytes=5,
        ),
    )
    accepted = {"team": "ééx"}
    assert normalize_run_metadata_labels(accepted) == accepted
    assert _run_metadata_attributes(accepted, ()) == {
        RUN_METADATA_LABEL_ATTRIBUTE_PREFIX + "team": "ééx"
    }
    for invalid in ({"teams": "x"}, {"team": "ééxx"}, {"one": "a", "two": "b"}):
        with pytest.raises(ValueError):
            normalize_run_metadata_labels(invalid)
        assert _run_metadata_attributes(invalid, ()) == {}
    # Placement labels have their own grammar and size policy.
    _require_runtime_labels("labels", ["team-label"])
