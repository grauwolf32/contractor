import asyncio
import logging

import pytest
from fakes.spec import allocation_spec
from starlette.testclient import TestClient

from contractor_runtime.contracts import API_VERSION, PrepareAllocationRequest
from contractor_runtime.server import create_app
from contractor_runtime.state import RuntimeState


def test_health_and_readiness_reflect_process_state() -> None:
    state = RuntimeState(instance_id="runtime-test")
    with TestClient(create_app(state, require_verified_peer=False)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "state": "starting"}
        readiness = client.get("/readyz")
        assert readiness.status_code == 503

        asyncio.run(state.mark_registered())
        readiness = client.get("/readyz")
        assert readiness.status_code == 200
        assert readiness.json() == {"status": "ready", "state": "idle"}


def test_routes_require_verified_peer_by_default() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")
        assert response.status_code == 401
        assert response.json()["code"] == "mtls_required"
        assert response.headers["x-request-id"] == response.json()["requestId"]


def test_unconfigured_lifecycle_service_is_typed_and_records_dispatch() -> None:
    state = RuntimeState(instance_id="runtime-test")
    with TestClient(create_app(state, require_verified_peer=False)) as client:
        response = client.post(
            "/private/v1/allocations/allocation-1/prepare",
            json={},
            headers={"X-Request-ID": "control-plane-request-1"},
        )
        assert response.status_code == 503
        assert response.json() == {
            "code": "allocation_service_unavailable",
            "message": "allocation lifecycle service is not configured",
            "retryable": True,
            "requestId": "control-plane-request-1",
        }
        assert response.headers["x-request-id"] == "control-plane-request-1"
    assert asyncio.run(state.snapshot()).route_dispatches == 1


def test_internal_failure_is_correlated_and_does_not_log_injected_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    injected_secret = "secret-do-not-log"

    class FailingAllocationService:
        async def prepare(self, _: object) -> object:
            raise RuntimeError(injected_secret)

        async def active_a2a_application(self, _: str) -> object:
            raise RuntimeError(injected_secret)

    request = PrepareAllocationRequest(
        apiVersion=API_VERSION,
        spec=allocation_spec(allocation_id=injected_secret),
    )
    with (
        caplog.at_level(logging.ERROR, logger="contractor_runtime.server"),
        TestClient(
            create_app(
                allocation_service=FailingAllocationService(),  # type: ignore[arg-type]
                require_verified_peer=False,
            )
        ) as client,
    ):
        response = client.post(
            f"/private/v1/allocations/{injected_secret}/prepare",
            content=request.model_dump_json(by_alias=True),
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": "failure-request-1",
            },
        )
        uncaught = client.post(
            f"/private/v1/allocations/{injected_secret}/a2a",
            content=b"{}",
            headers={"X-Request-ID": "gateway-failure-request-1"},
        )
    assert response.status_code == 500
    assert response.headers["x-request-id"] == "failure-request-1"
    assert response.json()["requestId"] == "failure-request-1"
    assert injected_secret not in response.text
    assert injected_secret not in caplog.text
    assert "failure-request-1" in caplog.text
    assert uncaught.status_code == 500
    assert uncaught.headers["x-request-id"] == "gateway-failure-request-1"
    assert uncaught.json()["requestId"] == "gateway-failure-request-1"
    assert injected_secret not in uncaught.text
