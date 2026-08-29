import asyncio

from starlette.testclient import TestClient

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


def test_unconfigured_lifecycle_service_is_typed_and_records_dispatch() -> None:
    state = RuntimeState(instance_id="runtime-test")
    with TestClient(create_app(state, require_verified_peer=False)) as client:
        response = client.post("/private/v1/allocations/allocation-1/prepare", json={})
        assert response.status_code == 503
        assert response.json() == {
            "code": "allocation_service_unavailable",
            "message": "allocation lifecycle service is not configured",
            "retryable": True,
        }
    assert asyncio.run(state.snapshot()).route_dispatches == 1
