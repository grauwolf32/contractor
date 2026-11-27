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
        assert response.headers["x-request-id"] == response.json()["requestId"]


def test_migrated_http_routes_are_removed():
    state = RuntimeState(instance_id="runtime-test")
    with TestClient(create_app(state, require_verified_peer=False)) as client:
        for operation in ("prepare", "finalize", "abort", "release", "a2a"):
            response = client.post(f"/private/v1/allocations/allocation-1/{operation}", json={})
            assert response.status_code == 404
    assert asyncio.run(state.snapshot()).route_dispatches == 0
