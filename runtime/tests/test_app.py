from starlette.testclient import TestClient

from contractor_runtime.app import create_app


def test_health_and_readiness() -> None:
    with TestClient(create_app()) as client:
        for path in ("/healthz", "/readyz"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


def test_health_rejects_post() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/healthz")
        assert response.status_code == 405
