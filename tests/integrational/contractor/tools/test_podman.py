import pytest
from pathlib import Path
from typing import Callable
from contractor.tools.podman import PodmanContainer


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "openapi"
    # adjust parents[...] if needed


@pytest.fixture(scope="session")
def test_pod(data_dir: Path) -> PodmanContainer:
    pod = PodmanContainer(
        name="test",
        image="ubuntu:jammy",
        mounts=[data_dir],
        ro_mode=True,
    )
    yield pod
    pod.stop()


def test_openapi_schema_present(test_pod: PodmanContainer):
    code_exec: Callable = test_pod.tools()[0]

    res = code_exec("ls openapi")
    assert "error" not in res, res
    result = res["result"]
    assert "petstore.json" in result
