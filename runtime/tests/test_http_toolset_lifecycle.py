from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.http_tools import HTTPToolError, HTTPToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


def test_close_erases_session_and_fresh_allocation_starts_empty(tmp_path: Path) -> None:
    clients: list[httpx.AsyncClient] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"ok", request=request)

    def direct() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        clients.append(client)
        return client

    async def create(allocation_id: str) -> dict[str, Any]:
        factory = HTTPToolsetFactory(lambda _allocation, _settings: FakeArtifactClient(), direct)
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            llmGatewayToken="gateway-token",
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        )
        result = await factory.create_selected(
            selected=sorted(factory.exported_tools),
            allocation_id=allocation_id,
            run_id="run-http",
            namespace="worker",
            runtime_settings=settings,
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path / allocation_id),
            state=WorkerState(),
        )
        return dict(result)

    async def scenario() -> None:
        first = await create("allocation-first")
        await first["http_session_set"](
            cookies={"sid": "private"}, auth={"kind": "bearer", "token": "private"}
        )
        await first["http_request"]("https://target.example/one")
        for tool in first.values():
            await tool.close()
        with pytest.raises(HTTPToolError):
            await first["http_session_get"]()
        assert clients[0].is_closed

        second = await create("allocation-second")
        assert await second["http_session_get"]() == {
            "auth_kind": "none",
            "default_headers": {},
            "cookie_names": [],
            "cookie_count": 0,
            "history_count": 0,
        }
        for tool in second.values():
            await tool.close()

    asyncio.run(scenario())


class FakeArtifactClient:
    def __init__(self) -> None:
        self.payloads: dict[tuple[str, str, str], bytes] = {}
        self.media_types: dict[tuple[str, str, str], str] = {}
        self.writes = 0

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        assert expected_revision is None
        self.writes += 1
        exact = target.model_copy(update={"revision": f"revision-{self.writes}"})
        key = (exact.namespace, exact.name, exact.revision)
        self.payloads[key] = data
        self.media_types[key] = media_type
        return type("Write", (), {"artifact": exact})()

    async def read_artifact(self, ref: ArtifactRef) -> Any:
        exact = ref.require_exact()
        key = (exact.namespace, exact.name, exact.revision)
        return type(
            "Value",
            (),
            {"data": self.payloads[key], "media_type": self.media_types[key]},
        )()
