"""Caido scope, workflow and Automate entry catalogs above one page (HTP-R147).

Every test drives the production caido-graphql@1 adapter and caido@1 tools
against a local GraphQL server, so provider bounds, strict validation and the
tool-result bound apply exactly as in an allocation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fakes.worker_state import WorkerState
from test_caido_adapter import adapter_context
from test_caido_read_tools import CAIDO_TOKEN, FakeArtifactClient, close_tools

from contractor_runtime.adapters import caido_graphql
from contractor_runtime.adapters.caido_graphql import CaidoGraphQLAdapter
from contractor_runtime.contracts import CaidoSettings, RuntimeSettings
from contractor_runtime.toolsets.caido import tools as caido_tools
from contractor_runtime.toolsets.caido.tools import CaidoToolError, CaidoToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

CARDINALITIES = [0, 100, 101, 107, 1000]
KINDS = ("convert", "active", "passive")
MAX_RESULT_BYTES = 1024 * 1024


def scope(index: int, **changes: Any) -> dict[str, Any]:
    return {
        "id": str(100 + index),
        "name": f"scope-{index}",
        "allowlist": [f"*.target-{index}.example"],
        "denylist": [],
        **changes,
    }


def workflow(index: int, **changes: Any) -> dict[str, Any]:
    return {
        "id": str(100 + index),
        "name": f"workflow-{index}",
        "kind": KINDS[index % 3].upper(),
        "enabled": True,
        "global": index % 2 == 0,
        **changes,
    }


def automate_session(entries: int, *, session_id: str = "session-1") -> dict[str, Any]:
    return {
        "automateSession": {
            "id": session_id,
            "name": "scan",
            "entries": [
                {"id": f"entry-{index}", "name": f"run {index}", "createdAt": 1788220800000 + index}
                for index in range(entries)
            ],
            "settings": {"strategy": "ALL", "placeholders": [{"start": 1, "end": 2}]},
        }
    }


def automate_entry(entry_id: str) -> dict[str, Any]:
    return {
        "automateEntry": {
            "id": entry_id,
            "name": "run",
            "requestsByOffset": {
                "count": {"value": 1},
                "nodes": [
                    {
                        "sequenceId": 1,
                        "error": None,
                        "payloads": [{"position": 0, "raw": base64.b64encode(b"probe").decode()}],
                        "request": None,
                    }
                ],
            },
        }
    }


@asynccontextmanager
async def caido_server(tmp_path: Path, responses: dict[str, Any]):
    """Serve each GraphQL operation's current data from responses."""

    requests: list[dict[str, Any]] = []
    tasks: set[asyncio.Task[Any]] = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            fields = {}
            for line in headers.splitlines()[1:]:
                name, separator, value = line.partition(":")
                if separator:
                    fields[name.lower()] = value.strip()
            assert fields["authorization"] == f"Bearer {CAIDO_TOKEN}"
            payload = json.loads(await reader.readexactly(int(fields["content-length"])))
            requests.append(payload)
            data = responses[payload["operationName"]]
            if callable(data):
                data = data(payload["variables"])
            body = json.dumps({"data": data}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    settings = CaidoSettings(
        adapter="caido-graphql@1",
        endpoint=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        bearerToken=CAIDO_TOKEN,
        requestTimeoutSeconds=5,
    )
    adapter = CaidoGraphQLAdapter(adapter_context(), settings)
    state = WorkerState()
    tools = dict(
        await CaidoToolsetFactory(lambda *_: FakeArtifactClient()).create_selected(
            selected=["caido_automate_results", "caido_scope", "caido_workflow_list"],
            allocation_id="catalog-allocation",
            run_id="catalog-run",
            namespace="analyst",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example",
                requestTimeoutSeconds=10,
                caido=settings,
            ),
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path / "allocation"),
            resources=state.resources,
            adapter_handles=adapter.handles,
        )
    )
    try:
        yield tools, requests, state
    finally:
        await close_tools(tools)
        await adapter.close()
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


async def read_catalog(call, field: str, total: int, **arguments: Any) -> list[str]:
    """Follow next_offset from offset 0 and check every page's metadata."""

    page = await call(**arguments)
    seen: list[str] = []
    while True:
        assert len(page[field]) <= 100
        assert len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode()) <= (
            MAX_RESULT_BYTES
        )
        assert page[f"total_{field}"] == total
        assert page["offset"] == len(seen)
        seen.extend(member["id"] for member in page[field])
        assert page["truncated"] == (len(seen) < total)
        if not page["truncated"]:
            assert "next_offset" not in page
            return seen
        assert page[field]
        assert page["next_offset"] == len(seen)
        page = await call(**arguments, offset=page["next_offset"])


@pytest.mark.parametrize("count", CARDINALITIES)
def test_scope_catalog_pages_every_member(tmp_path: Path, count: int) -> None:
    scopes = [scope(index) for index in range(count)]

    async def scenario():
        async with caido_server(tmp_path, {"Scopes": {"scopes": scopes}}) as (
            tools,
            requests,
            state,
        ):
            seen = await read_catalog(tools["caido_scope"], "scopes", count)
            assert seen == [item["id"] for item in scopes]
            assert len(requests) == max(1, -(-count // 100))
            assert {request["operationName"] for request in requests} == {"Scopes"}
            short = await read_catalog(tools["caido_scope"], "scopes", count, limit=7)
            assert short == seen
            assert CAIDO_TOKEN not in repr(state.metrics)

    asyncio.run(scenario())


@pytest.mark.parametrize("count", CARDINALITIES)
@pytest.mark.parametrize("kind", ["", "active"])
def test_workflow_catalog_pages_every_member_of_the_kind(
    tmp_path: Path, count: int, kind: str
) -> None:
    workflows = [workflow(index) for index in range(count)]
    selected = [item["id"] for item in workflows if not kind or item["kind"].lower() == kind]

    async def scenario():
        async with caido_server(tmp_path, {"Workflows": {"workflows": workflows}}) as (
            tools,
            requests,
            _state,
        ):
            seen = await read_catalog(
                tools["caido_workflow_list"], "workflows", len(selected), kind=kind
            )
            assert seen == selected
            assert {request["operationName"] for request in requests} == {"Workflows"}

    asyncio.run(scenario())


def test_scope_pages_fit_the_result_bound(tmp_path: Path) -> None:
    # 107 scopes of 20 multibyte terms each, about 40 KiB per scope in a tool
    # result, so a page holds far fewer than 100 scopes.
    scopes = [
        scope(index, allowlist=[f"{term}." + "界" * 680 for term in range(20)])
        for index in range(107)
    ]

    async def scenario():
        async with caido_server(tmp_path, {"Scopes": {"scopes": scopes}}) as (tools, _, _state):
            first = await tools["caido_scope"]()
            assert 0 < len(first["scopes"]) < 100
            seen = await read_catalog(tools["caido_scope"], "scopes", 107)
            assert seen == [item["id"] for item in scopes]

    asyncio.run(scenario())


def test_offset_past_the_end_returns_an_empty_final_page(tmp_path: Path) -> None:
    async def scenario():
        responses = {"Scopes": {"scopes": [scope(0)]}, "Workflows": {"workflows": []}}
        async with caido_server(tmp_path, responses) as (tools, _requests, _state):
            assert await tools["caido_scope"](offset=5) == {
                "scopes": [],
                "total_scopes": 1,
                "offset": 5,
                "truncated": False,
            }
            assert await tools["caido_workflow_list"](offset=1) == {
                "workflows": [],
                "total_workflows": 0,
                "offset": 1,
                "truncated": False,
            }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("caido_scope", {"limit": 0}),
        ("caido_scope", {"limit": 101}),
        ("caido_scope", {"limit": True}),
        ("caido_scope", {"offset": -1}),
        ("caido_scope", {"offset": 1.0}),
        ("caido_scope", {"action": "create", "name": "s", "allowlist": ["a"], "offset": 1}),
        ("caido_scope", {"action": "create", "name": "s", "allowlist": ["a"], "limit": 7}),
        ("caido_workflow_list", {"limit": 0}),
        ("caido_workflow_list", {"limit": 101}),
        ("caido_workflow_list", {"offset": -1}),
        ("caido_workflow_list", {"offset": "1"}),
    ],
)
def test_invalid_catalog_pages_fail_before_a_request(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    async def scenario():
        async with caido_server(tmp_path, {}) as (tools, requests, _state):
            with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                await tools[tool](**arguments)
            assert requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("operation", "field", "member", "malformed"),
    [
        ("Scopes", "scopes", scope, {"allowlist": "not-a-list"}),
        ("Workflows", "workflows", workflow, {"kind": "UNKNOWN"}),
        ("Workflows", "workflows", workflow, {"enabled": "yes", "kind": "CONVERT"}),
    ],
)
@pytest.mark.parametrize("failure", ["malformed", "oversized"])
def test_invalid_catalog_source_beyond_the_first_page_fails_safely(
    tmp_path: Path, monkeypatch, operation, field, member, malformed, failure
) -> None:
    members = [member(index) for index in range(150)]
    if failure == "malformed":
        # A member the active-kind filter would skip is still validated.
        members[140].update(malformed)
    else:
        monkeypatch.setattr(caido_graphql, "MAX_CAIDO_RESPONSE_BYTES", 1024)
    tool = "caido_scope" if field == "scopes" else "caido_workflow_list"
    arguments = {} if field == "scopes" else {"kind": "active"}

    async def scenario():
        async with caido_server(tmp_path, {operation: {field: members}}) as (
            tools,
            _requests,
            state,
        ):
            with pytest.raises(CaidoToolError) as error:
                await tools[tool](**arguments)
            assert error.value.code == (
                "caido_response_invalid" if failure == "malformed" else "caido_response_too_large"
            )
            assert CAIDO_TOKEN not in f"{error.value!r} {state.metrics!r}"

    asyncio.run(scenario())


@pytest.mark.parametrize("entry", ["", "entry-120", "entry-0"])
def test_automate_entry_is_found_among_every_session_entry(tmp_path: Path, entry: str) -> None:
    async def scenario():
        responses = {
            "AutomateSession": automate_session(150),
            "AutomateEntryRequests": lambda variables: automate_entry(variables["id"]),
        }
        async with caido_server(tmp_path, responses) as (tools, requests, _state):
            result = await tools["caido_automate_results"]("session-1", entry_id=entry)
            expected = entry or "entry-149"
            assert result["entry_id"] == expected
            assert result["results"][0]["payloads"] == ["probe"]
            assert [request["operationName"] for request in requests] == [
                "AutomateSession",
                "AutomateEntryRequests",
            ]
            assert requests[1]["variables"]["id"] == expected

    asyncio.run(scenario())


def test_automate_entry_selection_outcomes(tmp_path: Path) -> None:
    async def scenario():
        responses: dict[str, Any] = {
            "AutomateSession": automate_session(150),
            "AutomateEntryRequests": lambda variables: automate_entry(variables["id"]),
        }
        async with caido_server(tmp_path, responses) as (tools, requests, _state):
            call = tools["caido_automate_results"]
            # An entry of another session is not this session's entry.
            assert await call("session-1", entry_id="entry-150") == {
                "session_id": "session-1",
                "entry_id": "entry-150",
                "status": "not_found",
            }
            assert [request["operationName"] for request in requests] == ["AutomateSession"]
            responses["AutomateSession"] = automate_session(0)
            assert await call("session-1") == {"session_id": "session-1", "status": "no_entries"}
            # Identities must match the request before anything is exposed.
            responses["AutomateSession"] = automate_session(150, session_id="session-2")
            with pytest.raises(CaidoToolError, match="caido_response_invalid"):
                await call("session-1")
            responses["AutomateSession"] = automate_session(150)
            responses["AutomateEntryRequests"] = automate_entry("entry-7")
            with pytest.raises(CaidoToolError, match="caido_response_invalid"):
                await call("session-1", entry_id="entry-120")
            # A malformed entry after the first hundred fails the whole read.
            responses["AutomateEntryRequests"] = lambda variables: automate_entry(variables["id"])
            malformed = automate_session(150)
            malformed["automateSession"]["entries"][130]["createdAt"] = True
            responses["AutomateSession"] = malformed
            with pytest.raises(CaidoToolError, match="caido_response_invalid"):
                await call("session-1", entry_id="entry-10")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("tool", "target"),
    [
        ("caido_scope", "_catalog_page"),
        ("caido_workflow_list", "_catalog_page"),
        ("caido_automate_results", "_automate_session_entry"),
    ],
)
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_catalog_processing_runs_off_loop_and_joins_before_release(
    tmp_path: Path, monkeypatch, tool: str, target: str, stop: str
) -> None:
    async def scenario():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        original = getattr(caido_tools, target)

        def blocked(*args, **kwargs):
            assert threading.get_ident() != loop_thread, f"{target} ran on the event loop"
            loop.call_soon_threadsafe(started.set)
            try:
                # A guard against a hung test, not a latency assertion.
                assert release.wait(60), "test did not release processing"
                return original(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(caido_tools, target, blocked)
        responses = {
            "Scopes": {"scopes": [scope(index) for index in range(150)]},
            "Workflows": {"workflows": [workflow(index) for index in range(150)]},
            "AutomateSession": automate_session(150),
            "AutomateEntryRequests": lambda variables: automate_entry(variables["id"]),
        }
        async with caido_server(tmp_path, responses) as (tools, requests, _state):
            budget = asyncio.timeout(None)
            arguments = ("session-1",) if tool == "caido_automate_results" else ()

            async def call():
                async with budget:
                    return await tools[tool](*arguments)

            task = asyncio.create_task(call())
            try:
                await asyncio.wait_for(started.wait(), timeout=60)
                if stop == "deadline":
                    budget.reschedule(loop.time())
                else:
                    task.cancel()
                close = asyncio.create_task(tools[tool].close())
                await asyncio.sleep(0)
                assert not task.done()
                assert not close.done()
            finally:
                release.set()
            with pytest.raises(TimeoutError if stop == "deadline" else asyncio.CancelledError):
                await task
            await close
            assert finished.is_set()
            # Nothing further reaches Caido once the call stopped, and the
            # closed session refuses another call.
            assert len(requests) == 1
            with pytest.raises(CaidoToolError, match="caido_request_failed"):
                await tools[tool](*arguments)
            assert len(requests) == 1

    asyncio.run(scenario())
