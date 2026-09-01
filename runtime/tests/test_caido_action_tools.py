from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_caido_read_tools import (
    CAIDO_TOKEN,
    FakeArtifactClient,
    create_tools,
    request_detail,
)

from contractor_runtime.toolsets.caido import (
    CAIDO_OUTPUT_ARTIFACT_PREFIX,
    CAIDO_TOOL_NAMES,
    CaidoToolError,
    CaidoToolsetFactory,
)


def test_scope_replay_automate_and_workflow_actions_are_exact_and_bounded(
    tmp_path: Path,
) -> None:
    automate_raw = (
        "POST /fuzz HTTP/1.1\r\n"
        "Host: target.example\r\n"
        "X-Request-Id: caller-value\r\n\r\n"
        "prefix=π&TARGET=value"
    ).encode()
    replay_response = b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok"
    convert_output = ("converted-" + "x" * 9000).encode()
    observed: list[dict[str, Any]] = []
    replay_raw: str | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal replay_raw
        payload = json.loads(request.content)
        observed.append(payload)
        operation = payload["operationName"]
        variables = payload["variables"]
        if operation == "CreateScope":
            data = {
                "createScope": {
                    "error": None,
                    "scope": {
                        "id": "scope-new",
                        "name": "target",
                        "allowlist": ["*.target.example"],
                        "denylist": [],
                    },
                }
            }
        elif operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {
                        "id": "replay-session",
                        "name": "replay",
                        "activeEntry": {"id": "replay-entry"},
                    }
                }
            }
        elif operation == "StartReplayTask":
            replay_raw = variables["input"]["raw"]
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "replay-task", "replayEntry": {"id": "replay-entry"}},
                }
            }
        elif operation == "ReplayEntry":
            assert replay_raw is not None
            data = {
                "replayEntry": {
                    "id": "replay-entry",
                    "raw": replay_raw,
                    "error": None,
                    "request": {
                        "id": "replayed-request",
                        "method": "GET",
                        "host": "target.example",
                        "path": "/probe",
                        "query": "",
                        "response": {
                            "statusCode": 201,
                            "length": 2,
                            "roundtripTime": 7,
                            "raw": base64.b64encode(replay_response).decode(),
                        },
                    },
                }
            }
        elif operation == "RequestDetail":
            data = {"request": request_detail(automate_raw, b"HTTP/1.1 200 OK\r\n\r\n")}
        elif operation == "CreateAutomateSession":
            data = {
                "createAutomateSession": {
                    "session": {
                        "id": "automate-session",
                        "name": "scan",
                        "settings": {"strategy": "ALL"},
                    }
                }
            }
        elif operation == "UpdateAutomateSession":
            settings = variables["input"]["settings"]
            data = {
                "updateAutomateSession": {
                    "error": None,
                    "session": {
                        "id": "automate-session",
                        "name": "scan",
                        "settings": {
                            "placeholders": settings["placeholders"],
                            "strategy": settings["strategy"],
                        },
                    },
                }
            }
        elif operation == "StartAutomateTask":
            data = {
                "startAutomateTask": {
                    "automateTask": {
                        "id": "automate-task",
                        "paused": False,
                        "entry": {"id": "automate-entry", "name": "scan"},
                    }
                }
            }
        elif operation == "RunConvertWorkflow":
            data = {
                "runConvertWorkflow": {
                    "output": base64.b64encode(convert_output).decode(),
                    "error": None,
                }
            }
        elif operation == "RunActiveWorkflow":
            data = {
                "runActiveWorkflow": {
                    "task": {
                        "id": "workflow-task",
                        "createdAt": "2026-09-01T00:00:00Z",
                        "workflow": {"id": "workflow-active", "name": "CORS"},
                    },
                    "error": None,
                }
            }
        else:
            raise AssertionError(f"unexpected operation {operation}")
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, state, handle = await create_tools(
            tmp_path,
            handler,
            artifacts,
            selected=CAIDO_TOOL_NAMES,
        )

        scope = await tools["caido_scope"](
            action="create", name="target", allowlist=["*.target.example"]
        )
        assert scope["status"] == "created"

        replay = await tools["caido_replay"](
            raw_request="GET /probe HTTP/1.1\r\nHost: target.example\r\n\r\n",
            host="target.example",
            port=443,
            is_tls=True,
        )
        assert replay["status"] == "completed"
        assert replay["status_code"] == 201
        assert replay["request_tag"].endswith("-c000001")
        replay_submitted = base64.b64decode(
            next(item for item in observed if item["operationName"] == "StartReplayTask")[
                "variables"
            ]["input"]["raw"]
        )
        assert f"X-Request-Id: {replay['request_tag']}\r\n".encode() in replay_submitted

        automate = await tools["caido_automate_run"](
            "request-1",
            targets=["TARGET"],
            payloads=["recognizable-payload-one", "recognizable-payload-two"],
            strategy="ALL",
            workers=3,
        )
        assert automate["status"] == "started"
        assert automate["request_tag"].endswith("-c000002")
        update = next(item for item in observed if item["operationName"] == "UpdateAutomateSession")
        submitted = base64.b64decode(update["variables"]["input"]["raw"])
        assert submitted.count(b"X-Request-Id:") == 1
        assert b"caller-value" not in submitted
        placeholder = update["variables"]["input"]["settings"]["placeholders"][0]
        assert placeholder == {
            "start": submitted.index(b"TARGET"),
            "end": submitted.index(b"TARGET") + len(b"TARGET"),
        }
        assert update["variables"]["input"]["settings"]["retryOnFailure"] == {
            "maximumRetries": 0,
            "backoff": 0,
        }

        converted = await tools["caido_workflow_run"](
            "workflow-convert", input="recognizable-workflow-input"
        )
        assert converted["output_truncated"] is True
        assert converted["output"].startswith("converted-")
        assert converted["output_artifact"]["name"].startswith(CAIDO_OUTPUT_ARTIFACT_PREFIX)
        active = await tools["caido_workflow_run"]("workflow-active", request_id="request-1")
        assert active["status"] == "started"
        assert active["task_id"] == "workflow-task"

        names = [item["operationName"] for item in observed]
        assert names == [
            "CreateScope",
            "CreateReplaySession",
            "StartReplayTask",
            "ReplayEntry",
            "RequestDetail",
            "CreateAutomateSession",
            "UpdateAutomateSession",
            "StartAutomateTask",
            "RunConvertWorkflow",
            "RunActiveWorkflow",
        ]
        assert state.metrics.counters["tool_calls"] == 5
        retained = repr(state.metrics)
        for forbidden in (
            CAIDO_TOKEN,
            "TARGET=value",
            "caller-value",
            "recognizable-workflow-input",
            "recognizable-payload-one",
        ):
            assert forbidden not in retained
        assert len(artifacts.payloads) == 2
        await handle.close()

    asyncio.run(scenario())


def test_mutation_response_loss_is_not_retried(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("recognizable mutation response loss", request=request)

    async def scenario() -> None:
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_scope"},
        )
        with pytest.raises(CaidoToolError) as failure:
            await tools["caido_scope"](action="create", name="target")
        assert failure.value.code == "caido_request_failed"
        assert failure.value.retryable is False
        assert calls == 1
        await handle.close()

    asyncio.run(scenario())


def test_replay_blob_may_exceed_generic_graphql_string_limit(tmp_path: Path) -> None:
    raw_request = "POST /large HTTP/1.1\r\nHost: target.example\r\n\r\n" + "x" * (800 * 1024)
    observed_blob_sizes: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operation = payload["operationName"]
        if operation == "CreateReplaySession":
            raw = payload["variables"]["input"]["requestSource"]["raw"]["raw"]
            observed_blob_sizes.append(len(raw.encode()))
            data = {
                "createReplaySession": {
                    "session": {
                        "id": "large-session",
                        "name": "large",
                        "activeEntry": {"id": "large-entry"},
                    }
                }
            }
        elif operation == "StartReplayTask":
            raw = payload["variables"]["input"]["raw"]
            observed_blob_sizes.append(len(raw.encode()))
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "large-task", "replayEntry": {"id": "large-entry"}},
                }
            }
        else:
            raise AssertionError(f"unexpected operation {operation}")
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
        )
        result = await tools["caido_replay"](
            raw_request=raw_request,
            host="target.example",
            wait=False,
        )
        assert result["status"] == "started"
        assert observed_blob_sizes[0] > 1024 * 1024
        assert observed_blob_sizes[0] == observed_blob_sizes[1]
        await handle.close()

    asyncio.run(scenario())


def test_replay_polling_timeout_and_cancellation_are_bounded(tmp_path: Path) -> None:
    class Clock:
        value = 0.0

        def monotonic(self) -> float:
            return self.value

        async def sleep(self, seconds: float) -> None:
            self.value += seconds

    async def run(*, cancel: bool) -> tuple[int, str | None]:
        clock = Clock()
        polls = 0

        async def sleep(seconds: float) -> None:
            if cancel:
                raise asyncio.CancelledError
            await clock.sleep(seconds)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            operation = json.loads(request.content)["operationName"]
            if operation == "CreateReplaySession":
                data = {
                    "createReplaySession": {
                        "session": {
                            "id": "session",
                            "name": "replay",
                            "activeEntry": {"id": "entry"},
                        }
                    }
                }
            elif operation == "StartReplayTask":
                data = {
                    "startReplayTask": {
                        "error": None,
                        "task": {"id": "task", "replayEntry": {"id": "entry"}},
                    }
                }
            else:
                polls += 1
                data = {"replayEntry": None}
            return httpx.Response(200, json={"data": data}, request=request)

        artifacts = FakeArtifactClient()
        factory = CaidoToolsetFactory(
            lambda _allocation, _settings: artifacts,
            sleep=sleep,
            monotonic=clock.monotonic,
        )
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            artifacts,
            selected={"caido_replay"},
            factory=factory,
        )
        status = None
        try:
            result = await tools["caido_replay"](
                raw_request="GET / HTTP/1.1\r\nHost: target.example\r\n\r\n",
                host="target.example",
                timeout_seconds=1,
            )
            status = result["status"]
        finally:
            await handle.close()
        return polls, status

    polls, status = asyncio.run(run(cancel=False))
    assert status == "timeout"
    assert polls == 3
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run(cancel=True))


def test_automate_rejects_limits_and_overlapping_targets_before_mutation(
    tmp_path: Path,
) -> None:
    raw = b"POST / HTTP/1.1\r\nHost: target.example\r\n\r\nabcdef"
    observed: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        operation = json.loads(request.content)["operationName"]
        observed.append(operation)
        assert operation == "RequestDetail"
        return httpx.Response(
            200,
            json={"data": {"request": request_detail(raw, b"")}},
            request=request,
        )

    async def scenario() -> None:
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_automate_run"},
        )
        with pytest.raises(CaidoToolError) as too_many:
            await tools["caido_automate_run"]("request-1", targets=["a"], payloads=["x"] * 1001)
        assert too_many.value.code == "caido_request_invalid"
        assert observed == []
        with pytest.raises(CaidoToolError) as overlap:
            await tools["caido_automate_run"]("request-1", targets=["abc", "bc"], payloads=["x"])
        assert overlap.value.code == "caido_request_invalid"
        assert observed == ["RequestDetail"]
        await handle.close()

    asyncio.run(scenario())


def test_automate_accepts_maximum_matrix_with_tagged_utf8_byte_offsets(
    tmp_path: Path,
) -> None:
    targets = [f"T{index:02d}!" for index in range(32)]
    payloads = [""] + [f"payload-{index:04d}\r\n" for index in range(999)]
    raw = b"POST /fuzz HTTP/1.1\r\nHost: target.example\r\n\r\nprefix=\xcf\x80&" + b"&".join(
        target.encode() for target in targets
    )
    submitted: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operation = payload["operationName"]
        variables = payload["variables"]
        if operation == "RequestDetail":
            data = {"request": request_detail(raw, b"")}
        elif operation == "CreateAutomateSession":
            data = {
                "createAutomateSession": {
                    "session": {
                        "id": "maximum-session",
                        "name": "maximum",
                        "settings": {"strategy": "ALL"},
                    }
                }
            }
        elif operation == "UpdateAutomateSession":
            submitted.update(variables["input"])
            data = {
                "updateAutomateSession": {
                    "error": None,
                    "session": {
                        "id": "maximum-session",
                        "name": "maximum",
                        "settings": {
                            "placeholders": variables["input"]["settings"]["placeholders"],
                            "strategy": "ALL",
                        },
                    },
                }
            }
        elif operation == "StartAutomateTask":
            data = {
                "startAutomateTask": {
                    "automateTask": {
                        "id": "maximum-task",
                        "paused": False,
                        "entry": {"id": "maximum-entry", "name": "maximum"},
                    }
                }
            }
        else:
            raise AssertionError(f"unexpected operation {operation}")
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_automate_run"},
        )
        result = await tools["caido_automate_run"](
            "request-1",
            targets=targets,
            payloads=payloads,
        )
        assert result["target_count"] == 32
        assert result["payload_count"] == 1000
        tagged = base64.b64decode(submitted["raw"])
        assert b"prefix=\xcf\x80" in tagged
        assert submitted["settings"]["payloads"][0]["options"]["simpleList"]["list"] == payloads
        assert submitted["settings"]["placeholders"] == [
            {"start": tagged.index(target.encode()), "end": tagged.index(target.encode()) + 4}
            for target in targets
        ]
        await handle.close()

    asyncio.run(scenario())


def test_domain_rejection_is_bounded_and_mismatched_identity_is_invalid(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        operation = json.loads(request.content)["operationName"]
        if operation == "CreateScope":
            data = {
                "createScope": {
                    "error": {"code": "InvalidGlobTerms"},
                    "scope": None,
                }
            }
        elif operation == "RunActiveWorkflow":
            data = {
                "runActiveWorkflow": {
                    "task": {
                        "id": "task",
                        "createdAt": "2026-09-01T00:00:00Z",
                        "workflow": {"id": "different-workflow", "name": "wrong"},
                    },
                    "error": None,
                }
            }
        else:
            raise AssertionError(f"unexpected operation {operation}")
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_scope", "caido_workflow_run"},
        )
        rejected = await tools["caido_scope"](
            action="create", name="target", allowlist=["[invalid"]
        )
        assert rejected == {"status": "rejected", "error_code": "InvalidGlobTerms"}
        with pytest.raises(CaidoToolError) as invalid:
            await tools["caido_workflow_run"]("workflow", request_id="request-1")
        assert invalid.value.code == "caido_response_invalid"
        assert calls == 2
        await handle.close()

    asyncio.run(scenario())
