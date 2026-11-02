from __future__ import annotations

import asyncio
import base64
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_caido_read_tools import (
    CAIDO_TOKEN,
    FakeArtifactClient,
    close_tools,
    create_tools,
    representative_responses,
    request_detail,
)
from test_http_toolset import WorkerToolCallbacks

import contractor_runtime.toolsets.caido.tools as caido_module
from contractor_runtime.adapters.caido_graphql import MAX_CAIDO_RESPONSE_BYTES
from contractor_runtime.toolsets.caido.tools import (
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
                        "createdAt": "2026-09-01T00:03:00Z",
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
            "RequestDetail",
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
        assert len(artifacts.payloads) == 1
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(caido_module.secrets, "token_hex", lambda _length: "bc00000000000000")
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
                    "error": {
                        "__typename": "InvalidGlobTermsUserError",
                        "code": "InvalidGlobTerms",
                    },
                    "scope": None,
                }
            }
        elif operation == "RequestDetail":
            data = {
                "request": request_detail(b"GET / HTTP/1.1\r\nHost: target.example\r\n\r\n", b"")
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
        assert calls == 3
        await handle.close()

    asyncio.run(scenario())


def test_unlisted_user_error_types_are_rejections_not_invalid_responses(
    tmp_path: Path,
) -> None:
    automate_raw = b"POST /fuzz HTTP/1.1\r\nHost: target.example\r\n\r\nTARGET=value"

    def user_error(query: str) -> dict[str, Any]:
        # GraphQL returns only selected members: an error type without its own
        # fragment is an empty object unless __typename is selected.
        return {"__typename": "FutureUserError"} if "__typename" in query else {}

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operation = payload["operationName"]
        query = payload["query"]
        if operation == "CreateScope":
            data = {"createScope": {"error": user_error(query), "scope": None}}
        elif operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {"id": "replay", "name": "replay", "activeEntry": None}
                }
            }
        elif operation == "StartReplayTask":
            data = {"startReplayTask": {"error": user_error(query), "task": None}}
        elif operation == "RequestDetail":
            data = {"request": request_detail(automate_raw, b"HTTP/1.1 200 OK\r\n\r\n")}
        elif operation == "CreateAutomateSession":
            data = {
                "createAutomateSession": {
                    "session": {"id": "automate", "name": "scan", "settings": {"strategy": "ALL"}}
                }
            }
        elif operation == "UpdateAutomateSession":
            data = {"updateAutomateSession": {"error": user_error(query), "session": None}}
        else:
            raise AssertionError(f"unexpected operation {operation}")
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_scope", "caido_replay", "caido_automate_run"},
        )
        scope = await tools["caido_scope"](action="create", name="target")
        assert scope == {"status": "rejected", "error_code": "FutureUserError"}
        replay = await tools["caido_replay"](
            raw_request="GET / HTTP/1.1\r\nHost: target.example\r\n\r\n",
            host="target.example",
            port=443,
            is_tls=True,
        )
        assert (replay["status"], replay["error_code"]) == ("rejected", "FutureUserError")
        automate = await tools["caido_automate_run"](
            "request-1", targets=["TARGET"], payloads=["one"]
        )
        assert (automate["status"], automate["error_code"]) == ("rejected", "FutureUserError")
        await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # LF-framed head whose body contains CRLF CRLF.
        (
            b"POST /x HTTP/1.1\nHost: t\n\nbody\r\n\r\nmore",
            b"POST /x HTTP/1.1\nX-Request-Id: tag-1\nHost: t\n\nbody\r\n\r\nmore",
        ),
        # CRLF-framed head whose body contains LF LF.
        (
            b"POST /x HTTP/1.1\r\nHost: t\r\n\r\nbody\n\nmore",
            b"POST /x HTTP/1.1\r\nX-Request-Id: tag-1\r\nHost: t\r\n\r\nbody\n\nmore",
        ),
    ],
)
def test_request_tag_framing_follows_the_first_header_terminator(raw, expected) -> None:
    tagged, (start, end) = caido_module._inject_request_tag(raw, "tag-1")
    assert tagged == expected
    assert tagged[start:end] == b"X-Request-Id: tag-1"


EIGHT_MIB = 8 * 1024 * 1024
GATEWAY_AUTHORIZATION = b"Authorization: Bearer gateway-token\r\n"
STORED_REQUEST = b"POST /fuzz HTTP/1.1\r\nHost: target.example\r\n\r\nTARGET=value"
PAYLOAD_PATHS = {
    # Each path's payload helpers that must run on a worker thread.
    "request_detail": {"_blob", "message", "_raw_preview"},
    "replay": {"_inject_request_tag", "_blob", "message", "_raw_preview"},
    "workflow": {"_blob", "message"},
    "replay_stored": {"_blob", "_inject_request_tag"},
    "automate_run": {"_blob", "_inject_request_tag"},
    "automate_results": {"_blob"},
}


def large_message(head: bytes) -> bytes:
    """An 8 MiB raw message whose body echoes a long Runtime secret."""

    echo = f"echo {CAIDO_TOKEN}\n".encode()
    return head + echo + b"x" * (EIGHT_MIB - len(head) - len(echo))


def scrubbed(message: bytes) -> bytes:
    return message.replace(
        GATEWAY_AUTHORIZATION, b"Authorization: [runtime-credential]\r\n"
    ).replace(CAIDO_TOKEN.encode(), b"[REDACTED]")


class PayloadBarrier:
    """Hold the first off-loop Caido Blob decode and record where payload helpers run.

    On the event loop thread nothing blocks, so an inline implementation runs to
    completion and fails the test instead of deadlocking.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.started = asyncio.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.threads: dict[str, list[int]] = {}
        self._held = False
        original_blob = caido_module._blob

        def blob(value: object) -> bytes:
            self._record("_blob")
            if threading.get_ident() == self.loop_thread or self._held:
                return original_blob(value)
            self._held = True
            self.loop.call_soon_threadsafe(self.started.set)
            try:
                # A guard against a hung test, not a latency assertion.
                if not self.release.wait(60):
                    raise RuntimeError("the test did not release payload processing")
                return original_blob(value)
            finally:
                self.finished.set()

        monkeypatch.setattr(caido_module, "_blob", blob)
        for name in ("_raw_preview", "_inject_request_tag"):
            monkeypatch.setattr(
                caido_module, name, self._recorded(name, getattr(caido_module, name))
            )
        monkeypatch.setattr(
            caido_module._CaidoRedaction,
            "message",
            self._recorded("message", caido_module._CaidoRedaction.message),
        )

    def _record(self, name: str) -> None:
        self.threads.setdefault(name, []).append(threading.get_ident())

    def _recorded(self, name: str, function: Any) -> Any:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            self._record(name)
            return function(*args, **kwargs)

        return recorded

    def assert_off_loop(self, names: set[str]) -> None:
        for name in names:
            assert self.threads.get(name), f"{name} was not exercised"
        for name, idents in self.threads.items():
            assert self.loop_thread not in idents, f"{name} ran on the event loop"


def payload_handler(observed: list[str]) -> tuple[Any, dict[str, bytes]]:
    """Adapter-valid responses: every JSON body stays below the 16 MiB bound."""

    large_response = large_message(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n")
    large_output = large_message(b"converted ")
    detail_request = b"GET /large HTTP/1.1\r\nHost: target.example\r\n" + GATEWAY_AUTHORIZATION
    detail_request += b"\r\n"
    large_response_blob = base64.b64encode(large_response).decode()
    bodies = {
        "large_detail": json.dumps(
            {"data": {"request": request_detail(detail_request, large_response)}}
        ).encode(),
        "stored_detail": json.dumps(
            {"data": {"request": request_detail(STORED_REQUEST, b"HTTP/1.1 200 OK\r\n\r\n")}}
        ).encode(),
        "convert": json.dumps(
            {
                "data": {
                    "runConvertWorkflow": {
                        "output": base64.b64encode(large_output).decode(),
                        "error": None,
                    }
                }
            }
        ).encode(),
    }
    replay_raw: list[str] = []
    automate = representative_responses(b"", b"")

    def data_response(request: httpx.Request, data: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"data": data}, request=request)

    def body_response(request: httpx.Request, body: bytes) -> httpx.Response:
        assert len(body) < MAX_CAIDO_RESPONSE_BYTES
        return httpx.Response(
            200, content=body, headers={"content-type": "application/json"}, request=request
        )

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operation = payload["operationName"]
        variables = payload["variables"]
        observed.append(operation)
        if operation == "RequestDetail":
            large = variables["id"] == "large-request"
            return body_response(request, bodies["large_detail" if large else "stored_detail"])
        if operation == "RunConvertWorkflow":
            return body_response(request, bodies["convert"])
        if operation == "CreateReplaySession":
            return data_response(
                request,
                {
                    "createReplaySession": {
                        "session": {
                            "id": "replay-session",
                            "name": "replay",
                            "activeEntry": {"id": "replay-entry"},
                        }
                    }
                },
            )
        if operation == "StartReplayTask":
            replay_raw.append(variables["input"]["raw"])
            return data_response(
                request,
                {
                    "startReplayTask": {
                        "error": None,
                        "task": {"id": "replay-task", "replayEntry": {"id": "replay-entry"}},
                    }
                },
            )
        if operation == "ReplayEntry":
            entry = {
                "id": "replay-entry",
                "raw": replay_raw[-1],
                "error": None,
                "request": {
                    "id": "replayed-request",
                    "createdAt": "2026-09-01T00:03:00Z",
                    "method": "GET",
                    "host": "target.example",
                    "path": "/probe",
                    "query": "",
                    "response": {
                        "statusCode": 200,
                        "length": len(large_response),
                        "roundtripTime": 7,
                        "raw": large_response_blob,
                    },
                },
            }
            return body_response(request, json.dumps({"data": {"replayEntry": entry}}).encode())
        if operation == "CreateAutomateSession":
            return data_response(
                request,
                {
                    "createAutomateSession": {
                        "session": {
                            "id": "automate",
                            "name": "scan",
                            "settings": {"strategy": "ALL"},
                        }
                    }
                },
            )
        if operation == "UpdateAutomateSession":
            settings = variables["input"]["settings"]
            return data_response(
                request,
                {
                    "updateAutomateSession": {
                        "error": None,
                        "session": {
                            "id": "automate",
                            "name": "scan",
                            "settings": {
                                "placeholders": settings["placeholders"],
                                "strategy": settings["strategy"],
                            },
                        },
                    }
                },
            )
        if operation == "StartAutomateTask":
            return data_response(
                request,
                {
                    "startAutomateTask": {
                        "automateTask": {
                            "id": "automate-task",
                            "paused": False,
                            "entry": {"id": "automate-entry", "name": "scan"},
                        }
                    }
                },
            )
        if operation in {"AutomateSession", "AutomateEntryRequests"}:
            return data_response(request, automate[operation])
        raise AssertionError(f"unexpected operation {operation}")

    return handler, {"response": large_response, "output": large_output, "request": detail_request}


def payload_call(path: str, tools: dict[str, Any]) -> Any:
    if path == "request_detail":
        return tools["caido_request_detail"]("large-request")
    if path == "replay":
        return tools["caido_replay"](
            raw_request=(
                b"GET /probe HTTP/1.1\r\nHost: target.example\r\n" + GATEWAY_AUTHORIZATION + b"\r\n"
            ).decode(),
            host="target.example",
        )
    if path == "workflow":
        return tools["caido_workflow_run"]("workflow-convert", input="convert this")
    if path == "replay_stored":
        return tools["caido_replay"](request_id="request-1", wait=False)
    if path == "automate_run":
        return tools["caido_automate_run"]("request-1", targets=["TARGET"], payloads=["one"])
    return tools["caido_automate_results"]("session-1")


async def assert_payload_result(
    path: str,
    result: dict[str, Any],
    state: Any,
    artifacts: FakeArtifactClient,
    payloads: dict[str, bytes],
) -> None:
    if path == "request_detail":
        response = scrubbed(payloads["response"])
        assert result["response"]["raw"]["size"] == len(response) > EIGHT_MIB - 64
        assert result["response"]["raw"]["truncated"] is True
        assert "echo [REDACTED]" in result["response"]["raw"]["data"]
        assert "Authorization: [runtime-credential]" in result["raw"]["data"]
        capture = (await state.resources.http_captures.get("caido", "request-1"))["attempts"][0]
        message = capture["response"]["message"]["data_base64"]
        assert base64.b64decode(message) == response
        assert base64.b64decode(capture["request"]["message"]["data_base64"]) == scrubbed(
            payloads["request"]
        )
    elif path == "replay":
        response = scrubbed(payloads["response"])
        assert result["status"] == "completed"
        assert result["response_raw"]["size"] == len(response)
        assert "Authorization: [runtime-credential]" in result["raw"]["data"]
        capture = (await state.resources.http_captures.get("caido", "replayed-request"))[
            "attempts"
        ][0]
        assert base64.b64decode(capture["response"]["message"]["data_base64"]) == response
    elif path == "workflow":
        output = scrubbed(payloads["output"])
        assert result["output_truncated"] is True
        assert result["output_size"] == len(output)
        assert result["output"].startswith("converted echo [REDACTED]")
        assert list(artifacts.payloads.values()) == [output]
    elif path == "automate_results":
        assert result["results"][0]["payloads"] == ["probe"]
    else:
        assert result["status"] == "started"
    rendered = json.dumps(result)
    assert CAIDO_TOKEN not in rendered
    assert "gateway-token" not in rendered


@pytest.mark.parametrize("path", sorted(PAYLOAD_PATHS))
@pytest.mark.parametrize("stop", ["complete", "cancel"])
def test_caido_payload_processing_runs_off_loop_and_joins_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, stop: str
) -> None:
    async def scenario() -> None:
        observed: list[str] = []
        handler, payloads = payload_handler(observed)
        artifacts = FakeArtifactClient()
        tools, state, handle = await create_tools(
            tmp_path, handler, artifacts, selected=CAIDO_TOOL_NAMES
        )
        session = tools["caido_replay"]._session
        puts: list[str] = []
        original_put = session._captures.put

        async def recording_put(capture: dict[str, Any], invocation_id: str | None) -> None:
            puts.append(capture["request_id"])
            await original_put(capture, invocation_id)

        monkeypatch.setattr(session._captures, "put", recording_put)
        barrier = PayloadBarrier(monkeypatch)
        task = asyncio.create_task(payload_call(path, tools))
        waiter = asyncio.create_task(barrier.started.wait())
        try:
            done, _ = await asyncio.wait(
                {task, waiter}, timeout=60, return_when=asyncio.FIRST_COMPLETED
            )
            assert waiter in done, "payload processing did not leave the event loop"
            assert not task.done()
            operations = list(observed)
            if stop == "complete":
                # The loop keeps serving other work while the payload thread is held.
                progressed = await asyncio.create_task(asyncio.sleep(0, result="progress"))
                assert progressed == "progress"
                assert not barrier.finished.is_set()
                assert not task.done()
                barrier.release.set()
                result = await task
                assert barrier.finished.is_set()
                await assert_payload_result(path, result, state, artifacts, payloads)
            else:
                task.cancel()
                close = asyncio.create_task(tools["caido_replay"].close())
                for _ in range(3):
                    await asyncio.sleep(0)
                # Cancellation and release both wait for the joined thread.
                assert not task.done()
                assert not close.done()
                assert session._lock.locked()
                barrier.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await close
                assert barrier.finished.is_set()
                # Nothing after the held work ran: no capture, no further request.
                assert puts == []
                assert observed == operations
            # A cancelled call stops after the held step; everything that ran ran off loop.
            barrier.assert_off_loop(PAYLOAD_PATHS[path] if stop == "complete" else {"_blob"})
        finally:
            barrier.release.set()
            waiter.cancel()
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("tool_name", ["caido_history", "caido_replay"])
def test_unexpected_caido_failure_reaches_tool_error_callback_unclassified(
    tmp_path: Path, tool_name: str
) -> None:
    detail = "recognizable-injected-detail"
    failures = [
        TypeError(f"implementation failure {detail}"),
        "connect",
    ]
    operations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        operation = json.loads(request.content)["operationName"]
        operations.append(operation)
        if failures:
            failure = failures.pop(0)
            if failure == "connect":
                raise httpx.ConnectError(f"refused {detail}", request=request)
            raise failure
        if operation == "RequestsByOffset":
            data = {"requestsByOffset": {"count": {"value": 0}, "nodes": []}}
        elif operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {"id": "session", "name": "replay", "activeEntry": None}
                }
            }
        else:
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "task", "replayEntry": None},
                }
            }
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        tools, state, handle = await create_tools(
            tmp_path, handler, FakeArtifactClient(), selected={tool_name}
        )
        tool = tools[tool_name]
        callbacks = await WorkerToolCallbacks.start(state, tools)
        arguments = (
            {}
            if tool_name == "caido_history"
            else {
                "raw_request": "GET / HTTP/1.1\r\nHost: target.example\r\n\r\n",
                "host": "target.example",
                "wait": False,
            }
        )
        response, raised = await callbacks.call(tool, **arguments)
        assert type(raised) is TypeError
        assert response == {
            "ok": False,
            "error": {
                "code": "tool_call_failed",
                "message": f"{tool_name} failed (TypeError)",
                "retryable": False,
            },
        }
        response, raised = await callbacks.call(tool, **arguments)
        assert isinstance(raised, CaidoToolError)
        # A read stays retryable; a mutation keeps its unknown outcome and is
        # never retried, by the tool or by the model's retry signal.
        assert response == {
            "ok": False,
            "error": {
                "code": "caido_request_failed",
                "message": f"{tool_name} failed (CaidoToolError)",
                "retryable": tool_name == "caido_history",
            },
        }
        result, raised = await callbacks.call(tool, **arguments)
        assert raised is None
        if tool_name == "caido_replay":
            # Neither failure was replayed, and their request tags are not reused.
            assert operations == [
                "CreateReplaySession",
                "CreateReplaySession",
                "CreateReplaySession",
                "StartReplayTask",
            ]
            assert result["request_tag"].endswith("-c000003")
        else:
            assert operations == ["RequestsByOffset"] * 3
        retained = f"{state.metrics!r} {handle!r}"
        assert detail not in retained
        assert CAIDO_TOKEN not in retained
        await close_tools(tools)
        await handle.close()

    asyncio.run(scenario())
