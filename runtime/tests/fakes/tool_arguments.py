"""Real Worker dispatch and local wire receivers for tool argument contracts."""

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from http_cookie_store import HTTPToolsetFactory
from test_adk_runtime import create_runtime, stage_request
from test_audit_http_credentials import tools_for
from test_http_proxy_adapter import adapter_context, fake_proxy, proxy_settings, text_response
from test_http_toolset import make_tools

from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import HTTPProxyAdapter
from contractor_runtime.contracts import RuntimeSettings
from fakes.engagement_socket import SocketTarget, preflight
from fakes.model import scripted_model, text_result, tool_call
from fakes.worker_state import WorkerState


async def run_calls(tmp_path, tool, state, arguments, *, inspect_results=None):
    model = scripted_model(
        [
            *(tool_call(tool.name, value, call_id=f"arg-{i}") for i, value in enumerate(arguments)),
            text_result("Completed"),
        ]
    )
    worker = await create_runtime(
        tmp_path,
        state,
        {tool.name: tool},
        model,
        max_model_calls=len(arguments) + 3,
        max_tool_calls=len(arguments) + 1,
    )
    try:
        completion = await worker.invoke(
            stage_request().model_copy(update={"result_artifacts": {}})
        )
        assert completion.failure is None
        session = await worker._session_service.get_session(
            app_name=worker._app_name,
            user_id=worker._user_id,
            session_id=worker._session_lifecycle.shared_session_id,
        )
        responses = [
            part.function_response.response
            for event in session.events
            for part in (event.content.parts if event.content else []) or []
            if part.function_response is not None
        ]
        assert len(responses) == len(arguments)
        if inspect_results is not None:
            await inspect_results(responses)
        return responses
    finally:
        await worker.finalize(datetime.now(UTC) + timedelta(seconds=5))


@asynccontextmanager
async def http_receiver(tmp_path, route, *, status=200, limits=None):
    if route == "controlled":

        async def reply(_target, _reader, writer, _message):
            writer.write(f"HTTP/1.1 {status} Result\r\nContent-Length: 2\r\n\r\nok".encode())
            await writer.drain()

        async with SocketTarget(reply) as target, preflight(target, limits=limits) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            tools, state = await tools_for(owner)
            yield tools["http_request"], state, target.origin + "/allowed", target.requests
        return
    async with fake_proxy(lambda request: (status, text_response(request)[1])) as receiver:
        proxy = None
        state = WorkerState()
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            llmGatewayToken="test-token",
            artifactApiUrl="https://control.example",
            requestTimeoutSeconds=10,
            httpOriginTarget={"url": receiver.url},
        )
        handles = EMPTY_ADAPTER_HANDLES
        url = receiver.url + "/allowed"
        if route == "proxy":
            config = proxy_settings(receiver.url, targets=["tool-http"])
            proxy = HTTPProxyAdapter(adapter_context(), config)
            handles = proxy.handles
            settings = settings.model_copy(update={"http_proxy": config})
            url = "http://public.example/allowed"
        try:
            tools = await make_tools(
                HTTPToolsetFactory(),
                tmp_path,
                state=state,
                settings=settings,
                adapter_handles=handles,
            )
            yield tools["http_request"], state, url, receiver.requests
        finally:
            if proxy is not None:
                await proxy.close()


def received(request):
    if not isinstance(request, bytes):
        return request.target, request.headers, request.body
    head, body = request.split(b"\r\n\r\n", 1)
    lines = head.decode("ascii").split("\r\n")
    headers = dict(line.lower().split(": ", 1) for line in lines[1:])
    return lines[0].split(" ")[1], headers, body
