from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO

import httpx
import pytest
from fakes.worker_state import WorkerState
from google.genai import types
from PIL import Image
from test_adk_runtime import (
    FakeArtifactClient,
    assert_slot_released,
    build_context,
    stage_request,
)
from test_otlp_adapter import adapter_context, telemetry_settings

from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.artifacts import ArtifactTransportError, ArtifactValue
from contractor_runtime.contracts import API_VERSION, ArtifactMetadataResult
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.image_parts import (
    ARTIFACT_IMAGE_PART,
    MAX_CONTEXT_IMAGES,
    MAX_IMAGE_BYTES,
    image_result_metadata,
)
from contractor_runtime.llm.openai import (
    OpenAICompatibleGatewayLlm,
    _completion_request,
    _observed_input,
)
from contractor_runtime.toolsets.common.artifact_images import validate_image
from contractor_runtime.toolsets.common.artifact_visibility import ModelArtifactAccessError
from contractor_runtime.toolsets.run_artifacts.tools import ReadArtifactTool
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory


def picture(fmt="PNG", size=(2, 1)):
    image = Image.new("RGB", size, "red")
    if size == (2, 1):
        image.putpixel((1, 0), (0, 128, 255))
    output = BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()


class ImageClient(FakeArtifactClient):
    def __init__(self, data=None, media_type="image/png"):
        super().__init__()
        self.head = "image-r1"
        self.values = {self.head: data if data is not None else picture()}
        self.media_type = media_type
        self.metadata_calls = 0
        self.read_limits = []
        self.advertised_size = None

    async def read_artifact_metadata(self, ref):
        self.metadata_calls += 1
        revision = ref.revision or self.head
        return ArtifactMetadataResult(
            apiVersion=API_VERSION,
            artifact=ref.model_copy(update={"revision": revision}),
            contentCacheKey="private-cache-key-must-not-reach-model",
            mediaType=self.media_type,
            size=len(self.values[revision])
            if self.advertised_size is None
            else self.advertised_size,
        )

    async def read_artifact(self, ref, *, max_bytes=None):
        self.calls.append("read_artifact")
        self.read_limits.append(max_bytes)
        data = self.values[ref.revision]
        if max_bytes is not None and len(data) > max_bytes:
            raise ArtifactTransportError("response exceeds byte bound")
        self._remember(ref)
        now = datetime(2026, 11, 22, tzinfo=UTC)
        return ArtifactValue(ref, self.media_type, data, now, now)


def reader(client, support=True):
    state = WorkerState()
    budget = state.resources.image_input
    budget.supports_images = support
    return ReadArtifactTool(client, state.metrics, (), images=budget), budget


def response_part(result, call_id="read-1"):
    native = result[ARTIFACT_IMAGE_PART].inline_data
    return types.Part(
        function_response=types.FunctionResponse(
            id=call_id,
            name="read_artifact",
            response=image_result_metadata(result, replay=False),
            parts=[
                types.FunctionResponsePart(
                    inline_data=types.FunctionResponseBlob(
                        mime_type=native.mime_type,
                        data=native.data,
                    )
                )
            ],
        )
    )


@pytest.mark.parametrize(
    "support,reason", [(None, "image_support_unconfirmed"), (False, "text_only_route")]
)
def test_image_unavailable_is_metadata_only_before_download(support, reason):
    async def scenario():
        client = ImageClient()
        tool, budget = reader(client, support)
        result = await tool("inputs", "source")
        assert result["artifact"]["revision"] == "image-r1"
        assert result["imageDelivery"] == "unavailable" and result["reason"] == reason
        assert client.metadata_calls == 1 and client.calls == [] and budget.count == 0
        assert "private-cache-key" not in json.dumps(result)
        assert ARTIFACT_IMAGE_PART not in result

    asyncio.run(scenario())


@pytest.mark.parametrize("media_type", ["image/gif", "image/webp", "image/avif"])
def test_unsupported_image_formats_do_not_download(media_type):
    async def scenario():
        client = ImageClient(media_type=media_type)
        tool, _ = reader(client)
        assert (await tool("inputs", "source"))["reason"] == "unsupported_image_format"
        assert client.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("support", [None, False, True])
def test_text_and_svg_keep_text_pages_on_every_route(support):
    async def scenario():
        for mime, data in [("text/plain", "Привет 😀\n".encode()), ("image/svg+xml", b"<svg/>\n")]:
            client = ImageClient(data, mime)
            tool, budget = reader(client, support)
            assert (await tool("inputs", "source"))["text"] == data.decode()
            assert budget.count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_exact_image_bytes_replay_after_head_change_and_context_reset(fmt, mime):
    async def scenario():
        data = picture(fmt)
        client = ImageClient(data, mime)
        tool, budget = reader(client)
        result = await tool("inputs", "source")
        assert result["width"] == 2 and result["height"] == 1
        assert result[ARTIFACT_IMAGE_PART].inline_data.data == data
        client.head = "image-r2"
        client.values[client.head] = picture(fmt, (3, 1))
        budget.reconcile([])
        replay = await tool("inputs", "source", "image-r1")
        assert replay[ARTIFACT_IMAGE_PART].inline_data.data == data
        assert client.calls == ["read_artifact"] and client.read_limits == [MAX_IMAGE_BYTES]

    asyncio.run(scenario())


def test_invalid_images_limits_windows_and_hidden_refs_release_admission():
    async def scenario():
        for data, mime in [
            (b"broken", "image/png"),
            (picture("JPEG"), "image/png"),
            (picture()[:-12], "image/png"),
        ]:
            client = ImageClient(data, mime)
            tool, budget = reader(client)
            with pytest.raises(ToolInputError):
                await tool("inputs", "source")
            assert budget.pending == budget.count == budget.size == 0
        client = ImageClient()
        tool, budget = reader(client)
        for kwargs in [{"start_line": 2}, {"max_lines": 1}, {"line_offset": 1}]:
            with pytest.raises(ToolInputError, match="default line"):
                await tool("inputs", "source", **kwargs)
        with pytest.raises(ModelArtifactAccessError):
            await tool("builder", "http.body.hidden")
        assert client.calls == [] and budget.count == 0
        client.advertised_size = MAX_IMAGE_BYTES + 1
        with pytest.raises(ToolInputError, match="byte bound"):
            await tool("inputs", "source")
        assert client.calls == []
        client.advertised_size = 1
        client.values[client.head] = b"x" * (MAX_IMAGE_BYTES + 1)
        with pytest.raises(ArtifactTransportError, match="byte bound"):
            await tool("inputs", "source")
        assert budget.pending == budget.count == 0

    asyncio.run(scenario())


def test_decoded_pixel_dimension_and_single_frame_bounds():
    assert validate_image(picture(size=(2000, 2000)), "image/png") == (2000, 2000)
    assert validate_image(picture(size=(4096, 1)), "image/png") == (4096, 1)
    for size in [(2000, 2001), (4097, 1)]:
        with pytest.raises(ToolInputError):
            validate_image(picture(size=size), "image/png")
    output = BytesIO()
    Image.new("RGB", (2, 2), "red").save(
        output, format="PNG", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")]
    )
    with pytest.raises(ToolInputError):
        validate_image(output.getvalue(), "image/png")


def test_parallel_image_admission_is_bounded_and_context_can_release_it():
    async def scenario():
        client = ImageClient()
        tool, budget = reader(client)
        results = await asyncio.gather(
            *(tool("inputs", "source") for _ in range(MAX_CONTEXT_IMAGES))
        )
        assert budget.pending == 0 and budget.count == MAX_CONTEXT_IMAGES
        contents = [
            types.Content(
                role="user",
                parts=[response_part(result, str(i)) for i, result in enumerate(results)],
            )
        ]
        budget.reconcile(contents)
        with pytest.raises(ToolInputError) as error:
            await tool("inputs", "source", "image-r1")
        assert error.value.code == "image_context_limit" and client.calls == ["read_artifact"]
        budget.reconcile([])
        assert (await tool("inputs", "source", "image-r1"))["imageDelivery"] == "inline"
        assert client.calls == ["read_artifact"]

    asyncio.run(scenario())


def test_cancelled_decode_joins_before_releasing_image_capacity(monkeypatch):
    import contractor_runtime.toolsets.run_artifacts.tools as tools_module

    started, release = threading.Event(), threading.Event()

    def blocked(data, mime):
        started.set()
        if not release.wait(5):
            raise RuntimeError("test barrier was not released")
        return validate_image(data, mime)

    monkeypatch.setattr(tools_module, "validate_image", blocked)

    async def scenario():
        tool, budget = reader(ImageClient())
        task = asyncio.create_task(tool("inputs", "source"))
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and budget.pending == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert budget.pending == budget.count == 0
        await tool.close()

    asyncio.run(scenario())


def test_gateway_parallel_pairing_and_metadata_only_observer():
    async def scenario():
        tool, _ = reader(ImageClient())
        result = await tool("inputs", "source")
        contents = [
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(id="read-1", name="read_artifact", args={})
                    ),
                    types.Part(
                        function_call=types.FunctionCall(id="other-1", name="other", args={})
                    ),
                ],
            ),
            types.Content(role="user", parts=[response_part(result)]),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="other-1", name="other", response={"ok": True}
                        )
                    )
                ],
            ),
        ]
        from google.adk.models.llm_request import LlmRequest

        request = _completion_request("worker", LlmRequest(contents=contents), allow_images=True)
        assert [m["role"] for m in request["messages"]] == ["assistant", "tool", "tool", "user"]
        image_url = request["messages"][-1]["content"][1]["image_url"]["url"]
        assert base64.b64decode(image_url.partition(",")[2]) == picture()
        captured = json.dumps(_observed_input(request))
        assert "data:image" not in captured and base64.b64encode(picture()).decode() not in captured
        assert '"size": ' + str(len(picture())) in captured
        assert "private-cache-key" not in captured
        assert json.loads(request["messages"][1]["content"])["artifact"]["revision"] == "image-r1"

    asyncio.run(scenario())


def test_actual_worker_delivers_native_image_through_http_gateway(tmp_path):
    async def scenario():
        state, client = WorkerState(), ImageClient()
        tool = ReadArtifactTool(client, state.metrics, (), images=state.resources.image_input)
        sent, exports = [], []

        async def collector(request):
            exports.append(await request.aread())
            return httpx.Response(200, json={})

        adapter = await OTLPHTTPAdapterFactory(httpx.MockTransport(collector)).create(
            adapter_context(), telemetry_settings()
        )

        async def gateway(request):
            value = json.loads(request.content)
            sent.append(value)
            if len(sent) == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "read-1",
                            "type": "function",
                            "function": {
                                "name": "read_artifact",
                                "arguments": json.dumps(
                                    {
                                        "namespace": "inputs",
                                        "name": "source",
                                        "revision": "image-r1",
                                    }
                                ),
                            },
                        }
                    ],
                }
            elif len(sent) == 2:
                parts = [
                    part
                    for m in value["messages"]
                    if isinstance(m["content"], list)
                    for part in m["content"]
                ]
                native = next(
                    part["image_url"]["url"] for part in parts if part["type"] == "image_url"
                )
                actual = base64.b64decode(native.partition(",")[2])
                with (
                    Image.open(BytesIO(actual)) as decoded,
                    Image.open(BytesIO(picture())) as expected,
                ):
                    assert decoded.size == expected.size and decoded.mode == expected.mode
                    assert decoded.tobytes() == expected.tobytes()
                tool_message = next(m for m in value["messages"] if m["role"] == "tool")
                metadata = json.loads(tool_message["content"])
                assert (
                    tool_message["tool_call_id"] == "read-1"
                    and metadata["artifact"]["revision"] == "image-r1"
                )
                assert "data:image" not in tool_message["content"]
                message = {"role": "assistant", "content": "Image delivered"}
            else:
                message = {
                    "role": "assistant",
                    "content": json.dumps({"subtaskId": "0", "result": "Image delivered"}),
                }
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls" if len(sent) == 1 else "stop",
                            "message": message,
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )

        context = build_context(tmp_path, state, {"read_artifact": tool})
        context = replace(
            context,
            model_policy=context.model_policy.model_copy(update={"supports_images": True}),
            adapter_handles=adapter.handles,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            runtime = await AdkWorkerRuntimeFactory(
                lambda build: OpenAICompatibleGatewayLlm(
                    model=build.model_policy.model,
                    supports_images=build.model_policy.supports_images,
                    client_handle=new_gateway_client(
                        base_url="https://gateway.example/v1",
                        api_key="fixture-key",
                        timeout_seconds=10,
                        http_client=http,
                    ),
                )
            ).create(context)
            try:
                completion = await runtime.invoke(
                    stage_request().model_copy(update={"result_artifacts": {}})
                )
                assert completion.failure is None, completion.failure
                assert completion.result.result == "Image delivered" and len(sent) == 3
                assert_slot_released(runtime)
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

        await adapter.flush()
        await adapter.close()
        serialized = b"".join(exports)
        assert b"imageDelivery" in serialized and b"image/png" in serialized
        assert b"data:image" not in serialized and picture() not in serialized
        assert base64.b64encode(picture()) not in serialized

    asyncio.run(scenario())


def test_native_image_capacity_is_checked_before_encoding(monkeypatch):
    from google.adk.models.llm_request import LlmRequest

    import contractor_runtime.llm.openai as bridge
    from contractor_runtime.llm.image_parts import MAX_CONTEXT_IMAGE_BYTES

    def event(data, ordinal):
        return types.Part(
            function_response=types.FunctionResponse(
                id=str(ordinal),
                name="read_artifact",
                response={"imageDelivery": "inline"},
                parts=[
                    types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type="image/png",
                            data=data,
                        )
                    )
                ],
            )
        )

    data = b"x" * MAX_IMAGE_BYTES
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[event(data, i) for i in range(4)])]
    )
    encoded = _completion_request("vision", request, allow_images=True)
    observed = _observed_input(encoded)
    sizes = [
        part["image"]["size"]
        for content in observed["contents"]
        for part in content["parts"]
        if "image" in part
    ]
    assert len(sizes) == 4 and sum(sizes) == MAX_CONTEXT_IMAGE_BYTES

    def forbidden(_data):
        raise AssertionError("out-of-bound image was encoded")

    monkeypatch.setattr(bridge.base64, "b64encode", forbidden)
    for parts, allowed in [
        ([event(data, i) for i in range(5)], True),
        ([event(data + b"x", 0)], True),
        ([event(data, 0)], False),
    ]:
        with pytest.raises(RuntimeError, match="UnsupportedContentPart"):
            _completion_request(
                "vision",
                LlmRequest(contents=[types.Content(role="user", parts=parts)]),
                allow_images=allowed,
            )


def test_interrupted_history_seals_only_image_metadata():
    from test_worker_history import context, prepared, stored_responses

    async def scenario():
        tool, _ = reader(ImageClient())
        result = await tool("inputs", "source")
        history, service, session, _ = await prepared("read-1")
        history.admitted(context(session, "read-1"))
        await history.completed(context(session, "read-1"), result)
        await history.seal(service, session)
        sealed = (await stored_responses(service, session))[0]
        assert sealed.parts is None
        assert sealed.response["artifact"]["revision"] == "image-r1"
        assert sealed.response["imageDelivery"] == "reread_required"
        assert base64.b64encode(picture()).decode() not in json.dumps(sealed.response)

    asyncio.run(scenario())


def test_actual_compaction_releases_images_and_rereads_exact_ref(tmp_path):
    from test_worker_compaction import COMPACTOR_MODEL, compact_context

    async def scenario():
        state, client = WorkerState(), ImageClient()
        tool = ReadArtifactTool(client, state.metrics, (), images=state.resources.image_input)
        context = compact_context(
            tmp_path, state, {"read_artifact": tool}, keep=1, max_model_calls=8
        )
        context = replace(
            context, model_policy=context.model_policy.model_copy(update={"supports_images": True})
        )
        main, compactor = [], []
        runtime = None

        async def gateway(request):
            payload = json.loads(request.content)
            prompt = 10
            if payload["model"] == COMPACTOR_MODEL:
                compactor.append(payload)
                serialized = json.dumps(payload)
                assert "reread_required" in serialized and "image-r1" in serialized
                assert base64.b64encode(picture()).decode() not in serialized
                message = {
                    "role": "assistant",
                    "content": '{"summary":"Image ref retained; re-read exact source if needed"}',
                }
            elif "response_format" in payload:
                message = {
                    "role": "assistant",
                    "content": '{"subtaskId":"0","result":"Image replayed"}',
                }
            else:
                main.append(payload)
                ordinal = len(main)
                if ordinal == 3:
                    assert compactor
                    stored = [
                        session
                        for users in runtime._session_service.sessions.values()
                        for sessions in users.values()
                        for session in sessions.values()
                    ]
                    old = [
                        response
                        for session in stored
                        for event in session.events
                        for response in event.get_function_responses()
                        if response.id == "image-1"
                    ]
                    assert len(old) == 1 and old[0].parts is None
                    assert old[0].response["artifact"]["revision"] == "image-r1"
                    name, args = "read_context_history", {"group": 1}
                elif ordinal in (1, 2, 4):
                    if ordinal == 4:
                        assert "reread_required" in json.dumps(payload)
                    name, args = (
                        "read_artifact",
                        {"namespace": "inputs", "name": "source", "revision": "image-r1"},
                    )
                    if ordinal == 2:
                        prompt = 600
                elif ordinal == 5:
                    image_parts = [
                        part
                        for m in payload["messages"]
                        if isinstance(m["content"], list)
                        for part in m["content"]
                        if part["type"] == "image_url"
                    ]
                    assert image_parts and all(
                        base64.b64decode(part["image_url"]["url"].partition(",")[2]) == picture()
                        for part in image_parts
                    )
                    message = {"role": "assistant", "content": "Image replayed"}
                else:
                    raise AssertionError("unexpected model loop")
                if ordinal != 5:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"image-{ordinal}",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                    }
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"finish_reason": finish, "message": message}],
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": 5,
                        "total_tokens": prompt + 5,
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            runtime = await AdkWorkerRuntimeFactory(
                lambda build: OpenAICompatibleGatewayLlm(
                    model=build.model_policy.model,
                    supports_images=build.model_policy.supports_images,
                    client_handle=new_gateway_client(
                        base_url="https://gateway.example/v1",
                        api_key="fixture-key",
                        timeout_seconds=10,
                        http_client=http,
                    ),
                )
            ).create(context)
            try:
                completion = await runtime.invoke(
                    stage_request().model_copy(update={"result_artifacts": {}})
                )
                assert completion.failure is None, completion.failure
                assert completion.result.result == "Image replayed"
                assert len(main) == 5 and len(compactor) == 1
                assert client.calls == ["read_artifact"]
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())
