"""Small ADK model adapter for Contractor's OpenAI-compatible LLM Gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Protocol, override

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from contractor_runtime.llm.client import GatewayClientHandle, GatewayRequestError
from contractor_runtime.llm.errors import GatewayModelError, _AdapterFailure
from contractor_runtime.llm.request import completion_request
from contractor_runtime.llm.response import empty_stop_response
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.threads import to_thread_until_done

_SAFE_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_FINISH_REASON = {
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}
_NON_RETRYABLE_ADAPTER_FAILURES = frozenset(
    {
        "StreamingUnsupported",
        "UnsupportedContentPart",
        "UnsupportedSystemInstruction",
        "InvalidFunctionCall",
        "InvalidFunctionPayload",
        "InvalidFunctionDeclaration",
        "UnsupportedResponseSchema",
    }
)


class GatewayCallObserver(Protocol):
    def capture_model_input(self, content: Callable[[], dict[str, Any]]) -> None: ...

    def capture_model_finish_reason(self, reason: str, *, known: bool) -> None: ...


class OpenAICompatibleGatewayLlm(BaseLlm):
    """Translate the ADK model contract to OpenAI chat completions."""

    _client_handle: GatewayClientHandle = PrivateAttr()
    _allow_images: bool = PrivateAttr(default=False)
    _observer: ContextVar[GatewayCallObserver | None] = PrivateAttr(
        default_factory=lambda: ContextVar("gateway_call_observer", default=None)
    )

    def __init__(
        self, *, model: str, client_handle: GatewayClientHandle, supports_images: bool | None = None
    ) -> None:
        super().__init__(model=model)
        self._client_handle = client_handle
        self._allow_images = supports_images is True

    @property
    @override
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    @property
    def closed(self) -> bool:
        return self._client_handle.closed

    async def close(self) -> None:
        await self._client_handle.close()

    @contextlib.asynccontextmanager
    async def observe_calls(self, observer: GatewayCallObserver) -> AsyncGenerator[None]:
        token = self._observer.set(observer)
        try:
            yield
        finally:
            self._observer.reset(token)

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        provider_error_type: str | None = None
        retryable = False
        failure = None
        response: LlmResponse | None = None
        completion: Any = None
        received = False
        usage: types.GenerateContentResponseUsageMetadata | None = None
        request_invalid = False
        empty_stop = False
        try:
            if stream:
                raise _AdapterFailure("StreamingUnsupported")
            # Coordinated calls have per-attempt transport timeouts and remain
            # cancellable by the authoritative Stage deadline while waiting.
            operation = (
                contextlib.nullcontext()
                if self._client_handle.recovery is not None
                else asyncio.timeout(self._client_handle.operation_timeout_seconds)
            )
            async with operation:
                request = await to_thread_until_done(
                    lambda: completion_request(
                        self.model, llm_request, allow_images=self._allow_images
                    ),
                    name="gateway-request-assembly",
                )
                observer = self._observer.get()
                if observer is not None:
                    with contextlib.suppress(Exception):
                        observer.capture_model_input(lambda: _observed_input(request))
                completion = await self._client_handle.complete(request)
                received = True
                if observer is not None:
                    with contextlib.suppress(Exception):
                        choice = _first_choice(completion)
                        reason = choice.get("finish_reason")
                        if isinstance(reason, str) and reason:
                            observer.capture_model_finish_reason(
                                reason, known=reason in _FINISH_REASON
                            )
                response = await to_thread_until_done(
                    _to_llm_response, completion, name="gateway-adk-response"
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            provider_error_type = _safe_error_type(error)
            retryable = _retryable_gateway_error(error)
            request_invalid = isinstance(error, _AdapterFailure) and not received
            failure = error.failure if isinstance(error, GatewayRequestError) else None
            if isinstance(error, GatewayRequestError):
                received |= error.response_received
            if received:
                usage = _rejected_response_usage(completion)
                if provider_error_type == "EmptyGatewayResponse":
                    choice = _first_choice(completion)
                    message = choice.get("message")
                    empty_stop = (
                        choice.get("finish_reason") == "stop"
                        and isinstance(message, Mapping)
                        and not message.get("refusal")
                    )
        completion = None
        if provider_error_type is not None:
            # Raise outside the except suite so the provider exception is not
            # retained through __context__ and cannot keep headers/body alive.
            raise GatewayModelError(
                provider_error_type,
                retryable=retryable,
                failure=failure,
                response_received=received,
                usage_metadata=usage,
                request_invalid=request_invalid,
                empty_stop=empty_stop,
            ) from None
        if response is None:  # Defensive: every non-error call must produce one response.
            raise GatewayModelError("InvalidGatewayResponse") from None
        yield response


def _observed_input(request: dict[str, Any]) -> dict[str, Any]:
    contents = []
    instruction = None
    for message in request["messages"]:
        role = message["role"]
        text = message["content"]
        if role == "system":
            instruction = text
            continue
        if role == "tool":
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                response = text
            parts = [{"functionResponse": {"id": message["tool_call_id"], "response": response}}]
        else:
            if isinstance(text, list):
                parts = []
                for item in text:
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        marker = url.find(";base64,")
                        padding = 2 if url.endswith("==") else 1 if url.endswith("=") else 0
                        parts.append(
                            {
                                "image": {
                                    "mediaType": url[5:marker],
                                    "size": (len(url) - marker - 8) // 4 * 3 - padding,
                                }
                            }
                        )
            else:
                parts = [] if text is None else [{"text": text}]
            parts.extend(
                {
                    "functionCall": {
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "args": json.loads(call["function"]["arguments"]),
                    }
                }
                for call in message.get("tool_calls", [])
            )
        contents.append({"role": role, "parts": parts})
    return {
        "contents": contents,
        "systemInstruction": instruction,
        "tools": [
            {"functionDeclarations": [tool["function"] for tool in request.get("tools", [])]}
        ],
    }


def _first_choice(completion: Any) -> Mapping[str, Any]:
    if not isinstance(completion, Mapping):
        raise _AdapterFailure("InvalidGatewayResponse")
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise _AdapterFailure("InvalidGatewayResponse")
    return choices[0]


def _to_llm_response(completion: Any) -> LlmResponse:
    choice = _first_choice(completion)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise _AdapterFailure("InvalidGatewayResponse")
    finish_reason = _map_finish_reason(choice.get("finish_reason"))
    if message.get("refusal"):
        finish_reason = types.FinishReason.SAFETY
    parts: list[types.Part] = []
    reasoning = _reasoning_text(message)
    if reasoning:
        parts.append(types.Part(text=reasoning, thought=True))
    text = _response_text(message.get("content"))
    if text:
        parts.append(types.Part.from_text(text=text))
    # A length-limited response is never executable, even if a partial tool
    # call happens to parse. Keep the finish reason and usage instead of
    # misclassifying truncated arguments as an unaccounted adapter error.
    tool_calls = message.get("tool_calls")
    if tool_calls is None or finish_reason not in (None, types.FinishReason.STOP):
        tool_calls = []
    if not isinstance(tool_calls, list):
        raise _AdapterFailure("InvalidGatewayToolCall")
    for tool_call in tool_calls:
        function = tool_call.get("function") if isinstance(tool_call, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            raise _AdapterFailure("InvalidGatewayToolCall")
        arguments = _tool_arguments(function.get("arguments"))
        part = types.Part.from_function_call(name=name, args=arguments)
        call_id = tool_call.get("id") or ""
        if not isinstance(call_id, str):
            raise _AdapterFailure("InvalidGatewayToolCall")
        part.function_call.id = call_id
        parts.append(part)
    if not parts and finish_reason in (None, types.FinishReason.STOP):
        raise _AdapterFailure("EmptyGatewayResponse")

    response = LlmResponse(
        model_version=completion.get("model"),
        content=types.Content(role="model", parts=parts),
        partial=False,
        finish_reason=finish_reason,
        usage_metadata=_usage_metadata(completion.get("usage")),
    )
    # ADK also maps provider tool finishes to STOP. Only the provider's
    # explicit stop can authorize recovery of a blank or thought-only reply.
    if choice.get("finish_reason") != "stop" and empty_stop_response(response):
        raise _AdapterFailure("EmptyGatewayResponse")
    if finish_reason is not None and finish_reason != types.FinishReason.STOP:
        response.error_code = finish_reason.value
        response.error_message = (
            "Maximum tokens reached"
            if finish_reason == types.FinishReason.MAX_TOKENS
            else f"Finished with {finish_reason.name}"
        )
    return response


def _reasoning_text(message: Mapping[str, Any]) -> str:
    value = message.get("reasoning_content")
    if value is None:
        value = message.get("reasoning")
    extra = message.get("model_extra")
    if value is None and isinstance(extra, Mapping):
        value = extra.get("reasoning_content", extra.get("reasoning"))
    return "".join(_reasoning_fragments(value))


def _reasoning_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [text for item in value for text in _reasoning_fragments(item)]
    if isinstance(value, Mapping):
        for key in ("text", "content", "reasoning", "reasoning_content"):
            if key in value:
                return _reasoning_fragments(value[key])
    return []


def _response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        texts: list[str] = []
        for part in value:
            text = part.get("text") if isinstance(part, Mapping) else getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
        return "".join(texts)
    raise _AdapterFailure("InvalidGatewayContent")


def _tool_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if not isinstance(value, str):
        raise _AdapterFailure("InvalidGatewayToolArguments")
    try:
        result = strict_json_loads(value)
    except (ValueError, RecursionError):
        raise _AdapterFailure("InvalidGatewayToolArguments") from None
    if not isinstance(result, dict):
        raise _AdapterFailure("InvalidGatewayToolArguments")
    return result


def _usage_metadata(usage: Any) -> types.GenerateContentResponseUsageMetadata | None:
    if usage is None:
        return None
    if not isinstance(usage, Mapping):
        raise _AdapterFailure("InvalidGatewayUsage")
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(prompt_details, Mapping) or not isinstance(completion_details, Mapping):
        raise _AdapterFailure("InvalidGatewayUsage")
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage.get("prompt_tokens"),
        candidates_token_count=usage.get("completion_tokens"),
        total_token_count=usage.get("total_tokens"),
        cached_content_token_count=prompt_details.get("cached_tokens"),
        thoughts_token_count=completion_details.get("reasoning_tokens"),
    )


def _rejected_response_usage(
    completion: Any,
) -> types.GenerateContentResponseUsageMetadata | None:
    """Keep only numeric usage from a response the adapter could not accept."""

    if not isinstance(completion, Mapping):
        return None
    try:
        return _usage_metadata(completion.get("usage"))
    except Exception:
        return None


def _map_finish_reason(value: Any) -> types.FinishReason | None:
    if value is None:
        return None
    return _FINISH_REASON.get(str(value).lower(), types.FinishReason.OTHER)


def _safe_error_type(error: Exception) -> str:
    if isinstance(error, _AdapterFailure):
        return error.code
    if isinstance(error, GatewayRequestError):
        return error.provider_error_type
    name = type(error).__name__
    return name if _SAFE_ERROR_TYPE.fullmatch(name) else "UnknownProviderError"


def _retryable_gateway_error(error: Exception) -> bool:
    if isinstance(error, GatewayRequestError):
        return error.retryable
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, _AdapterFailure):
        return error.code not in _NON_RETRYABLE_ADAPTER_FAILURES
    return False
