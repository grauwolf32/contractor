"""Offline Go-to-Runtime model-selection probe using the real Gateway factory."""

import asyncio
import contextlib
import io
import json
import logging
import sys
from types import SimpleNamespace

_TOKEN_CANARY = "contractor-offline-gateway-token-canary"
_BODY_CANARY = "contractor-offline-provider-body-canary"
_MAX_INPUT_BYTES = 1024 * 1024


class ProbeFailure(RuntimeError):
    """A probe invariant failed; diagnostics never include inputs or provider data."""


def _deny_network(event: str, _arguments: tuple[object, ...]) -> None:
    if event in {"socket.connect", "socket.getaddrinfo"}:
        # A future factory regression must fail offline, not bypass the mock.
        raise ProbeFailure


async def _probe(raw: bytes) -> list[dict[str, object]]:
    import httpx
    from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
    from contractor_runtime.contracts import ResolvedModelPolicy, RuntimeSettings
    from contractor_runtime.llm.factory import gateway_model
    from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types
    from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter

    class ProbeCase(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        label: str = Field(min_length=1, max_length=160)
        policy: ResolvedModelPolicy
        gatewayUrl: str

    cases = TypeAdapter(list[ProbeCase]).validate_json(raw)
    if not 1 <= len(cases) <= 128:
        raise ProbeFailure
    results: list[dict[str, object]] = []
    for case in cases:
        settings = RuntimeSettings(
            llm_gateway_url=case.gatewayUrl,
            llm_gateway_token=SecretStr(_TOKEN_CANARY),
            artifact_api_url="https://artifact-probe.invalid",
            request_timeout_seconds=1,
        )
        expected_url = httpx.URL(case.gatewayUrl.rstrip("/") + "/chat/completions")
        sent_models: list[str] = []

        def respond(
            request: httpx.Request,
            case: ProbeCase = case,
            expected_url: httpx.URL = expected_url,
            sent_models: list[str] = sent_models,
        ) -> httpx.Response:
            payload = json.loads(request.content)
            if (
                request.method != "POST"
                or request.url != expected_url
                or request.headers.get("authorization") != f"Bearer {_TOKEN_CANARY}"
                or payload.get("model") != case.policy.model
            ):
                raise ProbeFailure
            for field, expected in (
                ("max_tokens", case.policy.max_output_tokens),
                ("temperature", case.policy.temperature),
            ):
                if (expected is None and field in payload) or (
                    expected is not None and payload.get(field) != expected
                ):
                    raise ProbeFailure
            sent_models.append(payload["model"])
            if len(sent_models) == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": "offline-completion",
                        "object": "chat.completion",
                        "created": 1,
                        "model": case.policy.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": f"{_BODY_CANARY} {_TOKEN_CANARY}",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), trust_env=False
        ) as http_client:
            proxy = ProxyHTTPClient(http_client)
            context = SimpleNamespace(
                model_policy=case.policy,
                runtime_settings=settings,
                adapter_handles=SimpleNamespace(model_http=proxy),
            )
            model = gateway_model(context)
            if not isinstance(model, OpenAICompatibleGatewayLlm):
                raise ProbeFailure
            request = LlmRequest(
                contents=[types.Content(role="user", parts=[types.Part(text="Reply ok.")])],
                config=types.GenerateContentConfig(
                    max_output_tokens=case.policy.max_output_tokens,
                    temperature=case.policy.temperature,
                ),
            )
            if request.model:
                raise ProbeFailure
            try:
                responses = [response async for response in model.generate_content_async(request)]
                if (
                    len(responses) != 1
                    or responses[0].content is None
                    or not responses[0].content.parts
                    or responses[0].content.parts[0].text != "ok"
                    or len(sent_models) != 1
                ):
                    raise ProbeFailure
                try:
                    async for _ in model.generate_content_async(request):
                        raise ProbeFailure
                except GatewayModelError as error:
                    if (
                        error.retryable is not False
                        or error.provider_error_type != "NotFoundError"
                        or error.__cause__ is not None
                        or error.__context__ is not None
                        or len(sent_models) != 2
                    ):
                        raise ProbeFailure from None
                    safe_error = f"{error!s} {error!r} {vars(error)!r}"
                    if any(canary in safe_error for canary in (_TOKEN_CANARY, _BODY_CANARY)):
                        raise ProbeFailure from None
                    rejected_type = type(error).__name__
                else:
                    raise ProbeFailure
            finally:
                await model.close()
                proxy.detach()
            if http_client.is_closed:
                # Factory must not close the adapter-owned HTTP client.
                raise ProbeFailure
        results.append(
            {
                "label": case.label,
                "models": sent_models,
                "rejectedType": rejected_type,
                "retryable": False,
            }
        )
    return results


def main() -> int:
    logging.disable(logging.CRITICAL)
    sys.addaudithook(_deny_network)
    try:
        raw = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
        if len(raw) > _MAX_INPUT_BYTES:
            raise ProbeFailure
        # Dependencies and provider adapters must not pollute the JSON bridge or
        # expose data through incidental logging, warnings or tracebacks.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results = asyncio.run(_probe(raw))
    except Exception as error:
        print(type(error).__name__, file=sys.stderr)
        return 1
    print(json.dumps(results, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
