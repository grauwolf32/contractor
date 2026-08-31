"""A Google ADK BaseLlm that yields a deterministic sequence without network I/O."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field, PrivateAttr


class ScriptedLlm(BaseLlm):
    responses: list[LlmResponse] = Field(exclude=True)
    block_first_call: bool = Field(default=False, exclude=True)
    requests: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    _started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    @property
    def started(self) -> asyncio.Event:
        return self._started

    def release(self) -> None:
        self._release.set()

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        self.requests.append(
            {
                "model": llm_request.model,
                "maxOutputTokens": llm_request.config.max_output_tokens,
                "temperature": llm_request.config.temperature,
                "responseMimeType": llm_request.config.response_mime_type,
                "hasResponseSchema": llm_request.config.response_schema is not None,
                "toolNames": sorted(
                    declaration.name
                    for tool in llm_request.config.tools or []
                    for declaration in tool.function_declarations or []
                ),
            }
        )
        self._started.set()
        if self.block_first_call and len(self.requests) == 1:
            await self._release.wait()
        if not self.responses:
            raise RuntimeError("ScriptedLlm has no response for this model call")
        yield self.responses.pop(0)


def tool_call(name: str, arguments: dict[str, Any], *, call_id: str) -> LlmResponse:
    return _with_usage(
        LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name=name,
                            args=arguments,
                        )
                    )
                ],
            )
        )
    )


def json_result(value: dict[str, Any]) -> LlmResponse:
    import json

    return _with_usage(
        LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=json.dumps(value, separators=(",", ":")))],
            )
        )
    )


def text_result(text: str) -> LlmResponse:
    return _with_usage(
        LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )
    )


def thought_result(text: str = "Task complete") -> LlmResponse:
    return _with_usage(
        LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=text, thought=True)],
            )
        )
    )


def scripted_model(responses: Sequence[LlmResponse], *, block: bool = False) -> ScriptedLlm:
    return ScriptedLlm(
        model="deterministic-fake", responses=list(responses), block_first_call=block
    )


def _with_usage(response: LlmResponse) -> LlmResponse:
    response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=7,
        candidates_token_count=3,
        total_token_count=10,
    )
    return response
