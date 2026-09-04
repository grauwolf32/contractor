"""A Google ADK BaseLlm that yields a deterministic sequence without network I/O."""

from __future__ import annotations

import asyncio
import json
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
    block_call_number: int | None = Field(default=None, exclude=True)
    auto_result_finalizer: bool = Field(default=True, exclude=True)
    result_finalizer_error: Exception | None = Field(default=None, exclude=True)
    requests: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    _started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _blocked: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    @property
    def started(self) -> asyncio.Event:
        return self._started

    @property
    def blocked(self) -> asyncio.Event:
        return self._blocked

    def release(self) -> None:
        self._release.set()

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        request_record = {
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
            "systemInstruction": llm_request.config.system_instruction,
            "contentText": "\n".join(
                part.text
                for content in llm_request.contents
                for part in content.parts or []
                if part.text is not None
            ),
        }
        self.requests.append(request_record)
        self._started.set()
        blocked_call = self.block_call_number or (1 if self.block_first_call else None)
        if blocked_call == len(self.requests):
            self._blocked.set()
            await self._release.wait()
        marker = "Contractor Worker result finalization input (JSON):\n"
        if marker in request_record["contentText"] and self.result_finalizer_error is not None:
            raise self.result_finalizer_error
        if self.auto_result_finalizer and marker in request_record["contentText"]:
            payload = json.loads(request_record["contentText"].split(marker, 1)[1])
            yield json_result(
                {
                    "subtaskId": payload["subtaskId"],
                    "result": payload["resultText"],
                }
            )
            return
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


def scripted_model(
    responses: Sequence[LlmResponse],
    *,
    block: bool = False,
    block_call_number: int | None = None,
    auto_result_finalizer: bool = True,
    result_finalizer_error: Exception | None = None,
    model: str = "deterministic-fake",
) -> ScriptedLlm:
    return ScriptedLlm(
        model=model,
        responses=list(responses),
        block_first_call=block,
        block_call_number=block_call_number,
        auto_result_finalizer=auto_result_finalizer,
        result_finalizer_error=result_finalizer_error,
    )


def _with_usage(response: LlmResponse) -> LlmResponse:
    response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=7,
        candidates_token_count=3,
        total_token_count=10,
    )
    return response
