"""Small ADK model adapter for Contractor's OpenAI-compatible LLM Gateway."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, override

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, PrivateAttr

from contractor_runtime.llm.client import GatewayClientHandle

_SAFE_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_FINISH_REASON = {
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}


class GatewayModelError(RuntimeError):
    """Secret-free boundary error for failures below the model adapter."""

    def __init__(self, provider_error_type: str, *, retryable: bool = True) -> None:
        self.provider_error_type = provider_error_type
        self.retryable = retryable
        super().__init__(f"LLM gateway call failed ({provider_error_type})")


class _AdapterFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class OpenAICompatibleGatewayLlm(BaseLlm):
    """Translate the ADK model contract to OpenAI chat completions."""

    _client_handle: GatewayClientHandle = PrivateAttr()

    def __init__(self, *, model: str, client_handle: GatewayClientHandle) -> None:
        super().__init__(model=model)
        self._client_handle = client_handle

    @property
    @override
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    @property
    def closed(self) -> bool:
        return self._client_handle.closed

    async def close(self) -> None:
        await self._client_handle.close()

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        provider_error_type: str | None = None
        retryable = False
        response: LlmResponse | None = None
        try:
            if stream:
                raise _AdapterFailure("StreamingUnsupported")
            async with asyncio.timeout(self._client_handle.operation_timeout_seconds):
                completion = await self._client_handle.client.chat.completions.create(
                    **_completion_request(self.model, llm_request)
                )
            response = _to_llm_response(completion)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            provider_error_type = _safe_error_type(error)
            retryable = _retryable_gateway_error(error)
        if provider_error_type is not None:
            # Raise outside the except suite so the provider exception is not
            # retained through __context__ and cannot keep headers/body alive.
            raise GatewayModelError(provider_error_type, retryable=retryable) from None
        if response is None:  # Defensive: every non-error call must produce one response.
            raise GatewayModelError("InvalidGatewayResponse") from None
        yield response


def _completion_request(model: str, request: LlmRequest) -> dict[str, Any]:
    config = request.config
    messages: list[dict[str, Any]] = []
    if config is not None and config.system_instruction is not None:
        instruction = _instruction_text(config.system_instruction)
        if instruction:
            messages.append({"role": "system", "content": instruction})
    for content in request.contents or []:
        messages.extend(_content_messages(content))
    if not messages:
        messages.append({"role": "user", "content": ""})

    tools = _tools(config.tools if config is not None else None)
    payload: dict[str, Any] = {
        "model": request.model or model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = _tool_choice(config)
    response_format = _response_format(config)
    if response_format is not None:
        payload["response_format"] = response_format
    if config is not None:
        _copy_generation_options(payload, config)
    return payload


def _instruction_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, types.Part):
        return _text_part(value)
    if isinstance(value, types.Content):
        return "\n".join(_text_part(part) for part in value.parts or [])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(text for item in value if (text := _instruction_text(item)))
    raise _AdapterFailure("UnsupportedSystemInstruction")


def _content_messages(content: types.Content) -> list[dict[str, Any]]:
    role = "assistant" if content.role in {"model", "assistant"} else "user"
    tool_messages: list[dict[str, Any]] = []
    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in content.parts or []:
        if part.function_response is not None:
            response = part.function_response
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": response.id or "",
                    "content": _json_text(response.response),
                }
            )
        elif part.function_call is not None:
            call = part.function_call
            if not call.name:
                raise _AdapterFailure("InvalidFunctionCall")
            tool_calls.append(
                {
                    "id": call.id or "",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _json_text(call.args or {}),
                    },
                }
            )
        elif part.text is not None:
            if _part_has_payload(part):
                raise _AdapterFailure("UnsupportedContentPart")
            (reasoning if part.thought else text).append(part.text)
        elif _part_has_payload(part):
            # Contractor transports non-text inputs as exact artifacts. Do not
            # silently drop a media/code part and change the model's task.
            raise _AdapterFailure("UnsupportedContentPart")

    ordinary: dict[str, Any] | None = None
    joined_text = "\n".join(text)
    if role == "assistant" and (joined_text or reasoning or tool_calls):
        ordinary = {"role": "assistant", "content": joined_text or None}
        if reasoning:
            ordinary["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            ordinary["tool_calls"] = tool_calls
    elif role == "user" and joined_text:
        ordinary = {"role": "user", "content": joined_text}

    if tool_messages:
        # ADK emits function responses as user-role Content. OpenAI requires
        # one tool-role message per response, followed by any ordinary content.
        return [*tool_messages, *([ordinary] if ordinary is not None else [])]
    return [ordinary] if ordinary is not None else []


def _text_part(part: types.Part) -> str:
    if part.text is None or part.thought or _part_has_payload(part):
        raise _AdapterFailure("UnsupportedSystemInstruction")
    return part.text


def _part_has_payload(part: types.Part) -> bool:
    return any(
        value is not None
        for name, value in part.model_dump(exclude_none=True).items()
        if name not in {"text", "thought"}
    )


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise _AdapterFailure("InvalidFunctionPayload") from None


def _tools(config_tools: Sequence[Any] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in config_tools or []:
        for declaration in tool.function_declarations or []:
            if not declaration.name:
                raise _AdapterFailure("InvalidFunctionDeclaration")
            if declaration.parameters_json_schema is not None:
                parameters = _schema_dict(declaration.parameters_json_schema)
            elif declaration.parameters is not None:
                parameters = _schema_dict(declaration.parameters)
            else:
                parameters = {"type": "object", "properties": {}}
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": parameters,
                    },
                }
            )
    return result


def _tool_choice(config: types.GenerateContentConfig | None) -> Any:
    calling = config.tool_config.function_calling_config if config and config.tool_config else None
    if calling is None or calling.mode in {
        None,
        types.FunctionCallingConfigMode.MODE_UNSPECIFIED,
        types.FunctionCallingConfigMode.AUTO,
        types.FunctionCallingConfigMode.VALIDATED,
    }:
        return "auto"
    if calling.mode == types.FunctionCallingConfigMode.NONE:
        return "none"
    allowed = calling.allowed_function_names or []
    if len(allowed) == 1:
        return {"type": "function", "function": {"name": allowed[0]}}
    return "required"


def _response_format(config: types.GenerateContentConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    schema = config.response_json_schema or config.response_schema
    if schema is not None:
        schema_dict = _schema_dict(schema)
        schema_name = str(schema_dict.get("title") or getattr(schema, "__name__", "response"))
        _enforce_strict_schema(schema_dict)
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema_dict},
        }
    if config.response_mime_type == "application/json":
        return {"type": "json_object"}
    return None


def _schema_dict(schema: Any) -> dict[str, Any]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        result = schema.model_json_schema()
    elif isinstance(schema, types.Schema):
        result = schema.model_dump(by_alias=True, exclude_none=True, mode="json")
    elif isinstance(schema, BaseModel):
        result = schema.__class__.model_json_schema()
    elif isinstance(schema, Mapping):
        result = copy.deepcopy(dict(schema))
    else:
        raise _AdapterFailure("UnsupportedResponseSchema")
    _normalize_schema(result)
    return result


def _normalize_schema(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _normalize_schema(item)
        return
    if not isinstance(value, dict):
        return
    if "defs" in value and "$defs" not in value:
        value["$defs"] = value.pop("defs")
    if "ref" in value and "$ref" not in value:
        value["$ref"] = str(value.pop("ref")).replace("#/defs/", "#/$defs/")
    value.pop("propertyOrdering", None)
    schema_type = value.get("type")
    if isinstance(schema_type, str):
        value["type"] = schema_type.lower()
    elif isinstance(schema_type, list):
        value["type"] = [str(item).lower() for item in schema_type]
    nullable = value.pop("nullable", False)
    if nullable and isinstance(value.get("type"), str):
        value["type"] = [value["type"], "null"]
    for child in value.values():
        _normalize_schema(child)


def _enforce_strict_schema(schema: dict[str, Any]) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        schema.clear()
        schema["$ref"] = reference
        return
    properties = schema.get("properties")
    schema_type = schema.get("type")
    if isinstance(properties, dict) and (
        schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)
    ):
        schema["additionalProperties"] = False
        schema["required"] = sorted(properties)
    for definitions_key in ("$defs", "definitions"):
        for child in schema.get(definitions_key, {}).values():
            if isinstance(child, dict):
                _enforce_strict_schema(child)
    for child in schema.get("properties", {}).values():
        if isinstance(child, dict):
            _enforce_strict_schema(child)
    for keyword in ("anyOf", "oneOf", "allOf"):
        for child in schema.get(keyword, []):
            if isinstance(child, dict):
                _enforce_strict_schema(child)
    items = schema.get("items")
    if isinstance(items, dict):
        _enforce_strict_schema(items)


def _copy_generation_options(payload: dict[str, Any], config: types.GenerateContentConfig) -> None:
    mappings = (
        ("max_tokens", config.max_output_tokens),
        ("temperature", config.temperature),
        ("top_p", config.top_p),
        ("stop", config.stop_sequences),
        ("presence_penalty", config.presence_penalty),
        ("frequency_penalty", config.frequency_penalty),
        ("seed", config.seed),
    )
    for name, value in mappings:
        if value is not None:
            payload[name] = value


def _to_llm_response(completion: Any) -> LlmResponse:
    choices = getattr(completion, "choices", None)
    if not choices:
        raise _AdapterFailure("InvalidGatewayResponse")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise _AdapterFailure("InvalidGatewayResponse")
    finish_reason = _map_finish_reason(getattr(choice, "finish_reason", None))
    output_limited = finish_reason == types.FinishReason.MAX_TOKENS
    parts: list[types.Part] = []
    reasoning = _reasoning_text(message)
    if reasoning:
        parts.append(types.Part(text=reasoning, thought=True))
    text = _response_text(getattr(message, "content", None))
    if text:
        parts.append(types.Part.from_text(text=text))
    # A length-limited response is never executable, even if a partial tool
    # call happens to parse. Keep the finish reason and usage instead of
    # misclassifying truncated arguments as an unaccounted adapter error.
    tool_calls = [] if output_limited else (getattr(message, "tool_calls", None) or [])
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        if function is None or not name:
            raise _AdapterFailure("InvalidGatewayToolCall")
        arguments = _tool_arguments(getattr(function, "arguments", None))
        part = types.Part.from_function_call(name=name, args=arguments)
        part.function_call.id = getattr(tool_call, "id", None) or ""
        parts.append(part)
    if not parts and not output_limited:
        raise _AdapterFailure("EmptyGatewayResponse")

    response = LlmResponse(
        model_version=getattr(completion, "model", None),
        content=types.Content(role="model", parts=parts),
        partial=False,
        finish_reason=finish_reason,
        usage_metadata=_usage_metadata(getattr(completion, "usage", None)),
    )
    if finish_reason is not None and finish_reason != types.FinishReason.STOP:
        response.error_code = finish_reason.value
        response.error_message = (
            "Maximum tokens reached"
            if finish_reason == types.FinishReason.MAX_TOKENS
            else f"Finished with {finish_reason.name}"
        )
    return response


def _reasoning_text(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None:
        value = getattr(message, "reasoning", None)
    if value is None and isinstance(getattr(message, "model_extra", None), dict):
        extra = message.model_extra
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
        result = json.loads(value)
    except json.JSONDecodeError:
        raise _AdapterFailure("InvalidGatewayToolArguments") from None
    if not isinstance(result, dict):
        raise _AdapterFailure("InvalidGatewayToolArguments")
    return result


def _usage_metadata(usage: Any) -> types.GenerateContentResponseUsageMetadata | None:
    if usage is None:
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=getattr(usage, "prompt_tokens", None),
        candidates_token_count=getattr(usage, "completion_tokens", None),
        total_token_count=getattr(usage, "total_tokens", None),
        cached_content_token_count=getattr(prompt_details, "cached_tokens", None),
        thoughts_token_count=getattr(completion_details, "reasoning_tokens", None),
    )


def _map_finish_reason(value: Any) -> types.FinishReason | None:
    if value is None:
        return None
    return _FINISH_REASON.get(str(value).lower(), types.FinishReason.OTHER)


def _safe_error_type(error: Exception) -> str:
    if isinstance(error, _AdapterFailure):
        return error.code
    name = type(error).__name__
    return name if _SAFE_ERROR_TYPE.fullmatch(name) else "UnknownProviderError"


def _retryable_gateway_error(error: Exception) -> bool:
    # Read only bounded scalar metadata; never retain response bodies or headers.
    if isinstance(error, APIStatusError):
        if isinstance(error.code, str) and error.code in {
            "insufficient_quota",
            "budget_exceeded",
            "context_length_exceeded",
        }:
            return False
        return error.status_code in {408, 409, 429} or 500 <= error.status_code < 600
    if isinstance(error, (APIConnectionError, TimeoutError)):
        return True
    if isinstance(error, _AdapterFailure):
        return error.code not in {"StreamingUnsupported"}
    return False
