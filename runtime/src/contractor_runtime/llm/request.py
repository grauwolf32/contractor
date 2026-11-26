"""Pure OpenAI-compatible request encoding shared by model transport and admission."""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from contractor_runtime.llm.errors import _AdapterFailure
from contractor_runtime.llm.image_parts import (
    image_context_size,
    image_response_metadata,
    response_images,
)


def completion_request(
    model: str, request: LlmRequest, *, allow_images: bool = False
) -> dict[str, Any]:
    try:
        image_count, _ = image_context_size(request.contents or [])
    except ValueError:
        raise _AdapterFailure("UnsupportedContentPart") from None
    if image_count and not allow_images:
        raise _AdapterFailure("UnsupportedContentPart")
    config = request.config
    messages: list[dict[str, Any]] = []
    if config is not None and config.system_instruction is not None:
        instruction = _instruction_text(config.system_instruction)
        if instruction:
            messages.append({"role": "system", "content": instruction})
    for content in request.contents or []:
        messages.extend(_content_messages(content))
    messages = _complete_tool_call_messages(messages)
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


def _complete_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair interrupted calls without claiming their external effects were rolled back."""

    completed: list[dict[str, Any]] = []
    pending: list[str] = []
    images: list[dict[str, Any]] = []
    skipped = _json_text({"ok": False, "error": {"code": "tool_call_outcome_unknown"}})

    def flush_images() -> None:
        if images:
            completed.append({"role": "user", "content": list(images)})
            images.clear()

    def settle_pending() -> None:
        for call_id in pending:
            completed.append({"role": "tool", "tool_call_id": call_id, "content": skipped})
        pending.clear()
        flush_images()

    for source in messages:
        message = dict(source)
        attached = message.pop("_artifact_images", [])
        if message["role"] == "tool" and message.get("tool_call_id") in pending:
            pending.remove(message["tool_call_id"])
        else:
            settle_pending()
        completed.append(message)
        images.extend(attached)
        if message["role"] == "assistant":
            pending = [call["id"] for call in message.get("tool_calls", [])]
        if not pending:
            flush_images()
    settle_pending()
    return completed


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
            attached = list(response_images(response))
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": response.id or "",
                "content": _json_text(
                    image_response_metadata(response.response, attached=bool(attached))
                ),
            }
            if attached:
                images = []
                for mime_type, data in attached:
                    encoded = base64.b64encode(data).decode("ascii")
                    images.extend(
                        [
                            {
                                "type": "text",
                                "text": f"Image returned by tool call {response.id or ''}.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                            },
                        ]
                    )
                tool_message["_artifact_images"] = images
            tool_messages.append(tool_message)
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
        # Earlier reasoning is not replayed. Qwen-style templates expect history
        # without it, and LM Studio renders a replayed reasoning_content as a
        # second <think> block inside the content: the model then copies that
        # paragraph turn after turn and loops. A reasoning-only turn keeps an
        # empty message so the role order is unchanged.
        ordinary = {"role": "assistant", "content": joined_text or (None if tool_calls else "")}
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
            if parameters.get("type") == "object":
                parameters.setdefault("properties", {})
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
