"""OTel GenAI v1.41.1 content projection, independent of the trace collector."""

from __future__ import annotations

import json
from typing import Any

from contractor_runtime.adapters.content import MAX_CONTENT_BYTES


def usage_attributes(usage: Any | None) -> dict[str, int]:
    """Project known per-call token counts; missing usage is never reported as zero."""
    result = {}
    for source, target in (
        ("prompt_token_count", "gen_ai.usage.input_tokens"),
        ("candidates_token_count", "gen_ai.usage.output_tokens"),
        ("cached_content_token_count", "gen_ai.usage.cache_read.input_tokens"),
    ):
        value = getattr(usage, source, None)
        if type(value) is int and value >= 0:
            result[target] = value
    return result


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "content": value}]
    if isinstance(value, dict) and value.get("truncated") is True:
        return [{"type": "text", "content": value.get("preview", ""), "truncated": True}]
    if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
        return [{"type": "text", "content": _text(value)}]
    result = []
    for part in value["parts"]:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            result.append(
                {
                    "type": "reasoning" if part.get("thought") is True else "text",
                    "content": part["text"],
                }
            )
        elif isinstance(call := part.get("functionCall"), dict):
            item = {
                "type": "tool_call",
                "name": call.get("name", ""),
                "arguments": call.get("args"),
            }
            if isinstance(call.get("id"), str):
                item["id"] = call["id"]
            result.append(item)
        elif isinstance(response := part.get("functionResponse"), dict):
            item = {"type": "tool_call_response", "response": response.get("response")}
            if isinstance(response.get("id"), str):
                item["id"] = response["id"]
            result.append(item)
        else:
            # Preserve unsupported modalities without claiming to understand their schema.
            result.append({"type": "provider_specific", "value": part})
    return result


def _message(value: Any, default_role: str) -> dict[str, Any]:
    role = value.get("role", default_role) if isinstance(value, dict) else default_role
    if not isinstance(role, str):
        role = default_role
    if role == "model":
        role = "assistant"
    parts = _parts(value)
    if parts and all(part["type"] == "tool_call_response" for part in parts):
        role = "tool"
    return {"role": role, "parts": parts}


def _bounded(key: str, value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode()) <= MAX_CONTENT_BYTES:
        return encoded
    preview = encoded.encode()[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
    # Keep the standard array shape even when a large field needs a preview.
    while True:
        part = {"type": "text", "content": preview, "truncated": True}
        if key == "gen_ai.system_instructions":
            bounded = [part]
        elif key == "gen_ai.tool.definitions":
            bounded = [{"type": "truncated", "name": "truncated", "preview": preview}]
        else:
            message = {
                "role": "assistant" if key == "gen_ai.output.messages" else "user",
                "parts": [part],
            }
            if key == "gen_ai.output.messages":
                message["finish_reason"] = "unknown"
            bounded = [message]
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode()) <= MAX_CONTENT_BYTES:
            return encoded
        excess = len(encoded.encode()) - MAX_CONTENT_BYTES + 16
        preview = preview.encode()[: max(0, len(preview.encode()) - excess)].decode(
            "utf-8", errors="ignore"
        )


def content_attributes(field: str, raw: str, *, model: bool) -> dict[str, str]:
    """Project one already bounded, opted-in SDK payload into collector-neutral attributes."""
    if not model:
        return {"contractor." + field: raw}
    value = json.loads(raw)
    if value is None:
        return {}
    if field == "output":
        values = {
            "gen_ai.output.messages": [{**_message(value, "assistant"), "finish_reason": "unknown"}]
        }
    elif isinstance(value, dict) and "contents" in value:
        contents = value["contents"]
        if contents is not None and not isinstance(contents, list):
            return {}
        values = {"gen_ai.input.messages": [_message(item, "user") for item in contents or []]}
        if value.get("systemInstruction") is not None:
            values["gen_ai.system_instructions"] = _parts(value["systemInstruction"])
        tools = []
        for tool in value.get("tools") or []:
            for declaration in tool.get("functionDeclarations", []):
                item = {"type": "function", "name": declaration["name"]}
                if declaration.get("description") is not None:
                    item["description"] = declaration["description"]
                parameters = declaration.get("parametersJsonSchema", declaration.get("parameters"))
                if parameters is not None:
                    item["parameters"] = parameters
                tools.append(item)
        if tools:
            values["gen_ai.tool.definitions"] = tools
    else:
        values = {"gen_ai.input.messages": [_message(value, "user")]}
    return {key: _bounded(key, item) for key, item in values.items()}
