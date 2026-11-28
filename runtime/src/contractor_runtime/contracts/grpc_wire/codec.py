"""Preserve domain presence and JSON number types at the protobuf boundary.

The protobuf decoder owns binary parsing. This projection never uses protojson's
int64-as-string convention and never fills absent scalar fields with defaults.
Domain models retain the single definition of grammars, bounds and closed sets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from contractor_runtime.strict_json import strict_json_loads


class InvalidWireValue(ValueError):
    def __init__(self) -> None:
        super().__init__("private protobuf value violates its contract")


def encode(value: Mapping[str, Any], message: Message) -> Message:
    """Populate a new wire message without dropping unknown domain fields."""
    try:
        _encode(value, message)
    except (ValueError, TypeError, OverflowError, AttributeError, RecursionError):
        raise InvalidWireValue from None
    return message


def _encode(value: Any, message: Message) -> None:
    name = message.DESCRIPTOR.name
    message.SetInParent()
    if name.endswith("Document"):
        if not isinstance(value, Mapping):
            raise InvalidWireValue
        message.json = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        return
    if name == "MapOfString":
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
        ):
            raise InvalidWireValue
        message.values.update(value)
        return
    if name.startswith("Nullable"):
        if value is None:
            message.null = 0
        elif message.DESCRIPTOR.fields_by_name["value"].type == FieldDescriptor.TYPE_MESSAGE:
            _encode(value, message.value)
        else:
            field = message.DESCRIPTOR.fields_by_name["value"]
            message.value = _scalar(field, value)
        return
    if name.startswith("ListOf"):
        if not isinstance(value, list):
            raise InvalidWireValue
        descriptor = message.DESCRIPTOR.fields_by_name["values"]
        values = message.values
        if descriptor.type == FieldDescriptor.TYPE_MESSAGE:
            for item in value:
                _encode(item, values.add())
        else:
            values.extend(_scalar(descriptor, item) for item in value)
        return
    if not isinstance(value, Mapping):
        raise InvalidWireValue
    fields = {field.json_name: field for field in message.DESCRIPTOR.fields}
    for key, item in value.items():
        field = fields.get(key)
        if field is None:
            raise InvalidWireValue
        if field.containing_oneof and message.WhichOneof(field.containing_oneof.name) is not None:
            raise InvalidWireValue
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            _encode(item, getattr(message, field.name))
        else:
            setattr(message, field.name, _scalar(field, item))


def _scalar(field: FieldDescriptor, value: Any) -> Any:
    if field.type in {FieldDescriptor.TYPE_INT64, FieldDescriptor.TYPE_UINT64}:
        if type(value) is not int:
            raise InvalidWireValue
    elif field.type == FieldDescriptor.TYPE_DOUBLE:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise InvalidWireValue
    elif field.type == FieldDescriptor.TYPE_BOOL:
        if type(value) is not bool:
            raise InvalidWireValue
    elif field.type == FieldDescriptor.TYPE_STRING:
        if not isinstance(value, str):
            raise InvalidWireValue
    else:
        raise InvalidWireValue
    return value


def decode(message: Message) -> dict[str, Any]:
    """Reject unknown wire fields recursively, then project only present fields."""
    known = type(message)()
    known.CopyFrom(message)
    known.DiscardUnknownFields()
    if known != message:
        raise InvalidWireValue
    value = _decode(message)
    if not isinstance(value, dict):
        raise InvalidWireValue
    return value


def _decode(message: Message) -> Any:
    name = message.DESCRIPTOR.name
    if name.endswith("Document"):
        if not message.HasField("json"):
            raise InvalidWireValue
        try:
            value = strict_json_loads(message.json.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise InvalidWireValue from None
        if not isinstance(value, dict):
            raise InvalidWireValue
        return value
    if name == "MapOfString":
        return dict(message.values)
    if name.startswith("Nullable"):
        choice = message.WhichOneof("kind")
        if choice == "null" and message.null == 0:
            return None
        if choice == "value":
            if message.DESCRIPTOR.fields_by_name["value"].type == FieldDescriptor.TYPE_MESSAGE:
                return _decode(message.value)
            return _scalar(message.DESCRIPTOR.fields_by_name["value"], message.value)
        raise InvalidWireValue
    if name.startswith("ListOf"):
        field = message.DESCRIPTOR.fields_by_name["values"]
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            return [_decode(item) for item in message.values]
        return [_scalar(field, value) for value in message.values]
    return {
        field.json_name: _decode(value)
        if field.type == FieldDescriptor.TYPE_MESSAGE
        else _scalar(field, value)
        for field, value in message.ListFields()
    }
