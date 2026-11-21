"""Preserve domain arguments and logical identity at the ADK callable boundary."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from google.adk.tools import FunctionTool
from google.genai import types

from contractor_runtime.core.input_errors import ToolInputError


class _LogicalToolContext:
    __slots__ = ("_context", "_logical_id")

    def __init__(self, context: Any, logical_id: str) -> None:
        self._context = context
        self._logical_id = logical_id

    @property
    def invocation_id(self) -> str:
        return self._logical_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)


class ContractorFunctionTool(FunctionTool):
    """Keep callable-owned validation ahead of SDK numeric coercion.

    ADK's documented subclass preprocessing hook still handles structured
    objects. Only the original int/list[int] inputs bypass its implicit casts;
    their owning callable or raw guard decides whether those values are valid.
    Native callbacks keep the SDK context, while business functions receive
    the same context surface with their logical Contractor invocation ID.
    """

    def __init__(
        self, func: Callable[..., Any], *, logical_invocation_id: Callable[[Any], str]
    ) -> None:
        super().__init__(func)
        self._integer_parameters = _integer_parameters(func)
        self._logical_invocation_id = logical_invocation_id

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        declaration = super()._get_declaration()
        if declaration is None:
            return None
        schema = declaration.parameters_json_schema
        if schema is None and declaration.parameters is not None:
            schema = declaration.parameters.model_dump(by_alias=True, exclude_none=True)
            declaration.parameters = None
        if schema is not None:
            _close_records(schema)
            declaration.parameters_json_schema = schema
        return declaration

    def contractor_raw_argument_error(self, args: dict[str, Any]) -> ToolInputError | None:
        declaration = self._get_declaration()
        if declaration is None:
            raise RuntimeError("selected callable has no declaration")
        schema = declaration.parameters_json_schema or {"additionalProperties": False}
        path = _unknown_field(args, schema, schema, "arguments")
        if path is not None:
            return ToolInputError(
                f"{path} contains an undeclared field; use only declared arguments"
            )
        return None

    def _preprocess_args(self, args: dict[str, Any]) -> dict[str, Any]:
        original = {key: value for key, value in args.items() if key in self._integer_parameters}
        remaining = {key: value for key, value in args.items() if key not in original}
        return super()._preprocess_args(remaining) | original

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        logical_id = self._logical_invocation_id(tool_context)
        rejection = self.contractor_raw_argument_error(args)
        if rejection is not None:
            raise rejection
        return await super().run_async(
            args=args, tool_context=_LogicalToolContext(tool_context, logical_id)
        )


def _close_records(schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" and "properties" in schema:
        schema.setdefault("additionalProperties", False)
    for value in schema.values():
        if isinstance(value, dict):
            _close_records(value)
        elif isinstance(value, list):
            for item in value:
                _close_records(item)


def _unknown_field(value: Any, schema: Any, root: dict[str, Any], path: str) -> str | None:
    if not isinstance(schema, dict):
        return None
    if reference := schema.get("$ref"):
        target: Any = root
        for component in reference.removeprefix("#/").split("/"):
            target = target[component.replace("~1", "/").replace("~0", "~")]
        return _unknown_field(value, target, root, path)
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            # Only object/array variants can contain unknown fields. A null
            # variant must not let a supplied structured value bypass checks.
            if not isinstance(value, dict | list):
                return None
            expected_type = "object" if isinstance(value, dict) else "array"
            variants = [
                option for option in schema[keyword] if option.get("type") in (None, expected_type)
            ]
            errors = [_unknown_field(value, option, root, path) for option in variants]
            return next(iter(errors), None) if all(errors) else None
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key not in properties and additional is False:
                return path
            child = properties.get(key, additional)
            # Paths use declaration-owned property names, never supplied keys.
            child_path = f"{path}.{key}" if key in properties else path
            if error := _unknown_field(item, child, root, child_path):
                return error
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if error := _unknown_field(item, schema.get("items"), root, f"{path}[{index}]"):
                return error
    return None


def _integer_parameters(func: Callable[..., Any]) -> frozenset[str]:
    signature = inspect.signature(func)
    target = func
    while isinstance(target, functools.partial):
        target = target.func
    if not inspect.isfunction(target) and not inspect.ismethod(target):
        target = target.__call__
    # The normal get_type_hints mode strips Annotated even inside list types,
    # matching the SDK's own annotation resolution without another global cache.
    hints = get_type_hints(target)
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if _integer_parameter(hints.get(name, parameter.annotation))
    )


def _integer_parameter(annotation: Any) -> bool:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in (Union, UnionType):
        choices = [value for value in get_args(annotation) if value is not type(None)]
        if len(choices) != 1:
            return False
        return _integer_parameter(choices[0])
    return annotation is int or (get_origin(annotation) is list and get_args(annotation) == (int,))
