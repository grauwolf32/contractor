"""Preserve domain arguments and logical identity at the ADK callable boundary."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from google.adk.tools import FunctionTool


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

    def _preprocess_args(self, args: dict[str, Any]) -> dict[str, Any]:
        original = {key: value for key, value in args.items() if key in self._integer_parameters}
        remaining = {key: value for key, value in args.items() if key not in original}
        return super()._preprocess_args(remaining) | original

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        logical_id = self._logical_invocation_id(tool_context)
        return await super().run_async(
            args=args, tool_context=_LogicalToolContext(tool_context, logical_id)
        )


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
