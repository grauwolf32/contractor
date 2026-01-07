from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from .base import BaseCallback, CallbackTypes


class CallbackDependencyException(Exception):
    def __init__(self, cb_name: str, cb_list: list[str]):
        ",".join(cb_list)
        super().__init__("Callback {cb_name} depends on {dep_list}")


class CallbackAlreadyExistsException(Exception):
    def __init__(self, cb_name: str):
        super().__init__("Callback {cb_name} already exists in middleware")


@dataclass(slots=True)
class CallbackChain:
    cb_type: CallbackTypes
    funcs: list[BaseCallback] = field(default_factory=list)
    calls: list[Any] = field(default_factory=list)

    def register(self, func: BaseCallback) -> None:
        self.funcs.append(func.validate())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.clear()
        result: Any = None
        for func in self.funcs:
            result = func(*args, **kwargs)
            self.calls.append(result)
        return result

    def as_names(self) -> list[str]:
        return [f.name for f in self.funcs]


@dataclass(slots=True)
class CallbackMiddleware:
    """
    Middleware хранит цепочки callback'ов по типам.
    """

    chains: dict[CallbackTypes, CallbackChain] = field(default_factory=dict)
    registry: dict[str, BaseCallback] = field(default_factory=dict)

    def register(self, func: BaseCallback) -> "CallbackMiddleware":
        if func.name in self.registry:
            raise

        missing = list(set(func.deps) - set(self.registry.keys()))
        if missing:
            raise CallbackDependencyException(func.name, missing)

        cb_type = func.cb_type
        chain = self.chains.get(cb_type)
        if chain is None:
            chain = CallbackChain(cb_type=cb_type)
            self.chains[cb_type] = chain

        chain.register(func)
        self.registry[func.name] = func

        return self

    def get_chain(self, cb_type: CallbackTypes) -> CallbackChain:
        chain = self.chains.get(cb_type)
        if chain is None:
            chain = CallbackChain(cb_type=cb_type)
            self.chains[cb_type] = chain
        return chain

    def __call__(self) -> dict[str, Callable[..., Any]]:
        return {cb_type.value: chain for cb_type, chain in self.chains.items()}
