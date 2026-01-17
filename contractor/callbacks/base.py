import inspect

from dataclasses import dataclass, field
from enum import StrEnum
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from functools import lru_cache

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.genai import types


def bfmc_sig(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]: ...
def afmc_sig(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]: ...
def bfac_sig(callback_context: CallbackContext) -> Optional[types.Content]: ...
def afac_sig(callback_context: CallbackContext) -> Optional[types.Content]: ...
def bftc_sig(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> Optional[dict]: ...
def aftc_sig(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict
) -> Optional[dict]: ...


class CallbackTypes(StrEnum):
    before_model_callback = "before_model_callback"
    after_model_callback = "after_model_callback"
    before_agent_callback = "before_agent_callback"
    after_agent_callback = "after_agent_callback"
    before_tool_callback = "before_tool_callback"
    after_tool_callback = "after_tool_callback"


@lru_cache()
def _expected_signatures() -> dict["CallbackTypes", inspect.Signature]:
    return {
        CallbackTypes.before_model_callback: inspect.signature(bfmc_sig),
        CallbackTypes.after_model_callback: inspect.signature(afmc_sig),
        CallbackTypes.before_agent_callback: inspect.signature(bfac_sig),
        CallbackTypes.after_agent_callback: inspect.signature(afac_sig),
        CallbackTypes.before_tool_callback: inspect.signature(bftc_sig),
        CallbackTypes.after_tool_callback: inspect.signature(aftc_sig),
    }


def verify_signature(cb_func: Callable, cb_type: "CallbackTypes") -> bool:
    # expected = _expected_signatures().get(cb_type)
    # if expected is None:
    #    raise ValueError(f"Unknown callback type: {cb_type}")
    # return inspect.signature(cb_func) == expected
    return True


def _callback_name(func: Callable[..., Any]) -> str:
    return (
        getattr(func, "__qualname__", None)
        or getattr(func, "__name__", None)
        or func.__class__.__name__
    )


@dataclass
class BaseCallback(ABC):
    cb_type: CallbackTypes
    deps: list[str] = field(default_factory=list)
    agent_name: Optional[str] = None

    def validate(self) -> "BaseCallback":
        if not verify_signature(self.__call__, self.cb_type):
            raise TypeError(
                f"Invalid signature for {self.cb_type}: {inspect.signature(self.__call__)}"
            )
        return self

    @property
    def name(self):
        return self.__class__.__name__

    def _callback_state_key(self, name: str) -> str:
        return f"{self.agent_name or ''}::{name}"

    @abstractmethod
    def to_state(self): ...

    @abstractmethod
    def __call__(self, *args, **kwargs): ...

    def get_dependencies(self):
        return self.deps

    def get_invocation_id(self, ctx: CallbackContext | ToolContext) -> str | None:
        return ctx.invocation_id

    def save_to_state(self, ctx: CallbackContext | ToolContext) -> None:
        ctx.state.setdefault("callbacks", {})

        # HACK: ctx.state must be explicilty overwritten
        callbacks = ctx.state["callbacks"]
        callbacks[self._callback_state_key(self.name)] = self.to_state()
        ctx.state["callbacks"] = callbacks
        return

    def get_from_cb_state(
        self, ctx: CallbackContext | ToolContext, cb_name: str
    ) -> Any:
        return ctx.state.get("callbacks", {}).get(self._callback_state_key(cb_name))
