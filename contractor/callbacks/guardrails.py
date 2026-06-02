import json
import logging
from collections.abc import Callable
from typing import Any, Final, Literal

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .base import BaseCallback, CallbackTypes
from .tokens import TokenUsageCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOKEN_USAGE_CALLBACK_NAME = TokenUsageCallback().name
TOKEN_BUDGET_DEFAULT_MESSAGE: Final[str] = (
    "I have reached the maximum thinking budget. I must stop."
)
TOOL_LIMIT_DEFAULT_RVALUE: dict[str, Any] = {"result": "Tool call limit reached."}

ADK_RESERVED_TOOLS: list[str] = ["transfer_to_agent"]

TOOL_MALFORMED_FORMAT: Final[str] = (
    "Tool call {name} has malformed or wrong format. Please, try again."
)

REPEATED_TOOL_CALL_DEFAULT_MESSAGE: Final[str] = (
    "You have called {tool_name} with the same arguments {count} times in a row. "
    "This is not making progress — try a different tool, different arguments, "
    "or stop and reconsider your approach."
)


def _format_llm_response(
    role: Literal["system", "user"],
    message: str,
    finish_reason: types.FinishReason | None = None,
):
    return LlmResponse(
        content=types.Content(
            role=role,
            parts=[types.Part(text=message)],
        ),
        finish_reason=finish_reason,
    )


class ThinkingBudgetGuardrailCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = [TOKEN_USAGE_CALLBACK_NAME]

    def __init__(
        self,
        token_budget: int,
        token_budget_key: str = "total",
        message: str | None = None,
    ):
        self.token_budget = token_budget
        self.token_budget_key = token_budget_key
        self.token_count: int | None = None
        self.message = TOKEN_BUDGET_DEFAULT_MESSAGE
        if message:
            self.message = message

        if token_budget_key not in {"input", "output", "total"}:
            raise ValueError(
                f"token_budget_key must be one of input/output/total, "
                f"got {token_budget_key!r}"
            )

    def to_state(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "token_budget_key": self.token_budget_key,
            "token_count": self.token_count or 0,
            "message": self.message,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        token_usage_stat = (
            self.get_from_cb_state(callback_context, TOKEN_USAGE_CALLBACK_NAME) or {}
        )
        token_count = token_usage_stat.get("counter", {}).get(self.token_budget_key, 0)
        self.token_count = token_count

        if token_count > self.token_budget:
            self.save_to_state(callback_context)
            return _format_llm_response(
                "system", self.message, types.FinishReason.MAX_TOKENS
            )

        self.save_to_state(callback_context)
        return None


class ToolMaxCallsGuardrailCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_tool_callback
    deps: list[str] = []

    def __init__(self, max_calls: int, tool_name: str, rvalue: dict | None):
        self.max_calls = max_calls
        self.tool_name = tool_name
        self.rvalue = TOOL_LIMIT_DEFAULT_RVALUE
        self.call_count: int = 0

        if rvalue:
            self.rvalue = rvalue

    def to_state(self):
        return {
            "max_calls": self.max_calls,
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "rvalue": self.rvalue,
        }

    @property
    def name(self):
        return ".".join((self.__class__.__name__, self.tool_name))

    def __call__(
        self, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        if tool.name != self.tool_name:
            return None
        self.call_count += 1
        self.save_to_state(tool_context)

        if self.call_count > self.max_calls:
            return self.rvalue
        return None


class InvalidToolCallGuardrailCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.after_model_callback
    deps: list[str] = []

    def __init__(
        self,
        tools: list[Callable],
        default_tool_name: str,
        default_tool_arg: str = "metadata",
    ):
        self.default_tool_name = default_tool_name
        self.default_tool_arg = default_tool_arg
        self.tool_names = {
            getattr(tool, "name", None)
            or getattr(tool, "__name__", None)
            or tool.__class__.__name__
            for tool in tools
        }
        self.tool_names.update(ADK_RESERVED_TOOLS)

        self.history: list[Any] = []
        if default_tool_name not in self.tool_names:
            raise ValueError(
                f"default_tool_name {default_tool_name!r} is not among the "
                f"provided tools: {sorted(self.tool_names)}"
            )

    def to_state(self):
        return {
            "default_tool_name": self.default_tool_name,
            "tool_names": sorted(self.tool_names),
            "history": self.history,
        }

    def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        content = llm_response.content
        if not content or not content.parts:
            return None

        new_parts: list[types.Part] = []

        for part in content.parts:
            fc = part.function_call

            if fc is None:
                new_parts.append(part)
                continue

            func_name = fc.name
            func_args = fc.args

            if func_name in self.tool_names and isinstance(func_args, dict):
                new_parts.append(part)
                continue

            metadata: dict[str, Any] = {}

            if func_name not in self.tool_names:
                metadata.update(
                    {
                        "func_name": func_name,
                        "func_args": func_args or {},
                    }
                )

            if not isinstance(func_args, dict):
                metadata["error"] = TOOL_MALFORMED_FORMAT.format(name=func_name)

            fc.name = self.default_tool_name
            fc.args = {
                self.default_tool_arg: metadata,
            }

            new_parts.append(part)
            self.history.append(metadata)

        content.parts = new_parts
        self.save_to_state(callback_context)

        return llm_response


MANDATORY_TOOL_DEFAULT_MESSAGE: Final[str] = (
    "You MUST call {tool_name} before finishing. "
    "You have used {step_count} tool calls so far. "
    "Call {tool_name} now with your verdict."
)


class MandatoryToolCallback(BaseCallback):
    """Intercepts the model's final response and redirects it if a
    required tool hasn't been called yet.

    Uses ``after_model_callback``: when the model produces a text-only
    response (no function_call parts) and the mandatory tool hasn't
    been observed, replaces the response with a nudge message.
    """

    cb_type: CallbackTypes = CallbackTypes.after_model_callback
    deps: list[str] = []

    def __init__(
        self,
        tool_names: list[str],
        message: str | None = None,
        max_nudges: int = 2,
    ):
        self.tool_names = set(tool_names)
        self.message_template = message or MANDATORY_TOOL_DEFAULT_MESSAGE
        self.called: set[str] = set()
        self.step_count: int = 0
        self.nudge_count: int = 0
        self.max_nudges = max_nudges

    def to_state(self) -> dict[str, Any]:
        return {
            "tool_names": sorted(self.tool_names),
            "called": sorted(self.called),
            "step_count": self.step_count,
            "nudge_count": self.nudge_count,
        }

    def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        content = llm_response.content
        if not content or not content.parts:
            return None

        has_function_call = any(
            getattr(p, "function_call", None) is not None
            for p in content.parts
        )

        if has_function_call:
            for part in content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name in self.tool_names:
                    self.called.add(fc.name)
            self.step_count += 1
            self.save_to_state(callback_context)
            return None

        missing = self.tool_names - self.called
        if not missing:
            self.save_to_state(callback_context)
            return None

        if self.nudge_count >= self.max_nudges:
            self.save_to_state(callback_context)
            return None

        self.nudge_count += 1
        self.save_to_state(callback_context)

        tool_name = sorted(missing)[0]
        msg = self.message_template.format(
            tool_name=tool_name,
            step_count=self.step_count,
        )
        return _format_llm_response("user", msg)


class RepeatedToolCallCallback(BaseCallback):
    """Detects when the agent calls the same tool with the same args repeatedly.

    On the threshold-th identical consecutive call, returns an advisory dict
    instead of executing the tool. Subsequent identical calls keep returning
    the advisory until the agent breaks the streak with a different call.

    Tools invoked without arguments (e.g. ``execute_current_subtask``, ``skip``)
    are passed through unchanged: they do not advance the streak, do not break
    it, and never trigger the advisory. Repetition guarantees for those tools
    must be enforced inside the tool implementation.
    """

    cb_type: CallbackTypes = CallbackTypes.before_tool_callback
    deps: list[str] = []

    def __init__(self, threshold: int = 5, message: str | None = None):
        if threshold <= 1:
            raise ValueError(f"threshold must be > 1, got {threshold}")
        self.threshold = threshold
        self.message_template = message or REPEATED_TOOL_CALL_DEFAULT_MESSAGE
        self.last_signature: str | None = None
        self.run_length: int = 0
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _signature(tool_name: str, args: dict[str, Any]) -> str:
        try:
            payload = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = repr(args)
        return f"{tool_name}::{payload}"

    def to_state(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "last_signature": self.last_signature,
            "run_length": self.run_length,
            "history": self.history,
        }

    def __call__(
        self, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        if not args:
            return None

        sig = self._signature(tool.name, args)

        if sig == self.last_signature:
            self.run_length += 1
        else:
            self.last_signature = sig
            self.run_length = 1

        if self.run_length < self.threshold:
            self.save_to_state(tool_context)
            return None

        if self.run_length == self.threshold:
            self.history.append(
                {
                    "tool_name": tool.name,
                    "args": args,
                    "count": self.run_length,
                }
            )

        self.save_to_state(tool_context)
        return {
            "warning": self.message_template.format(
                tool_name=tool.name, count=self.run_length
            )
        }
