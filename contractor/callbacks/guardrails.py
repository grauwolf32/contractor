import logging
from typing import Any, Optional, Callable, Literal, Final
from google.genai import types
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool

from .tokens import TokenUsageCallback
from .base import BaseCallback, CallbackTypes


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOKEN_USAGE_CALLBACK_NAME = TokenUsageCallback().name
TOKEN_BUDGET_DEFAULT_MESSAGE: Final[str] = (
    "I have reached the maximum thinking budget. I must stop."
)
TOOL_LIMIT_DEFAULT_RVALUE: dict[str, Any] = {"result": "Tool call limit reached."}

ADK_RESERVED_TOOLS: list[str] = ["transfer_to_agent"]


def _format_llm_response(
    role: Literal["system", "user"],
    message: str,
    finish_reason: Optional[types.FinishReason] = None,
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
        message: Optional[str] = None,
    ):
        self.token_budget = token_budget
        self.token_budget_key = token_budget_key
        self.token_count: int | None = None
        self.message = TOKEN_BUDGET_DEFAULT_MESSAGE
        if message:
            self.message = message

        assert token_budget_key in {"input", "output", "total"}

    def to_state(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "token_budget_key": self.token_budget_key,
            "token_count": self.token_count or 0,
            "message": self.message,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        token_usage_stat = (
            self.get_from_cb_state(callback_context, TOKEN_USAGE_CALLBACK_NAME) or {}
        )
        token_count = token_usage_stat.get("counter", {}).get(self.tpm_limit_key, 0)
        self.token_count = token_count

        if token_count > self.token_budget:
            return _format_llm_response(
                "system", self.message, types.FinishReason.MAX_TOKENS
            )

        self.save_to_state(callback_context)
        return


class ToolMaxCallsGuardrailCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_tool_callback
    deps: list[str] = []

    def __init__(self, max_calls: int, tool_name: str, rvalue: Optional[dict]):
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
        return ".".join(self.__class__.__name__, self.tool_name)

    def __call__(
        self, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict]:
        if tool.name != self.tool_name:
            return
        self.call_count += 1
        self.save_to_state(tool_context)

        if self.call_count > self.max_calls:
            return self.rvalue
        return


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
        assert default_tool_name in self.tool_names

    def to_state(self):
        return {
            "default_tool_name": self.default_tool_name,
            "tool_names": self.tool_names,
            "history": self.history,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if not llm_response.content or not llm_response.content.parts:
            return

        parts: list[types.Part] = []
        for part in llm_response.content.parts:
            if part.function_call is None:
                parts.append(part)
                continue

            func_name = part.function_call.name
            if func_name in self.tool_names:
                parts.append(part)
                continue

            part.function_call.name = self.default_tool_name
            metadata = {
                "func_name": func_name,
                "func_args": part.function_call.args or {},
            }

            part.function_call.args = {
                self.default_tool_arg: metadata,
            }

            parts.append(part)
            self.history.append(metadata)

        llm_response.content.parts = parts
        self.save_to_state(callback_context)
        return llm_response
