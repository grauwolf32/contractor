import logging
from dataclasses import asdict, dataclass
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.tool_context import ToolContext

from .base import BaseCallback, CallbackTypes

logger = logging.getLogger(__name__)


@dataclass
class TokenCounter:
    input: int = 0
    output: int = 0
    total: int = 0

    def add(self, other: "TokenCounter") -> None:
        self.input += other.input
        self.output += other.output
        self.total += other.total

    def is_empty(self) -> bool:
        return all([self.input == 0, self.output == 0, self.total == 0])


class TokenUsageCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.after_model_callback
    deps: list[str] = []

    def __init__(self) -> None:
        self.counter = TokenCounter()
        self.invocation_id: str | None = None

    @classmethod
    def global_counter_key(cls):
        return f"::{cls.__name__}"

    @classmethod
    def global_history_key(cls):
        return f"::{cls.__name__}::history"

    @staticmethod
    def get_global_counter(ctx: ToolContext | CallbackContext) -> TokenCounter:
        ctx.state.setdefault(
            TokenUsageCallback.global_counter_key(), asdict(TokenCounter())
        )
        return TokenCounter(**ctx.state[TokenUsageCallback.global_counter_key()])

    @staticmethod
    def _update_global_counter(
        ctx: ToolContext | CallbackContext, counter: TokenCounter
    ) -> TokenCounter:
        global_counter = TokenUsageCallback.get_global_counter(ctx)
        global_counter.add(counter)
        ctx.state[TokenUsageCallback.global_counter_key()] = asdict(global_counter)
        return global_counter

    @staticmethod
    def get_history(ctx: ToolContext | CallbackContext) -> dict[str, Any]:
        ctx.state.setdefault(TokenUsageCallback.global_history_key(), {})
        return ctx.state[TokenUsageCallback.global_history_key()]

    def _update_history(
        self, ctx: ToolContext | CallbackContext, counter: TokenCounter
    ):
        history = TokenUsageCallback.get_history(ctx)
        if self.invocation_id is None:
            return history
        history[self.invocation_id] = asdict(counter)
        ctx.state[TokenUsageCallback.global_history_key()] = history
        return history

    def is_empty(self):
        return self.counter.is_empty() and self.invocation_id is None

    def to_state(self) -> dict[str, Any]:
        return {
            "counter": asdict(self.counter),
            "invocation_id": self.invocation_id,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        usage = getattr(llm_response, "usage_metadata", None)
        if usage is None:
            return

        token_count = TokenCounter(
            input=usage.prompt_token_count or 0,
            output=usage.candidates_token_count or 0,
            total=usage.total_token_count or 0,
        )

        self._update_global_counter(callback_context, token_count)

        invocation_id = self.get_invocation_id(callback_context)

        # invocation_id not adopted yet — pick it up from the context.
        if self.invocation_id is None and self.is_empty():
            self.invocation_id = invocation_id

        if invocation_id == self.invocation_id:
            # Same invocation_id -> keep accumulating the current counter.
            self.counter.add(token_count)
        else:
            # invocation_id changed -> start a fresh counter. The previous
            # invocation's totals are already in history (flushed below on
            # every call).
            self.invocation_id = invocation_id
            self.counter = token_count

        # Flush the in-progress invocation to history on every call. The
        # history is keyed by invocation_id, so this overwrites (never
        # double-counts) and keeps the *final* invocation's entry up to date —
        # there is no "invocation changed" event after the last response, so
        # waiting for the id to change would undercount by one invocation.
        self._update_history(callback_context, self.counter)

        self.save_to_state(callback_context)
        return
