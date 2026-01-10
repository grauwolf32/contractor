import logging
from typing import Any
from dataclasses import dataclass, asdict
from google.adk.models import LlmResponse
from google.adk.agents.callback_context import CallbackContext

from .base import BaseCallback, CallbackTypes

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


class TokenUsageCallbackException(Exception):
    def __init__(self) -> None:
        super().__init__("token usage callback not found!")


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
        self.common = TokenCounter()
        self.current = TokenCounter()
        self.invocation_id: str | None = None
        self.history: dict[str, Any] = {}

    def is_empty(self):
        return self.common.is_empty() and self.current.is_empty()

    def to_state(self) -> dict[str, Any]:
        return {
            "common": asdict(self.common),
            "current": asdict(self.current),
            "invocation_id": self.invocation_id,
            "history": self.history,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        token_count = TokenCounter(
            input=llm_response.usage_metadata.prompt_token_count,
            output=llm_response.usage_metadata.candidates_token_count,
            total=llm_response.usage_metadata.total_token_count,
        )

        self.common.add(token_count)
        invocation_id = self.get_invocation_id(callback_context)

        # если invocation_id ещё не задан — пытаемся подхватить из ответа
        if self.invocation_id is None and self.is_empty():
            self.invocation_id = invocation_id

        # тот же invocation_id -> копим current
        if invocation_id == self.invocation_id:
            self.current.add(token_count)
            self.save_to_state(callback_context)
            return None

        # смена invocation_id -> сохраняем прошлый current и начинаем новый
        if self.invocation_id is not None:
            self.history[self.invocation_id] = asdict(self.current)

        self.invocation_id = invocation_id
        self.current = token_count

        self.save_to_state(callback_context)
        return None
