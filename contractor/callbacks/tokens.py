import logging
from typing import Optional, Any
from dataclasses import dataclass, asdict
from google.adk.models import LlmResponse
from google.adk.agents.callback_context import CallbackContext

from .base import BaseCallback, CallbackTypes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


class TokenUsageCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.after_model_callback
    deps: list[str] = []

    def __init__(self) -> None:
        self.common = TokenCounter()
        self.current = TokenCounter()
        self.interaction_id: str | None = None
        self.history: dict[str, Any] = {}

    def to_state(self) -> dict[str, Any]:
        return {
            "common": asdict(self.common),
            "current": asdict(self.current),
            "interaction_id": self.interaction_id,
            "history": self.history,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        token_count = TokenCounter(
            input=llm_response.usage_metadata.prompt_token_count,
            output=llm_response.usage_metadata.candidates_token_count,
            total=llm_response.usage_metadata.total_token_count,
        )

        self.common.add(token_count)

        interaction_id = llm_response.interaction_id

        # если interaction_id ещё не задан — пытаемся подхватить из ответа
        if self.interaction_id is None:
            self.interaction_id = interaction_id

        # если всё ещё None (resp_id тоже None) — current/history не трогаем
        if self.interaction_id is None:
            self.save_to_state(callback_context)
            return None

        # тот же interaction_id -> копим current
        if interaction_id == self.interaction_id:
            self.current.add(token_count)
            self.save_to_state(callback_context)
            return None

        # смена interaction_id -> сохраняем прошлый current и начинаем новый
        self.history[self.interaction_id] = asdict(self.current)
        self.interaction_id = interaction_id
        self.current = token_count

        self.save_to_state(callback_context)
        return None
