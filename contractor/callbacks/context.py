import time
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from .tokens import TokenUsageCallback
from .base import BaseCallback, CallbackTypes


TOKEN_USAGE_CALLBACK_NAME = TokenUsageCallback().name


class SummarizationLimitCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: [TOKEN_USAGE_CALLBACK_NAME]

    def __init__(
        self,
        ctx_length: int,
        message: str,
        max_usage: Optional[float],
        max_tokens: Optional[int],
        summarization_key: str = "total",
    ):
        self.ctx_length = ctx_length
        self.max_tokens = max_tokens
        self.message = message
        self.token_count: int = 0
        self.history: list[Any] = []

        assert any((max_tokens, max_usage))
        if max_usage:
            self.max_tokens = int(max_usage * ctx_length) + 1

    def to_state(self) -> dict[str, Any]:
        return {
            "ctx_length": self.ctx_length,
            "max_tokens": self.max_tokens,
            "token_count": self.token_count,
            "message": self.message,
            "history": self.history,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        token_usage_stat = (
            self.get_from_cb_state(callback_context, TOKEN_USAGE_CALLBACK_NAME) or {}
        )
        token_count = token_usage_stat.get("current", {}).get(self.tpm_limit_key, 0)
        self.token_count = token_count

        if token_count < self.max_tokens:
            self.save_to_state(callback_context)
            return

        llm_request.contents.append(
            types.Content(role="user", parts=[types.Part(text=self.message)])
        )

        self.history.append(int(time.time()))
        self.save_to_state(callback_context)
        return


class FunctionResultsRemovalCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = []

    def __init__(self, keep_last_n: int):
        self.keep_last_n = keep_last_n
        self.counter = 0
        assert keep_last_n > 1

    def to_state(self) -> dict[str, Any]:
        return {
            "keep_last_n": self.keep_last_n,
            "counter": self.counter,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        if not llm_request.content or not llm_request.content.parts:
            return

        parts: types.Part = []
        func_count: int = 0
        for part in llm_request.content.parts[::-1]:
            if part.function_response is None:
                parts.append(part)
                continue
            func_count += 1
            if func_count < self.keep_last_n:
                continue

            self.counter += 1
            part.function_response.parts = None
            part.function_response.response = {}
            parts.append(part)

        parts = parts[::-1]
        llm_request.content.parts = parts
        self.save_to_state(callback_context)
        return
