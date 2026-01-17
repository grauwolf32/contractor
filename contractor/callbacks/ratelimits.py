import time
import logging
from typing import Any
from google.adk.models import LlmRequest
from google.adk.agents.callback_context import CallbackContext
from .tokens import TokenUsageCallback, TokenCounter
from .base import BaseCallback, CallbackTypes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOKEN_USAGE_CALLBACK_NAME = TokenUsageCallback().name


class TpmRatelimitCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = [TOKEN_USAGE_CALLBACK_NAME]

    def __init__(self, tpm_limit: int, tpm_limit_key="input"):
        self.tpm_limit = tpm_limit
        self.tpm_limit_key = tpm_limit_key
        self.timer_start: int | None = None
        self.token_count: int | None = None
        self.history: list[Any] = []

        assert tpm_limit_key in {"input", "output", "total"}

    def to_state(self) -> dict[str, Any]:
        return {
            "tpm_limit": self.tpm_limit,
            "tpm_limit_key": self.tpm_limit_key,
            "timer_start": self.timer_start or 0,
            "token_count": self.token_count or 0,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        current_time = int(time.time())
        token_usage_stat: TokenCounter = TokenUsageCallback.get_global_counter(
            callback_context
        )
        token_count: int = getattr(token_usage_stat, self.tpm_limit_key, 0)

        if self.timer_start is None:
            self.timer_start = current_time
            self.token_count = token_count
            self.save_to_state(callback_context)
            return

        diff = token_count - self.token_count
        els = current_time - self.timer_start
        self.token_count = token_count

        if diff > self.tpm_limit:
            delay = 60 - els + 1
            if delay > 0:
                time.sleep(delay)

            self.history.append(
                {
                    **self.to_state(),
                    "elapsed_seconds": els,
                    "timer_end": current_time,
                    "diff": diff,
                    "delay": delay,
                }
            )
            self.timer_start = int(time.time())

        self.save_to_state(callback_context)
        return


class RpmRatelimitCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = []

    def __init__(self, rpm_limit: int):
        self.rpm_limit = rpm_limit
        self.timer_start: int | None = None
        self.request_count: int | None = None
        self.history: list[Any] = []

    def to_state(self) -> dict[str, Any]:
        return {
            "rpm_limit": self.rpm_limit,
            "timer_start": self.timer_start,
            "request_count": self.request_count,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        current_time = int(time.time())

        if self.request_count is None:
            self.timer_start = current_time
            self.request_count = 1
            self.save_to_state(callback_context)
            return

        self.request_count += 1
        els = current_time - self.timer_start

        if self.request_count > self.rpm_limit:
            delay = 60 - els + 1
            if delay > 0:
                time.sleep(delay)
            self.history.append(
                {
                    **self.to_state(),
                    "elapsed_seconds": els,
                    "timer_end": current_time,
                    "delay": delay,
                }
            )
            self.timer_start = int(time.time())

        self.save_to_state(callback_context)
        return
