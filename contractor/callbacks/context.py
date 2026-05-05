import time
from typing import Any, Iterable, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from .base import BaseCallback, CallbackTypes
from .tokens import TokenUsageCallback

TOKEN_USAGE_CALLBACK_NAME = TokenUsageCallback().name


class SummarizationLimitCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = [TOKEN_USAGE_CALLBACK_NAME]

    def __init__(
        self,
        message: str,
        max_tokens: int,
        summarization_key: str = "total",
    ):
        self.max_tokens = max_tokens
        self.message = message
        self.token_count: int = 0
        self.history: list[Any] = []
        self.summarization_key = summarization_key

    def to_state(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "token_count": self.token_count,
            "message": self.message,
            "history": self.history,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        token_usage_stat = (
            self.get_from_cb_state(callback_context, TOKEN_USAGE_CALLBACK_NAME) or {}
        )
        token_count = token_usage_stat.get("counter", {}).get(self.summarization_key, 0)
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
    """Elide stale function-call results from the prompt.

    target_tools — whitelist: only results of these tools are eligible for elision.
    exempt_tools — blacklist: all tools except these are eligible.
    Mutually exclusive; if neither is given, every tool is eligible (legacy behavior).
    keep_last_n is counted within the eligible set, so unrelated tools never affect
    the budget for the ones you actually want to truncate.
    """

    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = []

    def __init__(
        self,
        keep_last_n: int,
        target_tools: Optional[Iterable[str]] = None,
        exempt_tools: Optional[Iterable[str]] = None,
    ):
        assert keep_last_n > 1
        if target_tools is not None and exempt_tools is not None:
            raise ValueError("target_tools and exempt_tools are mutually exclusive")

        self.keep_last_n = keep_last_n
        self.target_tools: Optional[frozenset[str]] = (
            frozenset(target_tools) if target_tools is not None else None
        )
        self.exempt_tools: frozenset[str] = (
            frozenset(exempt_tools) if exempt_tools is not None else frozenset()
        )
        self.counter = 0

    def _is_eligible(self, tool_name: Optional[str]) -> bool:
        if self.target_tools is not None:
            return tool_name in self.target_tools
        return tool_name not in self.exempt_tools

    def to_state(self) -> dict[str, Any]:
        return {
            "keep_last_n": self.keep_last_n,
            "counter": self.counter,
            "target_tools": sorted(self.target_tools) if self.target_tools else None,
            "exempt_tools": sorted(self.exempt_tools) if self.exempt_tools else None,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        if not llm_request.contents:
            return

        eligible_seen: int = 0
        for content in reversed(llm_request.contents):
            if not content.parts:
                continue
            for part in reversed(content.parts):
                fr = part.function_response
                if fr is None:
                    continue
                if not self._is_eligible(fr.name):
                    continue

                eligible_seen += 1
                if eligible_seen < self.keep_last_n:
                    continue
                if fr.response and fr.response.get("elided"):
                    continue

                self.counter += 1
                fr.parts = None
                fr.response = {"elided": True, "tool": fr.name}

        self.save_to_state(callback_context)
        return
