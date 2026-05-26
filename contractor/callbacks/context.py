import json
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
    """Elide stale or excess function-call results from the prompt.

    Two independent pruning strategies, applied in order during a single
    reverse scan of the conversation:

    1. **Staleness** (``deduplicate=True``, default): if the same tool was
       called with identical arguments more than once, every response except
       the most recent is elided unconditionally.
    2. **Budget**: non-stale eligible results are kept while both limits hold:
       cumulative response size <= ``keep_budget_chars`` *and* count <=
       ``keep_last_n``.  Set either to ``0`` to disable that axis.  At least
       one must be positive.

    target_tools / exempt_tools filter which tools are eligible (mutually
    exclusive; omit both to consider every tool).
    """

    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = []

    def __init__(
        self,
        keep_last_n: int = 0,
        keep_budget_chars: int = 0,
        *,
        target_tools: Optional[Iterable[str]] = None,
        exempt_tools: Optional[Iterable[str]] = None,
        deduplicate: bool = True,
    ):
        if keep_last_n < 0 or keep_budget_chars < 0:
            raise ValueError("keep_last_n and keep_budget_chars must not be negative")
        if keep_last_n == 0 and keep_budget_chars == 0:
            raise ValueError(
                "at least one of keep_last_n or keep_budget_chars must be > 0"
            )
        if target_tools is not None and exempt_tools is not None:
            raise ValueError("target_tools and exempt_tools are mutually exclusive")

        self.keep_last_n = keep_last_n
        self.keep_budget_chars = keep_budget_chars
        self.deduplicate = deduplicate
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

    @staticmethod
    def _response_size(response: Optional[dict]) -> int:
        if not response:
            return 0
        try:
            return len(json.dumps(response, default=str))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _args_key(args: Optional[dict]) -> str:
        if not args:
            return ""
        try:
            return json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return ""

    def _build_call_signatures(
        self, contents: list,
    ) -> dict[tuple[int, int], tuple[str, str]]:
        """Map each function_response position to ``(name, args_key)``."""
        calls: list[tuple[str, str]] = []
        responses: list[tuple[int, int, str]] = []

        for ci, content in enumerate(contents):
            if not content.parts:
                continue
            for pi, part in enumerate(content.parts):
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    args = getattr(fc, "args", None) or {}
                    calls.append((fc.name, self._args_key(args)))
                fr = getattr(part, "function_response", None)
                if fr is not None and getattr(fr, "name", None):
                    responses.append((ci, pi, fr.name))

        result: dict[tuple[int, int], tuple[str, str]] = {}
        for i, (ci, pi, name) in enumerate(responses):
            if i < len(calls) and calls[i][0] == name:
                result[(ci, pi)] = calls[i]
            else:
                result[(ci, pi)] = (name, "")
        return result

    def to_state(self) -> dict[str, Any]:
        return {
            "keep_last_n": self.keep_last_n,
            "keep_budget_chars": self.keep_budget_chars,
            "deduplicate": self.deduplicate,
            "counter": self.counter,
            "target_tools": sorted(self.target_tools) if self.target_tools else None,
            "exempt_tools": sorted(self.exempt_tools) if self.exempt_tools else None,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        if not llm_request.contents:
            return

        call_sigs = (
            self._build_call_signatures(llm_request.contents)
            if self.deduplicate
            else {}
        )
        seen_sigs: set[tuple[str, str]] = set()
        budget_used: int = 0
        eligible_kept: int = 0

        for ci in range(len(llm_request.contents) - 1, -1, -1):
            content = llm_request.contents[ci]
            if not content.parts:
                continue
            for pi in range(len(content.parts) - 1, -1, -1):
                part = content.parts[pi]
                fr = getattr(part, "function_response", None)
                if fr is None:
                    continue
                if not self._is_eligible(fr.name):
                    continue
                if fr.response and fr.response.get("elided"):
                    continue

                sig = call_sigs.get((ci, pi))
                if self.deduplicate and sig is not None and sig in seen_sigs:
                    self.counter += 1
                    fr.response = {"elided": True, "tool": fr.name, "reason": "stale"}
                    continue
                if sig is not None:
                    seen_sigs.add(sig)

                size = self._response_size(fr.response)
                over_budget = (
                    self.keep_budget_chars > 0
                    and eligible_kept > 0
                    and budget_used + size > self.keep_budget_chars
                )
                over_count = (
                    self.keep_last_n > 0 and eligible_kept >= self.keep_last_n
                )

                if over_budget or over_count:
                    self.counter += 1
                    fr.response = {"elided": True, "tool": fr.name}
                    continue

                budget_used += size
                eligible_kept += 1

        self.save_to_state(callback_context)
        return
