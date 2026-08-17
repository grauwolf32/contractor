import json
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from contractor.utils.llm_compat import forced_response_format, forced_tool_choice

from .base import BaseCallback, CallbackTypes
from .tokens import TokenUsageCallback

logger = logging.getLogger(__name__)

TOKEN_USAGE_CALLBACK_NAME = TokenUsageCallback().name


_FINISH_TOOL_NAME = "set_model_response"


def _none_would_block_finish(llm_request: LlmRequest) -> bool:
    """True if forcing ``tool_choice="none"`` would block the model's only way
    to deliver its final result.

    Contractor workers run *without* an ADK ``output_schema``
    (``build_planning_agent(use_output_schema=False)``): they deliver their
    ``SubtaskExecutionResult`` as free text that the formatters parse. So
    forbidding tools is exactly what forces termination — there is no finish
    *tool* to block.

    The one exception is ADK's ``output_schema``-with-tools workaround, which
    injects a ``set_model_response`` tool the model must *call* to finish (used
    only when ``can_use_output_schema_with_tools`` is False — not contractor's
    LiteLlm path). If that tool is present, ``"none"`` would trap the worker, so
    we degrade to message-only instead.
    """
    config = getattr(llm_request, "config", None)
    tools = getattr(config, "tools", None) if config is not None else None
    for tool in tools or []:
        for decl in getattr(tool, "function_declarations", None) or []:
            if getattr(decl, "name", None) == _FINISH_TOOL_NAME:
                return True
    return False


class SummarizationLimitCallback(BaseCallback):
    cb_type: CallbackTypes = CallbackTypes.before_model_callback
    deps: list[str] = [TOKEN_USAGE_CALLBACK_NAME]

    def __init__(
        self,
        message: str,
        max_tokens: int,
        summarization_key: str = "total",
        force_tool_choice: str | None = None,
        force_response_format: dict | None = None,
    ):
        self.max_tokens = max_tokens
        self.message = message
        self.token_count: int = 0
        self.history: list[Any] = []
        self.summarization_key = summarization_key
        # Hard enforcement of the summarization request. When set (e.g. "none"),
        # once the token limit is crossed this callback publishes the value to
        # the per-task ``forced_tool_choice`` ContextVar before every model
        # call, and the model client forces that tool_choice. "none" forbids
        # tool calls — compelling the worker to emit its final result instead of
        # ignoring the message and running context to the ceiling. ``None`` keeps
        # the prior message-only behaviour (the model may simply keep working).
        self.force_tool_choice = force_tool_choice
        # Companion response_format (OpenAI JSON-schema dict) forced alongside
        # "none". Workers carry no output_schema, so without this the forced
        # text is unconstrained and often fails to parse; the schema grammar
        # makes the forced result valid JSON. None = don't force a format.
        self.force_response_format = force_response_format
        # Other enforcement callbacks can register a predicate that reports
        # whether forbidding tools would make their required finish action
        # impossible. Mandatory verdict callbacks use this to keep tools
        # available until one required persistence call has been observed.
        self._force_none_blockers: list[Callable[[], bool]] = []
        # Last tool_choice published to the ContextVar (None = not forced); kept
        # for observability — surfaced in to_state()/CALLBACK_SUMMARY metrics.
        self.last_forced: str | None = None
        # Latch: once the message has been injected for an invocation, do not
        # inject it again for that invocation. The per-invocation token
        # counter (TokenUsageCallback) only grows within an invocation and is
        # reset only when the invocation changes, so there is no mid-invocation
        # event to re-arm on — "once per invocation" is the correct semantics.
        # The latch re-arms automatically when invocation_id changes (the
        # counter resets then too).
        self.fired: bool = False
        self.fired_invocation_id: str | None = None

    def add_force_none_blocker(self, blocker: Callable[[], bool]) -> None:
        """Degrade forced ``none`` while ``blocker`` reports pending work."""
        if blocker not in self._force_none_blockers:
            self._force_none_blockers.append(blocker)

    def to_state(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "token_count": self.token_count,
            "message": self.message,
            "history": self.history,
            "fired_invocation_id": self.fired_invocation_id,
            "force_tool_choice": self.force_tool_choice,
            "last_forced": self.last_forced,
        }

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        invocation_id = self.get_invocation_id(callback_context)
        token_usage_stat = (
            self.get_from_cb_state(callback_context, TOKEN_USAGE_CALLBACK_NAME) or {}
        )
        token_invocation_id = token_usage_stat.get("invocation_id")
        # before_model runs before TokenUsageCallback's after_model reset. A
        # child AgentTool session is seeded from planner state, so its very first
        # request can otherwise inherit the previous worker invocation's final
        # token count and be terminated before doing any work.
        token_state_is_current = (
            token_invocation_id is None
            or invocation_id is None
            or token_invocation_id == invocation_id
        )
        token_count = (
            token_usage_stat.get("counter", {}).get(self.summarization_key, 0)
            if token_state_is_current
            else 0
        )
        self.token_count = token_count

        over_limit = token_count >= self.max_tokens

        # Compute the enforcement value. "none" forbids tool calls, forcing the
        # worker to emit its final result (contractor workers deliver it as free
        # text — there is no output_schema). The one case to avoid is a
        # set_model_response finish *tool*, which "none" would block; degrade to
        # message-only there. See _none_would_block_finish.
        forced = self.force_tool_choice if over_limit else None
        finish_tool_present = _none_would_block_finish(llm_request)
        mandatory_tool_pending = any(
            blocker() for blocker in self._force_none_blockers
        )
        degraded = forced == "none" and (
            finish_tool_present or mandatory_tool_pending
        )
        if degraded:
            forced = None
        # Publish on every call so it tracks the limit exactly (cleared while
        # under it, so a new invocation's first call clears any value the
        # previous one left in this task's ContextVar). The response_format is
        # paired ONLY with "none": its grammar pins the output to the worker's
        # final-result shape, which is right when tools are forbidden but would
        # corrupt tool-call generation under "auto"/"required" (the worker is
        # still expected to call tools there).
        if self.force_tool_choice is not None:
            forced_tool_choice.set(forced)
            forced_response_format.set(
                self.force_response_format if forced == "none" else None
            )
        self.last_forced = forced

        if not over_limit:
            self.save_to_state(callback_context)
            return

        if self.fired and self.fired_invocation_id == invocation_id:
            # Already injected for this invocation — don't append the message
            # to every subsequent request. (The enforcement signal above is
            # still refreshed on each call.)
            self.save_to_state(callback_context)
            return

        # First crossing of the limit this invocation: inject the summarize
        # message and log the enforcement decision for visibility.
        llm_request.contents.append(
            types.Content(role="user", parts=[types.Part(text=self.message)])
        )
        self.fired = True
        self.fired_invocation_id = invocation_id
        self.history.append(int(time.time()))

        if degraded:
            reason = (
                f"a {_FINISH_TOOL_NAME} finish tool is present"
                if finish_tool_present
                else "a mandatory persistence tool is still pending"
            )
            logger.warning(
                "summarization limit hit (tokens=%d >= %d) but force_tool_choice="
                "'none' DEGRADED to message-only: %s and 'none' would block it "
                "(invocation=%s) — worker may keep "
                "calling tools instead of terminating.",
                token_count,
                self.max_tokens,
                reason,
                invocation_id,
            )
        else:
            logger.info(
                "summarization limit hit (tokens=%d >= %d): injected summarize "
                "message; forced tool_choice=%r response_format=%s (invocation=%s)",
                token_count, self.max_tokens, forced,
                "yes" if self.force_response_format else "no", invocation_id,
            )

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
        target_tools: Iterable[str] | None = None,
        exempt_tools: Iterable[str] | None = None,
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
        self.target_tools: frozenset[str] | None = (
            frozenset(target_tools) if target_tools is not None else None
        )
        self.exempt_tools: frozenset[str] = (
            frozenset(exempt_tools) if exempt_tools is not None else frozenset()
        )
        self.counter = 0

    def _is_eligible(self, tool_name: str | None) -> bool:
        if self.target_tools is not None:
            return tool_name in self.target_tools
        return tool_name not in self.exempt_tools

    @staticmethod
    def _response_size(response: dict | None) -> int:
        if not response:
            return 0
        try:
            return len(json.dumps(response, default=str))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _args_key(args: dict | None) -> str:
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
                # No matching call: give the response a per-index sentinel
                # signature so unmatched responses never collide with each
                # other (or with a real argless call) and are never elided
                # as "stale" duplicates.
                result[(ci, pi)] = (name, f"<unmatched:{i}>")
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
