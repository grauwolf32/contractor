"""Shared factory for worker agents.

All worker agents follow the same callback-registration pattern:
TokenUsage -> SummarizationLimit -> (optional) FunctionResultsRemoval
-> InvalidToolCallGuardrail -> RepeatedToolCall.

``build_worker`` encapsulates this boilerplate so each ``build_*_agent``
only needs to assemble its tools and summarization bullets, then delegate
the LlmAgent construction here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import (
    FunctionResultsRemovalCallback,
    SummarizationLimitCallback,
)
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools import DEFAULT_HEAVY_TOOLS
from contractor.tools.tasks import SubtaskFormatter, _prepare_worker_instructions
from contractor.utils.settings import DEFAULT_MODEL


def build_summarization_message(
    bullets: str,
    _format: Literal["json", "xml", "yaml", "markdown"],
) -> str:
    """Build a full summarization-limit message from agent-specific bullets.

    *bullets* is the agent-specific numbered list (including the leading
    newline-separated prefix like ``"You have reached the context limit.
    Summarize your progress:\\n1. ...\\n2. ...\\n"``).  The common
    ``_prepare_worker_instructions`` suffix is appended automatically.
    """
    return bullets + _prepare_worker_instructions(SubtaskFormatter(_format=_format))


def build_worker(
    *,
    name: str,
    instruction: str,
    description: str,
    tools: list,
    _format: Literal["json", "xml", "yaml", "markdown"],
    summarization_bullets: str,
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    with_elide: bool = True,
    elide_tool_results: Iterable[str] | None = None,
    elide_keep_last_n: int = 15,
    repeated_call_threshold: int = 5,
) -> LlmAgent:
    """Construct an :class:`LlmAgent` with the standard callback stack.

    Parameters
    ----------
    name:
        Agent (and callback-adapter) name.
    instruction:
        The full prompt text for the agent.
    description:
        Short human-readable agent description.
    tools:
        Pre-assembled tool list (each agent assembles tools differently).
    _format:
        Output format used for summarization / subtask formatting.
    summarization_bullets:
        Agent-specific bullet list *including* the leading "You have
        reached ..." prefix.  ``_prepare_worker_instructions`` is
        appended automatically via :func:`build_summarization_message`.
    max_tokens:
        Token budget before the summarization message is injected.
    model:
        LiteLlm model override; falls back to ``DEFAULT_MODEL``.
    with_elide:
        Whether to register the ``FunctionResultsRemovalCallback``.
        Set to ``False`` for agents that have no heavy-tool results to
        elide (e.g. ``http_agent``, ``oas_linter_agent``).
    elide_tool_results:
        Explicit tool-name whitelist for elision.  When *None* (the
        default) and ``with_elide`` is *True*, ``DEFAULT_HEAVY_TOOLS``
        is used.
    elide_keep_last_n:
        Number of recent eligible results to keep un-elided.
    repeated_call_threshold:
        Number of identical consecutive calls before the guardrail
        fires.
    """
    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens,
            message=build_summarization_message(summarization_bullets, _format),
        )
    )

    if with_elide:
        elide_targets = (
            list(elide_tool_results)
            if elide_tool_results is not None
            else list(DEFAULT_HEAVY_TOOLS)
        )
        if elide_targets:
            callback_adapter.register(
                FunctionResultsRemovalCallback(
                    keep_last_n=elide_keep_last_n,
                    target_tools=elide_targets,
                )
            )

    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools,
            default_tool_name="default_tool",
            default_tool_arg="meta",
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=repeated_call_threshold))

    return LlmAgent(
        name=name,
        description=description,
        instruction=instruction,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )
