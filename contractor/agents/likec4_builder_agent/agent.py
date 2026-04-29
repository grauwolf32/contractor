from __future__ import annotations

import os
from typing import Final, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks import default_tool
from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.code import code_tools
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.likec4 import likec4_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

LIKEC4_BUILDER_PROMPT: Final[str] = load_prompt("likec4_builder_agent")


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Subtask objective as you understand it\n"
        "2. Current state of the LikeC4 source (top-level blocks present, "
        "elements/relationships/views added, last validate_likec4 result)\n"
        "3. External actors and integrations represented vs. still missing\n"
        "4. Trust boundaries and security tags encoded so far\n"
        "5. Code areas explored vs. relevant areas still unexplored\n"
        "6. Assumptions made (inferred kinds, tags, relationships) and the reason\n"
        "7. Open questions, ambiguous integrations, or blockers\n"
        "8. Smallest concrete next step to resume modeling\n"
        "Include only claims supported by tool output; mark anything inferred as such.\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


def build_likec4_builder_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
) -> LlmAgent:
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)
    c4_tools = likec4_tools(fs=fs)

    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *c4_tools]

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens, message=summarization_message(_format=_format)
        )
    )
    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    return LlmAgent(
        name=name,
        description=(
            "likec4 architecture-as-code builder — produces a validated "
            "LikeC4 model focused on trust boundaries, external "
            "interactions, and security-relevant data flows."
        ),
        instruction=LIKEC4_BUILDER_PROMPT,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )
