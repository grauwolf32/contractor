from __future__ import annotations

import os
from typing import Final, Optional, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.openapi.openapi import openapi_tools
from contractor.tools.code.tools import code_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

OAS_PROMPT: Final[str] = load_prompt("oas_builder_agent")


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Subtask objective as you understand it\n"
        "2. Endpoints upserted into the schema (method + path) and the source file:line they were derived from\n"
        "3. Components/schemas upserted, with the source they were derived from\n"
        "4. Code areas explored vs. relevant areas still unexplored\n"
        "5. Assumptions made (inferred types, optionality, auth, content types) and the reason for each\n"
        "6. Endpoints or components observed but not yet upserted, and why\n"
        "7. Open questions, ambiguous handlers, or blockers\n"
        "8. Smallest concrete next step to resume building the schema\n"
        "Include only claims supported by tool output; mark anything inferred as such.\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


def build_oas_builder_agent(
    name: str,
    fs,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
):
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(fs, fmt=FileFormat(_format=_format))
    oas_tools = openapi_tools(name=namespace, fs=fs)
    ctools = code_tools(fs=fs)

    tools = [default_tool, *fs_tools, *mem_tools, *oas_tools, *ctools]

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

    swe_agent = LlmAgent(
        name=name,
        description="software engineering agent",
        instruction=OAS_PROMPT,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return swe_agent
