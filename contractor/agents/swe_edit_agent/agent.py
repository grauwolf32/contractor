from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from fsspec import AbstractFileSystem

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.utils import load_prompt
from contractor.tools.fs import FileFormat, RootedLocalFileSystem, rw_file_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.code import code_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

SWE_EDIT_PROMPT: Final[str] = load_prompt("swe_edit_agent")


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached context limit. Summarize your progress:\n"
        "1. Subtask understanding and goal\n"
        "2. Files explored and key findings\n"
        "3. Edits applied so far (paths and brief description)\n"
        "4. Outstanding edits or verification steps\n"
        "5. Blockers or open questions\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


SWE_EDIT_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def build_swe_edit_agent(
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

    tools = [default_tool, *fs_tools, *mem_tools, *ctools]

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

    swe_edit_agent = LlmAgent(
        name=name,
        description="software engineering agent with edit capabilities",
        instruction=SWE_EDIT_PROMPT,
        model=model if model is not None else SWE_EDIT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return swe_edit_agent


playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"

fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_swe_edit_agent(
    name="swe_edit_agent",
    namespace="swe_edit",
    fs=fs,
)
