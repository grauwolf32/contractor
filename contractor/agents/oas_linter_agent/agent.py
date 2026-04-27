from __future__ import annotations

import os
from pathlib import Path
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
from contractor.tools.fs import FileFormat, RootedLocalFileSystem, ro_file_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.openapi import openapi_tools, openapi_linter_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

OAS_LINTER_PROMPT: Final[str] = load_prompt("oas_linter_agent")


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached context limit. Summarize your progress and call report tool."
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


OAS_LINTER_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def build_oas_linter_agent(
    name: str,
    fs,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
):
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(
        fs, fmt=FileFormat(_format=_format), with_interaction_tools=False
    )
    oas_tools = openapi_tools(name=namespace, fs=fs)
    linter_tools = openapi_linter_tools(name=namespace)

    tools = [default_tool, *fs_tools, *mem_tools, *oas_tools, *linter_tools]

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

    oas_linter_agent = LlmAgent(
        name=name,
        description="software engineering agent",
        instruction=OAS_LINTER_PROMPT,
        model=model if model is not None else OAS_LINTER_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return oas_linter_agent


playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"

fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_oas_linter_agent(
    name="oas_builder_agent",
    namespace="code_review",
    fs=fs,
)
