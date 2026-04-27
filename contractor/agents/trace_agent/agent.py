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
from contractor.tools.fs import FileFormat, rw_file_tools, RootedLocalFileSystem
from contractor.tools.vuln import vulnerability_report_tools, VulnerabilityReportFormat
from contractor.tools.code import code_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
langfuse = get_client()

TRACE_AGENT_PROMPT: Final[str] = load_prompt("trace_agent")


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Entrypoint identified (file, function)\n"
        "2. Functions traced and annotated so far (list with depths)\n"
        "3. Sinks and validations discovered\n"
        "4. Files modified\n"
        "5. Branches not yet traced or partially traced\n"
        "6. Suggested next steps to complete the trace\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


TRACE_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def build_trace_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    enable_vuln_reporting: bool = False,
) -> LlmAgent:
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )

    ctools = code_tools(fs)
    tools = [default_tool, *fs_tools, *mem_tools, *ctools]
    if enable_vuln_reporting:
        vuln_tools = vulnerability_report_tools(
            name=namespace,
            fmt=VulnerabilityReportFormat(_format=_format),
        )
        tools.extend(vuln_tools)

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens,
            message=summarization_message(_format=_format),
        )
    )
    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools,
            default_tool_name="default_tool",
            default_tool_arg="meta",
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    trace_agent = LlmAgent(
        name=name,
        description="request trace annotation agent",
        instruction=TRACE_AGENT_PROMPT,
        model=model if model is not None else TRACE_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return trace_agent


playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"
fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_trace_agent(
    name="trace_agent",
    namespace="code_review",
    fs=fs,
)
