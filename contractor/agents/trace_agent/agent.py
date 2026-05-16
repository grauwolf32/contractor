from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import (FunctionResultsRemovalCallback,
                                          SummarizationLimitCallback)
from contractor.callbacks.guardrails import (InvalidToolCallGuardrailCallback,
                                             RepeatedToolCallCallback)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools import DEFAULT_HEAVY_TOOLS
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import (SubtaskFormatter,
                                    _prepare_worker_instructions)
from contractor.tools.vuln import (VulnerabilityReportFormat,
                                   vulnerability_report_tools)
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

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

def build_trace_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    enable_vuln_reporting: bool = False,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    prompt: Optional[str] = None,
) -> LlmAgent:
    instruction = prompt if prompt is not None else TRACE_AGENT_PROMPT
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )

    ctools = code_tools(fs)
    gtools = attach_graph_tools_if_local(fs)
    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *gtools]
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
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    trace_agent = LlmAgent(
        name=name,
        description="request trace annotation agent",
        instruction=instruction,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return trace_agent
