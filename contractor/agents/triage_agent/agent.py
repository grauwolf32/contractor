"""Currently unused: ``build_triage_agent`` has no callers in production
pipelines (``cli/pipelines/``) or tests as of 2026-05. Kept for potential
future use (security-analysis workflows). If you wire this up, remove this
notice; if it remains unreferenced, consider deleting the directory along
with its prompts and tasks.
"""

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
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import (SubtaskFormatter,
                                    _prepare_worker_instructions)
from contractor.tools.vuln import (VulnerabilityReportFormat,
                                   vulnerability_report_tools)
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

TRIAGE_PROMPT: Final[str] = load_prompt("triage_agent")

def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Subtask objective as you understand it\n"
        "2. Findings triaged so far (name, decision, severity, confidence)\n"
        "3. Evidence collected (file:line citations) per decision\n"
        "4. Duplicate clusters identified (canonical name + members)\n"
        "5. Entry points and sanitizers observed\n"
        "6. Findings still untriaged and the reason\n"
        "7. Open questions or blockers\n"
        "8. Smallest concrete next step to resume triage\n"
        "Include only claims supported by tool output; mark anything inferred as such.\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )

def build_triage_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    with_graph_tools: bool = False,
) -> LlmAgent:
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(fs, fmt=FileFormat(_format=_format))
    ctools = code_tools(fs=fs)
    gtools = attach_graph_tools_if_local(fs) if with_graph_tools else []
    vuln_tools = vulnerability_report_tools(
        name=namespace,
        fmt=VulnerabilityReportFormat(_format=_format),
    )

    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *gtools, *vuln_tools]

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens, message=summarization_message(_format=_format)
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
            tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    return LlmAgent(
        name=name,
        description=(
            "security triage agent — confirms, refutes, deduplicates, "
            "and ranks vulnerability candidates against the codebase."
        ),
        instruction=TRIAGE_PROMPT,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

