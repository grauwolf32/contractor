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
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.vuln import (VulnerabilityReportFormat,
                                   vulnerability_report_tools)
from contractor.utils import load_prompt

from contractor.agents.worker_factory import build_worker

TRIAGE_PROMPT: Final[str] = load_prompt("triage_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
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

    return build_worker(
        name=name,
        instruction=TRIAGE_PROMPT,
        description=(
            "security triage agent — confirms, refutes, deduplicates, "
            "and ranks vulnerability candidates against the codebase."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
