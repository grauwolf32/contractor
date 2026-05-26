from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

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

VULN_SCAN_PROMPT: Final[str] = load_prompt("vuln_scan_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Subtask objective: scan the codebase for security vulnerabilities\n"
    "2. Entry points discovered and scanned so far\n"
    "3. Vulnerabilities reported (name, file, CWE)\n"
    "4. Files and areas not yet reviewed\n"
    "5. Vulnerability classes not yet checked\n"
    "6. Next step to continue the scan\n"
)


def build_vuln_scan_agent(
    name: str,
    fs,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80_000,
    model: Optional[LiteLlm] = None,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    with_graph_tools: bool = False,
    prompt: Optional[str] = None,
) -> LlmAgent:
    instruction = prompt if prompt is not None else VULN_SCAN_PROMPT

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
        instruction=instruction,
        description="vulnerability scanning agent",
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
