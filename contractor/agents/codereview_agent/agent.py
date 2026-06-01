from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.vuln import VulnerabilityReportFormat, vulnerability_report_tools
from contractor.utils import load_prompt

from contractor.agents.worker_factory import build_worker

CODEREVIEW_PROMPT: Final[str] = load_prompt("codereview_agent")

_ELIDE_TOOLS: list[str] = [
    "read_file", "grep", "glob", "list_symbols",
]

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Passes completed (structural scan, confirm leads, auth audit)\n"
    "2. Vulnerabilities reported so far (count by severity)\n"
    "3. Files/directories already scanned\n"
    "4. Grep patterns already tried\n"
    "5. Leads found but not yet confirmed\n"
    "6. Areas not yet scanned\n"
    "Resume from where you left off — do NOT re-scan completed areas.\n"
)


def build_codereview_agent(
    name: str,
    fs: AbstractFileSystem,
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
    instruction = prompt if prompt is not None else CODEREVIEW_PROMPT

    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)
    gtools = attach_graph_tools_if_local(fs) if with_graph_tools else []

    vuln_tools = vulnerability_report_tools(
        name=namespace,
        fmt=VulnerabilityReportFormat(_format=_format),
    )

    tools = [
        default_tool,
        *fs_tools,
        *mem_tools,
        *ctools,
        *gtools,
        *vuln_tools,
    ]

    return build_worker(
        name=name,
        instruction=instruction,
        description=(
            "rapid vulnerability discovery agent — scans a codebase for "
            "dangerous patterns, confirms findings with targeted reads, "
            "and reports all discovered vulnerabilities."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results or _ELIDE_TOOLS,
        elide_keep_last_n=elide_keep_last_n,
    )
