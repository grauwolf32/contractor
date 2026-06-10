"""Post-trace vulnerability analytics agent.

Second stage of the post-diff trace split: a prior annotate-only trace
stage drives ``@trace`` / ``@validate`` / ``@sink`` annotations onto the
execution paths of a target (the annotated state lives in the trace
overlay), and this agent reads the resulting annotation diff, judges the
flows against the finding-shape taxonomy, and persists the supported
findings via ``report_vulnerability``.

The agent never annotates or edits code — its filesystem view is
read-only (over the annotated overlay, so annotations are visible
in-place) and its only write surface is the vulnerability-report store
of its namespace.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.worker_factory import build_worker
from contractor.callbacks import default_tool
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.vuln import VulnerabilityReportFormat, vulnerability_report_tools
from contractor.utils import load_prompt

AnalyticsFormat = Literal["json", "xml", "yaml", "markdown"]

VULN_ANALYTICS_PROMPT: Final[str] = load_prompt("vuln_analytics_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Annotated flows already analyzed (entrypoint -> sink, verdict)\n"
    "2. Findings reported so far (name, shape, severity)\n"
    "3. Flows from the diff not yet analyzed\n"
    "4. Controls checked and their status per handler\n"
    "5. Suggested next steps to finish the analysis\n"
)


def build_vuln_analytics_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: AnalyticsFormat = "json",
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    elide_tool_results: Iterable[str] | None = None,
    elide_keep_last_n: int = 15,
    prompt: str | None = None,
    with_graph_tools: bool = False,
    graph_tools: list | None = None,
) -> LlmAgent:
    instruction = prompt if prompt is not None else VULN_ANALYTICS_PROMPT

    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)
    if graph_tools is not None:
        gtools = graph_tools
    elif with_graph_tools:
        gtools = attach_graph_tools_if_local(fs)
    else:
        gtools = []
    vuln_tools = vulnerability_report_tools(
        name=namespace,
        fmt=VulnerabilityReportFormat(_format=_format),
    )

    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *gtools, *vuln_tools]

    return build_worker(
        name=name,
        instruction=instruction,
        description=(
            "post-trace vulnerability analytics agent — reads a "
            "@trace-annotated diff, judges each annotated flow against the "
            "finding-shape taxonomy, and reports supported vulnerabilities."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
