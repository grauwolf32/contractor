from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.tools.code import (annotation_tools,
                                   attach_graph_tools_if_local, code_tools)
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.vuln import (VulnerabilityReportFormat,
                                   vulnerability_report_tools)
from contractor.utils import load_prompt

from contractor.agents.worker_factory import build_worker

# Output-format knob threaded into the memory/file/vuln formatters. Exported so
# callers can cast their (str-typed) template format to it at the boundary.
TraceFormat = Literal["json", "xml", "yaml", "markdown"]

TRACE_AGENT_PROMPT: Final[str] = load_prompt("trace_agent")

# Tools that mutate the filesystem in ways the trace_agent should no
# longer reach for now that ``annotate_trace`` / ``annotate_validate``
# / ``annotate_sink`` exist. Filtered out of the agent's tool registry
# so the LLM cannot bypass the structured annotation surface (which
# enforces argument schemas, indentation, comment style, and
# duplicate-refusal). ``restore`` / ``changed_paths`` / ``diff`` stay —
# they are read-side / undo, not source mutations.
_TRACE_DISALLOWED_FS_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "insert_line",
        "edit",
        "replace_range",
        "write_file",
        "append_file",
        "mkdir",
        "rm",
        "cp",
        "mv",
    }
)

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Entrypoint identified (file, function)\n"
    "2. Functions traced and annotated so far (list with depths)\n"
    "3. Sinks and validations discovered\n"
    "4. Files modified\n"
    "5. Branches not yet traced or partially traced\n"
    "6. Suggested next steps to complete the trace\n"
)

def build_trace_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: TraceFormat = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    enable_vuln_reporting: bool = False,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    prompt: Optional[str] = None,
    with_graph_tools: bool = False,
    graph_tools: Optional[list] = None,
) -> LlmAgent:
    instruction = prompt if prompt is not None else TRACE_AGENT_PROMPT
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )

    ctools = code_tools(fs)
    atools = annotation_tools(fs)
    if graph_tools is not None:
        gtools = graph_tools
    elif with_graph_tools:
        gtools = attach_graph_tools_if_local(fs)
    else:
        gtools = []
    fs_tools = [
        t for t in fs_tools
        if getattr(t, "__name__", "") not in _TRACE_DISALLOWED_FS_TOOLS
    ]
    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *atools, *gtools]
    if enable_vuln_reporting:
        vuln_tools = vulnerability_report_tools(
            name=namespace,
            fmt=VulnerabilityReportFormat(_format=_format),
        )
        tools.extend(vuln_tools)

    return build_worker(
        name=name,
        instruction=instruction,
        description="request trace annotation agent",
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
