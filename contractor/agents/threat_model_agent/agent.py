"""Currently unused: ``build_threat_model_agent`` has no callers in production
workflows (``contractor/workflows/``) or tests as of 2026-05. The matching
``contractor/tasks/threat_analysis*`` template is also orphaned. Kept for
potential future use (security-analysis workflows). If you wire this up,
remove this notice; if it remains unreferenced, consider deleting the
directory along with its prompts and tasks.
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
from contractor.tools.openapi.openapi import OpenAPIFormat, openapi_tools
from contractor.tools.vuln import VulnerabilityReportFormat, vulnerability_report_tools
from contractor.utils import load_prompt

THREAT_MODEL_PROMPT: Final[str] = load_prompt("threat_model_agent")

_OAS_READ_ONLY_TOOLS = frozenset(
    {
        "list_paths",
        "list_components",
        "list_servers",
        "get_info",
        "get_path",
        "get_component",
        "get_full_openapi_schema",
    }
)

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Subtask objective as you understand it\n"
    "2. Asset inventory built so far\n"
    "3. Entry points cataloged (handler, file:line, auth state)\n"
    "4. Trust boundaries identified\n"
    "5. Threats raised (name, STRIDE letter, severity, confidence)\n"
    "6. Areas of the system not yet modeled\n"
    "7. Open questions or blockers\n"
    "8. Smallest concrete next step to resume modeling\n"
    "Include only claims supported by tool output; mark anything inferred as such.\n"
)

def build_threat_model_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    with_openapi: bool = True,
    elide_tool_results: Iterable[str] | None = None,
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

    if with_openapi:
        oas_all = openapi_tools(name=namespace, fs=fs, fmt=OpenAPIFormat(_format=_format))
        oas_read = [t for t in oas_all if getattr(t, "__name__", "") in _OAS_READ_ONLY_TOOLS]
        tools.extend(oas_read)

    return build_worker(
        name=name,
        instruction=THREAT_MODEL_PROMPT,
        description=(
            "threat modeling agent — produces a STRIDE-aligned threat "
            "model from code and OpenAPI, persisted as findings."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
