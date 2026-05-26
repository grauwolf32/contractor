from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.tools.code import attach_graph_tools_if_local
from contractor.tools.code.tools import code_tools
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.openapi.openapi import openapi_tools
from contractor.utils import load_prompt

from contractor.agents.worker_factory import build_worker

OAS_PROMPT: Final[str] = load_prompt("oas_builder_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Subtask objective as you understand it\n"
    "2. Endpoints upserted into the schema (method + path) and the source file:line they were derived from\n"
    "3. Components/schemas upserted, with the source they were derived from\n"
    "4. Code areas explored vs. relevant areas still unexplored\n"
    "5. Assumptions made (inferred types, optionality, auth, content types) and the reason for each\n"
    "6. Endpoints or components observed but not yet upserted, and why\n"
    "7. Open questions, ambiguous handlers, or blockers\n"
    "8. Smallest concrete next step to resume building the schema\n"
    "Include only claims supported by tool output; mark anything inferred as such.\n"
)

def build_oas_builder_agent(
    name: str,
    fs,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    with_graph_tools: bool = False,
):
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(fs, fmt=FileFormat(_format=_format))
    oas_tools = openapi_tools(name=namespace, fs=fs)
    ctools = code_tools(fs=fs)
    gtools = attach_graph_tools_if_local(fs) if with_graph_tools else []

    tools = [default_tool, *fs_tools, *mem_tools, *oas_tools, *ctools, *gtools]

    return build_worker(
        name=name,
        instruction=OAS_PROMPT,
        description="software engineering agent",
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
