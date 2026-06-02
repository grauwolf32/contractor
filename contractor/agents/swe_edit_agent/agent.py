from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.worker_factory import build_worker
from contractor.callbacks import default_tool
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.utils import load_prompt

SWE_EDIT_PROMPT: Final[str] = load_prompt("swe_edit_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached context limit. Summarize your progress:\n"
    "1. Subtask understanding and goal\n"
    "2. Files explored and key findings\n"
    "3. Edits applied so far (paths and brief description)\n"
    "4. Outstanding edits or verification steps\n"
    "5. Blockers or open questions\n"
)

def build_swe_edit_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    elide_tool_results: Iterable[str] | None = None,
    elide_keep_last_n: int = 15,
    with_graph_tools: bool = False,
) -> LlmAgent:
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)
    gtools = attach_graph_tools_if_local(fs) if with_graph_tools else []

    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *gtools]

    return build_worker(
        name=name,
        instruction=SWE_EDIT_PROMPT,
        description="software engineering agent with edit capabilities",
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
