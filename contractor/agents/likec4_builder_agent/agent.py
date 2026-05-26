from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.likec4 import likec4_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.utils import load_prompt

from contractor.agents.worker_factory import build_worker

LIKEC4_BUILDER_PROMPT: Final[str] = load_prompt("likec4_builder_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Subtask objective as you understand it\n"
    "2. Current state of the LikeC4 source (top-level blocks present, "
    "elements/relationships/views added, last validate_likec4 result)\n"
    "3. External actors and integrations represented vs. still missing\n"
    "4. Trust boundaries and security tags encoded so far\n"
    "5. Code areas explored vs. relevant areas still unexplored\n"
    "6. Assumptions made (inferred kinds, tags, relationships) and the reason\n"
    "7. Open questions, ambiguous integrations, or blockers\n"
    "8. Smallest concrete next step to resume modeling\n"
    "Include only claims supported by tool output; mark anything inferred as such.\n"
)

def build_likec4_builder_agent(
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
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)
    gtools = attach_graph_tools_if_local(fs) if with_graph_tools else []
    c4_tools = likec4_tools(fs=fs)

    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *gtools, *c4_tools]

    return build_worker(
        name=name,
        instruction=LIKEC4_BUILDER_PROMPT,
        description=(
            "likec4 architecture-as-code builder — produces a validated "
            "LikeC4 model focused on trust boundaries, external "
            "interactions, and security-relevant data flows."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
