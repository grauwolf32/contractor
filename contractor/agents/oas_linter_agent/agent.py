from __future__ import annotations

from typing import Final, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.openapi import openapi_linter_tools, openapi_tools
from contractor.utils import load_prompt

from contractor.agents.worker_factory import build_worker

OAS_LINTER_PROMPT: Final[str] = load_prompt("oas_linter_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Subtask objective as you understand it\n"
    "2. Initial lint_openapi findings vs. current state (issues resolved, issues remaining)\n"
    "3. Schema mutations applied so far (upsert_path / upsert_component / remove_*) with the rule or finding that motivated each\n"
    "4. Code or schema evidence supporting each mutation (file:line, component name, or lint rule id)\n"
    "5. Lint issues intentionally left unresolved and the reason (e.g., needs human input, ambiguous spec)\n"
    "6. Assumptions made when reconciling lint output with code\n"
    "7. Open questions or blockers\n"
    "8. Smallest concrete next step (typically: re-run lint_openapi and address the highest-severity remaining issue)\n"
    "Include only claims supported by tool output; mark anything inferred as such.\n"
)

def build_oas_linter_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
):
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(
        fs, fmt=FileFormat(_format=_format), with_interaction_tools=False
    )
    oas_tools = openapi_tools(name=namespace, fs=fs)
    linter_tools = openapi_linter_tools(name=namespace)

    tools = [default_tool, *fs_tools, *mem_tools, *oas_tools, *linter_tools]

    return build_worker(
        name=name,
        instruction=OAS_LINTER_PROMPT,
        description="software engineering agent",
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        with_elide=False,
    )
