"""Knowledge-base librarian agent.

A read-mostly worker that curates the run's accumulated knowledge. It has two
jobs, selected by the task it is given:

* **discovery** — explore the project source and the existing artifact pool to
  surface security-relevant facts that are *not yet* recorded, and write them as
  concise, tagged memory notes in its working namespace.
* **consolidation** — read memories and task results across *every* namespace
  (via the cross-namespace ``pool_*`` tools), then merge overlapping facts,
  resolve duplicates, and flag contradictions into a clean consolidated note
  set.

Its only write surface is the memory store of its own namespace
(``memory_tools``); everything else — source files, other namespaces' memories,
task results — is read-only via ``ro_file_tools`` / ``code_tools`` /
``artifact_pool_tools``. ``pool_search`` transparently uses the pgvector RAG
backend when one is supplied, else a keyword ranker.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.artifacts import BaseArtifactService
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.worker_factory import build_worker
from contractor.callbacks import default_tool
from contractor.tools.artifact_pool import ArtifactPoolBackend, artifact_pool_tools
from contractor.tools.code import code_tools
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.utils import load_prompt

LibrarianFormat = Literal["json", "xml", "yaml", "markdown"]

LIBRARIAN_PROMPT: Final[str] = load_prompt("librarian_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Namespaces and artifacts already reviewed\n"
    "2. Consolidated/new memory notes written so far (name, tags)\n"
    "3. Duplicates merged and contradictions flagged\n"
    "4. Namespaces/artifacts not yet reviewed\n"
    "5. Suggested next steps to finish curation\n"
)

# pool_read / pool_search return large bodies; treat them as heavy so the
# context-elision callback can drop stale results (mirrors read_file/grep).
_HEAVY_POOL_TOOLS: Final[tuple[str, ...]] = ("pool_read", "pool_search")


def build_librarian_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    artifact_service: BaseArtifactService,
    app_name: str,
    user_id: str,
    pool_masks: list[str] | None = None,
    pool_backend: ArtifactPoolBackend | None = None,
    _format: LibrarianFormat = "json",
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    elide_tool_results: Iterable[str] | None = None,
    elide_keep_last_n: int = 15,
    prompt: str | None = None,
) -> LlmAgent:
    instruction = prompt if prompt is not None else LIBRARIAN_PROMPT

    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    pool_tools = artifact_pool_tools(
        artifact_service=artifact_service,
        app_name=app_name,
        user_id=user_id,
        masks=pool_masks,
        backend=pool_backend,
    )
    fs_tools = ro_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)

    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *pool_tools]

    elide_targets = (
        list(elide_tool_results)
        if elide_tool_results is not None
        else [*_HEAVY_POOL_TOOLS]
    )

    return build_worker(
        name=name,
        instruction=instruction,
        description=(
            "knowledge-base librarian — discovers new security-relevant facts "
            "and consolidates existing memories/artifacts across namespaces "
            "into a clean, deduplicated note set."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_targets,
        elide_keep_last_n=elide_keep_last_n,
    )
