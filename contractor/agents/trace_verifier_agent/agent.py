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
from contractor.tools.vuln import (
    READ_ONLY_VULN_TOOL_NAMES,
    VerifiedFindingFormat,
    VulnerabilityReportFormat,
    verification_tools,
    vulnerability_report_tools,
)
from contractor.utils import load_prompt

TRACE_VERIFIER_PROMPT: Final[str] = load_prompt("trace_verifier_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached the context limit. Summarize your progress:\n"
    "1. Finding under verification (name, sink place, claimed kind)\n"
    "2. Entry point and auth state being assumed\n"
    "3. Data-flow steps traced so far (file:line)\n"
    "4. Guards / validators discovered (passing or refuting)\n"
    "5. Approaches already refuted\n"
    "6. Current tentative verdict and what is missing to lock it\n"
    "7. Smallest next code lookup that would resolve the open question\n"
)


def build_trace_verifier_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    source_namespace: str | None = None,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    elide_tool_results: Iterable[str] | None = None,
    elide_keep_last_n: int = 15,
    with_graph_tools: bool = False,
    prompt: str | None = None,
) -> LlmAgent:
    """Static, code-evidence-based verifier of a single vulnerability finding.

    OpenAnt Stage-2 style: attacker role-play, no HTTP probes, no edits.
    Reads upstream findings from ``source_namespace`` (defaults to
    ``namespace`` for runs where verification shares scope with the
    upstream stage) and writes verdicts via ``verification_tools``.
    """
    instruction = prompt if prompt is not None else TRACE_VERIFIER_PROMPT
    src_ns = source_namespace if source_namespace is not None else namespace

    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(fs, fmt=FileFormat(_format=_format))
    ctools = code_tools(fs=fs)
    gtools = attach_graph_tools_if_local(fs) if with_graph_tools else []

    # Read-only access to upstream finding store. Filtered to drop the
    # writer tool so the verifier cannot fabricate new VulnerabilityReports.
    upstream_tools = [
        t
        for t in vulnerability_report_tools(
            name=src_ns,
            fmt=VulnerabilityReportFormat(_format=_format),
        )
        if t.__name__ in READ_ONLY_VULN_TOOL_NAMES
    ]

    verif_tools = verification_tools(
        name=src_ns,
        fmt=VerifiedFindingFormat(_format=_format),
    )

    tools = [
        default_tool,
        *fs_tools,
        *mem_tools,
        *ctools,
        *gtools,
        *upstream_tools,
        *verif_tools,
    ]

    return build_worker(
        name=name,
        instruction=instruction,
        description=(
            "trace verifier agent — static, attacker-role-play assessment of "
            "a single upstream vulnerability finding; produces a structured "
            "exploit_path and verdict, never authors new findings."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        elide_tool_results=elide_tool_results,
        elide_keep_last_n=elide_keep_last_n,
    )
