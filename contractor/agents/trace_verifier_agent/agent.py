from __future__ import annotations

from typing import Final, Iterable, Literal, Optional

from fsspec import AbstractFileSystem
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import (FunctionResultsRemovalCallback,
                                          SummarizationLimitCallback)
from contractor.callbacks.guardrails import (InvalidToolCallGuardrailCallback,
                                             RepeatedToolCallCallback)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools import DEFAULT_HEAVY_TOOLS
from contractor.tools.code import attach_graph_tools_if_local, code_tools
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import (SubtaskFormatter,
                                    _prepare_worker_instructions)
from contractor.tools.vuln import (VerifiedFindingFormat,
                                   VulnerabilityReportFormat,
                                   verification_tools,
                                   vulnerability_report_tools)
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

TRACE_VERIFIER_PROMPT: Final[str] = load_prompt("trace_verifier_agent")

# Tools we keep from `vulnerability_report_tools` for the verifier. The
# verifier reads upstream findings but must not author new ones.
_READ_ONLY_VULN_TOOL_NAMES: frozenset[str] = frozenset(
    {"get_vulnerability", "list_vulnerabilities"}
)


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Finding under verification (name, sink place, claimed kind)\n"
        "2. Entry point and auth state being assumed\n"
        "3. Data-flow steps traced so far (file:line)\n"
        "4. Guards / validators discovered (passing or refuting)\n"
        "5. Approaches already refuted\n"
        "6. Current tentative verdict and what is missing to lock it\n"
        "7. Smallest next code lookup that would resolve the open question\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


def build_trace_verifier_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    source_namespace: Optional[str] = None,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    elide_budget_chars: int = 0,
    with_graph_tools: bool = False,
    prompt: Optional[str] = None,
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
        if t.__name__ in _READ_ONLY_VULN_TOOL_NAMES
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

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens,
            message=summarization_message(_format=_format),
        )
    )
    elide_targets = (
        list(elide_tool_results)
        if elide_tool_results is not None
        else list(DEFAULT_HEAVY_TOOLS)
    )
    if elide_targets:
        callback_adapter.register(
            FunctionResultsRemovalCallback(
                keep_last_n=elide_keep_last_n,
                keep_budget_chars=elide_budget_chars,
                target_tools=elide_targets,
            )
        )
    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools,
            default_tool_name="default_tool",
            default_tool_arg="meta",
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    return LlmAgent(
        name=name,
        description=(
            "trace verifier agent — static, attacker-role-play assessment of "
            "a single upstream vulnerability finding; produces a structured "
            "exploit_path and verdict, never authors new findings."
        ),
        instruction=instruction,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )
