from __future__ import annotations

from pathlib import Path
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
from contractor.tools.code import (PathResolver, code_graph_tools, code_tools,
                                   strip_prefix_resolver)
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import (SubtaskFormatter,
                                    _prepare_worker_instructions)
from contractor.tools.vuln import (VulnerabilityReportFormat,
                                   vulnerability_report_tools)
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

CODE_GRAPH_AGENT_PROMPT: Final[str] = load_prompt("code_graph_agent")


def _build_path_resolver(
    fs: AbstractFileSystem,
    project_root: Path | str,
) -> Optional[PathResolver]:
    """Pick a graph→fs path translator from whatever ``fs`` exposes.

    The contract here is intentionally narrow: any fs that holds a
    ``root_path`` attribute (today: ``RootedLocalFileSystem`` and
    overlays wrapping it) is treated as a sandboxed view of the same
    project tree trailmark parsed, and graph paths get re-rooted so
    they match the fs's virtual ``/relative/path.py`` form. Anything
    else (plain ``LocalFileSystem``, remote backends) gets no
    translation — callers can pass a custom resolver via
    ``code_graph_tools`` if their FS has a different convention.
    """
    # Walk through known wrapper attributes — overlays may nest several
    # layers deep (MemoryOverlayFileSystem.base_fs → …).
    current = fs
    for _ in range(8):
        fs_root = getattr(current, "root_path", None)
        if fs_root is not None:
            return strip_prefix_resolver(str(fs_root))
        inner = (
            getattr(current, "base_fs", None)
            or getattr(current, "fs", None)
            or getattr(current, "_fs", None)
        )
        if inner is None or inner is current:
            return None
        current = inner
    return None


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Entrypoint identified (graph node id, file, function)\n"
        "2. Graph hops walked so far (callees/callers/paths_between)\n"
        "3. Functions annotated (list with depths)\n"
        "4. Sinks and validations discovered\n"
        "5. Files modified\n"
        "6. Branches not yet traced or partially traced\n"
        "7. Suggested next graph queries to complete the trace\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


def build_code_graph_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    project_root: Path | str,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    enable_vuln_reporting: bool = False,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
    prompt: Optional[str] = None,
    graph_language: str = "auto",
) -> LlmAgent:
    """Build a trace-annotation worker backed by a trailmark code graph.

    Identical in role to ``trace_agent`` but with the call-graph tool
    set added — and the prompt steers it toward graph-first navigation.
    ``project_root`` must be a host-filesystem path: trailmark parses
    real files, so passing a memory-overlay FS would not extend to graph
    building (the overlay still applies to ``read_file`` / ``insert_line``
    used by the agent for annotation).
    """
    instruction = prompt if prompt is not None else CODE_GRAPH_AGENT_PROMPT

    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs)
    # Trailmark returns host-FS absolute paths; the agent's `fs` here is
    # a sandboxed view (RootedLocalFileSystem under any overlays) where
    # files live at `/relative/path`. Translate so graph results compose
    # with the file tools sharing the same `fs`. When the underlying fs
    # has no `root_path` (e.g. plain LocalFileSystem) we leave graph
    # paths as-is.
    resolver = _build_path_resolver(fs, project_root)
    gtools = code_graph_tools(
        project_root,
        language=graph_language,
        path_resolver=resolver,
    )
    tools = [default_tool, *fs_tools, *mem_tools, *ctools, *gtools]
    if enable_vuln_reporting:
        vuln_tools = vulnerability_report_tools(
            name=namespace,
            fmt=VulnerabilityReportFormat(_format=_format),
        )
        tools.extend(vuln_tools)

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
        description="trace annotation worker with call-graph navigation",
        instruction=instruction,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )
