from __future__ import annotations

from typing import Final, Iterable, Optional, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from fsspec import AbstractFileSystem

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import (
    FunctionResultsRemovalCallback,
    SummarizationLimitCallback,
)
from contractor.tools import DEFAULT_HEAVY_TOOLS
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL
from contractor.tools.fs import FileFormat, rw_file_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.code import code_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

SWE_EDIT_PROMPT: Final[str] = load_prompt("swe_edit_agent")

def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached context limit. Summarize your progress:\n"
        "1. Subtask understanding and goal\n"
        "2. Files explored and key findings\n"
        "3. Edits applied so far (paths and brief description)\n"
        "4. Outstanding edits or verification steps\n"
        "5. Blockers or open questions\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )

def build_swe_edit_agent(
    name: str,
    fs: AbstractFileSystem,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    elide_tool_results: Optional[Iterable[str]] = None,
    elide_keep_last_n: int = 15,
) -> LlmAgent:
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = rw_file_tools(
        fs,
        fmt=FileFormat(_format=_format),
        with_interaction_tools=True,
    )
    ctools = code_tools(fs=fs)

    tools = [default_tool, *fs_tools, *mem_tools, *ctools]

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens, message=summarization_message(_format=_format)
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
            tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    swe_edit_agent = LlmAgent(
        name=name,
        description="software engineering agent with edit capabilities",
        instruction=SWE_EDIT_PROMPT,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return swe_edit_agent
