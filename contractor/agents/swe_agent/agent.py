from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.fs import FileFormat, RootedLocalFileSystem, file_tools
from contractor.tools.memory import memory_tools
from contractor.tools.podman import PodmanContainer
from contractor.tools.tasks import (
    SUBTASK_PLANNING_PROMPT,
    SubtaskFormatter,
    _prepare_worker_instructions,
    task_tools,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

SWE_PROMPT: Final[str] = (
    "You are a professional, helpful Software Engineer (SWE) agent.\n"
    "You must complete the currently assigned subtask to the best of your ability.\n"
    "\n"
    "Operating rules:\n"
    "- First, read the assignment\n"
    "- Use grep, ls, glob and read_file to inspect the file content."
    "- Prefer small, safe, verifiable steps. If something is unclear, infer reasonable defaults and proceed.\n"
    "- Do not stop early: keep working until the subtask is completed or you are blocked by missing inputs.\n"
    "- When blocked, report what you tried, what failed, and the smallest concrete next step.\n"
    "\n"
    "TOOLS:\n"
    "- grep: Regex search across a file or directory tree.\n"
    "- ls: List files and directories.\n"
    "- read_file: Read the contents of a file.\n"
    "- glob: Glob search across a file or directory tree.\n"
    "- read_memory: read the memory from the memory store.\n"
    "- write_memory: write the memory to the memory store.\n"
    "- list_memories: list all the memories in the memory store.\n"
    "- list_tags: list all the tags in the memory store.\n"
    "IMPORTANT: always write useful information to the memory\n"
    "\n"
)


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached context limit. Summarize your progress and call report tool."
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


SWE_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def default_tool(meta: dict[str, Any]) -> dict:
    """
    default_tool: You must not use this tool. This is safeguard mechanism.
        Args:
            meta: meta information about failed tool call
        Returns:
            instructions
    """

    return {"error": f"tool {meta.get('func_name')} is not available!"}


def build_swe_agent(
    name: str,
    namespace: str,
    fs,
    *,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
):
    mem_tools = memory_tools(namespace)
    fs_tools = file_tools(fs, fmt=FileFormat(_format=format))

    tools = [default_tool, *fs_tools, *mem_tools]

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        SummarizationLimitCallback(
            max_tokens=max_tokens, message=summarization_message(_format=_format)
        )
    )
    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
        )
    )

    swe_agent = LlmAgent(
        name=name,
        description="software engineering agent",
        instruction=SWE_PROMPT,
        model=model if model is not None else SWE_MODEL,
        tools=tools,
        **callback_adapter(),
    )


playground_path = (
    Path(__file__).parent.parent.parent.parent / "tests" / "playground" / "cloud"
)

fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_swe_agent(
    name="swe_agent",
    namespace="code_review",
    fs=fs,
)
