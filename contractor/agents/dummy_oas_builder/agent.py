from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final
from langfuse import get_client
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.tools.podman import PodmanContainer
from contractor.tools.fs import file_tools, RootedLocalFileSystem, FileFormat
from contractor.tools.openapi import openapi_tools
from contractor.tools.tasks import (
    task_tools,
    TASK_PLANNING_PROMPT,
    _prepare_worker_instructions,
    TaskFormat,
)

from contractor.tools.memory import memory_tools

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

DUMMY_SWE_PROMPT: Final[str] = (
    "You are a professional, helpful Software Engineer (SWE) agent.\n"
    "You must complete the currently assigned subtask to the best of your ability.\n"
    "\n"
    "Operating rules:\n"
    "- First, read the assignment\n"
    "- Use grep, ls, glob and read_file to inspect the file content."
    "- Use code_exexution to run custom commands, implement changes, and run checks/tests.\n"
    "- Prefer small, safe, verifiable steps. If something is unclear, infer reasonable defaults and proceed.\n"
    "- Do not stop early: keep working until the subtask is completed or you are blocked by missing inputs.\n"
    "- When blocked, report what you tried, what failed, and the smallest concrete next step.\n"
    "\n"
    "TOOLS:\n"
    "- grep: Regex search across a file or directory tree.\n"
    "- ls: List files and directories.\n"
    "- read_file: Read the contents of a file.\n"
    "- glob: Glob search across a file or directory tree.\n"
    "- code_execution: execute bash commands.\n"
    "- read_memory: read the memory from the memory store.\n"
    "- write_memory: write the memory to the memory store.\n"
    "- list_memories: list all the memories in the memory store.\n"
    "- list_tags: list all the tags in the memory store.\n"
    "IMPORTANT: always write useful information to the memory\n"
    "\n"
)


DUMMY_PLANNER_PROMPT: Final[str] = (
    "You are a helpful assistant, task manager, and execution planner.\n"
    "Your job is to translate the user's request into a small, reliable plan and drive it to completion.\n"
    f"\n\n{TASK_PLANNING_PROMPT}"
)

DUMMY_SWE_DESCRIPTION: Final[str] = (
    "Professional software engineer focused on implementing and validating assigned subtasks."
)

DUMMY_SUMMARIZATION_MESSAGE: Final[str] = (
    "You have reached context limit.Summarize your progress and call report tool."
    + _prepare_worker_instructions(TaskFormat("xml"))
)

DUMMY_PLANNER_DESCRIPTION: Final[str] = "Helpful asistant. Professional task manager."

DUMMY_MODEL = LiteLlm(
    model="tgpt-qwen3-235b-a22b-instruct-2507",
    timeout=300,
)

playground_path = (
    Path(__file__).parent.parent.parent.parent / "tests" / "playground" / "notes"
)

sandbox = PodmanContainer(
    name="contractor_planner_sandbox",
    image="docker.io/ubuntu:jammy",
    mounts=[playground_path],
    commands=None,
    ro_mode=False,
    workdir="/",
)

fs = RootedLocalFileSystem(root_path=playground_path)


def default_tool(meta: dict[str, Any]) -> dict:
    """
    default_tool: You must not use this tool. This is safeguard mechanism.
        Args:
            meta: meta information about failed tool call
        Returns:
            instructions
    """

    return {"error": f"tool {meta.get('func_name')} is not available!"}


mem_tools = memory_tools("swe")
fs_tools = file_tools(fs, fmt=FileFormat("json"))
oas_tools = openapi_tools("playground")

tools = [default_tool, *fs_tools, *sandbox.tools(), *mem_tools, *oas_tools]

callback_adapter = CallbackAdapter(agent_name="dummy_swe")
callback_adapter.register(TokenUsageCallback())
callback_adapter.register(
    SummarizationLimitCallback(max_tokens=50000, message=DUMMY_SUMMARIZATION_MESSAGE)
)
callback_adapter.register(
    InvalidToolCallGuardrailCallback(
        tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
    )
)

dummy_swe = LlmAgent(
    name="dummy_swe",
    description=DUMMY_SWE_DESCRIPTION,
    instruction=DUMMY_SWE_PROMPT,
    model=DUMMY_MODEL,
    tools=tools,
    **callback_adapter(),
)

fmt = TaskFormat("xml")
planning_tools = task_tools(
    name="dummy_planner",
    max_tasks=15,
    worker=dummy_swe,
    fmt=fmt,
    use_output_schema=False,
)


tools = [default_tool, *planning_tools, *mem_tools]

callback_adapter = CallbackAdapter(agent_name="dummy_planner")
callback_adapter.register(TokenUsageCallback())
callback_adapter.register(
    InvalidToolCallGuardrailCallback(
        tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
    )
)

dummy_planner = LlmAgent(
    name="dummy_planner",
    description=DUMMY_PLANNER_DESCRIPTION,
    instruction=DUMMY_PLANNER_PROMPT,
    model=DUMMY_MODEL,
    tools=tools,
    **callback_adapter(),
)

root_agent = dummy_planner
