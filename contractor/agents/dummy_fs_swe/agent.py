from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.tools.fs import FileFormat, RootedLocalFileSystem, ro_file_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.podman import PodmanContainer
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

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

DUMMY_SWE_DESCRIPTION: Final[str] = (
    "Professional software engineer focused on implementing and validating assigned subtasks."
)

DUMMY_SUMMARIZATION_MESSAGE: Final[str] = (
    "You have reached context limit.Summarize your progress and call report tool."
    + _prepare_worker_instructions(SubtaskFormatter("xml"))
)

DUMMY_MODEL = LiteLlm(
    model="tgpt-qwen-3.5",
    timeout=300,
)

playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"

sandbox = PodmanContainer(
    name="contractor_planner_sandbox",
    image="docker.io/ubuntu:jammy",
    mounts=[playground_path],
    commands=None,
    ro_mode=False,
    workdir="/",
)

fs = RootedLocalFileSystem(root_path=playground_path)


mem_tools = memory_tools(name="swe", fmt=MemoryFormat("json"))
fs_tools = ro_file_tools(fs, fmt=FileFormat("json"))

tools = [default_tool, *fs_tools, *mem_tools]

callback_adapter = CallbackAdapter(agent_name="dummy_fs_swe")
callback_adapter.register(TokenUsageCallback())
callback_adapter.register(
    SummarizationLimitCallback(max_tokens=80000, message=DUMMY_SUMMARIZATION_MESSAGE)
)
callback_adapter.register(
    InvalidToolCallGuardrailCallback(
        tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
    )
)

dummy_fs_swe = LlmAgent(
    name="dummy_fs_swe",
    description=DUMMY_SWE_DESCRIPTION,
    instruction=DUMMY_SWE_PROMPT,
    model=DUMMY_MODEL,
    tools=tools,
    **callback_adapter(),
)

root_agent = dummy_fs_swe
