from __future__ import annotations

from pathlib import Path
from typing import Final

from contractor.callbacks import default_tool
from cli.fs import RootedLocalFileSystem
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import memory_tools
from contractor.tools.openapi import openapi_tools
from contractor.tools.podman import PodmanContainer
from contractor.utils.settings import DEFAULT_MODEL

from contractor.agents.worker_factory import build_worker

DUMMY_SWE_PROMPT: Final[str] = (
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
    "- list_paths: List all API paths defined in the OpenAPI specification.\n"
    "- list_components: List all components available in the OpenAPI specification.\n"
    "- list_servers: List all configured API servers.\n"
    "- get_info: Retrieve general API metadata (title, version, description).\n"
    "- get_path: Get details for a specific API path.\n"
    "- get_component: Get details for a specific component.\n"
    "- set_info: Update general API metadata (title, version, description).\n"
    "- add_server: Add a new API server definition.\n"
    "- upsert_path: Create or update an API path definition.\n"
    "- upsert_component: Create or update a component definition.\n"
    "- remove_server: Remove an API server definition.\n"
    "- remove_path: Remove an API path definition.\n"
    "- remove_component: Remove a component definition.\n"
    "- get_full_openapi_schema: Retrieve the complete OpenAPI schema."
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

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached context limit.Summarize your progress and call report tool."
)

DUMMY_PLANNER_DESCRIPTION: Final[str] = "Helpful asistant. Professional task manager."

playground_path = Path(__file__).parent.parent.parent / "playground"

sandbox = PodmanContainer(
    name="contractor_oas_sandbox",
    image="docker.io/ubuntu:jammy",
    mounts=[playground_path],
    commands=None,
    ro_mode=False,
    workdir="/",
)

fs = RootedLocalFileSystem(root_path=playground_path)

mem_tools = memory_tools("swe")
fs_tools = ro_file_tools(fs=fs, fmt=FileFormat(_format="xml"))
oas_tools = openapi_tools("playground", fs)

tools = [default_tool, *fs_tools, *mem_tools, *oas_tools]

dummy_oas_builder = build_worker(
    name="dummy_oas_builder",
    instruction=DUMMY_SWE_PROMPT,
    description=DUMMY_SWE_DESCRIPTION,
    tools=tools,
    _format="xml",
    summarization_bullets=_SUMMARIZATION_BULLETS,
    max_tokens=80000,
    model=DEFAULT_MODEL,
    with_elide=False,
)

root_agent = dummy_oas_builder
