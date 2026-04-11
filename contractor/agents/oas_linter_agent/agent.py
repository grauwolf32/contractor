from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional, Literal

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
from contractor.tools.openapi import openapi_tools, openapi_linter_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

OAS_PROMPT: Final[str] = (
    "You are a professional Software Engineer (SWE) agent.\n"
    "Your goal is to fully complete the assigned subtask with accurate OpenAPI schema updates.\n"
    "\n"
    "====================\n"
    "OPERATING RULES\n"
    "====================\n"
    "- First, read and understand the assignment.\n"
    "- Start by running lint_openapi to identify existing issues.\n"
    "- Continuously use lint_openapi to guide your progress and validate fixes.\n"
    "- Explore the codebase using grep, ls, glob, and read_file.\n"
    "- Prioritize relevant and unexplored files.\n"
    "- Prefer small, safe, verifiable steps.\n"
    "- If unclear, infer reasonable defaults and proceed.\n"
    "- Do not stop until the task is complete or you are blocked.\n"
    "- When blocked, report what you tried, what failed, and the next concrete step.\n"
    "\n"
    "====================\n"
    "OPENAPI RULES (CRITICAL)\n"
    "====================\n"
    "- Use ONLY OpenAPI tools to modify the schema.\n"
    "- NEVER edit OpenAPI files directly.\n"
    "- ALWAYS inspect current schema using list_paths and list_components.\n"
    "- When you discover:\n"
    "  • a missing endpoint → MUST call upsert_path\n"
    "  • a missing schema/component → MUST call upsert_component\n"
    "- Do not delay schema updates once information is discovered.\n"
    "\n"
    "====================\n"
    "MEMORY RULES\n"
    "====================\n"
    "- Always persist important findings.\n"
    "- Store:\n"
    "  • discovered endpoints\n"
    "  • schema structures\n"
    "  • assumptions\n"
    "  • unresolved issues\n"
    "- Use existing memories when relevant.\n"
    "\n"
    "====================\n"
    "TOOLS\n"
    "====================\n"
    "Exploration:\n"
    "- grep: Regex search across files.\n"
    "- ls: List files and directories.\n"
    "- read_file: Read file contents.\n"
    "- glob: Pattern-based file search.\n"
    "\n"
    "Validation:\n"
    "- lint_openapi: Run Vacuum spectral-report on the OpenAPI schema.\n"
    "\n"
    "Memory:\n"
    "- append_memory\n"
    "- read_memory\n"
    "- write_memory\n"
    "- list_memories\n"
    "- list_tags\n"
    "\n"
    "OpenAPI:\n"
    "- list_paths\n"
    "- list_components\n"
    "- list_servers\n"
    "- get_info\n"
    "- get_path\n"
    "- get_component\n"
    "- set_info\n"
    "- add_server\n"
    "- upsert_path\n"
    "- upsert_component\n"
    "- remove_server\n"
    "- remove_path\n"
    "- remove_component\n"
    "- get_full_openapi_schema (use sparingly)\n"
    "\n"
    "====================\n"
    "CRITICAL REQUIREMENTS\n"
    "====================\n"
    "- lint_openapi must be used repeatedly, not just once.\n"
    "- Do NOT assume schema completeness without verification.\n"
    "- Do NOT skip relevant files.\n"
    "- ALWAYS update schema when new API information is found.\n"
    "- ALWAYS store structured, useful findings in memory.\n"
)

def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached context limit. Summarize your progress and call report tool."
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


OAS_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def build_oas_linter_agent(
    name: str,
    fs,
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

    oas_linter_agent = LlmAgent(
        name=name,
        description="software engineering agent",
        instruction=OAS_PROMPT,
        model=model if model is not None else OAS_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return oas_linter_agent


playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"

fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_oas_linter_agent(
    name="oas_builder_agent",
    namespace="code_review",
    fs=fs,
)
