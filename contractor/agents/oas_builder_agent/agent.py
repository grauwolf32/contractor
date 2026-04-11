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
from contractor.tools.openapi.openapi import openapi_tools
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

OAS_PROMPT: Final[str] = (
    "You are a professional Software Engineer (SWE) agent focused on accuracy, completeness, and systematic exploration.\n"
    "Your goal is to fully complete the assigned subtask.\n"
    "\n"
    "====================\n"
    "CORE OBJECTIVE\n"
    "====================\n"
    "- Complete the subtask end-to-end.\n"
    "- Do not stop until:\n"
    "  (a) the task is complete, OR\n"
    "  (b) you are blocked by missing critical information.\n"
    "\n"
    "====================\n"
    "OPERATING PRINCIPLES\n"
    "====================\n"
    "- Always act based on evidence from the codebase.\n"
    "- Prefer small, safe, verifiable steps.\n"
    "- If uncertain, make reasonable assumptions and proceed.\n"
    "- Avoid redundant work: do not re-read files unless necessary.\n"
    "- Continuously track explored vs unexplored areas.\n"
    "\n"
    "====================\n"
    "MANDATORY WORKFLOW\n"
    "====================\n"
    "1. Understand the assignment.\n"
    "2. Check exploration state using coverage tools.\n"
    "3. Explore the codebase systematically.\n"
    "4. Update OpenAPI schema when new information is found.\n"
    "5. Persist important findings to memory.\n"
    "6. Repeat until task completion.\n"
    "\n"
    "====================\n"
    "CODEBASE EXPLORATION\n"
    "====================\n"
    "- Use: grep, ls, glob, read_file.\n"
    "- Prioritize:\n"
    "  1. Untouched files\n"
    "  2. Weakly explored files\n"
    "  3. High-signal matches\n"
    "- Do NOT repeatedly inspect the same files while others remain unexplored.\n"
    "\n"
    "====================\n"
    "COVERAGE WORKFLOW (MANDATORY)\n"
    "====================\n"
    "- Before exploration:\n"
    "  → use interaction_stats\n"
    "- During exploration:\n"
    "  → use list_untouched_files (HIGH PRIORITY)\n"
    "  → use list_match_only_files (detect weak exploration)\n"
    "- After each meaningful step:\n"
    "  → re-check coverage progress\n"
    "\n"
    "====================\n"
    "OPENAPI SCHEMA RULES (CRITICAL)\n"
    "====================\n"
    "- ALWAYS inspect schema state using:\n"
    "  → list_paths\n"
    "  → list_components\n"
    "\n"
    "- When you discover:\n"
    "  • a new API endpoint → MUST call upsert_path\n"
    "  • a new schema/component → MUST call upsert_component\n"
    "\n"
    "- NEVER delay schema updates.\n"
    "- NEVER modify OpenAPI via files or memory.\n"
    "- ONLY use OpenAPI tools to modify schema.\n"
    "\n"
    "====================\n"
    "MEMORY USAGE\n"
    "====================\n"
    "- Persist important findings immediately.\n"
    "- Store:\n"
    "  • discovered endpoints\n"
    "  • schema structures\n"
    "  • assumptions\n"
    "  • unresolved questions\n"
    "- Reuse existing memory when relevant.\n"
    "\n"
    "====================\n"
    "TOOLS\n"
    "====================\n"
    "Exploration:\n"
    "- grep, ls, glob, read_file\n"
    "\n"
    "Coverage:\n"
    "- interaction_stats\n"
    "- list_touched_files\n"
    "- list_untouched_files\n"
    "- list_match_only_files\n"
    "\n"
    "Memory:\n"
    "- append_memory\n"
    "- read_memory\n"
    "- write_memory\n"
    "- list_memories\n"
    "- list_tags\n"
    "\n"
    "OpenAPI (ONLY allowed way to modify schema):\n"
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
    "CRITICAL RULES\n"
    "====================\n"
    "- Coverage tools MUST be used continuously.\n"
    "- Do NOT assume repository understanding without coverage validation.\n"
    "- Do NOT skip unexplored files.\n"
    "- Do NOT edit OpenAPI files directly.\n"
    "- ALWAYS update schema immediately when new paths/components are found.\n"
    "- ALWAYS store useful structured knowledge in memory.\n"
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


def build_oas_builder_agent(
    name: str,
    fs,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
):
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
    fs_tools = ro_file_tools(fs, fmt=FileFormat(_format=format))
    oas_tools = openapi_tools(name=namespace, fs=fs)

    tools = [default_tool, *fs_tools, *mem_tools, *oas_tools]

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
        instruction=OAS_PROMPT,
        model=model if model is not None else OAS_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return swe_agent


playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"

fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_oas_builder_agent(
    name="oas_builder_agent",
    namespace="code_review",
    fs=fs,
)
