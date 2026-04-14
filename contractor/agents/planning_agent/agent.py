from __future__ import annotations

import os
import re
from typing import Final, Optional, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.utils import load_prompt
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.agents.dummy_fs_swe.agent import dummy_fs_swe
from contractor.tools.tasks import (
    task_tools,
    SubtaskFormatter,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

SUBTASK_PLANNING_PROMPT: Final[str] = load_prompt("planning_agent")

PLANNING_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)

FINISH_MAX_CALLS_RVALUE: dict[str, str] = {
    "error": "The 'finish' tool has already been called once. Stop execution."
}


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return safe or "task"


def build_planning_agent(
    name: str,
    *,
    namespace: str,
    worker: LlmAgent,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_steps: int = 15,
    use_output_schema: bool = False,
    model: Optional[LiteLlm] = None,
):
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))

    planning_tools = task_tools(
        name=name,
        max_tasks=max_steps,
        worker=worker,
        fmt=SubtaskFormatter(_format=_format),
        use_output_schema=use_output_schema,
    )

    tools = [default_tool, *planning_tools, *mem_tools]

    agent_name = f"task_planner_{_safe_identifier(name)}"
    callback_adapter = CallbackAdapter(agent_name=agent_name)
    callback_adapter.register(TokenUsageCallback())

    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
        )
    )

    planning_agent = LlmAgent(
        name=agent_name,
        description=f"planner for queued task {name}",
        instruction=SUBTASK_PLANNING_PROMPT,
        model=model if model is not None else PLANNING_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return planning_agent


root_agent = build_planning_agent(
    name="swe",
    namespace="swe",
    model=PLANNING_MODEL,
    worker=dummy_fs_swe,
)
