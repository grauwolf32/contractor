from __future__ import annotations

import re
from typing import Final, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import (InvalidToolCallGuardrailCallback,
                                             RepeatedToolCallCallback)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import SubtaskFormatter, task_tools
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

SUBTASK_PLANNING_PROMPT: Final[str] = load_prompt("planning_agent")

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
    callback_adapter.register(RepeatedToolCallCallback(threshold=2))

    instruction = SUBTASK_PLANNING_PROMPT.replace(
        "<<MAX_SUBTASKS>>", str(max_steps)
    )

    planning_agent = LlmAgent(
        name=agent_name,
        description=f"planner for queued task {name}",
        instruction=instruction,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return planning_agent
