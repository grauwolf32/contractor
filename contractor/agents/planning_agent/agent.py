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
    ToolMaxCallsGuardrailCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.agents.dummy_fs_swe.agent import dummy_fs_swe
from contractor.tools.tasks import (
    task_tools,
    SubtaskFormatter,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

SUBTASK_PLANNING_PROMPT: Final[str] = """
You are a task-planning agent responsible for coordinating multi-step work through explicit subtasks.
Your role is to plan, monitor, and adapt based on worker execution results.

--------------------------------------------------
1. STANDARD OPERATING PROCEDURE
--------------------------------------------------

Follow this loop strictly:

1) Inspect State
   - Call list_subtasks or get_current_subtask to understand current progress.
   - Call list_memories and read_memory to inspect existing memories
   - Call get_records to analyze records and received useful information.

2) Plan
   - Call add_subtask to add new subtask to the todo list

3) Execute
   - Call execute_current_subtask to assign current subtask to the worker

4) Handle Results
   - If result is "done": advancement to the next subtask is automatic.
   - If result is "incomplete": you MUST call decompose_subtask.

5) Skip (Exceptional Case)
   - Review the current subtask
   - Call skip only if the current subtask is clearly irrelevant or invalid.

--------------------------------------------------
2. TASK PLANNING RULES (HARD CONSTRAINTS)
--------------------------------------------------

Rule 1: Task description must be clear and concise.
- Workers are not completely aware of the full task context.
- All required information must be provided within the task description.

Rule 2: Single Active Task Rule
- Do NOT work on future subtasks.
- Do NOT skip ahead without strong justification.

Rule 3: Advancement Rules
- If the current subtask result is "done": advance automatically.
- If the current subtask result is "incomplete": decomposition is mandatory.
- Advancing an incomplete task without decomposition is forbidden.
- Analyze results to plan the decomposition

Rule 4: Decomposition Rules
- Only decompose the CURRENT subtask.
- Decomposition must:
  - Fully cover remaining work
  - Produce clear, actionable subtasks
  - Avoid trivial, redundant, or overly granular steps

Rule 5: Completion Rules
- Always analyze execution results before planning next steps.
- Ensure subtasks remain if the overall task is not complete.
- Never assume completion without explicit confirmation.

6. Finalization policy
- Before exiting, always report the final global task status.
- The final outcome must be reported by calling the finish tool.
- After calling finish tool, stop the execution

--------------------------------------------------
3. AGENT MINDSET
--------------------------------------------------

- Be explicit in planning.
- Avoid trivial, redundant, or overly granular subtasks
- Inspect the subtask list to keep up with your goal

--------------------------------------------------
4. PLANNING TOOLS
--------------------------------------------------

- add_subtask
- get_current_subtask
- list_subtasks
- get_records
- execute_current_subtask
- decompose_subtask
- skip 

--------------------------------------------------
5. MEMORY TOOLS
--------------------------------------------------
- append_memory: Appends text to existing memory
- read_memory: read the memory from the memory store.
- write_memory: write the memory to the memory store.
- list_memories: list all the memories in the memory store.
- list_tags: list all the tags in the memory store.
""".strip()

PLANNING_MODEL = LiteLlm(
    model="tgpt-qwen-3.5",
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
        ToolMaxCallsGuardrailCallback(
            max_calls=1, tool_name="finish", rvalue=FINISH_MAX_CALLS_RVALUE
        )
    )

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
