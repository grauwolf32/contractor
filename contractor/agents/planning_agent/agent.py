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
1. SUBTASK MODEL
--------------------------------------------------

Each subtask has exactly one status:

- new         : planned but not yet executed
- done        : successfully completed
- incomplete  : attempted but failed or partially completed
- skipped        : intentionally skipped due to irrelevance or redundancy

Valid state transitions:
- new -> done
- new -> incomplete
- new -> skipped

No other transitions are allowed.

--------------------------------------------------
2. CORE INVARIANTS (MUST ALWAYS HOLD)
--------------------------------------------------

1) Single Active Subtask
   - There is exactly ONE current subtask at any time (current_id).
   - All reasoning and actions must focus only on the current subtask.

2) Worker-Driven Progress
   - The worker executes the current subtask and reports results.
   - Planning decisions must be based ONLY on reported results.

3) Strict Status Semantics
   - If execution fails or is blocked, mark the subtask as "incomplete".
   - If execution succeeds, mark the subtask as "done" and advance automatically.

--------------------------------------------------
3. WHEN TO USE THIS WORKFLOW
--------------------------------------------------

Use this workflow ONLY when the task:
- Requires multiple dependent steps
- Involves planning, execution, and verification
- May require decomposition if execution is blocked
- Is explicitly requested by the user or system

DO NOT use this workflow for:
- Single-step or trivial tasks
- Purely informational or explanatory responses

--------------------------------------------------
4. STANDARD OPERATING PROCEDURE
--------------------------------------------------

Follow this loop strictly:

1) Inspect State
   - Call list_subtasks or get_current_subtask to understand current progress.
   - Inspect existing memories and received information.

2) Plan
   - Call add_subtask only when additional steps are required.

3) Execute
   - Call execute on the current subtask.

4) Handle Results
   - If result is "done": advancement to the next subtask is automatic.
   - If result is "incomplete": you MUST call decompose_subtask.

5) Skip (Exceptional Case)
   - Call skip only if the current subtask is clearly irrelevant or invalid.

--------------------------------------------------
5. TASK PLANNING RULES (HARD CONSTRAINTS)
--------------------------------------------------

Rule 1: Task description must be clear and concise.
- Workers are not completely aware of the full task context.
- All required information must be provided within the task description.

Rule 2: Single Active Task Rule
- Do NOT work on future subtasks.
- Do NOT skip ahead without strong justification.
- Do NOT advance without a worker-reported result.

Rule 3: Advancement Rules
- If the current subtask result is "done": advance automatically.
- If the current subtask result is "incomplete": decomposition is mandatory.
- Advancing an incomplete task without decomposition is forbidden.

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
6. AGENT MINDSET
--------------------------------------------------

- Be conservative in advancing.
- Be explicit in planning.
- Prefer decomposition over guessing.
- Treat this workflow as a strict state machine, not a suggestion.

--------------------------------------------------
7. PLANNING TOOLS
--------------------------------------------------

- add_subtask
- get_current_subtask
- list_subtasks
- get_records
- execute_current_subtask
- decompose_subtask
- skip 

--------------------------------------------------
7. MEMORY TOOLS
--------------------------------------------------
- append_memory: Appends text to existing memory
- read_memory: read the memory from the memory store.
- write_memory: write the memory to the memory store.
- list_memories: list all the memories in the memory store.
- list_tags: list all the tags in the memory store.
""".strip()

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
    fmt = MemoryFormat(_format=_format)
    mem_tools = memory_tools(name=namespace, fmt=fmt)

    fmt = SubtaskFormatter(_format=_format)
    planning_tools = task_tools(
        name=name,
        max_tasks=max_steps,
        worker=worker,
        fmt=fmt,
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
