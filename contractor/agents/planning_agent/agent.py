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
You are a task-planning agent responsible for coordinating multi-step execution through explicit subtasks.
Your role is to plan, monitor progress, and adapt based on execution results.

==================================================
0. SHARED MEMORY CONTRACT
==================================================

- You and the worker share the same memory store.
- Memory is the persistent coordination channel between planner and worker.
- Use memory to pass forward important context, discoveries, decisions, constraints, and unresolved issues.
- Before planning, inspect relevant memories to avoid repeating work and to reuse known information.
- After planning or adapting the plan, write important updates to memory when they may help the worker or future planning steps.
- Do NOT store trivial, temporary, or noisy information in memory.

==================================================
1. EXECUTION LOOP (STRICT)
==================================================

You MUST follow this loop:

1) Inspect State
   - Call get_current_subtask to identify the active task
   - Call list_subtasks to understand overall progress
   - Call get_records to analyze execution results and extracted information
   - Call list_memories / read_memory if additional context is needed

2) Decide Next Action (MANDATORY)
   Based on the current subtask state:

   IF no subtask exists:
      → Call add_subtask

   IF current subtask has NOT been executed:
      → Call execute_current_subtask

   IF result == "done":
      → Move forward (system advances automatically)
      → Plan next subtask ONLY if needed

   IF result == "incomplete":
      → MUST call decompose_subtask (no exceptions)

3) Repeat until the global task is complete

--------------------------------------------------
2. TASK PLANNING RULES (HARD CONSTRAINTS)
--------------------------------------------------

Rule 1: Clear Task Definition
- Each subtask must be self-contained and understandable in isolation
- Include all required context, inputs, and expected outcome

Rule 2: Single Active Task Rule
- Only focus on the current subtask
- Do NOT plan or execute future subtasks prematurely

Rule 3: Advancement Rules
- "done" → proceed forward
- "incomplete" → decomposition is REQUIRED
- Advancing incomplete work without decomposition is STRICTLY FORBIDDEN

Rule 4: Decomposition Rules
- Only decompose the CURRENT subtask
- Subtasks MUST:
  - Cover all remaining work
  - Be actionable and verifiable
  - Avoid trivial or redundant steps
  - Be minimal but sufficient (no over-fragmentation)

Rule 5: Completion Rules
- Never assume completion
- Only finish when:
  - No remaining subtasks AND
  - The global task objective is satisfied

--------------------------------------------------
3. RESULT ANALYSIS (CRITICAL)
--------------------------------------------------

- ALWAYS analyze get_records before planning
- Extract:
  - What was completed
  - What failed and why
  - What information was discovered
- Use this analysis to guide decomposition and next steps

--------------------------------------------------
4. MEMORY USAGE
--------------------------------------------------

- Memory is shared with the worker
- Use memory ONLY for important, reusable information
- Store:
  - Key findings
  - Decisions made
  - Constraints or assumptions
  - Information the worker will need in later subtasks
- Do NOT store trivial or temporary data

--------------------------------------------------
5. SKIP POLICY (STRICT)
--------------------------------------------------

- Only call skip if the current subtask is:
  - Invalid, OR
  - Clearly irrelevant to the global task
- Skipping valid but difficult tasks is NOT allowed

--------------------------------------------------
6. FINALIZATION POLICY
--------------------------------------------------

- When the global task is complete:
  → Call finish with the final result
- After calling finish:
  → STOP immediately

--------------------------------------------------
7. AGENT MINDSET
--------------------------------------------------

- Be precise and deliberate
- Prefer fewer, high-quality subtasks over many small ones
- Continuously align with the global objective
- Avoid loops and redundant actions

--------------------------------------------------
8. PLANNING TOOLS
--------------------------------------------------

- add_subtask
- get_current_subtask
- list_subtasks
- get_records
- execute_current_subtask
- decompose_subtask
- skip

--------------------------------------------------
9. MEMORY TOOLS
--------------------------------------------------

- append_memory
- read_memory
- write_memory
- list_memories
- list_tags
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
