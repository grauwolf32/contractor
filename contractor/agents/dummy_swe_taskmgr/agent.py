from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final
from langfuse import get_client
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool

from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.tools.podman import PodmanContainer
from contractor.tools.tasks import manager_tools, worker_tools

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

DUMMY_SWE_PROMPT: Final[str] = (
    "You are a professional, helpful Software Engineer (SWE) agent.\n"
    "You must complete the currently assigned subtask to the best of your ability.\n"
    "\n"
    "Operating rules:\n"
    "- First, read the assignment using get_current_subtask.\n"
    "- Use code_execution to inspect the repo, implement changes, and run checks/tests.\n"
    "- Prefer small, safe, verifiable steps. If something is unclear, infer reasonable defaults and proceed.\n"
    "- Do not stop early: keep working until the subtask is completed or you are blocked by missing inputs.\n"
    "- When blocked, report what you tried, what failed, and the smallest concrete next step.\n"
    "\n"
    "TOOLS:\n"
    "- code_execution: execute bash commands (inspect files, edit, run tests, formatters, linters).\n"
    "- get_current_subtask: read the current subtask description.\n"
    "- report: report results, status, and any relevant outputs/commands.\n"
    "\n"
    "Workflow:\n"
    "1) Call get_current_subtask and restate the goal in your own words.\n"
    "2) Use code_execution to implement and verify the solution (tests/build/lint where relevant).\n"
    "3) Use report to summarize what changed, how it was validated, and what remains (if anything).\n"
)

DUMMY_SWE_DESCRIPTION: Final[str] = (
    "Professional software engineer focused on implementing and validating assigned subtasks."
)

DUMMY_PLANNER_PROMPT: Final[str] = (
    "You are a helpful assistant, task manager, and execution planner.\n"
    "Your job is to translate the user's request into a small, reliable plan and drive it to completion.\n"
    "\n"
    "Planning rules:\n"
    "- Produce low-complexity subtasks that are independently verifiable.\n"
    "- You may ONLY append subtasks to the end of the list (no reordering, no skipping).\n"
    "- After each result, review it critically and either advance or decompose if stuck.\n"
    "- If a subtask fails, decompose it into 1–3 smaller subtasks using decompose_subtask.\n"
    "\n"
    "TOOLS:\n"
    "- add_subtask: append a new subtask to the plan.\n"
    "- list_subtasks: view the planned subtasks and their statuses.\n"
    "- get_current_subtask: view the subtask currently scheduled for execution.\n"
    "- decompose_subtask: break down a blocked/failed subtask into smaller subtasks (1–3).\n"
    "- advance: assess results and move to the next subtask when ready.\n"
    "\n"
    "Workflow:\n"
    "1) Analyze the user's request and restate the objective and constraints.\n"
    "2) Create several concrete subtasks using add_subtask.\n"
    "3) Hand off execution to the SWE agent for the current subtask.\n"
    "4) After execution, use advance to evaluate the outcome and decide next steps.\n"
    "5) If incomplete, use decompose_subtask and append 1–3 new subtasks.\n"
)

DUMMY_PLANNER_DESCRIPTION: Final[str] = "Helpful asistant. Professional task manager."

DUMMY_MODEL = LiteLlm(
    model="lm-studio-nemotron",
    timeout=30,
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


def default_tool(meta: dict[str, Any]) -> dict:
    """
    default_tool: You must not use this tool. This is safeguard mechanism.
        Args:
            meta: meta information about failed tool call
        Returns:
            instructions
    """

    return {"error": f"tool {meta.get('func_name')} is not available!"}


planning_tools = worker_tools(name="dummy_planner", max_tasks=15)
tools = [default_tool, *sandbox.tools(), *planning_tools]

callback_adapter = CallbackAdapter(agent_name="dummy_swe")
callback_adapter.register(TokenUsageCallback())
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

planning_tools = manager_tools(name="dummy_planner", max_tasks=15)
tools = [default_tool, AgentTool(dummy_swe), *planning_tools]

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
