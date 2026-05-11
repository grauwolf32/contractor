from __future__ import annotations

from typing import Final, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks import default_tool
from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import (InvalidToolCallGuardrailCallback,
                                             RepeatedToolCallCallback)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.fs import FileFormat, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.openapi import openapi_linter_tools, openapi_tools
from contractor.tools.tasks import (SubtaskFormatter,
                                    _prepare_worker_instructions)
from contractor.utils import load_prompt
from contractor.utils.settings import DEFAULT_MODEL

OAS_LINTER_PROMPT: Final[str] = load_prompt("oas_linter_agent")

def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached the context limit. Summarize your progress:\n"
        "1. Subtask objective as you understand it\n"
        "2. Initial lint_openapi findings vs. current state (issues resolved, issues remaining)\n"
        "3. Schema mutations applied so far (upsert_path / upsert_component / remove_*) with the rule or finding that motivated each\n"
        "4. Code or schema evidence supporting each mutation (file:line, component name, or lint rule id)\n"
        "5. Lint issues intentionally left unresolved and the reason (e.g., needs human input, ambiguous spec)\n"
        "6. Assumptions made when reconciling lint output with code\n"
        "7. Open questions or blockers\n"
        "8. Smallest concrete next step (typically: re-run lint_openapi and address the highest-severity remaining issue)\n"
        "Include only claims supported by tool output; mark anything inferred as such.\n"
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
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
    callback_adapter.register(RepeatedToolCallCallback(threshold=5))

    oas_linter_agent = LlmAgent(
        name=name,
        description="software engineering agent",
        instruction=OAS_LINTER_PROMPT,
        model=model if model is not None else DEFAULT_MODEL,
        tools=tools,
        **callback_adapter(),
    )

    return oas_linter_agent
