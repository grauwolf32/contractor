from __future__ import annotations

import os
from typing import Final, Literal, Optional, Sequence

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.utils import load_prompt

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

ROUTER_PROMPT: Final[str] = load_prompt("router_agent")

ROUTER_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def build_router_agent(
    name: str,
    *,
    namespace: str,
    sub_agents: Sequence[LlmAgent],
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    model: Optional[LiteLlm] = None,
) -> LlmAgent:
    """Build a routing agent that dispatches subtasks to one of ``sub_agents``.

    Each sub-agent is exposed as an ``AgentTool``; the router picks one,
    forwards the subtask, and wraps the response as a SubtaskExecutionResult.
    """
    if not sub_agents:
        raise ValueError("router_agent requires at least one sub-agent")

    sub_agent_tools = [AgentTool(agent) for agent in sub_agents]
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))

    tools = [default_tool, *sub_agent_tools, *mem_tools]

    callback_adapter = CallbackAdapter(agent_name=name)
    callback_adapter.register(TokenUsageCallback())
    callback_adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
        )
    )
    callback_adapter.register(RepeatedToolCallCallback(threshold=3))

    return LlmAgent(
        name=name,
        description="routes subtasks to specialized sub-agents",
        instruction=ROUTER_PROMPT,
        model=model if model is not None else ROUTER_MODEL,
        tools=tools,
        **callback_adapter(),
    )
