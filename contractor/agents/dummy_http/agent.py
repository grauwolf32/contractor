from __future__ import annotations

import os
from typing import Any, Final

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.http import http_tools
from contractor.tools.memory import memory_tools, MemoryFormat

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

DUMMY_AGENT_PROMPT: Final[str] = (
    "You are helpfull assistent. You must comply with the user request."
)

DUMMY_AGENT_DESCRIPTION: Final[str] = (
    "software engineering agent to test tools and integrational scenarios."
)

DUMMY_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
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


httptools = http_tools(name="dummy")
mem_tools = memory_tools(name="dummy", fmt=MemoryFormat())
tools = [default_tool, *httptools, *mem_tools]

callback_adapter = CallbackAdapter()
callback_adapter.register(TokenUsageCallback())
callback_adapter.register(
    InvalidToolCallGuardrailCallback(
        tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
    )
)

dummy_http = LlmAgent(
    name="dummy_http",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    tools=tools,
    **callback_adapter(),
)

root_agent = dummy_http
