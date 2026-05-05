from __future__ import annotations

import os
from typing import Any, Final

from google.adk.agents import LlmAgent

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.http import http_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.utils.settings import DEFAULT_MODEL

DUMMY_AGENT_PROMPT: Final[str] = (
    "You are helpfull assistent. You must comply with the user request."
)

DUMMY_AGENT_DESCRIPTION: Final[str] = (
    "software engineering agent to test tools and integrational scenarios."
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
callback_adapter.register(RepeatedToolCallCallback(threshold=5))

dummy_http = LlmAgent(
    name="dummy_http",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DEFAULT_MODEL,
    tools=tools,
    **callback_adapter(),
)

root_agent = dummy_http
