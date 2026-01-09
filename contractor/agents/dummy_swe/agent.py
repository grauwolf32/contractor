from __future__ import annotations

import os
from typing import Any, Final
from langfuse import get_client
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.tools.podman import PodmanContainer

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    langfuse = get_client()

DUMMY_AGENT_PROMPT: Final[str] = (
    "You are helpfull assistent. You must comply with the user request."
)

DUMMY_AGENT_DESCRIPTION: Final[str] = (
    "software engineering agent to test tools and integrational scenarios."
)

DUMMY_MODEL = LiteLlm(
    model="lm-studio-openai",
    timeout=30,
)

sandbox = PodmanContainer(
    name="contractor_dummy_sandbox",
    image="docker.io/ubuntu:jammy",
    mounts=[],
    commands=None,
    ro_mode=True,
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


tools = [default_tool, *sandbox.tools()]

callback_adapter = CallbackAdapter()
callback_adapter.register(
    InvalidToolCallGuardrailCallback(
        tools=tools, default_tool_name="default_tool", default_tool_arg="meta"
    )
)

dummy_swe = LlmAgent(
    name="dummy_swe",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    tools=tools,
    **callback_adapter(),
)

root_agent = dummy_swe
