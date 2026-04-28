from __future__ import annotations

import os
from typing import Final, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import (
    InvalidToolCallGuardrailCallback,
    RepeatedToolCallCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.tools.http import http_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)
from contractor.utils import load_prompt

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

HTTP_PROMPT: Final[str] = load_prompt("http_agent")

HTTP_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
    return (
        "You have reached context limit. Summarize your progress and call report tool."
        + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
    )


def build_http_agent(
    name: str,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: Optional[LiteLlm] = None,
    proxy: Optional[str] = None,
) -> LlmAgent:
    httptools = http_tools(name=namespace, proxy=proxy)
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))

    tools = [default_tool, *httptools, *mem_tools]

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

    return LlmAgent(
        name=name,
        description=(
            "HTTP agent — issues HTTP requests, captures responses, and "
            "follows multi-step request flows (auth, sessions, redirects)."
        ),
        instruction=HTTP_PROMPT,
        model=model if model is not None else HTTP_MODEL,
        tools=tools,
        **callback_adapter(),
    )
