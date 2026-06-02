from __future__ import annotations

from typing import Final, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.worker_factory import build_worker
from contractor.callbacks import default_tool
from contractor.tools.http import http_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.utils import load_prompt

HTTP_PROMPT: Final[str] = load_prompt("http_agent")

_SUMMARIZATION_BULLETS: Final[str] = (
    "You have reached context limit. Summarize your progress and call report tool."
)

def build_http_agent(
    name: str,
    *,
    namespace: str,
    _format: Literal["json", "xml", "yaml", "markdown"] = "json",
    max_tokens: int = 80000,
    model: LiteLlm | None = None,
    proxy: str | None = None,
) -> LlmAgent:
    httptools = http_tools(name=namespace, proxy=proxy)
    mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))

    tools = [default_tool, *httptools, *mem_tools]

    return build_worker(
        name=name,
        instruction=HTTP_PROMPT,
        description=(
            "HTTP agent — issues HTTP requests, captures responses, and "
            "follows multi-step request flows (auth, sessions, redirects)."
        ),
        tools=tools,
        _format=_format,
        summarization_bullets=_SUMMARIZATION_BULLETS,
        max_tokens=max_tokens,
        model=model,
        with_elide=False,
    )
