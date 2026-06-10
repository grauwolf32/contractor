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
    "You have reached the context limit. Summarize your progress:\n"
    "1. Subtask objective as you understand it\n"
    "2. Requests issued so far (method + URL) and the key responses observed\n"
    "3. Findings worth keeping — persist them to memory before stopping\n"
    "4. Open questions or blockers\n"
    "5. Smallest concrete next step to resume the flow\n"
    "Then return the structured result. Include only claims supported by "
    "tool output; mark anything inferred as such.\n"
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
