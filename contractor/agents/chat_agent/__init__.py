from __future__ import annotations

from typing import Callable, Iterable, Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from contractor.callbacks import default_tool
from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.tools.fs import FileFormat, RootedLocalFileSystem, ro_file_tools
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.openapi.openapi import openapi_tools

from .models import ChatContext

PLANNER_PROMPT = """
You are the planning agent for a CLI coding assistant.

Your job:
- read the latest user request
- produce a compact plan with 1-5 steps
- pick the next concrete step
- do not call specialist tools directly

Output rules:
- first line: GOAL: <one sentence>
- second line: ROUTE_HINT: <repo|oas_build|oas_enrich|trace|artifacts|code_edit|direct>
- then a short numbered plan
""".strip()

ROUTER_PROMPT = """
You are the routing agent for a CLI coding assistant.

You receive the user request and the planner output.
Choose exactly one best target agent.

Valid targets:
- repo
- oas_build
- oas_enrich
- trace
- artifacts
- code_edit
- direct

Return exactly this format:
TARGET: <value>
REASON: <one sentence>
""".strip()

SUPERVISOR_PROMPT = """
You are Contractor Chat, the main conversational agent for this repository.

Operating model:
1. ALWAYS call the planner tool first for non-trivial requests.
2. Then call the router tool.
3. Then delegate to exactly one specialist tool unless the request is trivial.
4. Synthesize the final answer in a concise engineering style.

Behavior rules:
- Be practical and implementation-focused.
- When the user asks for code, prefer concrete code over abstract advice.
- When you need repository evidence, use the repo analyst agent.
- When the user asks to generate or update OpenAPI, use the OAS agents.
- When the user asks to trace request flow, use the trace agent.
- If the user asks to edit code, use the code editor agent.
- If an operation looks risky, say it clearly.
- Never mention hidden chain-of-thought.
""".strip()

REPO_ANALYST_PROMPT = """
You are a repository analysis specialist.

Focus on:
- structure
- files
- symbols
- architecture
- implementation advice grounded in the codebase

Use file tools actively. Quote paths precisely. Keep answers concrete.
""".strip()

OAS_SPECIALIST_PROMPT = """
You are an OpenAPI specialist.

Use file tools, memory tools, and OpenAPI tools to inspect and update schema-related work.
Prefer modifying the in-memory OpenAPI representation through OpenAPI tools.
""".strip()

TRACE_SPECIALIST_PROMPT = """
You are a trace specialist.

Your role is to identify entrypoints, request flow, validation points, and sinks.
Stay narrowly focused on the requested endpoint or feature.
""".strip()

CODE_EDITOR_PROMPT = """
You are a cautious code-editing specialist.

Rules:
- Prefer small diffs.
- Explain why each change is needed.
- If write tools are unavailable, produce a ready-to-apply patch.
""".strip()

ARTIFACT_SPECIALIST_PROMPT = """
You are an artifact specialist.

Focus on produced artifacts, OpenAPI outputs, diffs, summaries, and previous run outputs.
""".strip()


def _callback_config(agent_name: str, tools: list[object]) -> dict:
    adapter = CallbackAdapter(agent_name=agent_name)
    adapter.register(TokenUsageCallback())
    adapter.register(
        InvalidToolCallGuardrailCallback(
            tools=tools,
            default_tool_name="default_tool",
            default_tool_arg="meta",
        )
    )
    return adapter()


def _specialist_tools(
    *,
    fs: RootedLocalFileSystem,
    namespace: str,
    include_openapi: bool = False,
) -> list[object]:
    tools: list[object] = [default_tool]
    tools.extend(ro_file_tools(fs, fmt=FileFormat(_format="markdown")))
    tools.extend(memory_tools(name=namespace, fmt=MemoryFormat(_format="markdown")))
    if include_openapi:
        tools.extend(openapi_tools(name=namespace, fs=fs))
    return tools


def build_chat_agents(
    *,
    ctx: ChatContext,
    fs: RootedLocalFileSystem,
    model: Optional[LiteLlm] = None,
) -> LlmAgent:
    llm = model if model is not None else LiteLlm(model=ctx.model, timeout=300)

    planner = LlmAgent(
        name="planner_agent",
        description="Creates an execution plan for the next user turn.",
        instruction=PLANNER_PROMPT,
        model=llm,
    )

    router = LlmAgent(
        name="router_agent",
        description="Routes a planned request to the best specialist.",
        instruction=ROUTER_PROMPT,
        model=llm,
    )

    repo_tools = _specialist_tools(fs=fs, namespace="chat-repo")
    repo_analyst = LlmAgent(
        name="repo_analyst_agent",
        description="Explains repository structure and implementation details.",
        instruction=REPO_ANALYST_PROMPT,
        model=llm,
        tools=repo_tools,
        **_callback_config("repo_analyst_agent", repo_tools),
    )

    oas_tools = _specialist_tools(fs=fs, namespace="chat-oas", include_openapi=True)
    oas_builder = LlmAgent(
        name="oas_build_agent",
        description="Builds or updates OpenAPI schema details.",
        instruction=OAS_SPECIALIST_PROMPT,
        model=llm,
        tools=oas_tools,
        **_callback_config("oas_build_agent", oas_tools),
    )
    oas_enricher = LlmAgent(
        name="oas_enrich_agent",
        description="Enriches existing OpenAPI schema details.",
        instruction=OAS_SPECIALIST_PROMPT,
        model=llm,
        tools=oas_tools,
        **_callback_config("oas_enrich_agent", oas_tools),
    )

    trace_tools = _specialist_tools(fs=fs, namespace="chat-trace")
    trace_agent = LlmAgent(
        name="trace_specialist_agent",
        description="Traces requests and code flow.",
        instruction=TRACE_SPECIALIST_PROMPT,
        model=llm,
        tools=trace_tools,
        **_callback_config("trace_specialist_agent", trace_tools),
    )

    code_tools = _specialist_tools(fs=fs, namespace="chat-code")
    code_editor = LlmAgent(
        name="code_edit_agent",
        description="Produces safe code changes or patches.",
        instruction=CODE_EDITOR_PROMPT,
        model=llm,
        tools=code_tools,
        **_callback_config("code_edit_agent", code_tools),
    )

    artifact_tools = _specialist_tools(fs=fs, namespace="chat-artifacts", include_openapi=True)
    artifact_agent = LlmAgent(
        name="artifact_agent",
        description="Summarizes outputs and artifacts from prior runs.",
        instruction=ARTIFACT_SPECIALIST_PROMPT,
        model=llm,
        tools=artifact_tools,
        **_callback_config("artifact_agent", artifact_tools),
    )

    supervisor_tools = [
        AgentTool(agent=planner, skip_summarization=True),
        AgentTool(agent=router, skip_summarization=True),
        AgentTool(agent=repo_analyst, skip_summarization=True),
        AgentTool(agent=oas_builder, skip_summarization=True),
        AgentTool(agent=oas_enricher, skip_summarization=True),
        AgentTool(agent=trace_agent, skip_summarization=True),
        AgentTool(agent=code_editor, skip_summarization=True),
        AgentTool(agent=artifact_agent, skip_summarization=True),
    ]

    root_tools = [default_tool, *supervisor_tools]
    root = LlmAgent(
        name="contractor_chat_agent",
        description="Top-level planner/router/specialist chat agent for Contractor.",
        instruction=SUPERVISOR_PROMPT,
        model=llm,
        tools=root_tools,
        **_callback_config("contractor_chat_agent", root_tools),
    )
    return root