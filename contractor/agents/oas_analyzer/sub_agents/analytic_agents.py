from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Literal, Optional

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from pydantic import BaseModel
from typing_extensions import override

from contractor.agents.oas_analyzer.models import (
    EndpointVulnerability,
    ServiceBasicInfo,
)
from contractor.agents.oas_analyzer.prompts.factory import SectionPrompts

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()


DEFAULT_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


@dataclass
class BotFactory:
    @staticmethod
    def build(
        spec: Literal["appsec", "datasec", "ddos", "review"],
        *,
        model: Optional[LiteLlm] = None,
        tools: list[Callable] = [],
        output_schema: Optional[BaseModel] = None,
        output_key: Optional[str] = None,
    ) -> list[LlmAgent]:
        bots: list[LlmAgent] = []
        section = SectionPrompts().load(name=spec)
        for task_name, task in section.tasks.items():
            bots.append(
                LlmAgent(
                    model=model if model else DEFAULT_MODEL,
                    name=f"{spec}_{task_name}",
                    instruction=section.format(name=task_name),
                    description=section.role,
                    tools=tools,
                    output_schema=output_schema,
                    output_key=output_key,
                )
            )
        return bots


def save_vulnerability(
    path: str,
    method: str,
    parameters: list[str],
    vulnerability: str,
    description: str,
    severity: Literal["low", "medium", "high", "critical"],
    confidence: Literal["low", "medium", "high"],
    tool_context: ToolContext,
) -> dict[str, str]:
    """
    Save a vulnerability to the report.
    Args:
        path: str
            The path of the endpoint that is vulnerable.
        method: str
            The HTTP method of the endpoint that is vulnerable.
        parameters: list[str]
            The parameters of the endpoint that are vulnerable.
        vulnerability: str
            The type of vulnerability.
        description: str
            A description of the vulnerability.
        severity: Literal["low", "medium", "high", "critical"]
            The severity of the vulnerability.
        confidence: Literal["low", "medium", "high"]
            The confidence level of the vulnerability.

     Returns:
        dict[str, str]: result of the operation.
    """

    key = "oas_analyzer::vulnerabilities"
    state = tool_context.state
    state.setdefault(key, [])
    tag = tool_context.agent_name.split("_")[0]

    vulnerabilities = state[key]
    vulnerabilities.append(
        EndpointVulnerability(
            path=path,
            method=method.lower(),
            parameters=parameters,
            vulnerability=vulnerability,
            description=description,
            severity=severity.lower(),
            confidence=confidence.lower(),
            tag=tag,
        ).model_dump()
    )
    tool_context.state[key] = vulnerabilities

    return {"success": "true"}


def _create_branch_ctx_for_sub_agent(
    agent: BaseAgent,
    sub_agent: BaseAgent,
    invocation_context: InvocationContext,
) -> InvocationContext:
    """Create isolated branch for every sub-agent."""
    invocation_context = invocation_context.model_copy()
    branch_suffix = f"{agent.name}.{sub_agent.name}"
    invocation_context.branch = (
        f"{invocation_context.branch}.{branch_suffix}"
        if invocation_context.branch
        else branch_suffix
    )
    return invocation_context


class AnalyticAgent(BaseAgent):
    class Config:
        extra = "allow"

    def __init__(self):
        review_agent: LlmAgent = BotFactory.build(
            spec="review",
            output_schema=ServiceBasicInfo,
            output_key="oas_analyzer::service_information",
        )[0]
        sub_agents = []
        for spec in {"appsec", "datasec", "ddos"}:
            sub_agents.extend(BotFactory.build(spec=spec, tools=[save_vulnerability]))

        super().__init__(
            name="analytic_agent", sub_agents=sub_agents, review_agent=review_agent
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        async for event in self.review_agent.run_async(ctx):
            yield event

        for agent in self.sub_agents:
            async for event in agent.run_async(
                _create_branch_ctx_for_sub_agent(self, agent, ctx)
            ):
                yield event


analytic_agent = AnalyticAgent()
