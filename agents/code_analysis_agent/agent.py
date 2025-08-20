import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from agents.code_analysis_agent.prompts import basic_project_structure
from helpers import Settings
from tools import cdxgen_mock_tool, cdxgen_tool, serena_mcp_tools


class AgentConfig(Settings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
    )

    model_name: str
    api_key: str
    temp_dir: str = str(Path(__file__).parent.parent / "data" / "tmp")

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls):
        return cls()


config = AgentConfig.get_settings()

AGENT_MODEL = LiteLlm(model=config.model_name, api_key=config.api_key)

swe_agent = LlmAgent(
    model=AGENT_MODEL,
    name="swe_agent",
    description="Professional Software Engineer",
    instruction=(),
    tools=[serena_mcp_tools()]
)

code_analysis_agent = LlmAgent(
    model=AGENT_MODEL,
    name="project_analysis_agent",
    description="An agent to analyse project",
    instruction=basic_project_structure,
    output_key="code_analysis",
    tools=[AgentTool(swe_agent)]
)

