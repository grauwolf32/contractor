import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic_settings import SettingsConfigDict

from helpers import Settings
from tools import serena_mcp_tools


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
    instruction=(
        "You are professional software engineer. You goal is to complete assigned task"
    ),
    tools=[serena_mcp_tools()],
)
