from functools import lru_cache
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic_settings import SettingsConfigDict

from helpers import Settings


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

project_manager_agent = LlmAgent(
    model=AGENT_MODEL,
    name="project_manager_agent",
    description="An agent to manage assigned task",
    instruction=(
        "You are professional project manager. Your goal is to complete assigned task."
    ),
)

root_agent = project_manager_agent
