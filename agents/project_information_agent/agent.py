from functools import lru_cache
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic_settings import SettingsConfigDict

from agents.project_information_agent.models.project_information import \
    ProjectInformation, DependencyInformation
from agents.project_information_agent.prompts import \
    dependency_filtering_instructions, dependency_format_instructions
from helpers import Settings
from tools import cdxgen_mock_tool, cdxgen_tool


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

dependency_filtering_agent = LlmAgent(
    model=AGENT_MODEL,
    name="dependency_filtering_agent",
    description="An agent to filter only usefult dependencies for project analysis",
    instruction=dependency_filtering_instructions,
    output_schema=DependencyInformation,
    output_key="dependencies",
    tools=[],
)

project_information_formatting_agent = LlmAgent(
    model=AGENT_MODEL,
    name="project_information_formatting_agent",
    description="An agent to generate structured JSON output",
    instruction=(
        "{dependencies}\n" + \
        dependency_format_instructions 
    ),
    output_key="project_information",
    output_schema=ProjectInformation,
)

