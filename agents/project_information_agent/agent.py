import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from agents.project_information_agent.models.project_information import \
    ProjectInformation
from agents.project_information_agent.prompts import \
    dependency_filtering_instructions
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


class CodeAnalysisInputSchema(BaseModel):
    project_dir: str = Field(description="Path to the project for analysis")


config = AgentConfig.get_settings()

AGENT_MODEL = LiteLlm(model=config.model_name, api_key=config.api_key)


async def save_project_information(pd: ProjectInformation):
    """Save dependencies in JSON format with respect to provided JSON Schema
    Args:
        dependencies: ProjectInformation
            Project dependencies
    Schema:

    """
    pd = ProjectInformation.model_validate(pd)

    for i in range(len(pd.dependencies)):
        pd.dependencies[i]["tags"] = list(set(pd.dependencies[i].get("tags", [])))

    temp_dir = config.temp_dir

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "project_dependencies.json"), "w") as f:
        f.write(pd.model_dump_json())

    return


dependency_gathering_agent = LlmAgent(
    model=AGENT_MODEL,
    name="dependency_filtering_agent",
    description="An agent to filter only usefult dependencies for project analysis",
    instruction=dependency_filtering_instructions,
    tools=[cdxgen_mock_tool],
    output_key="project_dependencies",
)

project_information_agent = LlmAgent(
    name="dependency_gathering_agent",
    model=AGENT_MODEL,
    instruction=(
        "You are professional software engineer with application security background."
        "You goal is to get project information, based on its dependencies."
        "You need to determine all of the project’s dependencies using `dependency_gathering_agent`."
        "And save information using `save_project_information` tool."
        "Validate every step. If one of them fails, repeat it."
    ),
    input_schema=CodeAnalysisInputSchema,
    tools=[
        AgentTool(dependency_gathering_agent),
        save_project_information,
    ],
)

root_agent = dependency_gathering_agent
