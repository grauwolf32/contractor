import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.agents import callback_context as callback_context_module
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from contractor.agent.instructions import contract_analysis_global
from contractor.agent.models.project_analysis import (ProjectBasicInformation,
                                                      ProjectDependencies)
from contractor.helpers import Settings
from contractor.tools import serena_mcp_tools


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


def save_dependencies(
    callback_context: callback_context_module.CallbackContext,
) -> Optional[types.Content]:
    """Prints the current state of the callback context."""
    project_dependencies = callback_context.state.get("project_dependencies", "")
    temp_dir = config.temp_dir

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "project_dependencies.json"), "w") as f:
        json.dump(project_dependencies, f, indent=2)
    return None


dependency_gathering_agent = LlmAgent(
    name="dependency_gathering_agent",
    model=AGENT_MODEL,
    instruction=(
        "You need to determine all of the project’s dependencies"
        "— specifically which libraries are used and which functions "
        "from those libraries are called within the project."
    ),
    input_schema=CodeAnalysisInputSchema,
    tools=[serena_mcp_tools()],
)

dependency_formatter_agent = LlmAgent(
    name="dependency_formatter_agent",
    instruction=(
        "You need to determine all of the project’s dependencies"
        "— specifically which libraries are used and format "
        "output as JSON according to provided schema"
    ),
    output_schema=ProjectDependencies,
    output_key="project_dependencies",
)

dependency_analysis_agent = SequentialAgent(
    name="dependency_analysis_agent",
    description=(
        "An agent to gather information on project dependencies, library used and imported functions."
    ),
    sub_agents=[dependency_gathering_agent, dependency_formatter_agent],
    after_agent_callback=save_dependencies,
)


root_agent = Agent(
    name="project_analysis_agent",
    model=AGENT_MODEL,
    instruction=contract_analysis_global,
    sub_agents=[dependency_analysis_agent],
)
