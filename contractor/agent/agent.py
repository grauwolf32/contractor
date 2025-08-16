from os import getenv
from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool

from contractor.agent.models.project_analysis import ProjectBasicInformation
from contractor.tools import serena_mcp_tools

AGENT_MODEL = LiteLlm(
    model=getenv("OPENAI_MODEL_NAME"), api_key=getenv("OPENAI_API_KEY")
)

root_agent = Agent(
    name="project_analysis_agent",
    model=AGENT_MODEL,
    description=(
        "An agent to gather fundamental project details: directories, programming language, and framework in use"
    ),
    instruction=(
        "Your main goal is to acquire more details about project:"
        "project directory, programming language, and framework in use"
    ),
    tools=[
        serena_mcp_tools()
    ],
)

