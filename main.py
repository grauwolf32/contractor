import os
import asyncio
import uuid
import json
import argparse

from dotenv import load_dotenv
from pprint import pprint
from typing import Final, Any

from google.adk.orchestration import Runner
from google.adk.orchestration.session import InMemorySessionService
from google.generativeai.types import content_types
from google.generativeai.types.content_types import Part

# Import our agents
from agents.code_analysis_agent.agent import code_analysis_agent
from agents.project_information_agent.agent import project_information_agent
from agents.project_manager_agent.agent import project_manager_agent

# Load environment variables
load_dotenv()

async def run_agent(runner, user_id, session_id, agent_name, content):
    """Run a specific agent with the given content."""
    print(f"\nRunning {agent_name}...")

    # Create content object if string is provided
    if isinstance(content, str):
        content = content_types.Content(
            role="user",
            parts=[Part.from_text(content)]
        )

    # Run the agent
    response = await runner.run_async(
        user_id=user_id,
        session_id=session_id,
        content=content,
        agent_name=agent_name
    )

    # Process the response
    final_response_text = None
    for event in response.events:
        if event.type == "content" and event.content.role == "agent":
            final_response_text = event.content.parts[0].text

    # Get the session to access state
    session = runner.session_service.get_session(
        user_id=user_id,
        session_id=session_id
    )

    print(f"{agent_name} completed.")
    return final_response_text, session.state

async def security_analysis_sequence_workflow(project_dir: str)->Any:
    session_service = InMemorySessionService()

    # Create a session
    session_id = str(uuid.uuid4())
    user_id = "workflow_user"

    session = session_service.create_session(
        app_name="SecurityEngineer",
        user_id=user_id,
        session_id=session_id
    )

    agents = [project_manager_agent, code_analysis_agent, project_information_agent]
    runner = Runner(
        root_agent=project_manager_agent,  # This doesn't matter in our case as we specify agent_name
        agents=agents,
        session_service=session_service
    )

    project_information_query: Final[str] = f"You need to gather information about the project: {project_dir}"
    project_information_response, state = await run_agent(runner, user_id, session_id, "project_information_agent", project_information_query)

    print(f"Project information:\n{project_information_response}\n\n")
    
    project_information: str = project_information_response
    if "project_information" in state:
        project_information = state["project_information"]
    else:
        print("[ERROR] project_information not in the state")

    code_analysis_query: Final[str] = (
        f"You need to analyse code of the project: {project_dir}",
        "Here is information about code language, framework, and dependencies:\n",
        f"{project_information}"
    )

    code_analysis_response, state = await run_agent(runner, user_id, session_id, "code_analysis_agent", code_analysis_query)
    print("Code analysis:\n{code_analysis_response}\n\n")

    return code_analysis_response



async def main():
    parser = argparse.ArgumentParser(description="SoftwareEnfineer")
    parser.add_argument(
        "--project_dir",
        required=True,
        help="Path to project"
    )

    args = parser.parse_args()
    code_analysis = await security_analysis_sequence_workflow(args.project_dir)


if __name__ == "__main__":
    asyncio.run(main())
