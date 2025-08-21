import os
import asyncio
import uuid
import json
import argparse
import logging

from dotenv import load_dotenv
from pprint import pprint
from typing import Final, Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part
# Import our agents
from agents.code_analysis_agent.agent import code_analysis_agent
from agents.project_information_agent.agent import project_information_gathering_agent

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

APP_NAME = "SecurityEngineer"

async def run_agent(runner, user_id, session_id, content):
    print(f"\nRunning {runner.agent.name}...")

    if isinstance(content, (list, tuple)):
        content = "\n".join(map(str, content))
    if isinstance(content, str):
        content = Content(role="user", parts=[Part(text=content)])

    final_response_text = ""


    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            
    except Exception as e:
        logger.error(str(e))

    session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )

    print(f"{runner.agent.name} completed.")
    return final_response_text, session.state



async def security_analysis_sequence_workflow(project_dir: str) -> Any:
    session_service = InMemorySessionService()

    # Create a session
    session_id = str(uuid.uuid4())
    user_id = "workflow_user"

    pi_runner = Runner(
        agent=project_information_gathering_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    await pi_runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )

    project_information_query: Final[str] = f"You need to gather information about the project: {project_dir}"
    project_information_response, state = await run_agent(
        pi_runner, user_id, session_id, project_information_query
    )

    print(f"Project information:\n{project_information_response}\n\n")

    project_information: str = project_information_response or ""
    if isinstance(state, dict) and "project_information" in state:
        project_information = state["project_information"]
    else:
        print("[ERROR] project_information not in the state")

    await pi_runner.close()

    # Вторая сессия — анализ кода
    session_id = str(uuid.uuid4())
    ca_runner = Runner(
        agent=code_analysis_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    await ca_runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )

    # ВАЖНО: делаем одну строку, а не кортеж
    code_analysis_query: Final[str] = (
        f"You need to analyse code of the project: {project_dir}\n"
        f"Here is information about code language, framework, and dependencies:\n"
        f"{project_information}"
    )

    code_analysis_response, state = await run_agent(
        ca_runner, user_id, session_id, code_analysis_query
    )

    await ca_runner.close()
    print(f"Code analysis:\n{code_analysis_response}\n\n")  # f-строка

    return code_analysis_response

async def main():
    parser = argparse.ArgumentParser(description="SecurityEngineer")
    parser.add_argument("--project_dir", required=True, help="Path to project")
    args = parser.parse_args()
    code_analysis = await security_analysis_sequence_workflow(args.project_dir)
    print(f"Final response: \n{code_analysis}\n\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except* RuntimeError as eg:
        rest = [e for e in eg.exceptions
                if "Attempted to exit cancel scope in a different task" not in str(e)]
        if rest:
            # Поднимем обратно «чужие» ошибки, если они есть
            raise ExceptionGroup("Unhandled RuntimeError(s)", rest)