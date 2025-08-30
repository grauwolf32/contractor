import logging
from helpers import Settings
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part
from typing import Any

class WorkflowSettings(Settings):
    context_length: int = 80000
    app_name: str = "contractor"

settings = WorkflowSettings.get_settings()
logger = logging.getLogger(__file__)

async def run_agent(runner:Runner, user_id:str, session_id:str, content:Any):
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
        app_name=settings.app_name, user_id=user_id, session_id=session_id
    )

    return final_response_text, session.state