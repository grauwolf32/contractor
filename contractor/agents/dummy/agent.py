from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

DUMMY_AGENT_PROMPT: Final[str] = (
    "You are helpfull assistent. You must comply with user request."
)

DUMMY_MODEL = LiteLlm(
    model="lm-studio-openai",
    timeout=30,
)

dummy = LlmAgent(
    name="dummy",
    description="agent to test tools and integrational scenarios",
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
)

root_agent = dummy