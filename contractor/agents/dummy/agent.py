from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback

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

callback_adapter = CallbackAdapter()
callback_adapter.register(TokenUsageCallback())

dummy_with_token_count = LlmAgent(
    name="dummy_with_token_count",
    description="agent to test tools and integrational scenarios",
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    **callback_adapter(),
)

root_agent = dummy_with_token_count