from __future__ import annotations

from typing import Final
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks.ratelimits import RpmRatelimitCallback
from contractor.callbacks.context import SummarizationLimitCallback

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

callback_adapter.register(RpmRatelimitCallback(rpm_limit=5))

dummy_with_rpm_ratelimit = LlmAgent(
    name="dummy_with_rpm_ratelimit",
    description="agent to test tools and integrational scenarios",
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    **callback_adapter(),
)

dummy_summarizer_message: Final[str] = "Summarize our conversation"
callback_adapter.register(
    SummarizationLimitCallback(message=dummy_summarizer_message, max_tokens=4000)
)

dummy_summarizator = LlmAgent(
    name="dummy_summarizator",
    description="agent to test tools and integrational scenarios",
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    **callback_adapter(),
)

root_agent = dummy_summarizator
