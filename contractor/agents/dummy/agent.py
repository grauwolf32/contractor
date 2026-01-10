from __future__ import annotations

import os
from typing import Final
from langfuse import get_client
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks.ratelimits import RpmRatelimitCallback
from contractor.callbacks.context import SummarizationLimitCallback

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
    langfuse = get_client()

DUMMY_AGENT_PROMPT: Final[str] = (
    "You are helpfull assistent. You must comply with the user request."
)

DUMMY_AGENT_DESCRIPTION: Final[str] = "agent to test tools and integrational scenarios."

DUMMY_MODEL = LiteLlm(
    model="lm-studio-openai",
    timeout=30,
)

DUMMY_SUMMARIZER_MESSAGE: Final[str] = "Summarize our conversation."

dummy = LlmAgent(
    name="dummy",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
)

callback_adapter = CallbackAdapter()
callback_adapter.register(TokenUsageCallback())

dummy_with_token_count = LlmAgent(
    name="dummy_with_token_count",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    **callback_adapter(),
)

callback_adapter.register(RpmRatelimitCallback(rpm_limit=5))

dummy_with_rpm_ratelimit = LlmAgent(
    name="dummy_with_rpm_ratelimit",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    **callback_adapter(),
)

callback_adapter.register(
    SummarizationLimitCallback(message=DUMMY_SUMMARIZER_MESSAGE, max_tokens=4000)
)

dummy_summarizator = LlmAgent(
    name="dummy_summarizator",
    description=DUMMY_AGENT_DESCRIPTION,
    instruction=DUMMY_AGENT_PROMPT,
    model=DUMMY_MODEL,
    **callback_adapter(),
)

root_agent = dummy_summarizator
