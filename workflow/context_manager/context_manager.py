import json
from typing import Optional
from pydantic import BaseModel, Field
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.models.lite_llm import LiteLlm
from google.adk.events.event import Event
from google.genai import types

from pydantic_settings import SettingsConfigDict
from functools import lru_cache
from helpers import Settings


class LlmConfig(Settings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
    )

    model_name: str
    api_key: str

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls):
        return cls()


config = LlmConfig.get_settings()

AGENT_MODEL = LiteLlm(model=config.model_name, api_key=config.api_key)


class ContextManager(BaseModel):
    max_tokens: int = Field(
        default=64 * 10**3, description="context length limit to run agent"
    )
    storage: dict = Field(
        default_factory=dict, description="storage for context management"
    )
    llm_client: LiteLlm = Field(description="llm client for summarization")

    def __call__(self, context: InvocationContext):
        context.session

    def _estimate_token_count(text: str) -> int:
        return len(text) / 4

    def _get_best_fit_messages(
        self,
        context: InvocationContext,
        llm_request: LlmRequest,
        include_init_message: bool = True,
    ) -> list[types.Content]:
        messages: list[types.Content]
        message_tokens: list[int] = []
        init_message: str = ""

        if context.session.user_content and context.session.user_content.parts:
            if message := context.session.user_content.parts.text:
                init_message = message

        first_message_is_init_message: bool = False
        if llm_request.contents and llm_request.contents.parts:
            if (
                msg := llm_request.contents.parts[0]
                and llm_request.contents.parts[0].text
            ):
                first_message_is_init_message = msg.text == init_message

        init_message_tokens: int = 0
        if include_init_message and not first_message_is_init_message:
            init_message_tokens = self._estimate_token_count(init_message)

        reserved_tokens: int = init_message_tokens

        if llm_request.contents and llm_request.contents.parts:
            for part in llm_request.contents.parts[::-1]:
                if part.text:
                    message_tokens.append(self._estimate_token_count(part.text))
                elif part.function_call:
                    message_tokens.append(
                        self._estimate_token_count(
                            json.dumps(part.function_response.response)
                        )
                    )
                elif part.code_execution_result:
                    message_tokens.append(
                        self._estimate_token_count(part.code_execution_result.output)
                    )
                if sum(message_tokens) >= (self.max_tokens - reserved_tokens):
                    break

        if (
            init_message < self.max_tokens
            and include_init_message
            and not first_message_is_init_message
        ):
            messages.append(context.session.user_content)

        messages.extends(llm_request.contents[-len(message_tokens) :])

        if include_init_message and not first_message_is_init_message:
            message_tokens.insert(0, init_message_tokens)

        return messages, message_tokens

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ): ...


class BaseStrategy(BaseModel):
    def run(
        self, context: InvocationContext, llm_request: LlmRequest, storage: dict
    ) -> InvocationContext: ...

    def last_message_is_tool_call(self, llm_request: LlmRequest) -> bool:
        if llm_request.contents and llm_request.contents.parts:
            if (
                llm_request.contents.parts[-1].function_response
                or llm_request.parts[-1].code_execution_result
            ):
                return True
        return False


class SlidingWindowStrategy(BaseStrategy):
    keep_init_message: bool = Field(default=True)
    keep_function_response: bool = Field(default=True)
    fit_context: bool = Field(default=True)
    num_messages: Optional[int] = Field(description="number of messages to keep")

    def run(self, context: InvocationContext, llm_request: LlmRequest, storage: dict):
        ...



class SummarizationStrategy(BaseStrategy): ...


class ContextPrioritizatinStrategy(BaseStrategy): ...


def create_slice_history_callback(n_recent_turns):
    async def before_model_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ):
        if n_recent_turns < 1:
            return

        user_indexes = [
            i
            for i, content in enumerate(llm_request.contents)
            if content.role == "user"
        ]

        if n_recent_turns > len(user_indexes):
            return

        suffix_idx = user_indexes[-n_recent_turns]
        llm_request.contents = llm_request.contents[suffix_idx:]

    return before_model_callback
