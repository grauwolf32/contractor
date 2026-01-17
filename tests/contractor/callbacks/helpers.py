from uuid import uuid4
from typing import Any
from dataclasses import dataclass, field


@dataclass
class MockCtx:
    state: dict[str, Any] = field(default_factory=dict)
    invocation_id: str | None = None


def mk_callback_context(initial_state=None, invocation_id=None) -> MockCtx:
    """
    CallbackContext нам нужен только с полем .state (dict).
    """
    ctx = MockCtx()
    ctx.state = initial_state or {"callbacks": {}}
    ctx.invocation_id = invocation_id or str(uuid4())
    return ctx


@dataclass
class MockUsage:
    total_token_count: int = 0
    prompt_token_count: int = 0
    candidates_token_count: int = 0


@dataclass
class MockRespose:
    usage_metadata: MockUsage = field(default_factory=MockUsage)


def mk_llm_response(total, prompt, candidates) -> MockRespose:
    """
    LlmResponse с полями:
      - .usage_metadata.total_token_count
      - .usage_metadata.prompt_token_count
      - .usage_metadata.candidates_token_count
    """
    resp = MockRespose()

    usage = MockUsage()
    usage.total_token_count = total
    usage.prompt_token_count = prompt
    usage.candidates_token_count = candidates

    resp.usage_metadata = usage
    return resp
