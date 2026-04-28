from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4


@dataclass
class MockCtx:
    state: dict[str, Any] = field(default_factory=dict)
    invocation_id: str | None = None


def mk_callback_context(
    initial_state: Optional[dict[str, Any]] = None, invocation_id: Optional[str] = None
) -> MockCtx:
    """
    CallbackContext нам нужен только с полем .state (dict).
    """
    ctx = MockCtx()
    ctx.state = initial_state or {"callbacks": {}}
    ctx.invocation_id = invocation_id or str(uuid4())
    return ctx


def mk_tool_context(
    initial_state: Optional[dict[str, Any]] = None,
    invocation_id: Optional[str] = "invocation_id",
) -> MockCtx:
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


class MockAgentTool:
    """
    Minimal stand-in for google.adk.tools.AgentTool.
    task_tools() only needs an object with .run_async(...).
    """

    def __init__(self, agent):
        self.agent = agent

    async def run_async(self, args, tool_context):
        return await self.agent.run_async(args=args, tool_context=tool_context)


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


# ---------------------------------------------------------------------------
# Minimal LlmRequest / Content / Part stand-ins for callback tests.
# The callbacks only access a small slice of the real google.adk types:
#   request.contents (list)
#   content.parts (list or None)
#   part.text, part.function_response (with .parts and .response)
# ---------------------------------------------------------------------------


@dataclass
class MockFunctionResponse:
    parts: Any = None
    response: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockPart:
    text: Optional[str] = None
    function_response: Optional[MockFunctionResponse] = None


@dataclass
class MockContent:
    role: str = "user"
    parts: Optional[list[MockPart]] = None


@dataclass
class MockLlmRequest:
    contents: list[Any] = field(default_factory=list)


def mk_llm_request(contents: Optional[list[Any]] = None) -> MockLlmRequest:
    return MockLlmRequest(contents=list(contents or []))


def mk_function_response_part(response: Optional[dict[str, Any]] = None) -> MockPart:
    return MockPart(
        function_response=MockFunctionResponse(response=dict(response or {}))
    )


def mk_text_part(text: str) -> MockPart:
    return MockPart(text=text)
