"""Bounded ADK Planner loop over an injected model and execution host."""

from __future__ import annotations

import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import BaseTool
from google.genai import types

from contractor_runtime.adapters.content import bounded_content, model_request_content
from contractor_runtime.contracts.planner import (
    PlannerInvocation,
    PlannerOperationResult,
    PlannerTool,
)
from contractor_runtime.contracts.settings import ResolvedModelPolicy
from contractor_runtime.llm.errors import GatewayModelError
from contractor_runtime.llm.response import (
    empty_stop_response,
    model_response_failed,
    output_limit_reached,
)
from contractor_runtime.llm.usage import valid_token_usage
from contractor_runtime.planner.compaction import CheckpointRequested, Conversation
from contractor_runtime.planner.host import PlannerHost, PlannerStoppedError
from contractor_runtime.threads import to_thread_until_done


class _HostTool(BaseTool):
    def __init__(self, definition: PlannerTool, loop: PlannerLoop) -> None:
        super().__init__(name=definition.name, description=definition.description)
        self._definition, self._loop = definition, loop

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._definition.parameters,
        )

    async def run_async(self, *, args, tool_context):
        reply = await self._loop.operation(self.name, args)
        if reply.stop:
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = reply.escalate
        return reply.result


class PlannerLoop:
    def __init__(
        self,
        model: BaseLlm,
        host: PlannerHost,
        policy: ResolvedModelPolicy,
        *,
        compactor: BaseLlm | None = None,
        cost_snapshot=None,
    ) -> None:
        self._model, self._host, self._policy = model, host, policy
        self._compactor, self._cost_snapshot = compactor, cost_snapshot
        self._capture_model = False
        self.stopped = False

    async def operation(self, name, arguments) -> PlannerOperationResult:
        if self._cost_snapshot is not None and name in {
            "after_model",
            "model_failed",
            "after_compactor",
        }:
            arguments = {**arguments, "cost": self._cost_snapshot().model_dump(by_alias=True)}
        reply = await self._host.execute(name, arguments)
        if reply.failure is not None:
            raise PlannerStoppedError(
                reply.failure.code, reply.failure.message, reply.failure.retryable
            )
        self.stopped |= reply.stop
        return reply

    async def run(self, invocation: PlannerInvocation, instruction: str) -> None:
        conversation = Conversation(invocation.compaction) if invocation.compaction else None
        checkpoint_requested = recovery_requested = False
        if conversation is not None and self._compactor is None:
            raise RuntimeError("Planner compactor model is missing")

        async def model_failed(
            *, usage=None, kind="invalid_response", code="", empty=False, limited=False
        ):
            nonlocal recovery_requested
            if not valid_token_usage(usage):
                usage = None
            reply = await self.operation(
                "model_failed",
                {
                    "kind": kind,
                    "code": code,
                    "usage": {
                        "inputTokens": usage.prompt_token_count or 0,
                        "outputTokens": usage.candidates_token_count or 0,
                    }
                    if usage is not None
                    else None,
                    "emptyStop": empty,
                    "outputLimited": limited,
                },
            )
            if reply.result.get("recoverFinish"):
                recovery_requested = True
                return LlmResponse()
            raise PlannerStoppedError(
                "planner_gateway_invalid_response",
                "Planner Gateway returned an invalid response",
                True,
            )

        async def before_model(callback_context, llm_request):
            nonlocal checkpoint_requested
            if conversation is not None:
                try:
                    await to_thread_until_done(
                        conversation.prepare,
                        callback_context.session,
                        name="planner-context-ledger",
                    )
                except CheckpointRequested:
                    checkpoint_requested = True
                    # A content-free callback response ends this SDK turn before
                    # provider admission; the owner compacts after Runner joins.
                    return LlmResponse()
            reply = await self.operation(
                "before_model", {"retainedBytes": conversation.retained if conversation else 0}
            )
            if conversation is not None:
                projected = await to_thread_until_done(
                    conversation.project, reply.result["plan"], name="planner-context-projection"
                )
                if projected is not None:
                    llm_request.contents = projected
            if guidance := reply.result.get("guidance"):
                llm_request.contents = [
                    *llm_request.contents,
                    types.Content(role="user", parts=[types.Part(text=guidance)]),
                ]
            self._capture_model = reply.result.get("captureContent") is True
            if self._capture_model:
                content = await bounded_content(model_request_content(llm_request))
                if content is not None:
                    await self.operation("model_content", {"content": content})

        async def after_model(callback_context, llm_response):
            usage = llm_response.usage_metadata
            if (
                not valid_token_usage(usage)
                or output_limit_reached(llm_response)
                or empty_stop_response(llm_response)
                or model_response_failed(llm_response)
            ):
                return await model_failed(
                    usage=usage,
                    empty=empty_stop_response(llm_response),
                    limited=output_limit_reached(llm_response),
                )
            calls = [
                part.function_call.name
                for part in (llm_response.content.parts if llm_response.content else []) or []
                if part.function_call is not None
            ]
            arguments = {
                "inputTokens": (usage.prompt_token_count or 0) if usage else 0,
                "outputTokens": (usage.candidates_token_count or 0) if usage else 0,
                "cachedTokens": (usage.cached_content_token_count or 0) if usage else 0,
                # Two names preserve single/parallel selection without copying
                # an arbitrarily large provider call list into the host request.
                "calls": calls[:2],
            }
            if self._capture_model:
                content = await bounded_content(llm_response.content)
                if content is not None:
                    arguments["content"] = content
            reply = await self.operation("after_model", arguments)
            if conversation is not None:
                conversation.observe_main(usage)
            if reply.result.get("rejected"):
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Call exactly one declared tool, then continue.")],
                    ),
                    usage_metadata=usage,
                )
            for part in (llm_response.content.parts if llm_response.content else []) or []:
                if part.function_call is not None and not part.function_call.id:
                    part.function_call.id = uuid.uuid4().hex
            return None

        async def on_model_error(callback_context, llm_request, error):
            if not isinstance(error, GatewayModelError):
                raise error
            usage = error.usage_metadata
            kind = (
                "invalid_response"
                if error.response_received
                else "unavailable"
                if error.retryable
                else "rejected"
            )
            return await model_failed(
                usage=usage,
                kind=kind,
                code=error.failure.code if error.failure else error.provider_error_type,
                empty=error.empty_stop,
            )

        service = InMemorySessionService()
        session = await service.create_session(app_name="planner", user_id="execution")
        agent = LlmAgent(
            name="planner",
            model=self._model,
            instruction=lambda _: instruction,
            tools=[_HostTool(tool, self) for tool in invocation.tools],
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=self._policy.max_output_tokens,
                temperature=self._policy.temperature,
            ),
            before_model_callback=before_model,
            after_model_callback=after_model,
            on_model_error_callback=on_model_error,
        )
        runner = Runner(app_name="planner", agent=agent, session_service=service)
        message = types.Content(role="user", parts=[types.Part(text=invocation.input)])
        try:
            while not self.stopped:
                async for _ in runner.run_async(
                    user_id="execution",
                    session_id=session.id,
                    new_message=message,
                    run_config=RunConfig(max_llm_calls=self._policy.max_model_calls),
                ):
                    pass
                if checkpoint_requested:
                    checkpoint_requested = False
                    await conversation.compact(
                        model=self._compactor,
                        operation=self.operation,
                        on_model_error=on_model_error,
                    )
                if recovery_requested:
                    recovery_requested = False
                    message = types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=(
                                    "The document work is complete. Submit exactly one complete "
                                    "concise finish with the retained result artifacts."
                                )
                            )
                        ],
                    )
                    continue
                message = types.Content(
                    role="user", parts=[types.Part(text="Continue the plan or call finish.")]
                )
        finally:
            await runner.close()
            await service.delete_session(
                app_name="planner", user_id="execution", session_id=session.id
            )
