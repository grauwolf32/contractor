"""Allocation-scoped A2A 1.0 JSON-RPC binding for the in-process Worker."""

from __future__ import annotations

import re
import uuid
from typing import Any, Protocol

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    MutualTlsSecurityScheme,
    Part,
    Role,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    Task,
    TaskState,
    TaskStatus,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.types import Message as ASGIMessage

from contractor_runtime.contracts import (
    API_VERSION,
    StageContentRequest,
    StageContentResult,
    StageOutcome,
    TerminationError,
)

STAGE_CONTENT_MEDIA_TYPE = "application/vnd.contractor.stage-content+json"
MAX_A2A_REQUEST_BYTES = 1 << 20
ALLOCATION_A2A_PATH = re.compile(
    r"^/private/v1/allocations/(?P<allocation>[A-Za-z0-9_-]+)/a2a(?P<suffix>/.*)?$"
)


class InvocableWorker(Protocol):
    allocation_id: str

    async def invoke(self, request: StageContentRequest) -> StageContentResult: ...

    def cancel_active(self) -> None: ...


class ActiveA2AProvider(Protocol):
    async def active_a2a_application(self, allocation_id: str) -> ASGIApp | None: ...


def build_agent_card(
    *,
    allocation_id: str,
    endpoint: str,
    logical_agent_name: str,
    description: str,
    version: str,
) -> AgentCard:
    """Build the one-skill external card without allocation secrets or instructions."""

    return AgentCard(
        name=f"Contractor Worker {logical_agent_name}",
        description=description,
        supported_interfaces=[
            AgentInterface(
                url=endpoint,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                tenant=allocation_id,
            )
        ],
        version=version,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        security_schemes={
            "mutualTLS": SecurityScheme(
                mtls_security_scheme=MutualTlsSecurityScheme(
                    description="Deployment-CA mutual TLS with a Contractor Control Plane peer"
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={"mutualTLS": StringList(list=[])})],
        default_input_modes=[STAGE_CONTENT_MEDIA_TYPE],
        default_output_modes=[STAGE_CONTENT_MEDIA_TYPE],
        skills=[
            AgentSkill(
                id="contractor_stage_content",
                name="Execute Contractor stage content",
                description="Execute one strict Contractor StageContentRequest.",
                tags=["contractor", "stage"],
                input_modes=[STAGE_CONTENT_MEDIA_TYPE],
                output_modes=[STAGE_CONTENT_MEDIA_TYPE],
            )
        ],
    )


def agent_card_dict(card: AgentCard) -> dict[str, Any]:
    return MessageToDict(card)


def build_worker_a2a_application(worker: InvocableWorker, card: AgentCard) -> ASGIApp:
    executor = ContractorAgentExecutor(worker)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(owner_resolver=lambda _: worker.allocation_id),
        agent_card=card,
    )
    routes = create_jsonrpc_routes(handler, rpc_url="/")
    routes.extend(create_agent_card_routes(card, card_url="/.well-known/agent-card.json"))
    return Starlette(routes=routes)


class ContractorAgentExecutor(AgentExecutor):
    def __init__(self, worker: InvocableWorker) -> None:
        self._worker = worker

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("A2A task and context identities are required")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if context.current_task is None:
            initial = Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
            if context.message is not None:
                initial.history.append(context.message)
            await event_queue.enqueue_event(initial)
        else:
            await updater.start_work()
        if context.call_context.tenant != self._worker.allocation_id:
            result = _failed_result(
                "allocation_route_mismatch", "A2A tenant does not name the active allocation"
            )
        else:
            try:
                request = _stage_request(context)
            except (TypeError, ValueError, ValidationError):
                result = _failed_result(
                    "invalid_stage_content", "A2A message must contain one StageContentRequest"
                )
            else:
                result = await self._worker.invoke(request)
        message = _result_message(result, context)
        if result.outcome is StageOutcome.FAILED:
            await updater.failed(message)
        else:
            await updater.complete(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        del event_queue
        if context.call_context.tenant == self._worker.allocation_id:
            self._worker.cancel_active()


class AllocationA2AGateway:
    """Dispatch stable allocation URLs to the currently active SDK application."""

    def __init__(self, fallback: ASGIApp, provider: ActiveA2AProvider) -> None:
        self._fallback = fallback
        self._provider = provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._fallback(scope, receive, send)
            return
        match = ALLOCATION_A2A_PATH.fullmatch(scope.get("path", ""))
        if match is None:
            await self._fallback(scope, receive, send)
            return
        application = await self._provider.active_a2a_application(match.group("allocation"))
        if application is None:
            await _gateway_error(
                scope,
                receive,
                send,
                409,
                "allocation_not_active",
                "A2A allocation is stale, draining, or unavailable",
            )
            return
        bounded_receive = await _bounded_receive(scope, receive, send)
        if bounded_receive is None:
            return
        delegated = dict(scope)
        suffix = match.group("suffix") or "/"
        delegated["path"] = suffix
        delegated["raw_path"] = suffix.encode("ascii")
        base_path = (
            match.group(0) if match.group("suffix") is None else match.group(0)[: -len(suffix)]
        )
        delegated["root_path"] = scope.get("root_path", "") + base_path
        await application(delegated, bounded_receive, send)


def _stage_request(context: RequestContext) -> StageContentRequest:
    message = context.message
    if message is None or len(message.parts) != 1 or not message.parts[0].HasField("data"):
        raise ValueError("one DataPart is required")
    value = MessageToDict(message.parts[0].data)
    if not isinstance(value, dict):
        raise TypeError("StageContentRequest DataPart must be an object")
    return StageContentRequest.model_validate(value)


def _result_message(result: StageContentResult, context: RequestContext) -> Message:
    payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    return Message(
        role=Role.ROLE_AGENT,
        parts=[Part(data=ParseDict(payload, Value()))],
        message_id=uuid.uuid4().hex,
        task_id=context.task_id or "",
        context_id=context.context_id or "",
    )


def _failed_result(code: str, summary: str, *, retryable: bool = False) -> StageContentResult:
    return StageContentResult(
        apiVersion=API_VERSION,
        outcome=StageOutcome.FAILED,
        summary=summary,
        artifacts={},
        error=TerminationError(code=code, message=summary, retryable=retryable),
    )


async def _bounded_receive(scope: Scope, receive: Receive, send: Send) -> Receive | None:
    raw_headers = scope.get("headers", [])
    lengths = [value for name, value in raw_headers if name.lower() == b"content-length"]
    if len(lengths) > 1:
        await _gateway_error(scope, receive, send, 400, "invalid_request", "Invalid request")
        return None
    if lengths:
        try:
            length = int(lengths[0])
        except ValueError:
            length = -1
        if length < 0 or length > MAX_A2A_REQUEST_BYTES:
            await _gateway_error(
                scope, receive, send, 413, "request_too_large", "A2A request exceeds its limit"
            )
            return None
    messages: list[ASGIMessage] = []
    total = 0
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            break
        total += len(message.get("body", b""))
        if total > MAX_A2A_REQUEST_BYTES:
            await _gateway_error(
                scope, receive, send, 413, "request_too_large", "A2A request exceeds its limit"
            )
            return None
        if not message.get("more_body", False):
            break

    async def replay() -> ASGIMessage:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    return replay


async def _gateway_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    status: int,
    code: str,
    message: str,
) -> None:
    response = JSONResponse(
        {"code": code, "message": message, "retryable": False}, status_code=status
    )
    await response(scope, receive, send)
