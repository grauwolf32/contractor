"""Allocation-scoped A2A 1.0 gRPC binding for the in-process Worker."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import grpc
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.grpc_handler import GrpcHandler
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
    a2a_pb2,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message_factory import GetMessageClass
from google.protobuf.struct_pb2 import Value
from pydantic import ValidationError

from contractor_runtime._grpc_metadata import response_request_id
from contractor_runtime.contracts import (
    StageContentRequest,
    WorkerCompletion,
)
from contractor_runtime.contracts.base import RESOURCE_ID_PATTERN, parse_private_rpc_origin
from contractor_runtime.contracts.codec import decode_private_value
from contractor_runtime.contracts.grpc_wire.limits import MAX_A2A_BYTES

logger = logging.getLogger(__name__)

STAGE_CONTENT_MEDIA_TYPE = "application/vnd.contractor.stage-content+json"
WORKER_COMPLETION_MEDIA_TYPE = "application/vnd.contractor.worker-completion+json"
# Bounds the official A2A protobuf message, including Planner input.
# Worker and Planner DTOs retain their smaller independent content bounds.
MAX_A2A_REQUEST_BYTES = MAX_A2A_BYTES


class InvocableWorker(Protocol):
    allocation_id: str

    async def invoke(self, request: StageContentRequest) -> WorkerCompletion: ...

    async def failure_completion(
        self, code: str, message: str, *, retryable: bool = False
    ) -> WorkerCompletion: ...

    def cancel_active(self, owner: asyncio.Task[Any]) -> None:
        """Cancel the active invocation only when ``owner`` started it."""
        ...


class ActiveA2AProvider(Protocol):
    async def active_a2a_application(self, allocation_id: str) -> GrpcHandler | None: ...


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
                protocol_binding="GRPC",
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
        default_output_modes=[WORKER_COMPLETION_MEDIA_TYPE],
        skills=[
            AgentSkill(
                id="contractor_stage_content",
                name="Execute Contractor stage content",
                description="Execute one strict Contractor StageContentRequest.",
                tags=["contractor", "stage"],
                input_modes=[STAGE_CONTENT_MEDIA_TYPE],
                output_modes=[WORKER_COMPLETION_MEDIA_TYPE],
            )
        ],
    )


def agent_card_dict(card: AgentCard) -> dict[str, Any]:
    return MessageToDict(card)


def build_worker_a2a_application(worker: InvocableWorker, card: AgentCard) -> GrpcHandler:
    executor = ContractorAgentExecutor(worker)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(owner_resolver=lambda _: worker.allocation_id),
        agent_card=card,
    )
    return GrpcHandler(handler)


@dataclass
class _Invocation:
    owner: asyncio.Task[Any]
    cancel_requested: bool = False


class ContractorAgentExecutor(AgentExecutor):
    def __init__(self, worker: InvocableWorker) -> None:
        self._worker = worker
        # A2A Task ID -> the asyncio task currently inside ``worker.invoke``.
        self._invocations: dict[str, _Invocation] = {}

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
            result = await self._worker.failure_completion(
                "allocation_route_mismatch", "A2A tenant does not name the active allocation"
            )
        else:
            try:
                request = _stage_request(context)
            except (TypeError, ValueError, ValidationError):
                result = await self._worker.failure_completion(
                    "invalid_stage_content", "A2A message must contain one StageContentRequest"
                )
            else:
                try:
                    result = await self._invoke(context.task_id, request, updater)
                    if result is None:
                        return
                except Exception:
                    # A Worker defect must still end the Task with a bounded
                    # Contractor failure instead of leaving it working.
                    try:
                        result = await self._worker.failure_completion(
                            "worker_execution_failed", "Worker execution failed", retryable=True
                        )
                    except Exception:
                        await updater.failed()
                        return
        message = _result_message(result, context)
        if result.failure is not None:
            await updater.failed(message)
        else:
            await updater.complete(message)

    async def _invoke(
        self, task_id: str, request: StageContentRequest, updater: TaskUpdater
    ) -> WorkerCompletion | None:
        owner = asyncio.current_task()
        if owner is None or task_id in self._invocations:
            raise RuntimeError("A2A Task is already executing")
        invocation = _Invocation(owner)
        self._invocations[task_id] = invocation
        try:
            return await self._worker.invoke(request)
        except asyncio.CancelledError:
            if invocation.cancel_requested or owner.cancelling():
                # Tool Workers cancel an owned child for CancelTask, so the
                # producer's cancellation count alone cannot identify it.
                raise
            # A native child can cancel without cancelling this producer.
            # Publish before the SDK ignores its cancelled producer.
            await updater.cancel()
            return None
        finally:
            self._invocations.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        del event_queue
        if context.call_context.tenant != self._worker.allocation_id or not context.task_id:
            return
        # Cancel only the invocation this A2A Task started; a concurrent Task
        # rejected as busy must never cancel another Task's active invocation.
        invocation = self._invocations.get(context.task_id)
        if invocation is not None:
            invocation.cancel_requested = True
            self._worker.cancel_active(invocation.owner)


def grpc_endpoint(origin: str) -> str:
    """Derive native A2A authority only from a validated HTTPS origin."""
    parsed = parse_private_rpc_origin("A2A endpoint", origin)
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{host}:{parsed.port or 443}"


def register_a2a_gateway(server, provider: ActiveA2AProvider) -> None:
    """Route official SDK methods by the active allocation's tenant identity."""
    service = a2a_pb2.DESCRIPTOR.services_by_name["A2AService"]

    async def resolve(request, context):
        await response_request_id(context)
        tenant = getattr(request, "tenant", "")
        if not isinstance(tenant, str) or RESOURCE_ID_PATTERN.fullmatch(tenant) is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid_allocation")
        try:
            handler = await provider.active_a2a_application(tenant)
        except Exception as error:
            logger.error("A2A routing failed error_type=%s", type(error).__name__)
            await context.abort(grpc.StatusCode.INTERNAL, "allocation_lookup_failed")
        if handler is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "allocation_not_active")
        return handler

    def unary(name):
        async def call(request, context):
            handler = await resolve(request, context)
            response = await getattr(handler, name)(request, context)
            if response.ByteSize() > MAX_A2A_BYTES:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "a2a_response_too_large")
            return response

        return call

    def streaming(name):
        async def call(request, context):
            handler = await resolve(request, context)
            async for item in getattr(handler, name)(request, context):
                if item.ByteSize() > MAX_A2A_BYTES:
                    await context.abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED, "a2a_response_too_large"
                    )
                yield item

        return call

    methods = {}
    for method in service.methods:
        request_type = GetMessageClass(method.input_type)
        response_type = GetMessageClass(method.output_type)
        factory = (
            grpc.unary_stream_rpc_method_handler
            if method.server_streaming
            else grpc.unary_unary_rpc_method_handler
        )
        methods[method.name] = factory(
            streaming(method.name) if method.server_streaming else unary(method.name),
            request_deserializer=request_type.FromString,
            response_serializer=response_type.SerializeToString,
        )
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler(service.full_name, methods),)
    )


def _stage_request(context: RequestContext) -> StageContentRequest:
    message = context.message
    if message is None or len(message.parts) != 1 or not message.parts[0].HasField("data"):
        raise ValueError("one DataPart is required")
    part = message.parts[0]
    if part.media_type != STAGE_CONTENT_MEDIA_TYPE:
        raise ValueError("StageContentRequest media type is invalid")
    value = MessageToDict(part.data)
    if not isinstance(value, dict):
        raise TypeError("StageContentRequest DataPart must be an object")
    return decode_private_value(StageContentRequest, _struct_numbers(value))


def _struct_numbers(value):
    # google.protobuf.Struct has only a double for JSON numbers. Restore exact
    # integral values at this SDK boundary; ordinary private JSON stays strict.
    # The DTO codec still rejects fractions, non-finite and unsafe integers.
    if isinstance(value, dict):
        return {key: _struct_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_struct_numbers(item) for item in value]
    if type(value) is float and value.is_integer():
        return int(value)
    return value


def _result_message(result: WorkerCompletion, context: RequestContext) -> Message:
    payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    return Message(
        role=Role.ROLE_AGENT,
        parts=[
            Part(
                data=ParseDict(payload, Value()),
                media_type=WORKER_COMPLETION_MEDIA_TYPE,
            )
        ],
        message_id=uuid.uuid4().hex,
        task_id=context.task_id or "",
        context_id=context.context_id or "",
    )
