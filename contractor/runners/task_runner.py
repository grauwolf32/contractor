from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any, Awaitable, Callable, Literal, Optional, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.artifacts import BaseArtifactService
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.genai import types
from pydantic import BaseModel, Field

from contractor.agents.planning_agent.agent import build_planning_agent
from contractor.models.task import RenderedTask, TaskTemplate
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.tools.memory import MemoryNote, MemoryTools

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_GLOBAL_TASK_ID_KEY = "_global_task_id"

ArtifactKind = Literal["result", "summary", "records"]
_ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("result", "summary", "records")

WorkerBuilder = Callable[..., LlmAgent | AgentTool]
TaskRunnerEventHandler = Callable[["TaskRunnerEvent"], Awaitable[None]]

DEFAULT_MODEL = LiteLlm(model="lm-studio-qwen3.5", timeout=300)


# ─── Enums ────────────────────────────────────────────────────────────────────


@unique
class EventType(StrEnum):
    """All event types emitted by TaskRunner, in one discoverable place."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_FAILED = "task_failed"
    GLOBAL_TASK_FINISHED = "global_task_finished"
    ITERATION_STARTED = "iteration_started"
    ITERATION_FINISHED = "iteration_finished"
    ITERATION_RESULT = "iteration_result"
    FINAL_TEXT = "final_text"


@unique
class TaskStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"


# ─── Custom Exceptions ───────────────────────────────────────────────────────


class TaskNotCompletedError(Exception):
    """Raised when a task exhausts all retry attempts without completing."""

    def __init__(self, ref: str, iterations: int, max_attempts: int) -> None:
        self.ref = ref
        self.iterations = iterations
        self.max_attempts = max_attempts
        super().__init__(
            f"Task '{ref}' was not completed "
            f"{iterations} time(s) after {max_attempts} attempt(s)."
        )


class InvalidTemplateKeyError(ValueError):
    """Raised when an artifact template key is invalid."""


# ─── Data Structures ─────────────────────────────────────────────────────────


class TaskResult(TypedDict):
    """Strongly-typed dict returned from each iteration / task."""

    invocation_id: str
    task_ref: str
    task_key: str
    task_title: str
    template_key: str
    task_id: int
    session_id: str
    final_response: str
    state: dict[str, Any]
    carry_state: dict[str, Any]
    status: str | None
    result: Any
    summary: str | None
    records: list[Any]
    params: dict[str, Any]
    input_artifacts: dict[str, str]
    published_artifacts: dict[ArtifactKind, str]


@dataclass(slots=True, frozen=True)
class TaskRunnerEvent:
    type: EventType
    task_name: str
    task_id: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskInvocation:
    id: str
    ref: str
    template_key: str
    worker_builder: WorkerBuilder

    params: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    iterations: int = 1
    max_attempts: int = 1
    max_steps: int = 15

    namespace: str | None = None
    model: LiteLlm | None = None

    def effective_namespace(self, fallback: str) -> str:
        return self.namespace or fallback

    def effective_model(self, fallback: LiteLlm) -> LiteLlm:
        return self.model or fallback


# ─── Helpers (module-level, stateless) ────────────────────────────────────────


def _validate_template_key(template_key: str) -> str:
    """Return a cleaned template key or raise."""
    cleaned = (template_key or "").strip().strip("/")
    if not cleaned:
        raise InvalidTemplateKeyError("template_key must not be empty")
    if ".." in cleaned.split("/"):
        raise InvalidTemplateKeyError(
            "template_key must not contain path traversal segments"
        )
    return cleaned


def _artifact_filename(template_key: str, kind: ArtifactKind) -> str:
    return f"{_validate_template_key(template_key)}/{kind}"


def _artifact_names_for_task(template_key: str) -> dict[ArtifactKind, str]:
    return {kind: _artifact_filename(template_key, kind) for kind in _ARTIFACT_KINDS}


def _global_state_key(task_id: int, key: str) -> str:
    return f"task::{task_id}::{key}"


def _extract_final_text(event: Event) -> str:
    """Pull the concatenated text from a final-response event."""
    if not event.is_final_response():
        return ""

    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return "\n".join(
        text for part in parts if (text := getattr(part, "text", None))
    ).strip()


def _decode_part_text(part: types.Part | None) -> str:
    """Best-effort text extraction from an artifact Part."""
    if part is None:
        return ""

    text = getattr(part, "text", None)
    if text is not None:
        return text

    inline_data = getattr(part, "inline_data", None)
    data = getattr(inline_data, "data", None) if inline_data else None
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8")
    return ""


# ─── Main Runner ──────────────────────────────────────────────────────────────


class TaskRunner(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Runner name")
    artifact_service: BaseArtifactService

    output_format: Literal["json", "markdown", "yaml", "xml"] = "json"
    templates: dict[str, TaskTemplate] = Field(default_factory=dict)
    queue: list[TaskInvocation] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    default_model: LiteLlm = Field(default=DEFAULT_MODEL)
    session_service: InMemorySessionService = Field(
        default_factory=InMemorySessionService
    )

    # ── Public API ────────────────────────────────────────────────────────

    def add_variable(self, name: str, value: str) -> None:
        self.variables[name] = value

    def add_task(
        self,
        name: str,
        *,
        worker_builder: WorkerBuilder,
        ref: str | None = None,
        params: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        iterations: int | None = None,
        max_attempts: int | None = None,
        max_steps: int = 15,
        namespace: Optional[str] = None,
        model: Optional[LiteLlm] = None,
    ) -> str:
        template = self._ensure_template(name)
        task_ref = ref or f"{name}:{len(self.queue)}"
        self._assert_unique_ref(task_ref)

        eff_iterations, eff_max_attempts = self._resolve_retry_params(
            template, iterations, max_attempts
        )

        item = TaskInvocation(
            id=uuid4().hex,
            ref=task_ref,
            template_key=template.key,
            worker_builder=worker_builder,
            params=params or {},
            artifacts=list(artifacts or template.default_artifacts),
            iterations=eff_iterations,
            max_attempts=eff_max_attempts,
            max_steps=max_steps,
            namespace=namespace,
            model=model,
        )
        self.queue.append(item)
        return item.id


    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> list[TaskResult]:
        results: list[TaskResult] = []
        total_tasks = len(self.queue)

        await self._emit(
            on_event,
            type=EventType.RUN_STARTED,
            task_name="__runner__",
            task_id=-1,
            total_tasks=total_tasks,
            completed_tasks=0,
            user_id=user_id,
        )

        for task_id, item in enumerate(self.queue):
            result = await self._run_task_with_retries(
                item=item,
                task_id=task_id,
                user_id=user_id,
                on_event=on_event,
                total_tasks=total_tasks,
            )

            results.append(result)

            await self._emit(
                on_event,
                type=EventType.GLOBAL_TASK_FINISHED,
                task_name=item.ref,
                task_id=task_id,
                session_id=result["session_id"],
                status=result["status"],
                result=result["result"],
                summary=result["summary"],
                total_tasks=total_tasks,
                completed_tasks=task_id + 1,
            )

        return results

    # ── Template & validation helpers ─────────────────────────────────────

    def _ensure_template(self, key: str) -> TaskTemplate:
        if key not in self.templates:
            self.templates[key] = TaskTemplate.load(key)
        return self.templates[key]

    def _assert_unique_ref(self, ref: str) -> None:
        if any(item.ref == ref for item in self.queue):
            raise ValueError(f"Queued task ref '{ref}' already exists")

    @staticmethod
    def _resolve_retry_params(
        template: TaskTemplate,
        iterations: int | None,
        max_attempts: int | None,
    ) -> tuple[int, int]:
        eff_iterations = (
            iterations if iterations is not None else template.default_iterations
        )
        eff_max_attempts = (
            max_attempts if max_attempts is not None else max(1, eff_iterations)
        )
        if eff_iterations < 1:
            raise ValueError("iterations must be >= 1")
        if eff_max_attempts < eff_iterations:
            raise ValueError("max_attempts must be >= iterations")
        return eff_iterations, eff_max_attempts

    # ── Rendering & agent creation ────────────────────────────────────────

    def _render_task(
        self,
        template: TaskTemplate,
        params: dict[str, Any],
        artifacts: dict[str, dict[str, str]],
    ) -> RenderedTask:
        return RenderedTask.from_template(
            template=template,
            variables=self.variables,
            params=params,
            artifacts=artifacts,
        )

    def _spawn_planning_agent(
        self, item: TaskInvocation, task: RenderedTask
    ) -> LlmAgent:
        ns = item.effective_namespace(self.name)
        worker = item.worker_builder(namespace=ns, _format=task.format)
        return build_planning_agent(
            _format="xml",
            name=item.ref,
            namespace=ns,
            worker=worker,
            model=item.effective_model(self.default_model),
        )

    # ── Event emission ────────────────────────────────────────────────────

    @staticmethod
    async def _emit(
        handler: Optional[TaskRunnerEventHandler],
        *,
        type: EventType,
        task_name: str,
        task_id: int,
        **payload: Any,
    ) -> None:
        if handler is None:
            return
        await handler(
            TaskRunnerEvent(
                type=type, task_name=task_name, task_id=task_id, payload=payload
            )
        )

    # ── Session management ────────────────────────────────────────────────

    async def _ensure_session(
        self,
        user_id: str,
        session_id: str,
        initial_state: dict[str, Any] | None = None,
    ):
        existing = await self.session_service.get_session(
            app_name=self.name, user_id=user_id, session_id=session_id
        )
        if existing is not None:
            return existing
        return await self.session_service.create_session(
            app_name=self.name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state or {},
        )

    async def _get_session_state(self, user_id: str, session_id: str) -> dict[str, Any]:
        session = await self.session_service.get_session(
            app_name=self.name, user_id=user_id, session_id=session_id
        )
        return dict(session.state) if session else {}

    # ── State building ────────────────────────────────────────────────────

    def _build_task_initial_state(
        self,
        task_id: int,
        task: RenderedTask,
        item: TaskInvocation,
        carry_state: dict[str, Any],
        iteration: int,
        input_artifacts: dict[str, str],
    ) -> dict[str, Any]:
        def key(k):
            return _global_state_key(task_id, k)

        state = copy.deepcopy(carry_state)

        # Task-scoped keys
        state[_GLOBAL_TASK_ID_KEY] = task_id
        state[key("objective")] = task.objective
        state[key("status")] = TaskStatus.RUNNING
        state[key("current")] = None
        state[key("result")] = ""
        state[key("summary")] = ""
        state[key("pool")] = []

        # Runner-scoped keys
        state.update(
            {
                "runner:last_task_id": task_id,
                "runner:last_task_key": task.key,
                "runner:last_task_title": task.title,
                "runner:active_task_ref": item.ref,
                "runner:active_template_key": item.template_key,
                "runner:iteration": iteration,
                "runner:params": copy.deepcopy(item.params),
                "runner:input_artifacts": copy.deepcopy(input_artifacts),
            }
        )
        return state

    @staticmethod
    def _extract_carry_state(
        state: dict[str, Any], finished_task_id: int
    ) -> dict[str, Any]:
        carry = copy.deepcopy(state)

        def key(k):
            return _global_state_key(finished_task_id, k)

        carry.update(
            {
                "runner:previous_task_id": finished_task_id,
                "runner:previous_task_status": state.get(key("status")),
                "runner:previous_task_result": state.get(key("result")),
                "runner:previous_task_summary": state.get(key("summary")),
                "runner:previous_task_objective": state.get(key("objective")),
            }
        )
        return carry

    def _is_task_completed(self, task_id: int, state: dict[str, Any]) -> bool:
        return state.get(_global_state_key(task_id, "status")) == TaskStatus.DONE

    # ── Artifact I/O ──────────────────────────────────────────────────────

    async def _load_artifact_text(self, user_id: str, artifact_ref: str) -> str:
        part = await self.artifact_service.load_artifact(
            app_name=self.name,
            user_id=user_id,
            session_id=None,
            filename=artifact_ref,
        )
        return _decode_part_text(part)

    async def _load_artifacts(
        self, user_id: str, artifact_refs: list[str]
    ) -> dict[str, str]:
        return {
            ref: await self._load_artifact_text(user_id=user_id, artifact_ref=ref)
            for ref in artifact_refs
        }

    async def _publish_task_artifacts(
        self,
        user_id: str,
        template_key: str,
        result: TaskResult,
    ) -> None:
        records_raw = result.get("records", [])
        records_text = (
            records_raw
            if isinstance(records_raw, str)
            else json.dumps(records_raw, ensure_ascii=False)
        )

        payloads: dict[ArtifactKind, str] = {
            "result": result.get("result", "") or "",
            "summary": result.get("summary", "") or "",
            "records": records_text,
        }

        for kind, text in payloads.items():
            await self.artifact_service.save_artifact(
                app_name=self.name,
                user_id=user_id,
                session_id=None,
                filename=_artifact_filename(template_key, kind),
                artifact=types.Part.from_text(text=text),
            )

    async def _inject_artifacts(
        self, user_id: str, namespace: str, input_artifacts: dict[str, str]
    ) -> None:
        memories: list[MemoryNote] = []
        for name, text in input_artifacts.items():
            description = self._describe_artifact(name)
            memories.append(
                MemoryNote(
                    name=name,
                    memory=text,
                    description=description,
                    tags=[name, "inbox", "previous-task-result"],
                )
            )

        if not memories:
            return

        mem_tools = MemoryTools(name=namespace)
        await mem_tools.inject(
            memories=memories,
            artifact_service=self.artifact_service,
            app_name=self.name,
            user_id=user_id,
        )

    def _describe_artifact(self, name: str) -> str:
        """Derive a human-readable description for a loaded artifact."""
        if "/" in name:
            template_key = name.split("/", 1)[0]
            template = self.templates.get(template_key)
            if template is not None:
                return template.title
        return f"result from previous task {name}"

    # ── Iteration execution ───────────────────────────────────────────────

    def _build_iteration_result(
        self,
        item: TaskInvocation,
        rendered_task: RenderedTask,
        task_id: int,
        session_id: str,
        final_text: str,
        final_state: dict[str, Any],
        input_artifacts: dict[str, str],
    ) -> TaskResult:
        def key(k):
            return _global_state_key(task_id, k)

        carry_state = self._extract_carry_state(final_state, task_id)

        return TaskResult(
            invocation_id=item.id,
            task_ref=item.ref,
            task_key=rendered_task.key,
            task_title=rendered_task.title,
            template_key=item.template_key,
            task_id=task_id,
            session_id=session_id,
            final_response=final_text,
            state=final_state,
            carry_state=carry_state,
            status=final_state.get(key("status")),
            result=final_state.get(key("result")),
            summary=final_state.get(key("summary")),
            records=final_state.get(key("pool"), []),
            params=copy.deepcopy(item.params),
            input_artifacts=copy.deepcopy(input_artifacts),
            published_artifacts=_artifact_names_for_task(rendered_task.key),
        )

    async def _run_single_iteration(
        self,
        *,
        item: TaskInvocation,
        rendered_task: RenderedTask,
        input_artifacts: dict[str, str],
        task_id: int,
        user_id: str,
        carry_state: dict[str, Any],
        iteration: int,
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> TaskResult:
        agent = self._spawn_planning_agent(item, rendered_task)
        await self._inject_artifacts(
            user_id=user_id,
            namespace=item.effective_namespace(self.name),
            input_artifacts=input_artifacts,
        )

        session_id = uuid4().hex
        initial_state = self._build_task_initial_state(
            task_id=task_id,
            task=rendered_task,
            item=item,
            carry_state=carry_state,
            iteration=iteration,
            input_artifacts=input_artifacts,
        )

        await self._ensure_session(user_id, session_id, initial_state)

        await self._emit(
            on_event,
            type=EventType.ITERATION_STARTED,
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            objective=rendered_task.objective,
            initial_state=initial_state,
        )

        runner = Runner(
            agent=agent,
            app_name=self.name,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
            plugins=self._build_plugins(item, task_id, iteration, session_id, on_event),
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=rendered_task.objective)],
        )

        final_text = await self._consume_events(
            runner=runner,
            user_id=user_id,
            session_id=session_id,
            message=message,
            item=item,
            task_id=task_id,
            iteration=iteration,
            on_event=on_event,
        )

        final_state = await self._get_session_state(user_id, session_id)
        result = self._build_iteration_result(
            item,
            rendered_task,
            task_id,
            session_id,
            final_text,
            final_state,
            input_artifacts,
        )

        await self._emit(
            on_event,
            type=EventType.ITERATION_FINISHED,
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            result=result,
        )
        return result

    def _build_plugins(
        self,
        item: TaskInvocation,
        task_id: int,
        iteration: int,
        session_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> list:
        def emit_fn(**kw):
            return self._emit(on_event, **kw)

        common = dict(
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            emit=emit_fn,
        )
        return [AdkTracePlugin(**common), AdkMetricsPlugin(**common)]

    async def _consume_events(
        self,
        *,
        runner: Runner,
        user_id: str,
        session_id: str,
        message: types.Content,
        item: TaskInvocation,
        task_id: int,
        iteration: int,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> str:
        final_text = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            event_final = _extract_final_text(event)
            if not event_final:
                continue

            final_text = event_final
            state = await self._get_session_state(user_id, session_id)
            await self._emit(
                on_event,
                type=EventType.FINAL_TEXT,
                task_name=item.ref,
                task_id=task_id,
                iteration=iteration,
                session_id=session_id,
                text=event_final,
                state=state,
            )
        return final_text

    # ── Retry orchestration ───────────────────────────────────────────────

    async def _run_task_with_retries(
        self,
        *,
        item: TaskInvocation,
        task_id: int,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler] = None,
        total_tasks: int,
    ) -> TaskResult:
        template = self.templates[item.template_key]
        input_artifacts = await self._load_artifacts(user_id, item.artifacts)
        rendered_task = self._render_task(
            template,
            item.params,
            input_artifacts,
        )

        await self._emit(
            on_event,
            type=EventType.TASK_STARTED,
            task_name=item.ref,
            task_id=task_id,
            template_key=item.template_key,
            task_title=template.title,
            iterations=item.iterations,
            max_attempts=item.max_attempts,
            params=item.params,
            artifacts=item.artifacts,
            published_artifacts=_artifact_names_for_task(template.key),
            total_tasks=total_tasks,
            completed_tasks=task_id,
        )

        carry_state: dict[str, Any] = {}
        last_result: TaskResult | None = None
        successful_runs = 0

        for iteration in range(1, item.max_attempts + 1):
            result = await self._run_single_iteration(
                item=item,
                rendered_task=rendered_task,
                input_artifacts=input_artifacts,
                task_id=task_id,
                user_id=user_id,
                carry_state=carry_state,
                iteration=iteration,
                on_event=on_event,
            )

            last_result = result
            completed = self._is_task_completed(task_id, result["state"])

            next_successful_runs = successful_runs + (1 if completed else 0)

            await self._emit(
                on_event,
                type=EventType.ITERATION_RESULT,
                task_name=item.ref,
                task_id=task_id,
                iteration=iteration,
                session_id=result["session_id"],
                status=result["status"],
                result=result["result"],
                summary=result["summary"],
                completed=completed,
                iterations_required=item.iterations,
                max_attempts=item.max_attempts,
                successful_runs=next_successful_runs,
                total_tasks=total_tasks,
                completed_tasks=task_id,
            )

            if completed:
                successful_runs = next_successful_runs
                await self._publish_task_artifacts(user_id, template.key, result)

                if successful_runs >= item.iterations:
                    await self._emit(
                        on_event,
                        type=EventType.TASK_FINISHED,
                        task_name=item.ref,
                        task_id=task_id,
                        session_id=result["session_id"],
                        status=result["status"],
                        result=result["result"],
                        summary=result["summary"],
                        records=result["records"],
                        published_artifacts=result["published_artifacts"],
                        total_tasks=total_tasks,
                        completed_tasks=task_id,
                    )
                    return result

            carry_state = result["carry_state"]

        await self._emit(
            on_event,
            type=EventType.TASK_FAILED,
            task_name=item.ref,
            task_id=task_id,
            max_attempts=item.max_attempts,
            last_result=last_result,
            total_tasks=total_tasks,
            completed_tasks=task_id,
        )

        raise TaskNotCompletedError(
            ref=item.ref,
            iterations=item.iterations,
            max_attempts=item.max_attempts,
        )
