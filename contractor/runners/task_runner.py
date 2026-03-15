from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional
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

from contractor.models.task import RenderedTask, TaskTemplate
from contractor.runners.trace_plugin import AdkTracePlugin
from contractor.tools.memory import MemoryTools, MemoryNote

from contractor.agents.planning_agent.agent import build_planning_agent

load_dotenv()

_GLOBAL_TASK_ID_KEY = "_global_task_id"
ArtifactKind = Literal["result", "summary", "records"]
WorkerBuilder = Callable[..., LlmAgent | AgentTool]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_MODEL = LiteLlm(
    model="lm-studio-qwen3.5",
    timeout=300,
)


@dataclass(slots=True)
class TaskRunnerEvent:
    type: str
    task_name: str
    task_id: int
    payload: dict[str, Any] = field(default_factory=dict)


TaskRunnerEventHandler = Callable[[TaskRunnerEvent], Awaitable[None]]


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


class TaskRunner(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Runner name")
    artifact_service: BaseArtifactService

    output_format: Literal["json", "markdown", "yaml", "xml"] = Field(default="json")
    templates: dict[str, TaskTemplate] = Field(default_factory=dict)
    queue: list[TaskInvocation] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    default_model: LiteLlm = Field(default=DEFAULT_MODEL)

    session_service: InMemorySessionService = Field(
        default_factory=InMemorySessionService
    )

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
        template_key = name
        template = self.templates.get(template_key)
        if template is None:
            template = TaskTemplate.load(template_key)
            self.templates[template_key] = template

        task_ref = ref or f"{template_key}:{len(self.queue)}"
        if any(item.ref == task_ref for item in self.queue):
            raise ValueError(f"Queued task ref '{task_ref}' already exists")

        effective_iterations = (
            iterations if iterations is not None else template.default_iterations
        )
        effective_max_attempts = (
            max_attempts if max_attempts is not None else max(1, effective_iterations)
        )

        if effective_iterations < 1:
            raise ValueError("iterations must be >= 1")
        if effective_max_attempts < effective_iterations:
            raise ValueError("max_attempts must be >= iterations")

        item = TaskInvocation(
            id=uuid4().hex,
            ref=task_ref,
            template_key=template.key,
            worker_builder=worker_builder,
            params=params or {},
            artifacts=list(artifacts or template.default_artifacts),
            iterations=effective_iterations,
            max_attempts=effective_max_attempts,
            max_steps=max_steps,
            namespace=namespace,
            model=model,
        )

        self.queue.append(item)
        return item.ref

    @staticmethod
    def _artifact_filename(template_key: str, kind: ArtifactKind) -> str:
        clean_template_key = (template_key or "").strip().strip("/")
        if not clean_template_key:
            raise ValueError("template_key must not be empty")
        if ".." in clean_template_key.split("/"):
            raise ValueError("template_key must not contain path traversal segments")
        return f"{clean_template_key}/{kind}"

    @classmethod
    def _artifact_names_for_task(cls, template_key: str) -> dict[ArtifactKind, str]:
        return {
            "result": cls._artifact_filename(template_key, "result"),
            "summary": cls._artifact_filename(template_key, "summary"),
            "records": cls._artifact_filename(template_key, "records"),
        }

    def _render_task(
        self,
        *,
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
        self,
        *,
        item: TaskInvocation,
        task: RenderedTask,
    ) -> LlmAgent:
        worker = item.worker_builder(
            namespace=item.namespace or self.name,
            _format=task.format,
        )
        planner = build_planning_agent(
            name=item.ref,
            namespace=item.namespace or self.name,
            worker=worker,
            model=item.model or self.default_model,
        )
        return planner

    async def _emit(
        self,
        handler: Optional[TaskRunnerEventHandler],
        *,
        type: str,
        task_name: str,
        task_id: int,
        **payload: Any,
    ) -> None:
        if handler is None:
            return

        await handler(
            TaskRunnerEvent(
                type=type,
                task_name=task_name,
                task_id=task_id,
                payload=payload,
            )
        )

    async def _ensure_session(
        self,
        *,
        user_id: str,
        session_id: str,
        initial_state: Optional[dict[str, Any]] = None,
    ):
        session = await self.session_service.get_session(
            app_name=self.name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            session = await self.session_service.create_session(
                app_name=self.name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {},
            )
        return session

    async def _get_session_state(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        session = await self.session_service.get_session(
            app_name=self.name,
            user_id=user_id,
            session_id=session_id,
        )
        return dict(session.state) if session else {}

    @staticmethod
    def _global_state_key(task_id: int, key: str) -> str:
        return f"task::{task_id}::{key}"

    def _build_task_initial_state(
        self,
        *,
        task_id: int,
        task: RenderedTask,
        item: TaskInvocation,
        carry_state: dict[str, Any],
        iteration: int,
        input_artifacts: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        state = copy.deepcopy(carry_state)

        state[_GLOBAL_TASK_ID_KEY] = task_id
        state[self._global_state_key(task_id, "objective")] = task.objective
        state[self._global_state_key(task_id, "status")] = "running"
        state[self._global_state_key(task_id, "current")] = None
        state[self._global_state_key(task_id, "result")] = ""
        state[self._global_state_key(task_id, "summary")] = ""
        state[self._global_state_key(task_id, "pool")] = []

        state["runner:last_task_id"] = task_id
        state["runner:last_task_key"] = task.key
        state["runner:last_task_title"] = task.title
        state["runner:active_task_ref"] = item.ref
        state["runner:active_template_key"] = item.template_key
        state["runner:iteration"] = iteration
        state["runner:params"] = copy.deepcopy(item.params)
        state["runner:input_artifacts"] = copy.deepcopy(input_artifacts)

        return state

    def _extract_carry_state(
        self,
        *,
        state: dict[str, Any],
        finished_task_id: int,
    ) -> dict[str, Any]:
        carry = copy.deepcopy(state)

        result_key = self._global_state_key(finished_task_id, "result")
        summary_key = self._global_state_key(finished_task_id, "summary")
        status_key = self._global_state_key(finished_task_id, "status")
        objective_key = self._global_state_key(finished_task_id, "objective")

        carry["runner:previous_task_id"] = finished_task_id
        carry["runner:previous_task_status"] = state.get(status_key)
        carry["runner:previous_task_result"] = state.get(result_key)
        carry["runner:previous_task_summary"] = state.get(summary_key)
        carry["runner:previous_task_objective"] = state.get(objective_key)

        return carry

    @staticmethod
    def _extract_final_text(event: Event) -> str:
        if not event.is_final_response():
            return ""

        content = getattr(event, "content", None)
        if not content:
            return ""

        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []

        for part in parts:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)

        return "\n".join(chunks).strip()

    def _is_task_completed(
        self,
        *,
        task_id: int,
        state: dict[str, Any],
    ) -> bool:
        return state.get(self._global_state_key(task_id, "status")) == "done"

    async def _load_artifact_text(
        self,
        *,
        user_id: str,
        artifact_ref: str,
    ) -> str:
        part = await self.artifact_service.load_artifact(
            app_name=self.name,
            user_id=user_id,
            session_id=None,
            filename=artifact_ref,
        )
        if part is None:
            raise FileNotFoundError(f"Artifact '{artifact_ref}' not found")

        text = getattr(part, "text", None)
        if text is not None:
            return text

        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None and getattr(inline_data, "data", None) is not None:
            data = inline_data.data
            if isinstance(data, str):
                return data
            if isinstance(data, (bytes, bytearray)):
                return data.decode("utf-8")

        return ""

    async def _load_artifacts(
        self,
        *,
        user_id: str,
        artifact_refs: list[str],
    ) -> dict[str, str]:
        loaded: dict[str, str] = {}

        for artifact_ref in artifact_refs:
            loaded[artifact_ref] = await self._load_artifact_text(
                user_id=user_id,
                artifact_ref=artifact_ref,
            )

        return loaded

    async def _publish_task_artifacts(
        self,
        *,
        user_id: str,
        template_key: str,
        result: dict[str, Any],
    ) -> None:
        records = result.get("records", [])
        if isinstance(records, str):
            records_text = records
        else:
            records_text = json.dumps(records, ensure_ascii=False)

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
                filename=self._artifact_filename(template_key, kind),
                artifact=types.Part.from_text(text=text),
            )

    async def _inject_artifacts(
        self, user_id: str, namespace: str, input_artifacts: dict[str, str]
    ):
        mem_tools = MemoryTools(name=namespace)
        memories: list[MemoryNote] = []
        for name, text in input_artifacts.items():
            description: str = f"result from previous task {name}"
            if "/" in name:
                template_key = name.split("/")[0]
                if template_key in self.templates:
                    task_template = self.templates.get(template_key)
                    description = task_template.title

            memories.append(
                MemoryNote(
                    name=name,
                    memory=text,
                    description=description,
                    tags=[name, "inbox", "previous-task-result"],
                )
            )
        await mem_tools.inject(
            memories=memories,
            artifact_service=self.artifact_service,
            app_name=self.name,
            user_id=user_id,
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
    ) -> dict[str, Any]:
        agent = self._spawn_planning_agent(item=item, task=rendered_task)
        await self._inject_artifacts(
            user_id=user_id, namespace=item.ref, input_artifacts=input_artifacts
        )

        session_id = str(uuid4())
        initial_state = self._build_task_initial_state(
            task_id=task_id,
            task=rendered_task,
            item=item,
            carry_state=carry_state,
            iteration=iteration,
            input_artifacts=input_artifacts,
        )

        await self._ensure_session(
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
        )

        await self._emit(
            on_event,
            type="iteration_started",
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            objective=rendered_task.objective,
            initial_state=initial_state,
        )

        plugin = AdkTracePlugin(
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            emit=lambda **kw: self._emit(on_event, **kw),
        )

        runner = Runner(
            agent=agent,
            app_name=self.name,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
            plugins=[plugin],
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=rendered_task.objective)],
        )

        final_text = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            state = await self._get_session_state(
                user_id=user_id,
                session_id=session_id,
            )

            event_final = self._extract_final_text(event)
            if event_final:
                final_text = event_final
                await self._emit(
                    on_event,
                    type="final_text",
                    task_name=item.ref,
                    task_id=task_id,
                    iteration=iteration,
                    session_id=session_id,
                    text=event_final,
                    state=state,
                )

        final_state = await self._get_session_state(
            user_id=user_id,
            session_id=session_id,
        )

        next_carry_state = self._extract_carry_state(
            state=final_state,
            finished_task_id=task_id,
        )

        result = {
            "invocation_id": item.id,
            "task_ref": item.ref,
            "task_key": rendered_task.key,
            "task_title": rendered_task.title,
            "template_key": item.template_key,
            "task_id": task_id,
            "session_id": session_id,
            "final_response": final_text,
            "state": final_state,
            "carry_state": next_carry_state,
            "status": final_state.get(self._global_state_key(task_id, "status")),
            "result": final_state.get(self._global_state_key(task_id, "result")),
            "summary": final_state.get(self._global_state_key(task_id, "summary")),
            "records": final_state.get(self._global_state_key(task_id, "pool"), []),
            "params": copy.deepcopy(item.params),
            "input_artifacts": copy.deepcopy(input_artifacts),
            "published_artifacts": self._artifact_names_for_task(rendered_task.key),
        }

        await self._emit(
            on_event,
            type="iteration_finished",
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            result=result,
        )

        return result

    async def _run_task_with_retries(
        self,
        *,
        item: TaskInvocation,
        task_id: int,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> dict[str, Any]:
        template = self.templates[item.template_key]
        input_artifacts = await self._load_artifacts(
            user_id=user_id,
            artifact_refs=item.artifacts,
        )

        rendered_task = self._render_task(
            template=template,
            params=item.params,
            artifacts=input_artifacts,
        )

        current_carry_state: dict[str, Any] = {}
        last_result: Optional[dict[str, Any]] = None
        successful_runs = 0

        await self._emit(
            on_event,
            type="task_started",
            task_name=item.ref,
            task_id=task_id,
            template_key=item.template_key,
            task_title=template.title,
            iterations=item.iterations,
            max_attempts=item.max_attempts,
            params=item.params,
            artifacts=item.artifacts,
            published_artifacts=self._artifact_names_for_task(template.key),
        )

        for iteration in range(1, item.max_attempts + 1):
            result = await self._run_single_iteration(
                item=item,
                rendered_task=rendered_task,
                input_artifacts=input_artifacts,
                task_id=task_id,
                user_id=user_id,
                carry_state=current_carry_state,
                iteration=iteration,
                on_event=on_event,
            )
            last_result = result

            completed = self._is_task_completed(
                task_id=task_id,
                state=result["state"],
            )

            await self._emit(
                on_event,
                type="iteration_result",
                task_name=item.ref,
                task_id=task_id,
                iteration=iteration,
                session_id=result["session_id"],
                status=result["status"],
                result=result["result"],
                summary=result["summary"],
                completed=completed,
            )

            if completed:
                successful_runs += 1
                await self._publish_task_artifacts(
                    user_id=user_id,
                    template_key=template.key,
                    result=result,
                )

            if completed and successful_runs >= item.iterations:
                await self._emit(
                    on_event,
                    type="task_finished",
                    task_name=item.ref,
                    task_id=task_id,
                    session_id=result["session_id"],
                    status=result["status"],
                    result=result["result"],
                    summary=result["summary"],
                    records=result["records"],
                    published_artifacts=result["published_artifacts"],
                )
                return result

            current_carry_state = result["carry_state"]

        await self._emit(
            on_event,
            type="task_failed",
            task_name=item.ref,
            task_id=task_id,
            max_attempts=item.max_attempts,
            last_result=last_result,
        )

        raise RuntimeError(
            f"Task '{item.ref}' was not completed "
            f"{item.iterations} time(s) after {item.max_attempts} attempt(s)."
        )

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        await self._emit(
            on_event,
            type="run_started",
            task_name="__runner__",
            task_id=-1,
            total_tasks=len(self.queue),
            user_id=user_id,
        )

        for task_id, item in enumerate(self.queue):
            result = await self._run_task_with_retries(
                item=item,
                task_id=task_id,
                user_id=user_id,
                on_event=on_event,
            )
            results.append(result)

            await self._emit(
                on_event,
                type="global_task_finished",
                task_name=item.ref,
                task_id=task_id,
                session_id=result["session_id"],
                status=result["status"],
                result=result["result"],
                summary=result["summary"],
                published_artifacts=result["published_artifacts"],
            )

        await self._emit(
            on_event,
            type="run_finished",
            task_name="__runner__",
            task_id=-1,
            results=results,
        )

        return results
