from __future__ import annotations

import asyncio
import copy
import inspect
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.artifacts import BaseArtifactService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr

from contractor.agents.planning_agent.agent import build_planning_agent
from contractor.runners._helpers import _decode_part_text, _extract_final_text
from contractor.runners.artifacts import (
    artifact_names_for_key,
    save_result_artifacts,
    validate_artifact_key,
)
from contractor.runners.models import (
    Checkpoint,
    CheckpointEntry,
    EventType,
    RenderedTask,
    TaskInvocation,
    TaskResult,
    TaskRunnerEvent,
    TaskRunnerEventHandler,
    TaskScopedKeys,
    TaskStatus,
    TaskTemplate,
    WorkerBuilder,
    build_active_state,
)
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.sandbox_cleanup import SandboxCleanupPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.runners.skills import inject_skills
from contractor.tools.memory import MemoryNote, MemoryTools
from contractor.tools.observations import ObservationConfig
from contractor.tools.podman import teardown_all as _teardown_sandboxes
from contractor.utils import all_active_prompt_versions
from contractor.utils.settings import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ─── Custom Exceptions ───────────────────────────────────────────────────────


class TaskNotCompletedError(Exception):
    """Raised when a task exhausts all retry attempts without completing."""

    def __init__(
        self,
        ref: str,
        iterations: int,
        max_attempts: int,
        last_error: str | None = None,
    ) -> None:
        self.ref = ref
        self.iterations = iterations
        self.max_attempts = max_attempts
        self.last_error = last_error
        message = (
            f"Task '{ref}' was not completed "
            f"{iterations} time(s) after {max_attempts} attempt(s)."
        )
        if last_error:
            message += f" Last error: {last_error}"
        super().__init__(message)


# ─── Main Runner ──────────────────────────────────────────────────────────────


class TaskRunner(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Runner name")
    artifact_service: BaseArtifactService
    checkpoint_path: Path | None = Field(default=None)

    templates: dict[tuple[str, str], TaskTemplate] = Field(default_factory=dict)
    queue: list[TaskInvocation] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    default_model: LiteLlm = Field(default=DEFAULT_MODEL)
    session_service: InMemorySessionService = Field(
        default_factory=InMemorySessionService
    )
    # Runner-level default planner observations; a per-task ``observations`` on
    # ``add_task`` overrides it. Lets a workflow opt in once with
    # ``TaskRunner(..., observations=CFG.observations)`` instead of per task.
    observations: ObservationConfig = Field(default_factory=ObservationConfig)

    # Per-run event handler. Set at the start of run() and cleared in finally.
    # Re-entrant run() calls on the same instance are not supported — they
    # share self.queue and self.session_service already.
    _on_event: TaskRunnerEventHandler | None = PrivateAttr(default=None)

    # ── Public API ────────────────────────────────────────────────────────

    def add_variable(self, name: str, value: str) -> None:
        self.variables[name] = value

    def add_task(
        self,
        name: str,
        *,
        worker_builder: WorkerBuilder,
        version: str | None = None,
        ref: str | None = None,
        params: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        artifact_key: str | None = None,
        skills: list[str] | None = None,
        iterations: int | None = None,
        max_attempts: int | None = None,
        max_steps: int = 15,
        namespace: str | None = None,
        model: LiteLlm | None = None,
        observations: ObservationConfig | None = None,
    ) -> str:
        """Queue a task invocation.

        ``version`` pins a specific template version (must be declared in the
        task's manifest at ``contractor/tasks/<name>.yml``). When omitted, the
        manifest's ``active`` version is used.

        ``artifact_key`` overrides the key under which the task's
        ``result``/``summary``/``records`` artifacts are published (default:
        the template key). Workflows that queue multiple tasks from the same
        template must pass a unique, stable key per task — derived from the
        finding/operation identifier, not queue position — so siblings don't
        overwrite each other's artifacts and ``--resume`` validation stays
        per-task.
        """
        template = self._ensure_template(name, version=version)
        if artifact_key is not None:
            artifact_key = validate_artifact_key(artifact_key)
        task_ref = ref or f"{name}:{len(self.queue)}"
        self._assert_unique_ref(task_ref)

        eff_iterations, eff_max_attempts = self._resolve_retry_params(
            template, iterations, max_attempts
        )

        item = TaskInvocation(
            id=uuid4().hex,
            ref=task_ref,
            template_key=template.key,
            template_version=template.version,
            worker_builder=worker_builder,
            params=params or {},
            artifacts=list(
                artifacts if artifacts is not None else template.default_artifacts
            ),
            artifact_key=artifact_key,
            skills=list(skills if skills is not None else template.default_skills),
            iterations=eff_iterations,
            max_attempts=eff_max_attempts,
            max_steps=max_steps,
            namespace=namespace,
            model=model,
            observations=observations,
        )
        self.queue.append(item)
        return item.id

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: TaskRunnerEventHandler | None = None,
    ) -> list[TaskResult]:
        self._on_event = on_event
        checkpoint = self._load_checkpoint()

        try:
            results: list[TaskResult] = []
            total_tasks = len(self.queue)

            await self._emit(
                EventType.RUN_STARTED,
                task_name="__runner__",
                task_id=-1,
                total_tasks=total_tasks,
                completed_tasks=0,
                user_id=user_id,
                prompt_versions=all_active_prompt_versions(),
                observations=self.observations.as_tag(),
                task_invocations=[
                    {
                        "ref": item.ref,
                        "template_key": item.template_key,
                        "template_version": item.template_version,
                        "observations": (
                            item.observations or self.observations
                        ).as_tag(),
                    }
                    for item in self.queue
                ],
            )

            for task_id, item in enumerate(self.queue):
                restored = await self._try_restore_from_checkpoint(
                    checkpoint, item, task_id, user_id, total_tasks,
                )
                if restored is not None:
                    results.append(restored)
                    continue

                result = await self._run_task_with_retries(
                    item=item,
                    task_id=task_id,
                    user_id=user_id,
                    total_tasks=total_tasks,
                )

                results.append(result)
                self._save_checkpoint(checkpoint, item, result, task_id)

                await self._emit(
                    EventType.GLOBAL_TASK_FINISHED,
                    task_name=item.ref,
                    task_id=task_id,
                    template_key=item.template_key,
                    template_version=item.template_version,
                    session_id=result.session_id,
                    status=result.status,
                    result=result.result,
                    summary=result.summary,
                    total_tasks=total_tasks,
                    completed_tasks=task_id + 1,
                )

            await self._emit(
                EventType.RUN_FINISHED,
                task_name="__runner__",
                task_id=-1,
                total_tasks=total_tasks,
                completed_tasks=len(results),
                ok=True,
            )
            return results
        except BaseException:
            await self._emit(
                EventType.RUN_FINISHED,
                task_name="__runner__",
                task_id=-1,
                total_tasks=total_tasks,
                completed_tasks=len(results),
                ok=False,
            )
            raise
        finally:
            self._on_event = None
            # Reliable per-run teardown of any code-exec sandbox containers.
            # The ADK after_run_callback does not fire in the TaskRunner +
            # AgentTool nesting, so we sweep here, where run() always completes.
            try:
                _teardown_sandboxes()
            except Exception:
                logger.exception("sandbox teardown failed")

    # ── Template & validation helpers ─────────────────────────────────────

    def _ensure_template(
        self, key: str, version: str | None = None
    ) -> TaskTemplate:
        # Resolve once to determine the concrete version so the cache key is
        # stable even when callers pass ``version=None``.
        template = TaskTemplate.load(key, version=version)
        cache_key = (template.key, template.version)
        if cache_key not in self.templates:
            self.templates[cache_key] = template
        return self.templates[cache_key]

    def _assert_unique_ref(self, ref: str) -> None:
        if any(item.ref == ref for item in self.queue):
            raise ValueError(f"Queued task ref '{ref}' already exists")

    # ── Checkpoint helpers ──────────────────────────────────────────────

    def _load_checkpoint(self) -> Checkpoint | None:
        if self.checkpoint_path is None:
            return None
        return Checkpoint.load(self.checkpoint_path) or Checkpoint(workflow=self.name)

    def _save_checkpoint(
        self,
        checkpoint: Checkpoint | None,
        item: TaskInvocation,
        result: TaskResult,
        task_id: int,
    ) -> None:
        if checkpoint is None or self.checkpoint_path is None:
            return
        checkpoint.mark_done(
            CheckpointEntry(
                task_id=task_id,
                ref=item.ref,
                template_key=item.template_key,
                template_version=item.template_version,
                published_artifacts=dict(result.published_artifacts),
            )
        )
        checkpoint.save(self.checkpoint_path)

    async def _try_restore_from_checkpoint(
        self,
        checkpoint: Checkpoint | None,
        item: TaskInvocation,
        task_id: int,
        user_id: str,
        total_tasks: int,
    ) -> TaskResult | None:
        if checkpoint is None:
            return None
        entry = checkpoint.get(item.ref)
        if entry is None:
            return None

        # Validate against the invocation's *own* publish key, not whatever
        # the entry recorded — with shared template keys a sibling task's
        # artifacts would otherwise "validate" this task's restore.
        published_artifacts = artifact_names_for_key(item.effective_artifact_key)
        for artifact_name in published_artifacts.values():
            part = await self.artifact_service.load_artifact(
                app_name=self.name,
                user_id=user_id,
                session_id=None,
                filename=artifact_name,
            )
            if part is None:
                logger.info(
                    "checkpoint entry %s missing artifact %s — re-running",
                    item.ref,
                    artifact_name,
                )
                return None

        template = self.templates.get(item.template_cache_key)
        if template is None:
            template = self._ensure_template(
                item.template_key, version=item.template_version,
            )

        await self._emit(
            EventType.TASK_STARTED,
            task_name=item.ref,
            task_id=task_id,
            template_key=item.template_key,
            template_version=item.template_version,
            task_title=template.title,
            iterations=item.iterations,
            max_attempts=item.max_attempts,
            params=item.params,
            artifacts=item.artifacts,
            published_artifacts=published_artifacts,
            total_tasks=total_tasks,
            completed_tasks=task_id,
            restored=True,
        )

        result = TaskResult(
            invocation_id=item.id,
            task_ref=item.ref,
            task_key=item.template_key,
            task_title=template.title,
            template_key=item.template_key,
            task_id=task_id,
            session_id="",
            final_response="",
            state={},
            carry_state={},
            status=TaskStatus.DONE,
            result="(restored from checkpoint)",
            summary="",
            records=[],
            params=copy.deepcopy(item.params),
            input_artifacts={},
            published_artifacts=published_artifacts,
        )

        await self._emit(
            EventType.TASK_FINISHED,
            task_name=item.ref,
            task_id=task_id,
            template_key=item.template_key,
            template_version=item.template_version,
            session_id="",
            status=TaskStatus.DONE,
            result=result.result,
            summary="",
            records=[],
            published_artifacts=published_artifacts,
            total_tasks=total_tasks,
            completed_tasks=task_id,
            restored=True,
        )

        await self._emit(
            EventType.GLOBAL_TASK_FINISHED,
            task_name=item.ref,
            task_id=task_id,
            template_key=item.template_key,
            template_version=item.template_version,
            session_id="",
            status=TaskStatus.DONE,
            result=result.result,
            summary="",
            total_tasks=total_tasks,
            completed_tasks=task_id + 1,
            restored=True,
        )

        logger.info("restored task %s from checkpoint", item.ref)
        return result

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
        artifacts: dict[str, str],
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
        obs = item.observations or self.observations
        worker = item.worker_builder(
            namespace=ns,
            _format=task.format,
            **self._coverage_gap_kwargs(item.worker_builder, obs),
        )
        return build_planning_agent(
            _format="xml",
            name=item.ref,
            namespace=ns,
            worker=worker,
            model=item.effective_model(self.default_model),
            max_steps=item.max_steps,
            observations=obs,
        )

    @staticmethod
    def _coverage_gap_kwargs(
        worker_builder: WorkerBuilder, obs: ObservationConfig
    ) -> dict[str, bool]:
        """Extra ``worker_builder`` kwargs that turn on the in-scope walk.

        The coverage-gap observation needs the worker's fs tools to capture the
        full in-scope file set (``capture_in_scope=True``), which is otherwise
        off (zero traversal). We opt in *only* when ``track_coverage_gap`` is
        enabled **and** the builder actually accepts the kwarg — so the default
        (feature off) path passes nothing and stays byte-identical, and builders
        without fs tools (oas_linter, http, …) are never handed an unknown arg.
        """
        if not obs.track_coverage_gap:
            return {}
        try:
            params = inspect.signature(worker_builder).parameters
        except (TypeError, ValueError):
            return {}
        if "capture_in_scope" not in params:
            return {}
        return {"capture_in_scope": True}

    # ── Event emission ────────────────────────────────────────────────────

    async def _emit(
        self,
        type: EventType | str,
        *,
        task_name: str,
        task_id: int,
        **payload: Any,
    ) -> None:
        if self._on_event is None:
            return
        await self._on_event(
            TaskRunnerEvent(
                type=type, task_name=task_name, task_id=task_id, payload=payload
            )
        )

    # ── Session management ────────────────────────────────────────────────

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
        carry_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **copy.deepcopy(carry_state),
            **build_active_state(task_id=task_id, task=task),
        }

    def _is_task_completed(self, task_id: int, state: dict[str, Any]) -> bool:
        return state.get(TaskScopedKeys(task_id).status) == TaskStatus.DONE

    # ── Artifact I/O ──────────────────────────────────────────────────────

    async def _load_artifact_text(
        self, user_id: str, artifact_ref: str
    ) -> str | None:
        """Load an artifact's text, or ``None`` when it was never published."""
        part = await self.artifact_service.load_artifact(
            app_name=self.name,
            user_id=user_id,
            session_id=None,
            filename=artifact_ref,
        )
        if part is None:
            return None
        return _decode_part_text(part)

    async def _load_artifacts(
        self, user_id: str, artifact_refs: list[str], *, task_ref: str | None = None
    ) -> dict[str, str]:
        texts: dict[str, str] = {}
        for ref in artifact_refs:
            text = await self._load_artifact_text(user_id=user_id, artifact_ref=ref)
            if text is None:
                # Keep the empty-string substitution (seeds may legitimately
                # come from the persistent store), but never silently: a typo'd
                # ref or a never-published upstream looks identical otherwise.
                logger.warning(
                    "Task '%s' declares input artifact '%s' but it is not "
                    "present in the artifact store — substituting empty text",
                    task_ref or "<unknown>",
                    ref,
                )
                text = ""
            texts[ref] = text
        return texts

    async def _publish_task_artifacts(
        self,
        user_id: str,
        key: str,
        result: TaskResult,
    ) -> None:
        await save_result_artifacts(
            artifact_service=self.artifact_service,
            app_name=self.name,
            user_id=user_id,
            key=key,
            result=result.result or "",
            summary=result.summary or "",
            records=result.records or [],
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

    async def _inject_skills(
        self, user_id: str, namespace: str, skills: list[str]
    ) -> None:
        await inject_skills(
            skills,
            namespace=namespace,
            artifact_service=self.artifact_service,
            app_name=self.name,
            user_id=user_id,
        )

    def _describe_artifact(self, name: str) -> str:
        """Derive a human-readable description for a loaded artifact."""
        if "/" in name:
            template_key = name.split("/", 1)[0]
            # Templates are cached under (key, version); any cached entry
            # for this key is fine for description purposes.
            for (cached_key, _), template in self.templates.items():
                if cached_key == template_key:
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
        carry_state = copy.deepcopy(final_state)
        keys = TaskScopedKeys(task_id)

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
            result=final_state.get(keys.result, ""),
            summary=final_state.get(keys.summary, ""),
            records=final_state.get(keys.pool, []),
            status=final_state.get(keys.status),
            params=copy.deepcopy(item.params),
            input_artifacts=copy.deepcopy(input_artifacts),
            published_artifacts=artifact_names_for_key(item.effective_artifact_key),
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
    ) -> TaskResult:
        agent = self._spawn_planning_agent(item, rendered_task)
        namespace = item.effective_namespace(self.name)
        await self._inject_skills(
            user_id=user_id,
            namespace=namespace,
            skills=item.skills,
        )
        await self._inject_artifacts(
            user_id=user_id,
            namespace=namespace,
            input_artifacts=input_artifacts,
        )

        session_id = uuid4().hex
        initial_state = self._build_task_initial_state(
            task_id=task_id,
            task=rendered_task,
            carry_state=carry_state,
        )

        await self._emit(
            EventType.ITERATION_STARTED,
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            objective=rendered_task.objective,
            initial_state=initial_state,
            template_key=item.template_key,
            template_version=item.template_version,
        )

        runner = Runner(
            agent=agent,
            app_name=self.name,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
            plugins=self._build_plugins(item, task_id, iteration, session_id),
        )

        # Pre-create the session with initial state. ADK 2.0's node runner
        # path (chat-mode agents) drops the state_delta kwarg passed to
        # run_async, so we must seed the session state directly.
        await self.session_service.create_session(
            app_name=self.name,
            user_id=user_id,
            state=initial_state,
            session_id=session_id,
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=rendered_task._format_task())],
        )

        final_text = await self._consume_events(
            runner=runner,
            user_id=user_id,
            session_id=session_id,
            message=message,
            item=item,
            task_id=task_id,
            iteration=iteration,
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
            EventType.ITERATION_FINISHED,
            task_name=item.ref,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            result=result,
            template_key=item.template_key,
            template_version=item.template_version,
        )
        return result

    def _build_plugins(
        self,
        item: TaskInvocation,
        task_id: int,
        iteration: int,
        session_id: str,
    ) -> list:
        return [
            AdkTracePlugin(
                task_name=item.ref,
                task_id=task_id,
                iteration=iteration,
                session_id=session_id,
                emit=self._emit,
            ),
            AdkMetricsPlugin(
                task_name=item.ref,
                task_id=task_id,
                iteration=iteration,
                session_id=session_id,
                emit=self._emit,
            ),
            SandboxCleanupPlugin(),
        ]

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
                EventType.FINAL_TEXT,
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
        total_tasks: int,
    ) -> TaskResult:
        template = self.templates[item.template_cache_key]
        input_artifacts = await self._load_artifacts(
            user_id, item.artifacts, task_ref=item.ref
        )
        rendered_task = self._render_task(
            template,
            item.params,
            input_artifacts,
        )

        # Every per-task event carries the same six scoping fields. Bind them
        # once so each call site lists only its event-specific payload.
        async def emit(event_type: EventType, **extra: Any) -> None:
            await self._emit(
                event_type,
                task_name=item.ref,
                task_id=task_id,
                template_key=item.template_key,
                template_version=item.template_version,
                total_tasks=total_tasks,
                completed_tasks=task_id,
                **extra,
            )

        await emit(
            EventType.TASK_STARTED,
            task_title=template.title,
            iterations=item.iterations,
            max_attempts=item.max_attempts,
            params=item.params,
            artifacts=item.artifacts,
            published_artifacts=artifact_names_for_key(item.effective_artifact_key),
            observations=(item.observations or self.observations).as_tag(),
        )

        carry_state: dict[str, Any] = {}
        last_result: TaskResult | None = None
        last_exc: Exception | None = None
        successful_runs = 0

        for iteration in range(1, item.max_attempts + 1):
            try:
                result = await self._run_single_iteration(
                    item=item,
                    rendered_task=rendered_task,
                    input_artifacts=input_artifacts,
                    task_id=task_id,
                    user_id=user_id,
                    carry_state=carry_state,
                    iteration=iteration,
                )
            except asyncio.CancelledError:
                # Cancellation is not a task failure — let it unwind the run.
                raise
            except Exception as exc:
                # Transient LLM/network/tooling errors must consume an attempt
                # instead of aborting the whole multi-task workflow (documented
                # invariant: failures keep retrying until max_attempts).
                last_exc = exc
                logger.exception(
                    "Task '%s' iteration %d/%d raised %s; "
                    "counting as a failed attempt",
                    item.ref,
                    iteration,
                    item.max_attempts,
                    type(exc).__name__,
                )
                await emit(
                    EventType.ITERATION_RESULT,
                    iteration=iteration,
                    session_id=None,
                    status=None,
                    result=None,
                    summary=None,
                    completed=False,
                    iterations_required=item.iterations,
                    max_attempts=item.max_attempts,
                    successful_runs=successful_runs,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                continue

            last_result = result
            completed = self._is_task_completed(task_id, result.state)

            next_successful_runs = successful_runs + (1 if completed else 0)

            await emit(
                EventType.ITERATION_RESULT,
                iteration=iteration,
                session_id=result.session_id,
                status=result.status,
                result=result.result,
                summary=result.summary,
                completed=completed,
                iterations_required=item.iterations,
                max_attempts=item.max_attempts,
                successful_runs=next_successful_runs,
            )

            if completed:
                successful_runs = next_successful_runs
                await self._publish_task_artifacts(
                    user_id, item.effective_artifact_key, result
                )

                if successful_runs >= item.iterations:
                    await emit(
                        EventType.TASK_FINISHED,
                        session_id=result.session_id,
                        status=result.status,
                        result=result.result,
                        summary=result.summary,
                        records=result.records,
                        published_artifacts=result.published_artifacts,
                    )
                    return result

            carry_state = result.carry_state

        await emit(
            EventType.TASK_FAILED,
            max_attempts=item.max_attempts,
            last_result=last_result,
            last_error=str(last_exc) if last_exc is not None else None,
        )

        raise TaskNotCompletedError(
            ref=item.ref,
            iterations=item.iterations,
            max_attempts=item.max_attempts,
            last_error=str(last_exc) if last_exc is not None else None,
        ) from last_exc
