from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.genai import types
from pydantic import BaseModel, Field

from contractor.models.task import Task
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import SubtaskFormatter, task_tools

load_dotenv()


_GLOBAL_TASK_ID_KEY = "_global_task_id"
WorkerBuilder = Callable[..., LlmAgent | AgentTool]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PLANNER_MODEL = LiteLlm(
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


class TaskRunner(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Runner name")
    output_format: Literal["json", "markdown", "yaml", "xml"] = Field(default="json")

    tasks: dict[str, Task] = Field(default_factory=dict)
    queue: list[str] = Field(default_factory=list)
    task_agents: dict[str, LlmAgent] = Field(default_factory=dict)
    variables: dict[str, str] = Field(default_factory=dict)

    session_service: InMemorySessionService = Field(
        default_factory=InMemorySessionService
    )

    def add_variable(self, name: str, value: str):
        self.variables[name] = value

    def add_task(
        self,
        name: str,
        *,
        worker_builder: WorkerBuilder,
        max_iterations: int = 1,
        max_steps: int = 15,
        namespace: Optional[str] = None,
        model: Optional[LiteLlm] = None,
    ) -> None:
        task = Task.load(name, self.variables)

        task_name = name
        if task_name in self.tasks and max_iterations != getattr(
            self.tasks[task_name], "_max_iterations", 1
        ):
            task_name = f"{name}.x{max_iterations}"

        task._max_iterations = max_iterations
        self.tasks[task_name] = task
        self.queue.append(task_name)

        if namespace is None:
            namespace = self.name

        worker = worker_builder(namespace=namespace, _format=task._format)
        planner = self._spawn_planner_agent(
            task_name=task_name,
            task=task,
            worker=worker,
            max_steps=max_steps,
            namespace=namespace,
            model=model if model else PLANNER_MODEL,
        )

        self.task_agents[task_name] = planner

    def _format_task(self, task: Task) -> str:
        return (
            f"OBJECTIVE:\n{task.objective}\n\n"
            f"INSTRUCTIONS:\n{task.instructions}\n\n"
            f"OUTPUT FORMAT:\n{task.output_format}"
        )

    def _spawn_planner_agent(
        self,
        task_name: str,
        task: Task,
        worker: LlmAgent | AgentTool,
        *,
        model: LiteLlm,
        max_steps: int = 15,
        namespace: Optional[str] = None,
    ) -> LlmAgent:
        fmt = SubtaskFormatter(task._format)

        planning_tools = task_tools(
            name=task.name,
            max_tasks=max_steps,
            worker=worker,
            fmt=fmt,
            use_output_schema=False,
        )

        mem_tools = memory_tools(
            name=namespace,
            fmt=MemoryFormat(_format=task._format),
        )

        tools = [*planning_tools, *mem_tools]
        instruction = self._format_task(task)

        return LlmAgent(
            name=f"task_{task_name}_planner",
            description=f"Planner for global task {task_name}",
            instruction=instruction,
            tools=tools,
            model=model,
        )

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
        task: Task,
        carry_state: dict[str, Any],
        iteration: int,
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
        state["runner:last_task_name"] = task.name
        state["runner:active_task_name"] = task.name
        state["runner:iteration"] = iteration

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
    def _extract_function_call_from_event(
        event: Event,
    ) -> list[tuple[str, Any]]:
        """
        Best-effort extractor for ADK events that contain function/tool calls.
        Returns list of (function_name, args) pairs.
        """
        content = getattr(event, "content", None)
        if not content:
            return []

        parts = getattr(content, "parts", None) or []
        calls: list[tuple[str, Any]] = []

        for part in parts:
            function_call = getattr(part, "function_call", None)
            if function_call:
                name = getattr(function_call, "name", None) or "<unknown_function>"
                args = getattr(function_call, "args", None)
                calls.append((name, args))

        return calls

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
        final_response: str,
    ) -> bool:
        status = state.get(self._global_state_key(task_id, "status"))
        return status == "done"

    async def _run_single_iteration(
        self,
        *,
        task_name: str,
        task_id: int,
        user_id: str,
        carry_state: dict[str, Any],
        iteration: int,
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> dict[str, Any]:
        task = self.tasks[task_name]
        agent = self.task_agents[task_name]

        runner = Runner(
            agent=agent,
            app_name=self.name,
            session_service=self.session_service,
        )

        session_id = str(uuid4())
        initial_state = self._build_task_initial_state(
            task_id=task_id,
            task=task,
            carry_state=carry_state,
            iteration=iteration,
        )

        await self._ensure_session(
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
        )

        await self._emit(
            on_event,
            type="iteration_started",
            task_name=task_name,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            objective=task.objective,
            initial_state=initial_state,
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=task.objective)],
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

            await self._emit(
                on_event,
                type="adk_event",
                task_name=task_name,
                task_id=task_id,
                iteration=iteration,
                session_id=session_id,
                author=getattr(event, "author", None),
                state=state,
                event=event,
            )

            calls = self._extract_function_call_from_event(event)
            for tool_name, tool_args in calls:
                await self._emit(
                    on_event,
                    type="tool_call",
                    task_name=task_name,
                    task_id=task_id,
                    iteration=iteration,
                    session_id=session_id,
                    author=getattr(event, "author", None),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    state=state,
                )

            event_final = self._extract_final_text(event)
            if event_final:
                final_text = event_final
                await self._emit(
                    on_event,
                    type="final_text",
                    task_name=task_name,
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
            "task_name": task_name,
            "task_id": task_id,
            "session_id": session_id,
            "final_response": final_text,
            "state": final_state,
            "carry_state": next_carry_state,
            "status": final_state.get(self._global_state_key(task_id, "status")),
            "result": final_state.get(self._global_state_key(task_id, "result")),
            "summary": final_state.get(self._global_state_key(task_id, "summary")),
            "records": final_state.get(self._global_state_key(task_id, "pool"), []),
        }

        await self._emit(
            on_event,
            type="iteration_finished",
            task_name=task_name,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
            result=result,
        )

        return result

    async def _run_task_with_retries(
        self,
        *,
        task_name: str,
        task_id: int,
        user_id: str,
        carry_state: dict[str, Any],
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> dict[str, Any]:
        task = self.tasks[task_name]
        max_iterations = getattr(task, "_max_iterations", 1)

        current_carry_state = copy.deepcopy(carry_state)
        last_result: Optional[dict[str, Any]] = None

        await self._emit(
            on_event,
            type="task_started",
            task_name=task_name,
            task_id=task_id,
            max_iterations=max_iterations,
        )

        for iteration in range(1, max_iterations + 1):
            result = await self._run_single_iteration(
                task_name=task_name,
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
                final_response=result["final_response"],
            )

            await self._emit(
                on_event,
                type="iteration_result",
                task_name=task_name,
                task_id=task_id,
                iteration=iteration,
                session_id=result["session_id"],
                status=result["status"],
                result=result["result"],
                summary=result["summary"],
                completed=completed,
            )

            if completed:
                await self._emit(
                    on_event,
                    type="task_finished",
                    task_name=task_name,
                    task_id=task_id,
                    session_id=result["session_id"],
                    status=result["status"],
                    result=result["result"],
                    summary=result["summary"],
                    records=result["records"],
                )
                return result

            current_carry_state = result["carry_state"]

        await self._emit(
            on_event,
            type="task_failed",
            task_name=task_name,
            task_id=task_id,
            max_iterations=max_iterations,
            last_result=last_result,
        )

        raise RuntimeError(
            f"Task '{task_name}' was not completed after {max_iterations} iterations. "
            "Execution aborted."
        )

    async def run(
        self,
        *,
        user_id: str = "cli-user",
        on_event: Optional[TaskRunnerEventHandler] = None,
    ) -> list[dict[str, Any]]:
        """
        Run all registered global tasks sequentially.

        Each task gets its own session_id.
        State is transferred between tasks through carry_state.
        If a task is not completed after max_iterations, execution stops.
        """
        results: list[dict[str, Any]] = []
        carry_state: dict[str, Any] = {}

        await self._emit(
            on_event,
            type="run_started",
            task_name="__runner__",
            task_id=-1,
            total_tasks=len(self.queue),
            user_id=user_id,
        )

        for task_id, task_name in enumerate(self.queue):
            result = await self._run_task_with_retries(
                task_name=task_name,
                task_id=task_id,
                user_id=user_id,
                carry_state=carry_state,
                on_event=on_event,
            )
            results.append(result)
            carry_state = result["carry_state"]

            await self._emit(
                on_event,
                type="global_task_finished",
                task_name=task_name,
                task_id=task_id,
                session_id=result["session_id"],
                status=result["status"],
                result=result["result"],
                summary=result["summary"],
            )

        await self._emit(
            on_event,
            type="run_finished",
            task_name="__runner__",
            task_id=-1,
            results=results,
        )

        return results
