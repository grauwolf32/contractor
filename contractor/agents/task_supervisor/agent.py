from __future__ import annotations

import logging
from typing import AsyncGenerator, Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events.event import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.agent_tool import AgentTool

from contractor.models.task import Task
from contractor.tools.tasks import SubtaskFormatter, task_tools
from contractor.tools.memory import memory_tools

from pydantic import Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


_GLOBAL_TASK_ID_KEY = "_global_task_id"


def default_tool(meta: dict[str, Any]) -> dict:
    return {"error": f"tool {meta.get('func_name')} is not available!"}


class TaskSupervisor(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}

    tasks: dict[str, Task] = Field(default_factory=dict)
    task_agents: dict[str, LlmAgent] = Field(default_factory=dict)

    def __init__(self, name: str):
        super().__init__(name=name)

    def add_task(
        self,
        name: str,
        worker: LlmAgent | AgentTool,
        *,
        repeats: int = 1,
        max_steps: int = 15,
        shared_memory: bool = True,
    ):
        """
        Registers a task configuration and spawns its planner agent.
        """

        task = Task.load(name)

        if task.name in self.tasks:
            raise ValueError(f"Task {task.name} already registered")

        self.tasks[task.name] = task

        planner = self._spawn_streamline_agent(
            task_name=name,
            task=task,
            worker=worker,
            max_steps=max_steps,
            shared_memory=shared_memory,
        )

        self.task_agents[task.name] = planner

        # store repeats inside task metadata
        task._repeats = repeats

    def _spawn_streamline_agent(
        self,
        task_name: str,
        task: Task,
        worker: LlmAgent | AgentTool,
        *,
        max_steps: int = 15,
        shared_memory: bool = True,
    ) -> LlmAgent:
        fmt = SubtaskFormatter(task._format)

        planning_tools = task_tools(
            name=task.name,
            max_tasks=max_steps,
            worker=worker,
            fmt=fmt,
            use_output_schema=False,
        )

        mem_tools = memory_tools(self.name if shared_memory else task.name)

        tools = [default_tool, *planning_tools, *mem_tools]

        instruction = (
            f"{task.instructions}\n\n"
            f"OBJECTIVE:\n{task.objective}\n\n"
            f"OUTPUT FORMAT:\n{task.output_format}"
        )

        return LlmAgent(
            name=f"task_{task_name}_planner",
            description=f"Planner for task {task_name}",
            instruction=instruction,
            tools=tools,
        )

    def _spawn_summarizer_agent(self, task: Task) -> LlmAgent:
        return LlmAgent(
            name=f"{task.name}_summarizer",
            description="Summarizes task results",
            instruction=(
                "Summarize execution results and produce final output "
                f"in format:\n{task.output_format}"
            ),
        )

    async def _task_runner(
        self,
        task_name: str,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        task = self.tasks[task_name]
        agent = self.task_agents[task_name]

        repeats = task._repeats

        for iteration in range(repeats):
            logger.debug(f"Running task {task_name}, iteration {iteration}")

            # isolate state per run
            ctx.session.state[_GLOBAL_TASK_ID_KEY] = (
                ctx.session.state.get(_GLOBAL_TASK_ID_KEY, 0) + 1
            )

            async for event in agent.run_async(ctx):
                yield event

        # after all iterations — finalize
        await self._finalize_task(task, ctx)

    async def _finalize_task(
        self,
        task: Task,
        ctx: InvocationContext,
    ):
        """
        Collect task records and summarize.
        """

        pool_key = f"task::{ctx.session.state.get(_GLOBAL_TASK_ID_KEY, 0)}::pool"
        records = ctx.session.state.get(pool_key, [])
        summarizer = self._spawn_summarizer_agent(task)
        summary_prompt = "Below are task execution records:\n\n" + "\n".join(
            str(r) for r in records
        )

        await summarizer.run_async(
            args={"request": summary_prompt},
            tool_context=ctx,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        for task_name in self.tasks:
            async for event in self._task_runner(task_name, ctx):
                yield event
