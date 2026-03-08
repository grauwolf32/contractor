from __future__ import annotations

from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from pydantic import BaseModel, Field

from contractor.models.task import Task
from contractor.tools.memory import MemoryFormat, memory_tools
from contractor.tools.tasks import SubtaskFormatter, task_tools


class TaskRunner(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="runner name")
    _fromat: Literal["json", "xml", "yaml"] = Field(default="json")
    tasks: dict[str, Task] = Field(default_factory=dict)
    queue: list[str] = Field(default_factory=list)
    task_agents: dict[str, LlmAgent] = Field(default_factory=dict)
    idx: list[int] = Field(default_factory=list)

    def add_task(
        self,
        name: str,
        worker: LlmAgent | AgentTool,
        *,
        repeats: int = 1,
        max_steps: int = 15,
        namespace: Optional[str] = None,
    ):
        """
        Registers a task configuration and spawns its planner agent.
        """

        task = Task.load(name)
        self.queue.append(name)

        if name in self.tasks and name in self.task_agents:
            if repeats == self.tasks[name]._repeats:
                return
            name = f"{name}.x{repeats}"

        task._repeats = repeats
        self.tasks[name] = task

        planner = self._spawn_streamline_agent(
            task_name=name,
            task=task,
            worker=worker,
            max_steps=max_steps,
            namespace=namespace,
        )

        self.task_agents[name] = planner

    def _format_task(task: Task) -> str:
        return (
            f"{task.instructions}\n\n"
            f"OBJECTIVE:\n{task.objective}\n\n"
            f"OUTPUT FORMAT:\n{task.output_format}"
        )

    def _spawn_streamline_agent(
        self,
        task_name: str,
        task: Task,
        worker: LlmAgent | AgentTool,
        *,
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

        if namespace is None:
            namespace = self.name

        mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_fromat=self._fromat))

        tools = [default_tool, *planning_tools, *mem_tools]

        instruction = self._format_task(task)

        return LlmAgent(
            name=f"task_{task_name}_planner",
            description=f"Planner for task {task_name}",
            instruction=instruction,
            tools=tools,
        )

    def _spawn_summarizer_agent(self, task: Task, namespace: str) -> LlmAgent:
        return LlmAgent(
            name=f"{task.name}_summarizer",
            description="Summarizes task results",
            instruction=(
                "Summarize execution results and produce final output "
                f"in format:\n{task.output_format}"
            ),
        )

    async def summarize(self, task_name: str) -> str:
        task = self.tasks[task_name]
