from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from google.adk.models import LiteLlm

from contractor.agents.http_agent.agent import build_http_agent
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.planning_agent.agent import build_planning_agent
from contractor.agents.router_agent.agent import build_router_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.agent_runner import AgentRunner
from contractor.runners.models import (
    GLOBAL_TASK_ID_KEY,
    TaskRunnerEventHandler,
    TaskScopedKeys,
    TaskStatus,
)
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.tools.fs import RootedLocalFileSystem

from cli.pipelines import Pipeline


ROUTER_NAMESPACE = "router"
ROUTER_TASK_NAME = "router"
ROUTER_TASK_ID = 0
ROUTER_MAX_STEPS = 20
ROUTER_MAX_TOKENS = 120_000


class RouterPipeline(Pipeline):
    """Prompt-driven pipeline: planner → router → specialised sub-agent.

    The planner decomposes the user's prompt into subtasks. For each subtask
    the planner invokes the router (its worker), which dispatches to one of
    the available sub-agents (swe, oas_builder, oas_linter, trace).
    """

    async def _run_impl(
        self,
        *,
        user_id: str,
        on_event: Optional[TaskRunnerEventHandler],
    ) -> Any:
        ctx = self.ctx
        prompt = (ctx.prompt or "").strip()
        if not prompt:
            raise ValueError("RouterPipeline requires ctx.prompt to be set")

        llm = LiteLlm(model=ctx.model)
        fs = RootedLocalFileSystem(root_path=ctx.project_path)

        sub_agents = [
            build_swe_agent(
                name="swe_agent",
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=ROUTER_MAX_TOKENS,
            ),
            build_oas_builder_agent(
                name="oas_builder",
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=ROUTER_MAX_TOKENS,
            ),
            build_oas_linter_agent(
                name="oas_linter",
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=ROUTER_MAX_TOKENS,
            ),
            build_trace_agent(
                name="trace_agent",
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=ROUTER_MAX_TOKENS,
                enable_vuln_reporting=True,
            ),
            build_http_agent(
                name="http_agent",
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=ROUTER_MAX_TOKENS,
            ),
        ]

        router = build_router_agent(
            name="router",
            namespace=ROUTER_NAMESPACE,
            sub_agents=sub_agents,
            model=llm,
        )

        planner = build_planning_agent(
            name=ROUTER_TASK_NAME,
            namespace=ROUTER_NAMESPACE,
            worker=router,
            model=llm,
            max_steps=ROUTER_MAX_STEPS,
        )

        runner = AgentRunner(
            name=ctx.app_name,
            artifact_service=ctx.artifact_service,
        )

        keys = TaskScopedKeys(ROUTER_TASK_ID)
        initial_state: dict[str, Any] = {
            GLOBAL_TASK_ID_KEY: ROUTER_TASK_ID,
            keys.objective: prompt,
            keys.status: TaskStatus.RUNNING,
            keys.current: None,
            keys.result: "",
            keys.summary: "",
            keys.pool: [],
        }

        session_id = uuid4().hex
        plugin_kwargs = dict(
            task_name=ROUTER_TASK_NAME,
            task_id=ROUTER_TASK_ID,
            iteration=1,
            session_id=session_id,
            emit=runner._emit,
        )
        plugins = [
            AdkTracePlugin(**plugin_kwargs),
            AdkMetricsPlugin(**plugin_kwargs),
        ]

        return await runner.run(
            agent=planner,
            message=prompt,
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
            plugins=plugins,
            on_event=on_event,
            event_name=ROUTER_TASK_NAME,
        )
