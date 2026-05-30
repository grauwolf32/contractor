from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from contractor.agents.http_agent.agent import build_http_agent
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.planning_agent.agent import build_planning_agent
from contractor.agents.router_agent.agent import build_router_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.agent_runner import AgentRunner
from contractor.runners.models import (GLOBAL_TASK_ID_KEY,
                                       TaskRunnerEventHandler, TaskScopedKeys,
                                       TaskStatus)
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.runners.skills import inject_skills
from contractor.tools.tasks.models import Subtask, SubtaskExecutionResult
from contractor.utils.settings import build_model
from contractor.workflows import Workflow
from contractor.workflows.config import WorkflowConfig

CFG = WorkflowConfig.load(__file__)

ROUTER_NAMESPACE = "router"
ROUTER_TASK_NAME = "router"
ROUTER_TASK_ID = 0


class RouterWorkflow(Workflow):
    """Prompt-driven workflow: planner → router → specialised sub-agent.

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
            raise ValueError("RouterWorkflow requires ctx.prompt to be set")

        llm = build_model(ctx.model, ctx.timeout)
        fs = ctx.fs

        sub_agents = [
            build_swe_agent(
                name="swe_agent",
                _format=CFG.agent("swe_agent").output_format,
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=CFG.budgets.max_tokens,
            ),
            build_oas_builder_agent(
                name="oas_builder",
                _format=CFG.agent("oas_builder").output_format,
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=CFG.budgets.max_tokens,
            ),
            build_oas_linter_agent(
                name="oas_linter",
                _format=CFG.agent("oas_linter").output_format,
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=CFG.budgets.max_tokens,
            ),
            build_trace_agent(
                name="trace_agent",
                _format=CFG.agent("trace_agent").output_format,
                fs=fs,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=CFG.budgets.max_tokens,
                enable_vuln_reporting=True,
                with_graph_tools=CFG.agent("trace_agent").with_graph_tools,
            ),
            build_http_agent(
                name="http_agent",
                _format=CFG.agent("http_agent").output_format,
                namespace=ROUTER_NAMESPACE,
                model=llm,
                max_tokens=CFG.budgets.max_tokens,
            ),
        ]

        router = build_router_agent(
            name="router",
            _format=CFG.agent("router").output_format,
            namespace=ROUTER_NAMESPACE,
            sub_agents=sub_agents,
            model=llm,
        )

        # The router already has its own dispatch protocol (prompts/v1.md)
        # that produces SubtaskExecutionResult. Set schemas so ADK's AgentTool
        # parses Subtask input correctly, but skip instrument_worker to avoid
        # appending conflicting generic worker instructions.
        router.input_schema = Subtask
        router.output_schema = SubtaskExecutionResult

        planner = build_planning_agent(
            name=ROUTER_TASK_NAME,
            namespace=ROUTER_NAMESPACE,
            worker=router,
            model=llm,
            max_steps=CFG.budgets.max_steps,
            worker_instrumentation=False,
        )

        runner = AgentRunner(
            name=ctx.app_name,
            artifact_service=ctx.artifact_service,
        )

        await inject_skills(
            ["trace"],
            namespace=ROUTER_NAMESPACE,
            artifact_service=ctx.artifact_service,
            app_name=ctx.app_name,
            user_id=user_id,
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
        # Heterogeneous kwargs shared by both plugin constructors; typed Any so
        # the **splat checks against each plugin's individually-typed params.
        plugin_kwargs: dict[str, Any] = dict(
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
