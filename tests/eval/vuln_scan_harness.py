"""Vulnerability-scan eval harness.

Runs either the baseline ``codereview_agent`` or the ``trace_agent`` (with
vuln reporting enabled) against a fixture and returns the reported findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from google.adk.models.lite_llm import LiteLlm

from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.skills import inject_skills
from contractor.utils import observability
from tests.eval.harness import AgentRun, run_agent

AgentKind = Literal["vuln_scan", "trace"]

# Maps the eval's agent_kind to the agent the harness actually builds — used as
# the eval/v1 `unit` so envelopes are labelled by the real agent under test.
UNIT_FOR_KIND: dict[str, str] = {"vuln_scan": "codereview_agent", "trace": "trace_agent"}


def _make_metrics_plugin(
    task_name: str,
    namespace: str,
) -> tuple[AdkMetricsPlugin, list[dict]]:
    events: list[dict] = []

    async def _emit(event_type: str, **payload) -> None:
        events.append({"event_type": event_type, **payload})

    plugin = AdkMetricsPlugin(
        task_name=task_name,
        task_id=0,
        iteration=1,
        session_id=namespace,
        emit=_emit,
    )
    return plugin, events


@dataclass
class VulnScanRun:
    agent_run: AgentRun
    agent_kind: AgentKind
    prompt_version: str


async def run_vuln_scan(
    *,
    fixture_root: Path,
    user_message: str,
    model: LiteLlm,
    agent_kind: AgentKind = "vuln_scan",
    namespace: str = "vuln-eval",
    timeout_s: float = 900.0,
    prompt_version: str | None = None,
    with_graph_tools: bool = False,
    artifact_dir: Path | None = None,
) -> VulnScanRun:
    from cli.fs import RootedLocalFileSystem

    base_fs = RootedLocalFileSystem(str(fixture_root))

    if agent_kind == "trace":
        from contractor.agents.trace_agent.agent import build_trace_agent
        from contractor.tools.fs import MemoryOverlayFileSystem
        from contractor.utils import load_prompt_with_version

        prompt_text, resolved_version = load_prompt_with_version(
            "trace_agent", prompt_version
        )
        overlay = MemoryOverlayFileSystem(base_fs, skip_instance_cache=True)
        agent = build_trace_agent(
            name="trace_agent",
            fs=overlay,
            namespace=namespace,
            model=model,
            max_tokens=80_000,
            with_graph_tools=with_graph_tools,
            enable_vuln_reporting=True,
            prompt=prompt_text,
        )
    else:
        from contractor.agents.codereview_agent.agent import build_codereview_agent
        from contractor.utils import load_prompt_with_version

        prompt_text, resolved_version = load_prompt_with_version(
            "codereview_agent", prompt_version
        )
        agent = build_codereview_agent(
            name="codereview_agent",
            fs=base_fs,
            namespace=namespace,
            model=model,
            max_tokens=80_000,
            with_graph_tools=with_graph_tools,
            prompt=prompt_text,
        )

    skills = ["vuln_scan"] if agent_kind == "vuln_scan" else ["trace"]

    async def _setup(artifact_service, app_name: str, user_id: str) -> None:
        await inject_skills(
            skills,
            namespace=namespace,
            artifact_service=artifact_service,
            app_name=app_name,
            user_id=user_id,
        )

    plugin, metrics_events = _make_metrics_plugin(
        task_name=f"eval.{agent_kind}",
        namespace=namespace,
    )

    with observability.run_context(
        name=f"eval.{agent_kind}",
        session_id=namespace,
        tags=["eval", f"agent:{agent_kind}", f"prompt:{agent_kind}@{resolved_version}"],
        metadata={
            "agent": agent_kind,
            "prompt_version": resolved_version,
            "namespace": namespace,
            "fixture_root": str(fixture_root),
        },
    ):
        result = await run_agent(
            agent,
            user_message=user_message,
            timeout_s=timeout_s,
            setup=_setup,
            plugins=[plugin],
            metrics_events=metrics_events,
            artifact_dir=artifact_dir,
        )

    return VulnScanRun(
        agent_run=result,
        agent_kind=agent_kind,
        prompt_version=resolved_version,
    )
