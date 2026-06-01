"""Typed loader for per-workflow tuning configuration.

Each workflow folder carries a sibling ``config.yaml`` holding its tunable
shape — agent token budgets, per-task retry/iteration/step counts, and
per-agent tool options — instead of literals scattered through the workflow
module. A workflow loads its config with ``WorkflowConfig.load(__file__)`` and
reads:

    CFG = WorkflowConfig.load(__file__)
    CFG.budgets.scan_max_tokens        # int — summarization-trigger budget
    CFG.tasks.scan.as_kwargs()         # splat into TaskRunner.add_task(...)
    CFG.agent("codereview_agent")       # AgentToolConfig for that agent

This is deliberately *not* in global ``Settings``: these are per-workflow shape
decisions, not environment config. Global/tool defaults and LLM sampling live
in ``contractor.utils.settings``.

``*_max_tokens`` values are the summarization-trigger budget passed to
``build_worker`` (context retained before compression), not a generation cap.
``TaskBudget`` mirrors ``TaskRunner.add_task`` semantics: ``iterations``
successful runs required, retried up to ``max_attempts``, each attempt capped
at ``max_steps`` planner subtasks.

``AgentToolConfig`` carries the per-agent tool knobs threaded into the
``build_<agent>`` factories: ``output_format`` (the ``_format`` knob shared by
the fs/memory/openapi/report tool formatters) and ``with_graph_tools`` (attach
the trailmark call-graph tools). Workflows read these via ``CFG.agent(name)``,
which falls back to an all-default ``AgentToolConfig`` when the agent is not
declared in ``config.yaml`` — so behaviour is unchanged unless tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml

_CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class TaskBudget:
    """Retry/iteration/step budget for a single ``add_task`` call."""

    iterations: int = 1
    max_attempts: int = 1
    max_steps: int = 15

    def as_kwargs(self) -> dict[str, int]:
        """Splat into ``TaskRunner.add_task(..., **budget.as_kwargs())``."""
        return {
            "iterations": self.iterations,
            "max_attempts": self.max_attempts,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class AgentToolConfig:
    """Per-agent tool options threaded into a ``build_<agent>`` factory.

    ``output_format`` is the shared ``_format`` knob (json / xml / yaml /
    markdown — fs/openapi formatters fall back to json for unsupported
    renderers). ``with_graph_tools`` attaches the trailmark call-graph tools
    (only honoured by factories that accept it). ``with_code_exec`` attaches
    the podman-backed ``run_python`` / ``execute_bash`` sandbox tools (exploit
    agents only).
    """

    output_format: str = "json"
    with_graph_tools: bool = False
    with_code_exec: bool = False


class WorkflowConfig:
    """Parsed ``config.yaml`` for one workflow.

    ``budgets`` is a namespace of scalar knobs (token budgets, ``max_steps``,
    ``max_concurrency``, …). ``tasks`` is a namespace of :class:`TaskBudget`
    objects keyed by task name. ``agents`` is a namespace of
    :class:`AgentToolConfig` keyed by agent name. Call sites read as
    ``cfg.budgets.scan_max_tokens`` / ``cfg.tasks.scan.as_kwargs()`` /
    ``cfg.agent("swe_agent").with_graph_tools``.
    """

    def __init__(
        self,
        *,
        budgets: dict[str, int] | None = None,
        tasks: dict[str, TaskBudget] | None = None,
        agents: dict[str, AgentToolConfig] | None = None,
    ) -> None:
        self.budgets = SimpleNamespace(**(budgets or {}))
        self.tasks = SimpleNamespace(**(tasks or {}))
        self.agents = SimpleNamespace(**(agents or {}))

    def agent(self, name: str) -> AgentToolConfig:
        """Return the :class:`AgentToolConfig` for ``name``.

        Falls back to an all-default ``AgentToolConfig`` when the agent is not
        declared in ``config.yaml``, so workflows can read tool options
        uniformly without each YAML having to enumerate every agent.
        """
        return getattr(self.agents, name, AgentToolConfig())

    @classmethod
    def load(cls, anchor: str | Path) -> "WorkflowConfig":
        """Load the ``config.yaml`` sibling of ``anchor``.

        ``anchor`` is normally a workflow module's ``__file__``; the config is
        resolved next to it. A directory or a direct path to a ``config.yaml``
        also works.
        """
        path = Path(anchor)
        if path.is_file() and path.name == _CONFIG_FILENAME:
            yaml_path = path
        else:
            base = path if path.is_dir() else path.parent
            yaml_path = base / _CONFIG_FILENAME
        if not yaml_path.is_file():
            raise FileNotFoundError(f"workflow config not found: {yaml_path}")

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        raw_budgets = data.get("budgets") or {}
        raw_tasks = data.get("tasks") or {}
        raw_agents = data.get("agents") or {}
        tasks = {name: TaskBudget(**spec) for name, spec in raw_tasks.items()}
        agents = {
            name: AgentToolConfig(**(spec or {}))
            for name, spec in raw_agents.items()
        }
        return cls(budgets=raw_budgets, tasks=tasks, agents=agents)


__all__ = ["TaskBudget", "AgentToolConfig", "WorkflowConfig"]
