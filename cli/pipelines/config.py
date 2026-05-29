"""Per-pipeline tuning configuration.

Pipeline-level budgets (agent token budgets + per-task retry/iteration/step
counts) live here as frozen dataclasses instead of scattered module-level
literals in each ``cli/pipelines/<mode>.py``. Each pipeline references its
config singleton (e.g. ``VULN_SCAN``); to tune a pipeline, change it here.

This is deliberately *not* in global ``Settings``: these are per-pipeline
shape decisions, not environment config. Global/tool defaults and LLM
sampling live in ``contractor.utils.settings``.

``max_tokens`` here is the summarization-trigger budget passed to
``build_worker`` (context retained before compression), not a generation cap.
``TaskBudget`` mirrors ``TaskRunner.add_task`` semantics:
``iterations`` successful runs required, retried up to ``max_attempts``,
each attempt capped at ``max_steps`` planner subtasks.
"""

from __future__ import annotations

from dataclasses import dataclass


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


# ── shared preamble budgets (project/dependency discovery tasks) ─────────
_PREAMBLE_2_20 = TaskBudget(iterations=1, max_attempts=2, max_steps=20)
_PREAMBLE_3_20 = TaskBudget(iterations=1, max_attempts=3, max_steps=20)


@dataclass(frozen=True)
class ExploitabilityConfig:
    max_tokens: int = 80_000
    assess: TaskBudget = TaskBudget(1, 2, 25)


@dataclass(frozen=True)
class LikeC4BuildingConfig:
    swe_max_tokens: int = 100_000
    builder_max_tokens: int = 120_000
    dependency_information: TaskBudget = _PREAMBLE_3_20
    project_information: TaskBudget = _PREAMBLE_3_20
    likec4_build: TaskBudget = TaskBudget(3, 6, 20)
    likec4_validate: TaskBudget = TaskBudget(1, 2, 20)


@dataclass(frozen=True)
class OasBuildingConfig:
    swe_max_tokens: int = 100_000
    builder_max_tokens: int = 100_000
    validator_max_tokens: int = 100_000
    dependency_information: TaskBudget = _PREAMBLE_2_20
    project_information: TaskBudget = _PREAMBLE_2_20
    oas_update: TaskBudget = TaskBudget(2, 4, 20)
    oas_validate: TaskBudget = TaskBudget(1, 1, 20)


@dataclass(frozen=True)
class OasEnrichmentConfig:
    builder_max_tokens: int = 120_000
    validator_max_tokens: int = 120_000
    oas_enrich: TaskBudget = TaskBudget(3, 6, 30)
    oas_validate: TaskBudget = TaskBudget(2, 2, 20)


@dataclass(frozen=True)
class RouterConfig:
    max_tokens: int = 120_000
    max_steps: int = 20


@dataclass(frozen=True)
class TraceAnnotationConfig:
    max_tokens: int = 80_000
    annotate: TaskBudget = TaskBudget(1, 3, 20)


@dataclass(frozen=True)
class TraceAnnotationDirectConfig:
    max_tokens: int = 100_000


@dataclass(frozen=True)
class TraceGraphConfig:
    max_tokens: int = 100_000


@dataclass(frozen=True)
class TraceGraphPathParConfig:
    max_tokens: int = 100_000
    max_concurrency: int = 3


@dataclass(frozen=True)
class TraceVerifyConfig:
    max_tokens: int = 80_000
    verify: TaskBudget = TaskBudget(1, 2, 20)


@dataclass(frozen=True)
class VulnAssessConfig:
    swe_max_tokens: int = 100_000
    builder_max_tokens: int = 100_000
    validator_max_tokens: int = 100_000
    dependency_information: TaskBudget = _PREAMBLE_2_20
    project_information: TaskBudget = _PREAMBLE_2_20
    oas_update: TaskBudget = TaskBudget(2, 4, 20)
    oas_validate: TaskBudget = TaskBudget(1, 1, 20)


@dataclass(frozen=True)
class VulnScanConfig:
    scan_max_tokens: int = 80_000
    scan: TaskBudget = TaskBudget(1, 2, 75)


@dataclass(frozen=True)
class VulnScanFastConfig:
    scan_max_tokens: int = 80_000
    swe_max_tokens: int = 100_000
    dependency_information: TaskBudget = _PREAMBLE_2_20
    project_information: TaskBudget = _PREAMBLE_2_20
    scan: TaskBudget = TaskBudget(1, 2, 50)


@dataclass(frozen=True)
class VulnScanTraceConfig:
    scan_max_tokens: int = 80_000
    trace_max_tokens: int = 80_000
    scan: TaskBudget = TaskBudget(1, 2, 75)
    trace: TaskBudget = TaskBudget(1, 1, 30)


# ── singletons referenced by each pipeline ───────────────────────────────
EXPLOITABILITY = ExploitabilityConfig()
LIKEC4_BUILDING = LikeC4BuildingConfig()
OAS_BUILDING = OasBuildingConfig()
OAS_ENRICHMENT = OasEnrichmentConfig()
ROUTER = RouterConfig()
TRACE_ANNOTATION = TraceAnnotationConfig()
TRACE_ANNOTATION_DIRECT = TraceAnnotationDirectConfig()
TRACE_GRAPH = TraceGraphConfig()
TRACE_GRAPH_PATHPAR = TraceGraphPathParConfig()
TRACE_VERIFY = TraceVerifyConfig()
VULN_ASSESS = VulnAssessConfig()
VULN_SCAN = VulnScanConfig()
VULN_SCAN_FAST = VulnScanFastConfig()
VULN_SCAN_TRACE = VulnScanTraceConfig()
