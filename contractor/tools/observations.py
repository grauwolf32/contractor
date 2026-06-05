"""Deterministic worker-usage observations for the streamline planner.

A worker run produces hard, non-LLM facts — which tools it called (and how
often / with what error rate), which files it touched, which skill files it
read. Capturing those and surfacing them back to the planner (alongside, but
visually distinct from, the worker's self-reported ``output``/``summary``) gives
the planner ground truth to reason over and reduces hallucinated planning.

Design: **capture is always-on and cheap** — the metrics plugin and the
``skills_read`` tool write raw facts into ADK session state regardless of this
config (the writes never reach the LLM). These knobs only gate the *projection*
of those facts into the planner-visible record / tool result, so the disabled
default reproduces the pre-feature behaviour exactly. That single-point gating
is what makes the feature cleanly A/B-testable: one config object, one
consumption site (``execute_current_subtask``), no multi-layer threading.

This module is a dependency-free leaf so the plugin layer
(``runners/plugins/metrics_plugin``), the memory tools, the task tools, the
planner factory, the task runner, and the workflow config loader can all import
it downward without an import cycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

# State keys under which raw, per-subtask usage is accumulated during a worker
# run. Written inside the worker invocation; forwarded to the planner's
# ``tool_context.state`` via ADK ``AgentTool`` state-delta propagation.
WORKER_USAGE_STATE_KEY: Final[str] = "worker_usage"
SKILLS_READ_STATE_KEY: Final[str] = "skills_read"
MEMORIES_WRITTEN_STATE_KEY: Final[str] = "memories_written"
MEMORIES_READ_STATE_KEY: Final[str] = "memories_read"

# Env var carrying a JSON object that overlays the workflow's ``observations:``
# block — flip A/B arms (or ablate knobs) without editing any config.yaml.
OBSERVATIONS_ENV_VAR: Final[str] = "CONTRACTOR_EVAL_OBSERVATIONS"


@dataclass(frozen=True)
class ObservationConfig:
    """Per-planner toggles for injecting deterministic worker-usage facts.

    All-default is *disabled* — identical to the pre-feature behaviour — so a
    workflow only opts in via its ``config.yaml`` ``observations:`` block. The
    granular knobs exist to ablate the feature in A/B runs.

    Fields:
        enabled: master switch (``with_observations``).
        track_tools / track_files / track_skills: which signals to project.
        tracked_tools: ``None`` projects every tool; a tuple restricts to an
            allowlist (e.g. only ``skills_read`` / fs tools).
        include_tool_errors: project per-tool error counts, not just call
            counts (most diagnostic in the malformed path).
        malformed_only: project observations *only* for malformed/unparseable
            worker results, never for successful ones — the A/B arm that adds
            ground truth exactly where the worker output can't be trusted.
        in_record / in_result: which surfaces receive the projection — the
            persisted task record (history + summarizer) and/or the immediate
            ``observations`` field of the tool result.
    """

    enabled: bool = False
    track_tools: bool = True
    tracked_tools: tuple[str, ...] | None = None
    include_tool_errors: bool = False
    track_skills: bool = True
    track_files: bool = True
    track_memories: bool = False
    malformed_only: bool = False
    in_record: bool = True
    in_result: bool = True

    @classmethod
    def resolve(
        cls, section: Mapping[str, Any] | None, env: Mapping[str, str]
    ) -> ObservationConfig:
        """Build a config from a YAML ``observations:`` block + env overlay.

        The ``CONTRACTOR_EVAL_OBSERVATIONS`` env var (a JSON object) overlays the
        YAML section field-by-field, so an A/B harness can flip arms or ablate
        individual knobs without touching any ``config.yaml``. ``tracked_tools``
        is normalised to a tuple (or ``None``) so the result stays frozen and
        hashable. Raises ``ValueError`` on malformed input (bad JSON, wrong
        types, or unknown keys).
        """
        merged: dict[str, Any] = dict(section or {})

        raw = env.get(OBSERVATIONS_ENV_VAR)
        if raw:
            try:
                overlay = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{OBSERVATIONS_ENV_VAR} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overlay, dict):
                raise ValueError(
                    f"{OBSERVATIONS_ENV_VAR} must be a JSON object, got "
                    f"{type(overlay).__name__}"
                )
            merged.update(overlay)

        tracked = merged.get("tracked_tools")
        if tracked is not None:
            if not (isinstance(tracked, list) and all(isinstance(x, str) for x in tracked)):
                raise ValueError(
                    f"observations.tracked_tools must be a list[str] or null, "
                    f"got {tracked!r}"
                )
            merged["tracked_tools"] = tuple(tracked)

        try:
            return cls(**merged)
        except TypeError as exc:
            raise ValueError(f"invalid observations config: {exc}") from exc

    def as_tag(self) -> dict[str, Any]:
        """JSON-friendly dict identifying the A/B arm, for tagging metrics runs.

        Emitted into ``metrics.jsonl`` (RUN_STARTED / TASK_STARTED) so each run
        is self-describing — analysis can group/compare arms without out-of-band
        knowledge of which config produced a run. ``tracked_tools`` is rendered
        as a list (or ``None``) so the record round-trips through JSON.
        """
        return {
            "enabled": self.enabled,
            "track_tools": self.track_tools,
            "tracked_tools": (
                list(self.tracked_tools) if self.tracked_tools is not None else None
            ),
            "include_tool_errors": self.include_tool_errors,
            "track_skills": self.track_skills,
            "track_files": self.track_files,
            "track_memories": self.track_memories,
            "malformed_only": self.malformed_only,
            "in_record": self.in_record,
            "in_result": self.in_result,
        }


def project_usage(state: Any, cfg: ObservationConfig) -> dict[str, Any] | None:
    """Project raw per-subtask usage from session ``state`` into a compact dict.

    Returns ``None`` when observations are disabled. Otherwise returns a dict
    holding only the sections enabled by ``cfg`` — each section may be empty
    (e.g. the worker made no tool calls), which is itself signal in the
    malformed path ("produced unparseable output *and* did no work"). Callers
    decide whether to prune empties (success path) or keep them (malformed).

    Pure: reads ``state`` via ``.get`` (works for an ADK ``State`` or a plain
    dict), no mutation, no I/O — unit-testable with a literal ``state`` dict.
    """
    if not cfg.enabled:
        return None

    snapshot = state.get(WORKER_USAGE_STATE_KEY) or {}
    out: dict[str, Any] = {}

    if cfg.track_tools:
        tools = snapshot.get("tools") or {}
        if cfg.tracked_tools is not None:
            allow = set(cfg.tracked_tools)
            tools = {name: data for name, data in tools.items() if name in allow}
        out["tools"] = {
            name: (
                {"calls": int(data.get("calls", 0)), "errors": int(data.get("errors", 0))}
                if cfg.include_tool_errors
                else int(data.get("calls", 0))
            )
            for name, data in tools.items()
        }

    if cfg.track_files:
        fs = snapshot.get("fs_coverage")
        out["files"] = dict(fs) if isinstance(fs, dict) else fs

    if cfg.track_skills:
        out["skills_read"] = list(state.get(SKILLS_READ_STATE_KEY) or [])

    if cfg.track_memories:
        out["memories_written"] = list(state.get(MEMORIES_WRITTEN_STATE_KEY) or [])
        out["memories_read"] = list(state.get(MEMORIES_READ_STATE_KEY) or [])

    return out


def has_observations(usage: dict[str, Any] | None) -> bool:
    """True when ``usage`` carries at least one non-empty section.

    Used by the success path to prune empty projections (no point telling the
    planner "the worker did nothing" when it succeeded); the malformed path
    deliberately ignores this, since emptiness there is diagnostic.
    """
    if not usage:
        return False
    return any(bool(section) for section in usage.values())
