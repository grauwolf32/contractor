"""Trace-agent eval harness.

`trace_agent` mutates the source filesystem by inserting `# @trace ...`
comments above relevant function definitions. We wrap the fixture's read-only
FS with `MemoryOverlayFileSystem` so the writes land in memory; after the
agent run we walk the overlay's modified files and extract the
`(file, function_name)` set of annotations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fsspec import AbstractFileSystem
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.trace_agent.agent import build_trace_agent
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.skills import inject_skills
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.utils import load_prompt_with_version, observability
from tests.eval.harness import AgentRun, run_agent


def _make_metrics_plugin(
    *,
    task_name: str,
    namespace: str,
) -> tuple[AdkMetricsPlugin, list[dict]]:
    """Build an in-process AdkMetricsPlugin whose emit appends to a list.

    Returned plugin can be passed to ``run_agent(plugins=...)``; the list
    is captured into the resulting ``AgentRun.metrics_events`` so callers
    have token usage, exec time, and arg/result sizes per tool call.
    """
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

TRACE_MARKER = re.compile(r"@trace\b")

# Per-language extractors: applied to the joined window of N lines that
# follow a `@trace` marker. The first capture group must be the function /
# method / class name. Order matters — first match wins.
_LANG_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ".py": [
        re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE),
        re.compile(r"^\s*class\s+(\w+)\s*[:(]", re.MULTILINE),
    ],
    ".ts": [
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]",
            re.MULTILINE,
        ),
        # class methods: `name(args): ReturnType {` — exclude control kws.
        re.compile(
            r"^\s*(?!if|for|while|switch|return|throw|await|new\b)"
            r"(?:public|private|protected|static|async|\s)*"
            r"(\w+)\s*[<(]",
            re.MULTILINE,
        ),
    ],
    ".js": [
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]",
            re.MULTILINE,
        ),
    ],
    ".java": [
        re.compile(
            r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"
            r"(?:public|private|protected|static|final|abstract|synchronized|\s)+"
            r"[\w<>\[\],?\s.]+\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        re.compile(r"^\s*class\s+(\w+)\b", re.MULTILINE),
    ],
    ".php": [
        # functions / methods (with optional visibility, static/final/abstract,
        # by-ref ``&``). Excludes anonymous closures.
        re.compile(
            r"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*"
            r"function\s+&?(\w+)\s*\(",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)\b",
            re.MULTILINE,
        ),
        re.compile(r"^\s*(?:trait|interface)\s+(\w+)\b", re.MULTILINE),
    ],
    ".go": [
        # func FuncName(args)
        re.compile(r"^\s*func\s+(\w+)\s*\(", re.MULTILINE),
        # method: func (r *Receiver) MethodName(args)
        re.compile(
            r"^\s*func\s+\([^)]+\)\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
    ],
}

ANNOTATION_LOOKAHEAD_LINES = 6


@dataclass(frozen=True)
class Annotation:
    file: str
    function: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.file, self.function)


def _patterns_for(path: str) -> list[re.Pattern[str]]:
    ext = os.path.splitext(path)[1].lower()
    return _LANG_PATTERNS.get(ext, [])


def _function_after_marker(
    text: str, marker_match: re.Match[str], patterns: list[re.Pattern[str]]
) -> Optional[str]:
    end = marker_match.end()
    # Advance to start of next line.
    next_nl = text.find("\n", end)
    if next_nl == -1:
        return None
    cursor = next_nl + 1

    # Window of ANNOTATION_LOOKAHEAD_LINES non-empty content lines.
    window_lines: list[str] = []
    while cursor < len(text) and len(window_lines) < ANNOTATION_LOOKAHEAD_LINES:
        nl = text.find("\n", cursor)
        line = text[cursor : nl if nl != -1 else len(text)]
        if line.strip() and not line.lstrip().startswith(("#", "//")):
            window_lines.append(line)
        if nl == -1:
            break
        cursor = nl + 1

    if not window_lines:
        return None
    window = "\n".join(window_lines)
    for pat in patterns:
        m = pat.search(window)
        if m:
            return m.group(1)
    return None


def _extract_from_text(file_path: str, text: str) -> set[Annotation]:
    patterns = _patterns_for(file_path)
    if not patterns:
        return set()
    out: set[Annotation] = set()
    for marker in TRACE_MARKER.finditer(text):
        # The marker must live in a comment line, not inside a string.
        line_start = text.rfind("\n", 0, marker.start()) + 1
        line_prefix = text[line_start : marker.start()].lstrip()
        if not line_prefix.startswith(("#", "//", "/*", "*")):
            continue
        name = _function_after_marker(text, marker, patterns)
        if name is not None:
            out.add(Annotation(file=file_path, function=name))
    return out


def extract_annotations_from_overlay(
    overlay: MemoryOverlayFileSystem,
) -> set[Annotation]:
    """Return every `(file, function)` annotated in the overlay.

    Only files written through the overlay are inspected; the base FS is
    treated as untouched.
    """
    annotations: set[Annotation] = set()
    for path, blob in overlay._files.items():
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        annotations.update(_extract_from_text(path, text))
    return annotations


def overlay_modified_files(overlay: MemoryOverlayFileSystem) -> set[str]:
    return set(overlay._files.keys())


@dataclass
class TraceAgentRun:
    agent_run: AgentRun
    annotations: set[Annotation]
    modified_files: set[str]
    prompt_version: str


async def run_trace_agent(
    *,
    fixture_root: Path,
    user_message: str,
    model: LiteLlm,
    namespace: str = "trace-eval",
    enable_vuln_reporting: bool = False,
    timeout_s: float = 900.0,
    prompt_version: Optional[str] = None,
    with_graph_tools: bool = False,
) -> TraceAgentRun:
    """Build a trace agent over a fresh overlay of the fixture, run it, and
    return both the raw `AgentRun` and the parsed annotations.

    `prompt_version` pins a specific version from
    `contractor/agents/trace_agent/prompt.yml`; `None` resolves to the
    manifest's `active` version. The resolved id is recorded on the result.

    `with_graph_tools` opts the agent into the trailmark-backed
    call-graph toolset (see ``contractor.tools.code.graph``). Off by
    default to keep v5 prompts free of tool-description tax.
    """
    from cli.fs import RootedLocalFileSystem

    prompt_text, resolved_version = load_prompt_with_version(
        "trace_agent", prompt_version
    )

    # fsspec instance-caches AbstractFileSystem subclasses by their init
    # signature; without ``skip_instance_cache`` two trace-eval cases
    # against the same fixture_root would share the SAME overlay (and
    # therefore the same ``_files`` dict), letting annotations from
    # case N leak into case N+1's extraction. The flag forces a fresh
    # overlay per call. RootedLocalFileSystem is stateless (just a
    # sandboxing wrapper around the local FS) so its instance cache
    # stays on.
    base_fs: AbstractFileSystem = RootedLocalFileSystem(str(fixture_root))
    overlay = MemoryOverlayFileSystem(base_fs, skip_instance_cache=True)

    agent = build_trace_agent(
        name="trace_agent",
        fs=overlay,
        namespace=namespace,
        model=model,
        max_tokens=80_000,
        with_graph_tools=with_graph_tools,
        enable_vuln_reporting=enable_vuln_reporting,
        prompt=prompt_text,
    )

    async def _setup(artifact_service, app_name: str, user_id: str) -> None:
        await inject_skills(
            ["trace"],
            namespace=namespace,
            artifact_service=artifact_service,
            app_name=app_name,
            user_id=user_id,
        )

    plugin, metrics_events = _make_metrics_plugin(
        task_name="eval.trace_agent",
        namespace=namespace,
    )

    with observability.run_context(
        name="eval.trace_agent",
        session_id=namespace,
        tags=[
            "eval",
            "agent:trace_agent",
            f"prompt:trace_agent@{resolved_version}",
        ],
        metadata={
            "agent": "trace_agent",
            "prompt_version": resolved_version,
            "namespace": namespace,
            "fixture_root": str(fixture_root),
        },
    ):
        run = await run_agent(
            agent,
            user_message=user_message,
            timeout_s=timeout_s,
            setup=_setup,
            plugins=[plugin],
            metrics_events=metrics_events,
        )
    return TraceAgentRun(
        agent_run=run,
        annotations=extract_annotations_from_overlay(overlay),
        modified_files=overlay_modified_files(overlay),
        prompt_version=resolved_version,
    )
