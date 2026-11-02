"""Worker task rendering, prompt budgets and private-value policy."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from contractor_runtime.contracts import (
    ResolvedModelPolicy,
    StageContentRequest,
    WorkerSummarizerConfig,
)
from contractor_runtime.redaction import (
    contains_private_value,
    runtime_secrets,
    runtime_setting_values,
    substring_values,
)

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext


def _task_prompt(request: StageContentRequest) -> str:
    """Render Planner-supplied task data; wire/lifecycle contracts stay Runtime-private."""

    inputs = {
        name: ref.model_dump(mode="json", by_alias=True)
        for name, ref in sorted(request.artifacts.items())
    }
    return (
        f"Subtask ID:\n{request.subtask_id}\n\n"
        f"Objective:\n{request.objective}\n\n"
        f"Task instructions:\n{request.instructions}\n\n"
        "String parameters:\n"
        + json.dumps(request.parameters, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nNamed input artifacts:\n"
        + json.dumps(inputs, ensure_ascii=False, sort_keys=True, indent=2)
    )


def _summary_prompt_boundary(
    policy: ResolvedModelPolicy,
    config: WorkerSummarizerConfig | None,
) -> int | None:
    """Derive the normal-loop prompt boundary from pinned model metadata."""

    if config is None:
        return None
    if policy.context_window_tokens is None or policy.max_output_tokens is None:
        raise RuntimeError("summarized Worker policy has no context-window metadata")
    ratio_boundary = math.floor(policy.context_window_tokens * config.context_window_ratio)
    output_safe_boundary = policy.context_window_tokens - policy.max_output_tokens
    return min(ratio_boundary, output_safe_boundary)


def _gateway_token(context: WorkerBuildContext) -> str:
    token = context.runtime_settings.llm_gateway_token
    return token.get_secret_value() if token is not None else ""


def _exposes_private_value(text: str, context: WorkerBuildContext) -> bool:
    """Whether Worker-authored text contains an allocation credential.

    Covers every RuntimeSettings credential (Gateway token, telemetry headers,
    proxy, Caido and HTTP origin target credentials) under the Agent Card
    matching policy; the LLM Gateway token additionally matches at any length.
    Endpoints are not credentials: a result may name them, for example when a
    same-host deployment audits a service next to its own.
    """

    gateway_token = _gateway_token(context)
    if gateway_token and gateway_token in text:
        return True
    return contains_private_value(text, runtime_secrets(context.runtime_settings))


def _summarizer_secrets(context: WorkerBuildContext) -> tuple[str, ...]:
    """Return allocation-private values that must not enter summarizer input."""

    candidates = (
        _gateway_token(context),
        *substring_values(runtime_setting_values(context.runtime_settings)),
        str(context.workspace.path),
        str(context.workspace.root),
    )
    # Replacing '/' would destroy every path-like value rather than protect a
    # useful host path, so only non-root concrete paths are admitted. Short
    # settings values (for example a proxy username) are not replaced inside
    # ordinary transcript text for the same reason.
    return tuple(dict.fromkeys(value for value in candidates if len(value) > 1))
