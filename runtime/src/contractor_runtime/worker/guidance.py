"""Bounded request-only facts, independent of the stored conversation."""

import json
from typing import Any

from google.genai import types

# LIM-WRK-39; api/testdata/request-guidance.json pins both language consumers.
MAX_GUIDANCE_BYTES = 2048
GUIDANCE_PREAMBLE = "Current execution facts (authoritative for this request):\n"


def capacity_content(budget: Any, mode: str) -> types.Content:
    facts = {
        "contextManagement": mode,
        "committedGeneration": budget.checkpoint_generation,
        "callScope": "current context window" if mode == "compact@1" else "logical invocation",
        "additionalModelCallsAfterThisCall": max(
            0, budget.max_model_calls - (budget.model_calls - budget.checkpoint_model_calls)
        ),
        "remainingToolCalls": max(
            0, budget.max_tool_calls - (budget.tool_calls - budget.checkpoint_tool_calls)
        ),
        "completionCapacity": (
            "Ordinary finalization shares model-call capacity. "
            "Context management uses its existing "
            "policy; no additional capacity is promised."
        ),
    }
    text = GUIDANCE_PREAMBLE + json.dumps(facts, separators=(",", ":"))
    if len(text.encode()) > MAX_GUIDANCE_BYTES:
        raise RuntimeError("Worker guidance exceeds its byte limit")
    return types.Content(role="user", parts=[types.Part(text=text)])
