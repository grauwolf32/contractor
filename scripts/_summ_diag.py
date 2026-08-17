#!/usr/bin/env python3
"""No-LLM diagnostic: does the instrumented worker actually carry output_schema,
and would ADK set response_schema (vs add the set_model_response tool)?"""
from __future__ import annotations

import sys

REPO = "/home/ruslan/src/contractor"
sys.path.insert(0, REPO)

from cli.fs import RootedLocalFileSystem
from contractor.agents.planning_agent.agent import build_planning_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.utils.settings import build_model
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools

fs = RootedLocalFileSystem(f"{REPO}/contractor/runners")
model = build_model("llamacpp-qwen3.6-35b-a3b", timeout=60)

worker = build_swe_agent(
    name="swe_agent", _format="json", fs=fs, model=model,
    max_tokens=8000, namespace="diag",
)

print("=== BEFORE instrumentation (raw build_swe_agent) ===")
print(f"  output_schema: {worker.output_schema}")
print(f"  input_schema : {getattr(worker, 'input_schema', None)}")
print(f"  #tools       : {len(worker.tools)}")
print(f"  model type   : {type(worker.model).__name__}")
print(f"  before_model_callback set: {worker.before_model_callback is not None}")

# Build the planner the way TaskRunner does — this instruments the worker.
planner = build_planning_agent(
    name="diag_task", namespace="diag", worker=worker, model=model,
    max_steps=10, worker_instrumentation=True,
)

print("\n=== AFTER build_planning_agent (worker_instrumentation=True) ===")
print(f"  worker.output_schema: {worker.output_schema}")
print(f"  worker.input_schema : {getattr(worker, 'input_schema', None)}")
cm = worker.canonical_model
print(f"  worker.canonical_model type: {type(cm).__name__}")
print(f"  can_use_output_schema_with_tools(canonical_model): "
      f"{can_use_output_schema_with_tools(cm)}")
print(f"  worker.mode: {getattr(worker, 'mode', None)!r}")
print(f"  worker #tools: {len(worker.tools)}")

# Decision basic.py would make:
has_os = bool(worker.output_schema)
not_task = getattr(worker, "mode", None) != "task"
will_set_response_schema = (
    not_task and has_os and (not worker.tools or can_use_output_schema_with_tools(cm))
)
will_add_set_model_response = (
    has_os and bool(worker.tools)
    and not can_use_output_schema_with_tools(cm)
    and getattr(worker, "mode", None) != "task"
)
print("\n=== ADK structured-output routing decision ===")
print(f"  basic.py sets response_schema : {will_set_response_schema}")
print(f"  processor adds set_model_response tool: {will_add_set_model_response}")
print("\n=> structured output delivered via:",
      "response_format (content)" if will_set_response_schema
      else ("set_model_response TOOL" if will_add_set_model_response
            else "NEITHER (??)"))
