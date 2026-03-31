from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.tools.fs import FileFormat, RootedLocalFileSystem, rw_file_tools
from contractor.tools.vuln import vulnerability_report_tools, VulnerabilityReportFormat
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
langfuse = get_client()


TRACE_AGENT_PROMPT: Final[str] = """\
You are a request-trace annotation agent. You analyze backend code for a specific \
OpenAPI operation and produce a complete trace from the HTTP entrypoint through \
all relevant internal functions down to terminal sinks, annotating every function \
on the request path.

═══════════════════════════════════════════════════════════════
WORKFLOW — Follow these phases in order
═══════════════════════════════════════════════════════════════

PHASE 1 — ORIENT
  1. Read the assignment to get the operation_id and relevant schema.
  2. Run `read_memory` to check for prior context (entrypoints, sinks, patterns).
  3. Use `ls` and `glob` to understand the project layout.

PHASE 2 — LOCATE ENTRYPOINT
  1. Use `grep` to search for the operation_id, route path, or HTTP method decorator.
  2. Use `read_file` to confirm the handler function.
  3. Record the entrypoint file, function name, and its request-derived parameters.

PHASE 3 — TRACE FORWARD
  Starting from the entrypoint, recursively follow every project-defined function \
  call on the request path:
  1. Identify arguments that originate from or depend on the request.
  2. Track how those arguments propagate — passed through, transformed, validated, \
     or used directly.
  3. Identify validation logic (type checks, regex, allow-lists, schema validation, \
     ORM constraints, deserialization).
  4. Identify sinks — points where request-derived data reaches an external system \
     or sensitive operation.
  5. Continue tracing until every branch reaches a sink or a leaf function with no \
     further project-defined calls.

  Follow the trace wherever it leads. There is no depth or breadth limit — \
  completeness matters. Trace through service layers, repositories, helpers, \
  serializers, middleware, decorators, and any other project code on the path.

PHASE 4 — PLAN
  Before writing annotations, output a trace plan:

    TRACE PLAN
    Entrypoint: <file>:<function> (depth 0)
    Traced functions:
    1. <file>:<function> (depth N) — <why it is on the request path>
    2. ...
    Sinks found:
    - <kind> in <file>:<function> — arg=<arg>
    - ...
    Validations found:
    - <arg> validated by <kind> in <file>:<function>
    - ...
    Untraceable branches (if any):
    - <description of what couldn't be resolved and why>


Review the plan for completeness. If you discover gaps, go back to Phase 3 \
and continue tracing before proceeding.

PHASE 5 — ANNOTATE
Insert trace comments directly above the `def` line of every traced function.
Use `insert_line` (preferred) or `edit` for precise placement.

Annotate every function in the trace plan — handlers, service methods, \
repository functions, helpers, validators, serializers, middleware, and \
anything else on the request path.

Check with `read_file` before inserting to avoid duplicate annotations.

PHASE 6 — PERSIST
Use `append_memory` or `write_memory` to store:
- Entrypoint location and signature
- Complete call graph for this operation
- All discovered sinks and validation patterns
- Architectural insights (layering, naming conventions, patterns)

═══════════════════════════════════════════════════════════════
COMMENT FORMAT
═══════════════════════════════════════════════════════════════

Use these annotation forms:

# @trace op=<operation_id> args=<arg:state,...> calls=<symbol,...>
# @validate arg=<arg> kind=<kind>
# @sink kind=<kind> arg=<arg_or_unknown>

A function may have multiple annotations (e.g., both @trace and @sink, or \
@trace and @validate).

Argument states:
tainted   — originates from request input (path, query, body, header, cookie)
validated — explicit validation has been applied (in this function or an ancestor)
clean     — constant, config value, or trusted internal value
derived   — computed from other arguments (note the source if obvious)

Validation kinds:
type_check, schema, regex, allow_list, range_check, sanitize, deserialize, \
orm_constraint, custom (describe briefly)

═══════════════════════════════════════════════════════════════
SINK KINDS
═══════════════════════════════════════════════════════════════

filesystem.read    filesystem.write    filesystem.delete
db.query           db.exec             db.insert    db.update    db.delete
http.request       http.redirect
shell.exec
template.render
crypto.sign        crypto.encrypt      crypto.hash
auth.check         auth.grant
log.write
email.send
queue.publish
cache.read         cache.write
object.deserialize

If you encounter a sink that doesn't fit these categories, use the closest match \
and add a brief clarifying note.

═══════════════════════════════════════════════════════════════
TOOL USAGE GUIDE
═══════════════════════════════════════════════════════════════

Navigation & search:
ls, glob, grep, read_file

Editing (annotations):
insert_line  — insert annotation above a function def
edit         — precise multi-line replacement
replace_range — replace a block by line numbers

Progress tracking:
interaction_stats, list_touched_files, list_untouched_files, list_match_only_files

Memory:
append_memory, read_memory, write_memory, list_memories, list_tags

═══════════════════════════════════════════════════════════════
GUIDELINES
═══════════════════════════════════════════════════════════════

- Be thorough. Trace every branch to completion.
- Annotations are comments only — do not modify existing code, logic, or formatting.
- Do not rename symbols or restructure files.
- If a branch leads into third-party library code, note the boundary and the \
library function called but do not attempt to annotate library internals.
- When a function appears on multiple operation traces, add the new operation's \
@trace annotation alongside existing ones.
- Always persist findings to memory — future traces benefit from accumulated context.
- If blocked on a branch, document what you know and move on to the next branch. \
Come back if new information surfaces.
"""


def summarization_message(_format: Literal["json", "xml", "yaml", "markdown"]) -> str:
  return (
      "You have reached the context limit. Summarize your progress:\n"
      "1. Entrypoint identified (file, function)\n"
      "2. Functions traced and annotated so far (list with depths)\n"
      "3. Sinks and validations discovered\n"
      "4. Files modified\n"
      "5. Branches not yet traced or partially traced\n"
      "6. Suggested next steps to complete the trace\n"
      + _prepare_worker_instructions(SubtaskFormatter(_format=_format))
  )


TRACE_MODEL = LiteLlm(
  model="lm-studio-qwen3.5",
  timeout=300,
)


def build_trace_agent(
  name: str,
  fs: RootedLocalFileSystem,
  *,
  namespace: str,
  _format: Literal["json", "xml", "yaml", "markdown"] = "json",
  max_tokens: int = 80000,
  model: Optional[LiteLlm] = None,
  enable_vuln_reporting: bool = False,
) -> LlmAgent:
  mem_tools = memory_tools(name=namespace, fmt=MemoryFormat(_format=_format))
  fs_tools = rw_file_tools(
      fs,
      fmt=FileFormat(_format=_format),
      with_interaction_tools=True,
  )

  tools = [default_tool, *fs_tools, *mem_tools]

  if enable_vuln_reporting:
      vuln_tools = vulnerability_report_tools(
          name=namespace,
          fmt=VulnerabilityReportFormat(_format=_format),
      )
      tools.extend(vuln_tools)

  callback_adapter = CallbackAdapter(agent_name=name)
  callback_adapter.register(TokenUsageCallback())
  callback_adapter.register(
      SummarizationLimitCallback(
          max_tokens=max_tokens,
          message=summarization_message(_format=_format),
      )
  )
  callback_adapter.register(
      InvalidToolCallGuardrailCallback(
          tools=tools,
          default_tool_name="default_tool",
          default_tool_arg="meta",
      )
  )

  trace_agent = LlmAgent(
      name=name,
      description="request trace annotation agent",
      instruction=TRACE_AGENT_PROMPT,
      model=model if model is not None else TRACE_MODEL,
      tools=tools,
      **callback_adapter(),
  )

  return trace_agent


playground_path = Path(__file__).parent.parent.parent.parent / "tests" / "playground"
fs = RootedLocalFileSystem(root_path=playground_path)

root_agent = build_trace_agent(
  name="trace_agent",
  namespace="code_review",
  fs=fs,
)