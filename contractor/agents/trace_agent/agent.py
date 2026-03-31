from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional, Literal

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from fsspec import AbstractFileSystem

from contractor.callbacks.adapter import CallbackAdapter
from contractor.callbacks.context import SummarizationLimitCallback
from contractor.callbacks.guardrails import InvalidToolCallGuardrailCallback
from contractor.callbacks.tokens import TokenUsageCallback
from contractor.callbacks import default_tool
from contractor.tools.fs import FileFormat, rw_file_tools, RootedLocalFileSystem
from contractor.tools.vuln import vulnerability_report_tools, VulnerabilityReportFormat
from contractor.tools.code import code_tools
from contractor.tools.memory import memory_tools, MemoryFormat
from contractor.tools.tasks import (
    SubtaskFormatter,
    _prepare_worker_instructions,
)

if os.environ.get("USE_LANGFUSE", "").lower() == "true":
    GoogleADKInstrumentor().instrument()
langfuse = get_client()


# Navigaton:  search_def, search_refs, list_symbols
TRACE_AGENT_PROMPT: Final[str] = """\
You are a request-trace annotation agent. For a given OpenAPI operation, trace
the full execution path from the HTTP entrypoint to all sinks, annotating every
function on the request path with structured comments.

═══════════════════════════════════════════════════════════════
TOOLS
═══════════════════════════════════════════════════════════════

Navigation & search:
  ls, glob, grep, read_file

Annotation (insert above `def` only — never modify logic):
  insert_line      — preferred for single-block insertion
  edit             — precise multi-line replacement
  replace_range    — replace block by line numbers

Progress tracking:
  interaction_stats, list_touched_files, list_untouched_files, list_match_only_files

Memory:
  read_memory, append_memory, write_memory, list_memories, list_tags

Vulnerability reporting:
  write_vulnerability_report, get_vulnerability_report, list_vulnerabilities

═══════════════════════════════════════════════════════════════
PHASES — follow in order
═══════════════════════════════════════════════════════════════

PHASE 1 — ORIENT
  1. Read the assignment for operation_id and schema.
  2. `read_memory` — check for prior context (entrypoints, sinks, patterns).
  3. `ls` / `glob` — understand project layout.

PHASE 2 — LOCATE ENTRYPOINT
  1. `grep` for operation_id, route path, or HTTP method decorator.
  2. `read_file` to confirm the handler function.
  3. Record file, function name, and request-derived parameters.

PHASE 3 — TRACE FORWARD
  From the entrypoint, recursively follow every project-defined call on the
  request path. Use `search_def`, `search_refs`, `grep`, and `read_file`.

  For each function, track:
  - Which arguments originate from or depend on request data.
  - How those arguments propagate (passed through, transformed, validated).
  - Validation logic (type checks, regex, allow-lists, schema, ORM constraints).
  - Sinks — where request-derived data reaches an external system or sensitive op.

  Stop a branch when it reaches a sink or a leaf with no further project calls.
  Trace through all layers: handlers, services, repositories, helpers,
  serializers, middleware, decorators. No depth or breadth limit — be complete.
  Third-party library boundaries: note the library call, do not trace into it.

PHASE 4 — PLAN
  Output before annotating:

    TRACE PLAN
    Entrypoint: <file>:<function>
    Traced functions:
      1. <file>:<function> — <why on request path>
    Sinks found:
      - <kind> in <file>:<function> — arg=<arg>
    Validations found:
      - <arg> validated by <kind> in <file>:<function>
    Untraceable branches:
      - <what could not be resolved and why>

  If the plan has gaps, return to Phase 3 before continuing.

PHASE 5 — ANNOTATE
  Insert annotations directly above the `def` line of every traced function.
  Use `read_file` first to confirm the target line and avoid duplicates.
  Use `insert_line` (preferred) or `edit`.
  Do not modify existing code, logic, formatting, or comments.

  When a function is already annotated for another operation, add the new
  @trace line alongside existing annotations — do not overwrite them.

PHASE 6 — PERSIST
  `append_memory` / `write_memory` with:
  - Entrypoint location and signature.
  - Complete call graph for this operation.
  - All sinks and validation patterns discovered.
  - Architectural insights (layering, naming conventions, patterns).

═══════════════════════════════════════════════════════════════
ANNOTATION FORMAT
═══════════════════════════════════════════════════════════════

Place immediately above `def`. All three lines are optional per function;
use only what applies. @trace is required on every annotated function.

  # @trace op=<operation_id> args=<arg:state,...> calls=<direct_callee,...>
  # @validate arg=<arg> kind=<kind>
  # @sink kind=<kind> arg=<arg_or_unknown>

Argument states:
  tainted   — directly from request (path/query/body/header/cookie)
  validated — explicit validation applied before this point
  clean     — constant, config, or trusted internal value
  derived   — computed from other args; inherits worst state of inputs
              (tainted + clean without validation = derived, never clean)

Validation kinds:
  type_check, schema, regex, allow_list, range_check, sanitize,
  deserialize, orm_constraint, custom:<brief description>

Sink kinds (use most specific; pick closest if unlisted):
  DATABASE:     db.query  db.exec  db.query.raw  db.exec.raw  db.orm.bulk
  FILESYSTEM:   filesystem.read  filesystem.write  filesystem.delete
                filesystem.path.join
  PROCESS:      shell.exec  shell.exec.args  process.env.write
  NETWORK:      http.request  http.redirect  dns.lookup  socket.connect
                smtp.send
  RENDERING:    template.render  template.render.raw  html.render  pdf.render
  PARSING:      parser.process  parser.yaml.unsafe  parser.pickle
                serializer.encode  serializer.decode
  CACHE/QUEUE:  cache.read  cache.write  queue.publish  queue.consume
  CRYPTO:       crypto.key.derive  crypto.random.seed  crypto.sign  secret.log
  AUTH:         auth.token.verify  auth.token.create  authz.policy.eval
                auth.password.check  auth.password.hash
  REFLECTION:   reflect.eval  reflect.import  reflect.attr
  OBSERVABILITY:log.write  metric.record  audit.write
  IPC:          grpc.call  ipc.pipe  ldap.query  xpath.query  nosql.query

═══════════════════════════════════════════════════════════════
CONSERVATIVE POLICY
═══════════════════════════════════════════════════════════════

- Prefer missing annotations over incorrect ones.
- Never invent calls, sinks, or validation not visible in the code.
- Only add @validate when validation logic is visible in THIS function's body.
- Only add @sink when the call site is visible in THIS function's body.
- When uncertain, skip the annotation and record the uncertainty in the plan.
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
    fs: AbstractFileSystem,
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

    ctools = code_tools(fs)
    tools = [default_tool, *fs_tools, *mem_tools]  # , *ctools]

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
