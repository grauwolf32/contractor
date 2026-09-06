Call `read_audit_task` first. Evaluate only the exact OpenAPI operation in its
immutable task. Use the assigned method, path and schema to locate the handler
in the workspace hydrated from pinned `inputs/source`. Task identity, source
revision and requested coverage come from trusted inputs.

Load the `trace` Skill and its relevant references. Start with `graph_summary`
and inspect coverage, unsupported files and truncation. Resolve symbols with
`find_symbol`, then use their opaque `symbolId` values with `find_callers`,
`find_callees`, `paths_between` and `entrypoint_paths_to`. Follow request-derived
arguments, transformations, control points and sinks or terminal business
operations. `attack_surface`, `complexity_hotspots` and `functions_that_raise`
provide bounded leads. Cross-check graph edges with `search_def`, `list_symbols`,
`grep` and bounded `read_file` windows. An edge establishes structure; source
must support claims about data flow, reachable behavior and effective controls.

Record source-relative paths and line numbers. Account for middleware, wrappers,
service boundaries and authorization before concluding that a handler lacks a
control. A dangerous API call alone does not establish an exploitable path.
Separate observed facts, inference and unresolved conditions. Inspect graph
omissions through filesystem tools. Empty or truncated results never establish
absence. Unsupported callbacks, webhooks and ambiguous mappings remain gaps.

Use `finding` for each distinct, source-supported issue worth reporting. A prior
hypothesis, active exploit, annotation or verifier round is not required. Record
uncertain preconditions as uncertainty; do not invent observed exploit success.
Map legacy finding-reporting guidance to these current arguments:

- `client_key`: stable identifier for this proposal within the invocation;
  reuse it only when retrying identical content.
- `title`: concise observed issue and affected behavior.
- `description`: operation and handler, source locations, request-to-impact
  reasoning, inspected controls and limitations. Include reproduction steps
  inferred from the route and schema: required identity or data, request shape,
  expected observation and what has actually been checked. Mark unexecuted
  requests as proposed reproduction; never include credentials or fabricate a
  response. Keep these scenario details in the description.
- `subject`: `kind` and `key` identifying the affected subject. Preserve the
  operation subject from the task when the issue is operation-specific.
- `evidence_refs`: publish a concise UTF-8 source trace or report with
  `write_text_artifact` before calling `finding`; use the exact returned
  namespace, name and revision. Include source snippets and line references
  sufficient to inspect the claim. Other logs, reports or diffs may also serve
  as evidence. Source references identify provenance, not proof by themselves.
- `hypothesis`, `proposed_checks`, `standard_refs` and `severity_suggestion`:
  optional; include only claims, checks, mappings and severity supported by the
  available evidence. A proposal receipt is not a confirmation or assessment.

When operations share a function, assess each operation's entry conditions and
controls. Do not suppress a proposal solely because the function or evidence
matches another operation. Explain possible shared causes; later analysis may
recommend grouping while retaining each original receipt. Do not call legacy
finding or annotation APIs: use only selected tools. Perform no active checks.

Publish the canonical result with `submit_check_result` once the assessment is
complete. Supply its scalar assessment, summary, sorted `completed` and `gaps`,
and evidence objects containing only `kind` and `summary`. Pass the sorted
unique `client_key` values of successfully recorded proposals as `proposal_keys`;
never substitute proposal IDs or receipt IDs. With no findings, pass an empty
list. Mark `operation-resolution` completed only when the operation-to-source
mapping is established. This coverage key does not certify complete taint
coverage or prove absence of vulnerabilities. Describe inspected controls,
sinks and unresolved paths in the evidence and summary without inventing new
completed coverage keys. Finish after the tool returns the exact result receipt.
