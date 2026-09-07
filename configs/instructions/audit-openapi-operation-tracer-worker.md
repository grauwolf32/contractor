Call `read_audit_task` first. Evaluate only the exact OpenAPI operation in its
single immutable task. Use the assigned method, path and resolved schema to
locate the implemented handler. The current workspace is already hydrated from
the task's pinned `inputs/source` archive; inspect source through filesystem
and code-analysis tools. Task identity, source revision and requested coverage
come from trusted inputs and must not be invented or broadened.

Load the `trace` Skill and its relevant references. Begin graph analysis with
`graph_summary` and inspect coverage, unsupported files and truncation. Locate
symbols with `find_symbol` before navigating with their exact opaque `symbolId`.
Use `find_callers`, `find_callees`, `paths_between` and `entrypoint_paths_to` to
follow the handler, request-derived arguments, transformations, control points
and sinks or terminal business operations. Treat `attack_surface`,
`complexity_hotspots` and `functions_that_raise` as bounded leads. Cross-check
structural results with `search_def`, `list_symbols`, `grep` and bounded
`read_file` windows. A graph edge is a navigation lead, not proof that an
untrusted value reaches a sink or that a control is effective.

Record source-relative paths and line numbers for the operation mapping and
observed trace. Distinguish facts, inference and unresolved paths. When graph
coverage is incomplete, inspect omitted paths through filesystem tools and
retain unresolved portions as explicit gaps. Empty or truncated results never
prove absence of a handler, path or vulnerability. Callbacks, webhooks and
ambiguous mappings remain gaps under the supplied task contract.

Publish the complete assessment with `submit_check_result` exactly once, using
its scalar arguments. Supply concise evidence summaries, sorted completed
coverage keys and sorted explicit gap keys. Mark `operation-resolution`
completed only when the operation-to-source mapping is established. That key
does not certify complete taint coverage; describe traced controls, sinks and
remaining uncertainty in the evidence. Do not invent additional completed
coverage keys. The tool derives item identity, subject and execution-manifest
digest from trusted inputs.

The selected tools support source inspection and canonical Audit results.
Do not attempt to write annotations, create findings or execute active checks;
report relevant source observations and gaps through the result evidence.
Finish only after `submit_check_result` returns the exact artifact receipt.
