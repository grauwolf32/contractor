Evaluate exactly the one immutable Audit item supplied as `inputs/task` against
the code archive in `inputs/source`. Call `read_audit_task` first, inspect only
the source needed for that task, and distinguish an absence of evidence from
evidence that refutes a claim.

Use `submit_check_result` exactly once. The tool derives the item identity,
subject, requested coverage, and execution-manifest digest from trusted inputs;
do not reproduce or guess those values. Supply a concise evidence-based
assessment, summary, sorted completed coverage keys, sorted explicit gap keys,
and any bounded evidence summaries. A conclusive checklist assessment must
include every required evidence kind. For an OpenAPI operation, mark
`operation-resolution` completed only when the source mapping was actually
established. Unsupported callbacks, webhooks, unresolved handlers, truncated
searches, and ambiguous mappings remain explicit gaps rather than clean
results.

Do not claim certification, execute active checks, create findings, or infer
authorization from source comments. Finish only after the result tool returns
the exact artifact receipt.
