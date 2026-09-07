Evaluate every immutable Audit item supplied as the ordered task set in
`inputs/task` against the code archive in `inputs/source`. Call
`read_audit_task` first, inspect only the source needed for those tasks, and
distinguish an absence of evidence from evidence that refutes a claim.

Use `submit_check_result` exactly once. For one task, use its scalar arguments.
For multiple tasks, pass one `results` entry per task in the returned order;
do not include or reorder item identities. The tool derives each item identity,
subject, requested coverage, and execution-manifest digest from trusted inputs.
Supply a concise evidence-based assessment, summary, sorted completed coverage
keys, sorted explicit gap keys, and any bounded evidence summaries for every
task. A conclusive checklist assessment must include every required evidence
kind. For an OpenAPI operation, mark `operation-resolution` completed only when
the source mapping was actually established. Unsupported callbacks, webhooks,
unresolved handlers, truncated searches, and ambiguous mappings remain explicit
gaps rather than clean results. A partial batch is invalid; do not submit until
every assigned task has a truthful result.

Do not claim certification, execute active checks, create findings, or infer
authorization from source comments. Finish only after the result tool returns
the exact artifact receipt.
