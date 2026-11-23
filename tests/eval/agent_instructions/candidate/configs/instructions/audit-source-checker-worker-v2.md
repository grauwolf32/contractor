Evaluate every immutable Audit item supplied as the ordered task set in
`inputs/task` against the code archive in `inputs/source`. Call
`read_audit_task` first, inspect only the source needed for those tasks, and
distinguish an absence of evidence from evidence that refutes a claim.

Open the exact `inputs/source` revision with `open_source_archive`. Resolve each
assigned operation through registration and effective middleware to its handler
and relevant callees. Verify the control's implementation, its applicability to
the sensitive operation, and any visible bypass branch. Search misses, reassuring
names, and a protected sibling handler do not prove this operation safe. Keep
source observations separate from unexecuted runtime hypotheses. Reuse verified
evidence across items only where each item's scope and evidence contract match.

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
