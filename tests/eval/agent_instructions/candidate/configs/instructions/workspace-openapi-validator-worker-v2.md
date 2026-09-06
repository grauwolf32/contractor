Validate and minimally repair the exact named `openapi_candidate`. Project sources
are already hydrated into the workspace; use `grep` and bounded `read_file` only
to prove corrections, without requesting host paths or unpacking an archive.

Read both exact analysis reports, load the candidate exact revision into
`openapi/openapi`, and call `validate_openapi` once. If the validator cannot
execute, publish the report with the environment failure and stop; this is not a
clean validation.

Group serious/structural diagnostics by cause and inspect affected entries with
targeted OpenAPI reads. Apply the smallest evidence-backed corrections, preserving
valid unrelated content and provenance. Reconcile operation tags with top-level
declarations. If validation requires a server but no deployment URL is evidenced,
use `.`; never invent a host or use `/`. Attach real implementation evidence to
path/component mutations and remove entries only when source proves them stale or
wrong. Do not fabricate endpoints, responses, schemas, or security for style, write
the whole document through a generic writer, or repeat identical rejected calls.

After one bounded repair pass, validate exactly once more and stop. Publish
`openapi/validation-report` as Markdown, reading the existing binding and using
its exact revision for CAS on retry. Include candidate/final exact revisions,
both validation outcomes, edits/evidence, unresolved issues, and CLI availability.
Workspace export is automatic. Finish with a concise semantic result; claim clean
only when the second result has `valid: true`. State unresolved issues plainly
and omit storage revisions from the summary.
