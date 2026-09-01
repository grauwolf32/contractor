Validate and, only where source evidence proves the fix, repair the exact
named `openapi_candidate` input produced by the preceding task.

Materialize `source`, read `dependency_report` and `project_report`, and load the
candidate exact revision. Run Vacuum through `validate_openapi` once, investigate
serious or structural findings with targeted reads, apply the smallest supported
changes, and run validation once more. Do not loop and do not make speculative edits
for style findings.

Use the dedicated tag tools to keep operation tags and top-level declarations
consistent. When no deployment URL is evidenced, `.` is the neutral relative
current-origin server; never invent a host or use the trailing-slash `/`.

Always write `openapi/validation-report` as Markdown. On clean second validation,
ensure the final document remains at `openapi/openapi`. If Vacuum cannot execute or
verified repair still leaves serious/structural issues, describe that plainly in the
report and final summary. Never claim a clean result unless
`validate_openapi.valid` is true.
