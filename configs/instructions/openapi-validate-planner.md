Validate and, only where source evidence proves the fix, repair the exact
`artifacts.openapi_candidate` produced by the preceding Stage.

Materialize `source`, read `dependency_report` and `project_report`, and load the
candidate exact revision. Run Vacuum through `validate_openapi` once, investigate
serious or structural findings with targeted reads, apply the smallest supported
changes, and run validation once more. Do not loop and do not make speculative edits
for style findings.

Always write `openapi/validation-report` as Markdown. On clean second validation,
return a successful StageContentResult with exact result slots:

- `openapi` -> the final `openapi/openapi` revision;
- `validation_report` -> the written `openapi/validation-report` revision.

If Vacuum cannot execute, return a retryable failed result. If verified repair still
leaves serious/structural issues, return a non-retryable failed result and describe
them in the report. Never return success unless `validate_openapi.valid` is true.
