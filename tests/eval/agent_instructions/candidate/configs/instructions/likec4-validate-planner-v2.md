Validate the exact `architecture_candidate` using the exact `source`,
`dependency_report`, and `project_report`. Keep one bounded repair task without
broadening architecture scope. The Worker owns source access, targeted DSL
corrections, and validator execution.

Acceptance: `likec4/validation-report` records candidate/final revisions, initial
and final issue counts, verified edits, unresolved diagnostics, and CLI failures.
Keep the final model at `likec4/architecture`. Allow one initial validation and
one validation after repair; a CLI execution failure ends with an explicit
report. Claim clean only with final `validate_likec4.valid: true`. Return a
concise semantic result without pasting DSL or the report body.
