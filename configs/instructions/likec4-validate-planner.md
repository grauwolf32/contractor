Repair-validate the exact `artifacts.architecture_candidate` without extending its
scope. Materialize `source`, read both reports, load the candidate exact revision, and
call `validate_likec4` once.

Verify diagnostics with bounded document/source reads, apply one minimal bounded
repair pass, and call validation exactly once more. Never claim success when errors
remain or when the direct LikeC4 CLI is unavailable/failed.

Write `likec4/validation-report` as Markdown. A clean result returns exact slots:

- `architecture` -> final `likec4/architecture` revision;
- `validation_report` -> exact `likec4/validation-report` revision.

CLI/runtime environment failures are retryable; remaining DSL errors after the repair
pass are non-retryable. Do not paste the architecture source into the report or Stage
summary.
