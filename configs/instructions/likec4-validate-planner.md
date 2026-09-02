Repair-validate the exact named `architecture_candidate` input without extending its
scope. Materialize `source`, read both reports, load the candidate exact revision, and
call `validate_likec4` once.

Verify diagnostics with bounded document/source reads, apply one minimal bounded
repair pass, and call validation exactly once more. Never claim success when errors
remain or when the direct LikeC4 CLI is unavailable/failed.

Write `likec4/validation-report` as Markdown and keep the clean final model at
`likec4/architecture`. State CLI/environment failures or remaining DSL errors plainly
without claiming a clean result. Do not paste the architecture source into the report
or final result.
