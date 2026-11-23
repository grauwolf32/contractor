Validate the exact `openapi_candidate` using the exact `source`,
`dependency_report`, and `project_report` inputs. Keep one bounded repair task;
do not delegate repeated lint cycles or expand the API scope. The Worker owns
source access, domain edits, and validator execution.

Acceptance: `openapi/validation-report` records the candidate/final revisions,
initial/final validation, minimal verified changes, and unresolved issues. Keep
the final document at `openapi/openapi`. Allow one initial validation and one
validation after repair; a CLI execution failure ends with an explicit report.
Claim clean only with final `validate_openapi.valid: true`. Return a concise
semantic result that distinguishes clean, unresolved, and environment failure.
