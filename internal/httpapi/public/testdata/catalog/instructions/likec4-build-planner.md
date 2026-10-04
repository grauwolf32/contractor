Build a single-file architecture model from the exact `source`, `dependency_report`,
and `project_report` artifacts. An exact optional `existing_likec4` seed may also be
present.

Resume `likec4/architecture` on retry; otherwise copy the exact seed, or create a new
artifact when no seed was supplied. Use the reports as an index, verify material facts
with source, and build in persisted `specification` -> `model` -> `views` phases. Run
`validate_likec4` after each phase and fix errors before adding more content.

Cover evidenced actors, inbound entry points, deployable processes/services, stores,
queues, jobs, outbound integrations, identity/authentication/authorization, secrets,
sensitive-data classes, protocols, and trust-boundary crossings without limiting the
model to security use cases. Put `relative/path:line` evidence in element and material
relationship descriptions. Prefer a smaller accurate model to speculative detail.

Before finishing, compare the model with both reports, record justified omissions,
and require a clean final validator result. Persist the latest model at
`likec4/architecture` (`text/vnd.likec4`). End with a concise semantic result.
