# Audit completion fixtures

`task-local-validation.json` contains full trusted task documents and named
submission cases referencing those documents. `valid` is task-local admission,
not final Audit acceptance. The Python tests package the task and submit through
the v2 tool; Go validates the task/result schemas and runs the importer's actual
semantic coverage checks. No operator catalog, live model or database is needed.

Keep positive cases with partial coverage or gaps: valid data can still yield
inconclusive Audit coverage. Standard minimum evidence applies only to conclusive
results; its assessment, evidence-kind and maximum-count restrictions always apply.

## Completion diagnostics

`diagnostics.json` is shared by Go and Python report/live-State consumers. It
covers bounded counts/phases, malformed known fields, and safe omission of
unknown optional kinds/phases without retaining their extra content.
