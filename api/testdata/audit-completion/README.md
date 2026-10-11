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

`diagnostics.json` is shared by the Go and Python report and live-State
consumers and by the private JSON Schema, which give every case the same
verdict (AWC-R115). It covers bounded counts and phases, malformed known
fields, and the `unknown-*` cases: an unknown kind or phase is rejected like any
other malformed object and rejects the whole report or State, never only the
object. `semanticOnly` marks the invalid cases whose count invariants JSON
Schema cannot express; the schema accepts them and every codec rejects them.
