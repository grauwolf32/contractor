# Audit completion fixtures

Current common Task/context/conclusion cases live in `../audit-common/`.

## Completion diagnostics

`diagnostics.json` is shared by the Go and Python report and live-State
consumers and by the private JSON Schema, which give every case the same
verdict (AWC-R115). It covers bounded counts and phases, malformed known
fields, and the `unknown-*` cases: an unknown kind or phase is rejected like any
other malformed object and rejects the whole report or State, never only the
object. `semanticOnly` marks the invalid cases whose count invariants JSON
Schema cannot express; the schema accepts them and every codec rejects them.
