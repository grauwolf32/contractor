# Shared test configuration data

These static fixtures belong to configuration, scheduler and HTTP API unit tests.
They are not part of the operator catalog in `configs/` or the e2e catalog.

- `test-worker@1`: small Worker budget for validation and digest tests.
- `test-planner@1`: modeled Planner policy, without a Worker tool budget.
- `test-domain-worker@1`: alternate Worker budget for override precedence tests.
- `test-strong-worker@1`: different model and limits for escalation tests.

`configtest.CopyWithPolicies` copies the supplied catalog into `t.TempDir()` and
overlays these policies, the test artifact builder and the test escalation
profile. The other workflows, templates and instructions remain copied from the
supplied catalog. This preserves realistic wiring while keeping the budgets and
model variants used in behavioral tests independent of production policy tuning.
Tests of the actual shipped catalog load `configs/` directly.

Minimal loader fixtures and malformed manifests remain in
`internal/config/testdata/`. Neither helper edits the source catalog or e2e files.

`escalation/workflows/` contains a minimal validation Workflow used exclusively
by escalation tests. `CopyWithEscalation` installs it after the common fixtures.
It retains the selector used by those tests without copying the production
OpenAPI topology or depending on production retry settings.
