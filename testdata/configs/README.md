# Server test configuration fixtures

This small catalog supplies stable, resolved Workflow, AgentTemplate, ModelPolicy and
AuditProfile inputs for Scheduler, Control Plane and trusted Run Service tests.
Tests load this directory instead of the operator-editable `configs/` catalog.
Tests that mutate manifests must first copy the fixture into `t.TempDir()`.

The initial fixtures were selected from commit `8dc9929c`: the artifact-copy
and Audit source-check workflows, their dependencies and the policies used by
retry/escalation tests. The trace Skill is a minimal test-only document.
These are test data,
not a deployment catalog. Keep their exact versions and budget expectations
stable; change only the relevant fixtures alongside a contract/test change.
Do not automatically synchronize this directory with `configs/` or point it at
local deployment settings. Smaller loader-only cases remain in
`internal/config/testdata/valid`.
