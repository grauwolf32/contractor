# Configuration loader fixtures

`valid/` is a complete, dependency-resolved configuration root. Files in
`invalid/` are focused replacements for the valid ModelPolicy manifest; tests
install one replacement at a time so every failure remains attributable to one
contract violation. The valid root also carries an unconsumed ExecutionConfig
profile so every required immutable manifest subtree is represented.

Shared policy-override and escalation fixtures live in
[`../../configtest/testdata/`](../../configtest/testdata/README.md). They use
`test-*` policy names and are installed only into temporary test catalogs.

These fixtures replace the operator catalog in `configs/` as test input; the
shipped catalog's validity is checked by `make test-config`. `copyCoreFixture`
copies `valid/` into `t.TempDir()`, overlays the shared `test-*` fixtures and
then adds the named catalog slices below. A slice only adds files, so it cannot
replace a core manifest. Expected values and digests are pinned to these frozen
copies, not to the shipped catalog.

- `audit-completion-catalog/`: the `source-checklist@1` AuditProfile with its
  `audit-source-check@1` Workflow, checker AgentTemplate, instructions and
  ModelPolicy; completion escalation tests select the shared
  `test-strong-worker@1` policy.
- `audit-scan-catalog/`: the OpenAPI SQLMap and Nuclei scan AuditProfiles with
  their Workflows, tool AgentTemplates and instructions.
- `audit-preparation-catalog/`: a contract-only `openapi-from-workspace@7`
  prepare Workflow for the shared `api/testdata/audit-composition` profile.
- `audit-profile-catalog/`: the `taint-trace-from-workspace@4` and
  `security-analysis@4` Workflow closures selected by AuditProfile tests.
- `workflow-examples/`: the Streamline and Router examples retargeted to the
  core fixture's builder and Planner policy.

`scan-catalog/` is a standalone, model-free root with the Nuclei, Naabu and
SQLMap tool AgentTemplates, their single-tool Workflows and the scan-plan
Workflows that bind them.
