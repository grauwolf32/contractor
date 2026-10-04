# Scan test configuration fixtures

This model-free catalog supplies stable, resolved scan Workflow and tool
AgentTemplate inputs for Planner, Scheduler, Control Plane and Audit domain
tests. Tests load this directory instead of the operator-editable
`configs/scan` catalog. Tests that mutate manifests must first copy the fixture
into `t.TempDir()`.

The fixtures were copied from commit `96a90101`:

- `nuclei-target@1`, `request-set-scan@1` and `target-scan-plan@1` with their
  `nuclei-scan@1`, `naabu-scan@1` and `sqlmap-scan@1` tool templates and
  `instructions/scan.md` from `configs/scan`;
- `audit-openapi-sqlmap-scan@1` and `audit-openapi-nuclei-scan@1` with their
  `audit-sqlmap-scan@1` and `audit-nuclei-scan@1` templates and
  `instructions/audit-openapi-scan.md` from the main catalog;
- the OpenAPI document and scanner settings in `audit-openapi-scan/` from
  `configs/scan/examples/audit-openapi-scan`.

The empty catalog directories are required by the configuration loader.
These are test data, not a deployment catalog. Keep their exact versions,
budgets and inputs stable; change only the relevant fixtures alongside a
contract/test change. Do not automatically synchronize this directory with
`configs/` or point it at local deployment settings.
