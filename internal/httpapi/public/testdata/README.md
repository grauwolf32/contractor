# Public API test fixtures

`catalog/` is a frozen configuration root for Workflow query, configuration
projection, Repeat, Audit and Eval handler tests. It replaces the operator
catalog in `configs/` as test input; the shipped catalog's validity is checked
by `make test-config`.

The files were copied unchanged from `configs/` at commit `35ff8f60`: the
`openapi-from-workspace@7`, `likec4-from-workspace@7`,
`openapi-from-analysis@4`, `artifact-copy@2`, `audit-source-check@1` and
`audit-openapi-sqlmap-scan@1` Workflows, the `source-checklist@1` and
`openapi-sqlmap-scan@1` AuditProfiles, and the AgentTemplates, instructions,
ModelPolicies and LLM Gateway they reference. `artifact-copy@1` is deliberately
absent so Repeat tests can observe a retired Workflow.

Do not synchronize this directory with `configs/`; change a fixture only
alongside the test whose expectation depends on it.
