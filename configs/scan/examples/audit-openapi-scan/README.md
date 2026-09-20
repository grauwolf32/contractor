# Supplied OpenAPI scan inputs

These inputs exercise V62-009 preparation, inventory and trusted scan execution.
The catalog contains `openapi-sqlmap-scan@1` and `openapi-nuclei-scan@1` candidate
profiles; production-process acceptance is still pending.

Both settings files explicitly select `#/paths/~1pets~1{id}/post` and the local
server `http://127.0.0.1:8080/api`. The prepared URL is
`http://127.0.0.1:8080/api/pets/7?search=Milo`.

- SQLMap retains POST, the JSON body and the supplied fixture bearer token. It
  selects only the `name` and `search` test parameters. `id` fixes the path.
- Nuclei receives the URL. It does not replay POST, the JSON body or bearer
  authentication. Those limitations remain in the generated Audit task.
  Template selection is fixed by the resolved Workflow/Worker policy.

The inventory builder is `openapi-scans@1`. The consuming Workflow separately
declares `auditTask: {contract: openapi-scan@1, stage: scan}`. Resource names and
input aliases can change without changing Go code; the executor Stage may be
surrounded by ordinary Workflow Stages. Exact inputs, approval and the canonical
result producer remain enforced by the execution contract.

Run the preparation/inventory example check from the repository root:

```sh
go test ./internal/auditdomain -run '^TestOpenAPIScanDocumentedInputs$'
go test ./internal/planner/scan -run '^TestAuditExecutor'
```

This check performs no scanner or network calls. It verifies the concrete
request/URL and the exact source/settings provenance against the files here.
See the [adapter contract](../../../../docs/spec/openapi-audit-scans.md) for
execution, result and recovery requirements that remain before profile release.
