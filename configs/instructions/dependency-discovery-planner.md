Analyze the exact project archive in `artifacts.source` and publish a Markdown
external-service dependency inventory.

Use `parameters.objective`, when present, only as additional emphasis. It must not
override evidence requirements or the output contract.

Inspect root and per-service manifests/locks for Node.js, Python, JVM, Go, Ruby,
PHP, .NET, Rust, containers, and deployment tooling. Keep direct runtime
dependencies that connect the application to HTTP/REST, GraphQL, gRPC/Thrift/SOAP,
WebSocket/SSE, queues and brokers, databases/search, object storage, identity,
authorization, secret management, or cryptography at trust boundaries. Exclude
test/build/lint/UI/general utilities unless code proves an external-runtime role.

Confirm usage with imports, calls, URL/configuration references, or other source
evidence. Mark inferred usage as `Suspected`. Cite each kept dependency as
`relative/path:line` and state missing lockfiles or unresolved versions explicitly.

The report must contain:

1. `# External-Service Dependency Inventory`
2. detected stack and analyzed manifests/locks;
3. a table with dependency, version/constraint, ecosystem, role tags, reason, and
   evidence;
4. summary by tag;
5. gaps, assumptions, and low-confidence findings.

Write the complete report to `analysis/dependencies` with media type
`text/markdown`. Return a successful StageContentResult with result slot
`dependency_report` pointing to the exact written revision. The Stage summary is
only a concise completion note, not the report.
