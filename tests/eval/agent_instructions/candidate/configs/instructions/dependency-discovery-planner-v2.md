Analyze the exact project archive supplied as the named `source` input and publish a Markdown
external-service dependency inventory.

Use the string parameter `objective`, when present, only as additional emphasis. It must not
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
`text/markdown`. Finish only after that durable write succeeds, using a concise
semantic completion result rather than the report body.
