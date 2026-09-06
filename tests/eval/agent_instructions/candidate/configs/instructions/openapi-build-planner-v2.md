Produce `openapi/openapi` (`application/yaml`) from the exact `source`,
`dependency_report`, and `project_report` inputs, preserving the optional exact
`existing_openapi` seed. The Worker owns source access, incremental construction,
retry recovery, and domain-tool validation.

Keep work centered on this artifact. Any delegated subtask must have a verifiable
outcome; do not split reading from construction or repeat completed work without
an unresolved requirement. Resolve report/source conflicts in favor of source.

Acceptance: implemented inbound method/path pairs are represented or have explicit
evidence gaps; operations/components cite source, local references resolve, and
tags and servers are consistent. The latest document is durable and one builder
validation pass is recorded. The following validation task owns final repair.
Return a concise semantic result with omissions and remaining validation issues.
