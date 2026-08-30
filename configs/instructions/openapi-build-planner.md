Construct the Run-scoped OpenAPI document from the exact artifacts in this request.

Required inputs are `source`, `dependency_report`, and `project_report`; an exact
`existing_openapi` may also be present. Materialize source, read both reports, then
resume `openapi/openapi` if this is a retry. Otherwise copy the optional seed exactly,
or initialize a new document when no seed was supplied.

Discover the implemented inbound HTTP API and incrementally model supported info,
servers, top-level tags, reusable components, paths, operations, parameters,
request bodies, responses, status codes, and security. Every operation tag must
have a matching top-level declaration. If no deployment URL is evidenced, `.` is
the neutral relative current-origin server; do not invent a host or use `/`.
Resolve conflicts in favor of source code and record uncertainty in the concise
summary. Every path/component mutation must cite existing implementation files.
Use only OpenAPI domain mutations; never write the whole document through generic
or text artifact tools.

Before success, enumerate the resulting paths/components, tags, and servers, ensure
local references resolve, and run one validation pass. Return result slot `openapi` with the latest
exact `openapi/openapi` ArtifactRef and media type `application/yaml`.
